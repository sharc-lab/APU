"""Minimal Level Zero Sysman reader (ctypes) for iGPU frequency and package power.

Reads, once per call to sample():
  zesFrequencyGetState  -> actual, request, tdp, efficient MHz, throttle reason bit mask
  zesPowerGetEnergyCounter -> energy (uJ) and timestamp (us); power_w = delta energy / delta time (uJ per us = W)

Struct layouts and enum values follow zes_api.h. They could not be checked against a header on the target machine, so
the smoke test must confirm that every call returns ZE_RESULT_SUCCESS (0) and sane values. Any failure is recorded in
self.errors and available stays False for that part; nothing raises, so the experiment can fall back to Windows
counters and say so.
"""

from __future__ import annotations

import ctypes
import os
import time

STYPE_FREQ_PROPERTIES = 0x9
STYPE_POWER_PROPERTIES = 0xD
STYPE_FREQ_STATE = 0x1B


class FreqProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int32), ("pNext", ctypes.c_void_p), ("type", ctypes.c_int32),
                ("onSubdevice", ctypes.c_uint8), ("subdeviceId", ctypes.c_uint32), ("canControl", ctypes.c_uint8),
                ("isThrottleEventSupported", ctypes.c_uint8), ("min", ctypes.c_double), ("max", ctypes.c_double)]


class FreqState(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int32), ("pNext", ctypes.c_void_p), ("currentVoltage", ctypes.c_double),
                ("request", ctypes.c_double), ("tdp", ctypes.c_double), ("efficient", ctypes.c_double),
                ("actual", ctypes.c_double), ("throttleReasons", ctypes.c_uint32)]


class PowerProps(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int32), ("pNext", ctypes.c_void_p), ("onSubdevice", ctypes.c_uint8),
                ("subdeviceId", ctypes.c_uint32), ("canControl", ctypes.c_uint8),
                ("isEnergyThresholdSupported", ctypes.c_uint8), ("defaultLimit", ctypes.c_int32),
                ("minLimit", ctypes.c_int32), ("maxLimit", ctypes.c_int32)]


class EnergyCounter(ctypes.Structure):
    _fields_ = [("energy", ctypes.c_uint64), ("timestamp", ctypes.c_uint64)]


class Sysman:
    def __init__(self):
        os.environ["ZES_ENABLE_SYSMAN"] = "1"
        self.errors: list[str] = []
        self.freq_handles: list = []
        self.power_handles: list = []
        self.temp_handles: list = []
        self.freq_props: list[dict] = []
        self.power_props: list[dict] = []
        self.n_drivers = self.n_devices = 0
        self._last_energy: dict[int, tuple[int, int]] = {}
        self.available = False
        try:
            self._init()
        except Exception as e:
            self.errors.append(f"init exception: {e!r}")

    def _fn(self, name, argtypes):
        f = getattr(self.lib, name)
        f.argtypes = argtypes
        f.restype = ctypes.c_int32
        return f

    def _init(self):
        self.lib = ctypes.WinDLL("ze_loader.dll")
        u32p = ctypes.POINTER(ctypes.c_uint32)
        vpp = ctypes.POINTER(ctypes.c_void_p)
        rc = self._fn("zesInit", [ctypes.c_uint32])(0)
        if rc != 0:
            self.errors.append(f"zesInit rc={rc:#x}")
            return
        drv_get = self._fn("zesDriverGet", [u32p, vpp])
        n = ctypes.c_uint32(0)
        rc = drv_get(ctypes.byref(n), None)
        self.n_drivers = n.value
        if rc != 0 or n.value == 0:
            self.errors.append(f"zesDriverGet rc={rc:#x} count={n.value}")
            return
        drivers = (ctypes.c_void_p * n.value)()
        drv_get(ctypes.byref(n), drivers)
        dev_get = self._fn("zesDeviceGet", [ctypes.c_void_p, u32p, vpp])
        enum_f = self._fn("zesDeviceEnumFrequencyDomains", [ctypes.c_void_p, u32p, vpp])
        enum_p = self._fn("zesDeviceEnumPowerDomains", [ctypes.c_void_p, u32p, vpp])
        enum_t = self._fn("zesDeviceEnumTemperatureSensors", [ctypes.c_void_p, u32p, vpp])
        self._temp_state = self._fn("zesTemperatureGetState", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)])
        self._freq_props = self._fn("zesFrequencyGetProperties", [ctypes.c_void_p, ctypes.POINTER(FreqProps)])
        self._freq_state = self._fn("zesFrequencyGetState", [ctypes.c_void_p, ctypes.POINTER(FreqState)])
        self._power_props = self._fn("zesPowerGetProperties", [ctypes.c_void_p, ctypes.POINTER(PowerProps)])
        self._energy = self._fn("zesPowerGetEnergyCounter", [ctypes.c_void_p, ctypes.POINTER(EnergyCounter)])
        for d in range(n.value):
            nd = ctypes.c_uint32(0)
            rc = dev_get(drivers[d], ctypes.byref(nd), None)
            if rc != 0 or nd.value == 0:
                self.errors.append(f"zesDeviceGet rc={rc:#x} count={nd.value}")
                continue
            devs = (ctypes.c_void_p * nd.value)()
            dev_get(drivers[d], ctypes.byref(nd), devs)
            self.n_devices += nd.value
            for i in range(nd.value):
                self._enum(enum_f, devs[i], self.freq_handles)
                self._enum(enum_p, devs[i], self.power_handles)
                self._enum(enum_t, devs[i], self.temp_handles)
        for h in self.freq_handles:
            fp = FreqProps(stype=STYPE_FREQ_PROPERTIES)
            rc = self._freq_props(h, ctypes.byref(fp))
            self.freq_props.append({"rc": rc, "type": fp.type, "onSubdevice": fp.onSubdevice, "min": fp.min, "max": fp.max})
        for h in self.power_handles:
            pp = PowerProps(stype=STYPE_POWER_PROPERTIES)
            rc = self._power_props(h, ctypes.byref(pp))
            self.power_props.append({"rc": rc, "onSubdevice": pp.onSubdevice, "canControl": pp.canControl,
                                     "defaultLimit_mW": pp.defaultLimit, "minLimit_mW": pp.minLimit,
                                     "maxLimit_mW": pp.maxLimit})
        self.available = bool(self.freq_handles or self.power_handles)
        if not self.available:
            self.errors.append("no frequency or power domains enumerated")

    def _enum(self, fn, dev, out):
        c = ctypes.c_uint32(0)
        rc = fn(dev, ctypes.byref(c), None)
        if rc != 0 or c.value == 0:
            self.errors.append(f"{fn.__name__} rc={rc:#x} count={c.value}")
            return
        arr = (ctypes.c_void_p * c.value)()
        fn(dev, ctypes.byref(c), arr)
        out.extend(arr[i] for i in range(c.value))

    def sample(self) -> list[dict]:
        rows = []
        if not self.available:
            return rows
        for i, h in enumerate(self.freq_handles):
            st = FreqState(stype=STYPE_FREQ_STATE)
            rc = self._freq_state(h, ctypes.byref(st))
            rows.append({"kind": "freq", "domain": i, "rc": rc, "actual_mhz": st.actual, "request_mhz": st.request,
                         "tdp_mhz": st.tdp, "efficient_mhz": st.efficient, "throttle_reasons": st.throttleReasons})
        for i, h in enumerate(self.power_handles):
            ec = EnergyCounter()
            rc = self._energy(h, ctypes.byref(ec))
            power = None
            prev = self._last_energy.get(i)
            if rc == 0 and prev and ec.timestamp > prev[1]:
                power = (ec.energy - prev[0]) / (ec.timestamp - prev[1])
            if rc == 0:
                self._last_energy[i] = (ec.energy, ec.timestamp)
            rows.append({"kind": "power", "domain": i, "rc": rc, "energy_uj": ec.energy, "timestamp_us": ec.timestamp,
                         "power_w": power})
        for i, h in enumerate(self.temp_handles):
            t = ctypes.c_double(0.0)
            rc = self._temp_state(h, ctypes.byref(t))
            rows.append({"kind": "temp", "domain": i, "rc": rc, "temp_c": t.value if rc == 0 else None})
        return rows


if __name__ == "__main__":
    s = Sysman()
    print("available", s.available, "drivers", s.n_drivers, "devices", s.n_devices)
    print("freq props", s.freq_props)
    print("power props", s.power_props)
    print("errors", s.errors)
    for _ in range(3):
        print(s.sample())
        time.sleep(1)
