"""Windows memory-tier readouts for the CC cells (Task BM, amendment A19).

- GlobalMemoryStatusEx: ullTotalPhys / ullAvailPhys.
- DXGI adapters (IDXGIFactory1::EnumAdapters1 -> GetDesc1): DedicatedVideoMemory,
  DedicatedSystemMemory, SharedSystemMemory.
- IDXGIAdapter3::QueryVideoMemoryInfo for the CALLING process, LOCAL and NON_LOCAL:
  Budget, CurrentUsage, AvailableForReservation, CurrentReservation.
- BudgetWatch: RegisterVideoMemoryBudgetChangeNotificationEvent on an auto-reset event;
  poll() counts signals.

Plain ctypes COM through vtable slots; no extra package. Every function returns an
"error" field instead of raising, and off Windows returns {"error": "not windows"}.
Vtable slots (dxgi.h / dxgi1_4.h): IDXGIFactory1::EnumAdapters1 = 12;
IDXGIAdapter1::GetDesc1 = 10; IDXGIAdapter3::QueryVideoMemoryInfo = 14,
RegisterVideoMemoryBudgetChangeNotificationEvent = 16,
UnregisterVideoMemoryBudgetChangeNotification = 17; IUnknown QueryInterface = 0, Release = 2.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any

DXGI_ERROR_NOT_FOUND = 0x887A0002
SEGMENT_GROUPS = {"LOCAL": 0, "NON_LOCAL": 1}
INTEL_VENDOR = 0x8086
SOFTWARE_FLAG = 2  # DXGI_ADAPTER_FLAG_SOFTWARE


class _GUID(ctypes.Structure):
    _fields_ = [  # noqa: RUF012 (ctypes needs a plain class attribute)
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> _GUID:
    h = text.replace("-", "")
    g = _GUID()
    g.Data1 = int(h[0:8], 16)
    g.Data2 = int(h[8:12], 16)
    g.Data3 = int(h[12:16], 16)
    for i in range(8):
        g.Data4[i] = int(h[16 + 2 * i : 18 + 2 * i], 16)
    return g


IID_IDXGIFactory1 = "770aae78-f26f-4dba-a829-253c83d1b387"
IID_IDXGIAdapter3 = "645967a4-1392-4310-a798-8053ce3e93fd"


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]  # noqa: RUF012 (ctypes needs a plain class attribute)


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [  # noqa: RUF012 (ctypes needs a plain class attribute)
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint32),
    ]


class _VidmemInfo(ctypes.Structure):
    _fields_ = [  # noqa: RUF012 (ctypes needs a plain class attribute)
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
    ]


class _MemStatusEx(ctypes.Structure):
    _fields_ = [  # noqa: RUF012 (ctypes needs a plain class attribute)
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def _vcall(obj: ctypes.c_void_p, slot: int, argtypes: list[Any], *args: Any) -> int:
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)  # type: ignore[attr-defined]
    fn = proto(vtbl[slot])
    return int(fn(obj, *args)) & 0xFFFFFFFF


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])(obj)  # type: ignore[attr-defined]


def global_memory() -> dict[str, Any]:
    if sys.platform != "win32":
        return {"error": "not windows"}
    try:
        st = _MemStatusEx()
        st.dwLength = ctypes.sizeof(_MemStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return {"error": "GlobalMemoryStatusEx failed"}
        return {"ullTotalPhys": int(st.ullTotalPhys), "ullAvailPhys": int(st.ullAvailPhys)}
    except Exception as exc:  # recorded, never raised
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def _factory() -> ctypes.c_void_p:
    dxgi = ctypes.windll.dxgi  # type: ignore[attr-defined]
    fac = ctypes.c_void_p()
    iid = _guid(IID_IDXGIFactory1)
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(fac)) & 0xFFFFFFFF
    if hr != 0:
        raise OSError(f"CreateDXGIFactory1 hr=0x{hr:08x}")
    return fac


def _adapters(fac: ctypes.c_void_p) -> list[tuple[ctypes.c_void_p, dict[str, Any]]]:
    out = []
    i = 0
    while True:
        ad = ctypes.c_void_p()
        hr = _vcall(fac, 12, [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)], i, ctypes.byref(ad))
        if hr == DXGI_ERROR_NOT_FOUND:
            break
        if hr != 0:
            raise OSError(f"EnumAdapters1({i}) hr=0x{hr:08x}")
        d = _AdapterDesc1()
        hr = _vcall(ad, 10, [ctypes.POINTER(_AdapterDesc1)], ctypes.byref(d))
        info: dict[str, Any] = {"index": i, "hr_desc": hr}
        if hr == 0:
            info.update(
                {
                    "description": d.Description,
                    "vendor_id": int(d.VendorId),
                    "dedicated_video": int(d.DedicatedVideoMemory),
                    "dedicated_system": int(d.DedicatedSystemMemory),
                    "shared_system": int(d.SharedSystemMemory),
                    "software": bool(d.Flags & SOFTWARE_FLAG),
                }
            )
        out.append((ad, info))
        i += 1
    return out


def _query(adapter: ctypes.c_void_p) -> dict[str, Any]:
    a3 = ctypes.c_void_p()
    iid = _guid(IID_IDXGIAdapter3)
    hr = _vcall(
        adapter,
        0,
        [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)],
        ctypes.byref(iid),
        ctypes.byref(a3),
    )
    if hr != 0:
        return {"error": f"QueryInterface(IDXGIAdapter3) hr=0x{hr:08x}"}
    try:
        res: dict[str, Any] = {}
        for name, group in SEGMENT_GROUPS.items():
            info = _VidmemInfo()
            hr = _vcall(
                a3,
                14,
                [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(_VidmemInfo)],
                0,
                group,
                ctypes.byref(info),
            )
            res[name] = (
                {
                    "Budget": int(info.Budget),
                    "CurrentUsage": int(info.CurrentUsage),
                    "AvailableForReservation": int(info.AvailableForReservation),
                    "CurrentReservation": int(info.CurrentReservation),
                }
                if hr == 0
                else {"error": f"QueryVideoMemoryInfo hr=0x{hr:08x}"}
            )
        return res
    finally:
        _release(a3)


def snapshot() -> dict[str, Any]:
    """System memory, every adapter's desc, and this process's budget per adapter."""
    if sys.platform != "win32":
        return {"error": "not windows"}
    out: dict[str, Any] = {"global": global_memory(), "adapters": []}
    fac = ctypes.c_void_p()
    try:
        fac = _factory()
        for ad, info in _adapters(fac):
            try:
                if not info.get("software"):
                    info["budget"] = _query(ad)
            finally:
                _release(ad)
            out["adapters"].append(info)
    except Exception as exc:  # recorded, never raised
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        _release(fac)
    return out


def primary_budget() -> dict[str, Any]:
    """The first non-software Intel adapter's budget for this process (others if no Intel)."""
    snap = snapshot()
    ads = [a for a in snap.get("adapters", []) if not a.get("software") and "budget" in a]
    pick = next((a for a in ads if a.get("vendor_id") == INTEL_VENDOR), ads[0] if ads else None)
    if pick is None:
        return {"error": snap.get("error") or "no hardware adapter"}
    return {"index": pick["index"], "shared_system": pick.get("shared_system"), **pick["budget"]}


class BudgetWatch:
    """Counts budget-change notifications for the primary adapter (auto-reset event)."""

    def __init__(self) -> None:
        self.count = 0
        self.error: str | None = None
        self._a3 = ctypes.c_void_p()
        self._fac = ctypes.c_void_p()
        self._event: int | None = None
        self._cookie = ctypes.c_uint32(0)
        if sys.platform != "win32":
            self.error = "not windows"
            return
        try:
            self._open()
        except Exception as exc:  # recorded, never raised
            self.error = f"{type(exc).__name__}: {exc}"[:300]

    def _open(self) -> None:
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k32.CreateEventW.restype = ctypes.c_void_p
        self._event = k32.CreateEventW(None, False, False, None)
        self._fac = _factory()
        ads = _adapters(self._fac)
        pick = next(
            (ad for ad, i in ads if i.get("vendor_id") == INTEL_VENDOR and not i.get("software")),
            ads[0][0] if ads else None,
        )
        for ad, _ in ads:
            if ad is not pick:
                _release(ad)
        if pick is None:
            raise OSError("no adapter")
        iid = _guid(IID_IDXGIAdapter3)
        hr = _vcall(
            pick,
            0,
            [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)],
            ctypes.byref(iid),
            ctypes.byref(self._a3),
        )
        _release(pick)
        if hr != 0:
            raise OSError(f"QueryInterface hr=0x{hr:08x}")
        hr = _vcall(
            self._a3,
            16,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
            self._event,
            ctypes.byref(self._cookie),
        )
        if hr != 0:
            raise OSError(f"RegisterVideoMemoryBudgetChangeNotificationEvent hr=0x{hr:08x}")

    def poll(self) -> int:
        if self.error or self._event is None:
            return self.count
        if ctypes.windll.kernel32.WaitForSingleObject(ctypes.c_void_p(self._event), 0) == 0:  # type: ignore[attr-defined]
            self.count += 1
        return self.count

    def query(self) -> dict[str, Any]:
        """This process's LOCAL / NON_LOCAL budget and usage on the watched adapter."""
        if self.error or not self._a3:
            return {"error": self.error or "no adapter"}
        res: dict[str, Any] = {}
        for name, group in SEGMENT_GROUPS.items():
            info = _VidmemInfo()
            hr = _vcall(
                self._a3,
                14,
                [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(_VidmemInfo)],
                0,
                group,
                ctypes.byref(info),
            )
            if hr == 0:
                res[name] = (int(info.Budget), int(info.CurrentUsage))
        return res

    def close(self) -> None:
        if sys.platform != "win32":
            return
        try:
            if self._a3:
                _vcall(self._a3, 17, [ctypes.c_uint32], self._cookie.value)
                _release(self._a3)
            _release(self._fac)
            if self._event is not None:
                ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self._event))  # type: ignore[attr-defined]
        except Exception as exc:  # recorded, never raised
            self.error = f"close {type(exc).__name__}: {exc}"[:300]
