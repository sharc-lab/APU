"""Read-only Windows CPU topology: which logical processors share an L2 cache, via
GetLogicalProcessorInformationEx(RelationCache). Needed because B4's E-core cluster masks must match the real L2
groups, not an assumed 4-7/8-11 split -- P-cores typically have a private L2 per core, E-cores share one L2 across a
cluster of 4 on many Intel hybrid parts, but the exact grouping and which logical IDs it covers is model-specific and
must be read, not assumed.

Usage: python win_cpu_topology.py   (prints one JSON document; changes nothing)
"""

from __future__ import annotations

import ctypes
import json
import sys

RelationCache = 2
LOGICAL_PROCESSOR_RELATIONSHIP = ctypes.c_int


class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [("Mask", ctypes.c_uint64), ("Group", ctypes.c_uint16), ("Reserved", ctypes.c_uint16 * 3)]


class CACHE_RELATIONSHIP(ctypes.Structure):
    _fields_ = [("Level", ctypes.c_uint8), ("Associativity", ctypes.c_uint8), ("LineSize", ctypes.c_uint16),
                ("CacheSize", ctypes.c_uint32), ("Type", ctypes.c_int), ("Reserved", ctypes.c_uint8 * 20),
                ("GroupCount", ctypes.c_uint16), ("GroupMask", GROUP_AFFINITY)]


class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
    _fields_ = [("Relationship", LOGICAL_PROCESSOR_RELATIONSHIP), ("Size", ctypes.c_uint32),
                ("Cache", CACHE_RELATIONSHIP)]


def read_cache_groups():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_uint32(0)
    k32.GetLogicalProcessorInformationEx(RelationCache, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    ok = k32.GetLogicalProcessorInformationEx(RelationCache, buf, ctypes.byref(size))
    if not ok:
        return {"error": f"GetLogicalProcessorInformationEx failed, GetLastError={ctypes.get_last_error()}"}
    groups = []
    offset = 0
    while offset < size.value:
        entry = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX.from_buffer_copy(buf, offset)
        if entry.Relationship == RelationCache and entry.Cache.Level == 2:
            mask = entry.Cache.GroupMask.Mask
            cpus = [i for i in range(64) if mask & (1 << i)]
            groups.append({"logical_cpus": cpus, "cache_size_kb": entry.Cache.CacheSize // 1024,
                          "type": entry.Cache.Type})
        offset += entry.Size
    return {"l2_groups": sorted(groups, key=lambda g: g["logical_cpus"][0] if g["logical_cpus"] else -1)}


if __name__ == "__main__":
    print(json.dumps(read_cache_groups(), indent=1))
