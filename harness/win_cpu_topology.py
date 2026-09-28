"""Read-only Windows CPU topology: which logical processors share an L2 cache, via
GetLogicalProcessorInformationEx(RelationCache). Needed because B4's E-core cluster masks must match the real L2
groups, not an assumed 4-7/8-11 split -- P-cores typically have a private L2 per core, E-cores share one L2 across a
cluster on many Intel hybrid parts, but the exact grouping and which logical IDs it covers is model-specific and must
be read, not assumed.

Parses the returned buffer with struct.unpack_from rather than a ctypes.Structure mirror of
SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX/CACHE_RELATIONSHIP: only the 8-byte (Relationship: int32, Size: uint32)
header is trusted for advancing through the buffer (unambiguous, no compiler padding to get wrong). The
CACHE_RELATIONSHIP field offsets below were confirmed live (not assumed) by dumping raw entry bytes on evo-t2s and
matching known values (L1D 48 KB, L1I 64 KB, L2 3072/4096 KB, L3 18432 KB shared across a 0xfff mask):
Level/Associativity/LineSize/CacheSize/Type at entry-relative 8..40 (=8+1+1+2+4+4+18+2, i.e. Reserved really is
BYTE[18]), then GroupMask.Mask (UINT64) at entry-relative 40..48 with NO padding -- offset 40 is already 8-byte
aligned, so the compiler inserts none.

Usage: python win_cpu_topology.py   (prints one JSON document; changes nothing)
"""

from __future__ import annotations

import ctypes
import json
import struct
import sys

RelationCache = 2
HEADER_FMT = "<iI"  # Relationship (int32), Size (uint32)
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CACHE_FMT = "<BBHIi18sHQ"  # Level,Assoc,LineSize,CacheSize,Type,Reserved[18],GroupCount,Mask(uint64) -- no padding
CACHE_MIN_SIZE = struct.calcsize(CACHE_FMT)


def read_cache_groups():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_uint32(0)
    k32.GetLogicalProcessorInformationEx(RelationCache, None, ctypes.byref(size))
    if size.value == 0:
        return {"error": f"GetLogicalProcessorInformationEx (size query) returned 0, GetLastError={ctypes.get_last_error()}"}
    buf = ctypes.create_string_buffer(size.value)
    ok = k32.GetLogicalProcessorInformationEx(RelationCache, buf, ctypes.byref(size))
    if not ok:
        return {"error": f"GetLogicalProcessorInformationEx failed, GetLastError={ctypes.get_last_error()}"}
    raw = buf.raw
    groups = []
    offset = 0
    while offset + HEADER_SIZE <= size.value:
        relationship, entry_size = struct.unpack_from(HEADER_FMT, raw, offset)
        if entry_size <= 0:
            break
        if relationship == RelationCache and offset + HEADER_SIZE + CACHE_MIN_SIZE <= size.value:
            level, assoc, line_size, cache_size, ctype, _reserved, group_count, mask = \
                struct.unpack_from(CACHE_FMT, raw, offset + HEADER_SIZE)
            if level == 2:
                cpus = [i for i in range(64) if mask & (1 << i)]
                groups.append({"logical_cpus": cpus, "cache_size_kb": cache_size // 1024, "type": ctype,
                              "associativity": assoc, "line_size": line_size, "group_count": group_count})
        offset += entry_size
    return {"l2_groups": sorted(groups, key=lambda g: g["logical_cpus"][0] if g["logical_cpus"] else -1),
            "buffer_bytes": size.value}


if __name__ == "__main__":
    print(json.dumps(read_cache_groups(), indent=1))
