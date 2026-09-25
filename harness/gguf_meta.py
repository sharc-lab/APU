"""Minimal GGUF metadata reader (v2 and v3). Reads the key/value section only, never the tensors.

    meta = read_gguf_meta(path)
    kv_bytes_per_token_f16(meta)   # 2 (K and V) x n_layer x n_head_kv x head_dim x 2 bytes
"""

from __future__ import annotations

import struct

_SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4),
            7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_MAX_ARRAY_KEEP = 16


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, typ):
    if typ in _SCALARS:
        fmt, size = _SCALARS[typ]
        return struct.unpack(fmt, f.read(size))[0]
    if typ == 8:
        return _read_str(f)
    if typ == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        keep = []
        if et in _SCALARS:
            fmt, size = _SCALARS[et]
            blob = f.read(size * n)
            if n <= _MAX_ARRAY_KEEP:
                keep = list(struct.unpack("<" + fmt[1] * n, blob))
        else:
            for i in range(n):
                v = _read_value(f, et)
                if i < _MAX_ARRAY_KEEP:
                    keep.append(v)
        return {"array_len": n, "head": keep}
    raise ValueError(f"unknown GGUF value type {typ}")


def read_gguf_meta(path: str) -> dict:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
        meta = {"_version": version, "_tensor_count": tensor_count}
        for _ in range(kv_count):
            key = _read_str(f)
            (typ,) = struct.unpack("<I", f.read(4))
            meta[key] = _read_value(f, typ)
    return meta


def arch_params(meta: dict) -> dict:
    a = meta["general.architecture"]
    g = lambda k, d=None: meta.get(f"{a}.{k}", d)
    n_head = g("attention.head_count")
    n_embd = g("embedding_length")
    head_dim = g("attention.key_length") or (n_embd // n_head if n_head and n_embd else None)
    return {"arch": a, "name": meta.get("general.name"), "n_layer": g("block_count"), "n_head": n_head,
            "n_head_kv": g("attention.head_count_kv") or n_head, "head_dim": head_dim, "n_embd": n_embd,
            "context_length": g("context_length"), "expert_count": g("expert_count"),
            "expert_used_count": g("expert_used_count"), "file_type": meta.get("general.file_type")}


def kv_bytes_per_token_f16(meta: dict) -> int:
    p = arch_params(meta)
    return 2 * p["n_layer"] * p["n_head_kv"] * p["head_dim"] * 2


if __name__ == "__main__":
    import json
    import sys
    m = read_gguf_meta(sys.argv[1])
    p = arch_params(m)
    p["kv_bytes_per_token_f16"] = kv_bytes_per_token_f16(m)
    p["basename"] = m.get("general.basename")
    p["finetune"] = m.get("general.finetune")
    p["size_label"] = m.get("general.size_label")
    print(json.dumps(p, indent=1))
