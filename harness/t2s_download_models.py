"""Resumable model downloader for the evo-t2s overnight run. Runs on evo-t2s (uses curl.exe).

For each URL: HEAD for the published SHA-256 (the x-linked-etag of a Hugging Face LFS file) and size, then
`curl.exe -L -C -` with up to 3 resumes, then SHA-256 of the file compared with the published value. One JSON line per
event is appended to the log with fsync. Two workers run in parallel, each taking its own list in order. A model that
fails after 3 resumes is dropped and logged; the run continues.

Usage: python t2s_download_models.py <log.jsonl>
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

DEST = r"C:\apu\models"
MODELS = {
    "Qwen3-8B-Q4_K_M.gguf": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf":
        "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "Qwen3-14B-Q4_K_M.gguf": "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf",
    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf":
        "https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
    "Qwen3-32B-Q4_K_M.gguf": "https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_K_M.gguf",
}
WORKERS = [["Qwen3-8B-Q4_K_M.gguf", "Qwen3-14B-Q4_K_M.gguf", "Qwen3-32B-Q4_K_M.gguf"],
           ["Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"]]
_lock = threading.Lock()


def log(path, row):
    row = {"ts_utc": datetime.now(timezone.utc).isoformat(), **row}
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def head(url):
    r = subprocess.run(["curl.exe", "-sIL", url], capture_output=True, text=True, timeout=120)
    etag = None
    size = None
    for line in r.stdout.splitlines():
        low = line.lower()
        if low.startswith("x-linked-etag:"):
            m = re.search(r'"?([0-9a-f]{64})"?', line)
            etag = m.group(1) if m else None
        if low.startswith("x-linked-size:"):
            size = int(line.split(":", 1)[1].strip())
    return etag, size


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def fetch(name, logp):
    url = MODELS[name]
    path = os.path.join(DEST, name)
    etag, size = head(url)
    log(logp, {"event": "start", "model": name, "url": url, "published_sha256": etag, "published_size": size})
    t0 = time.time()
    for attempt in range(1, 5):
        r = subprocess.run(["curl.exe", "-L", "-C", "-", "-sS", "--retry", "2", "-o", path, url],
                           capture_output=True, text=True)
        have = os.path.getsize(path) if os.path.exists(path) else 0
        log(logp, {"event": "curl_exit", "model": name, "attempt": attempt, "rc": r.returncode, "bytes": have,
                   "stderr": r.stderr[-300:]})
        if r.returncode == 0 and (size is None or have == size):
            break
    else:
        log(logp, {"event": "dropped", "model": name, "reason": "download failed after 3 resumes"})
        return
    got = sha256_file(path)
    ok = etag is None or got == etag
    log(logp, {"event": "done" if ok else "sha_mismatch", "model": name, "sha256": got, "published_sha256": etag,
               "bytes": os.path.getsize(path), "seconds": round(time.time() - t0, 1), "verified_against_published": etag is not None})


def main():
    logp = sys.argv[1]
    os.makedirs(DEST, exist_ok=True)
    ts = [threading.Thread(target=lambda ws=w: [fetch(n, logp) for n in ws]) for w in WORKERS]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log(logp, {"event": "all_finished"})


if __name__ == "__main__":
    main()
