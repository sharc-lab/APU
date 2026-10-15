"""Queue job x2_model_pulls_scale (evo-x2): pulls the Ollama library tags llama3.3:70b (H-scale) and
mistral-small3.2:24b (B2 model family), verifies every manifest layer's blob against its digest, records the digests and
the chat template source, then exits. Pre-registered in docs/FINDINGS.md ("llama3.3:70b library pull and the B2 model
family", 2026-10-14); queued immediately before x2_scale_timing_probe. The harness/x2_model_pulls.py pattern
(pull_one / currently_present_tags / verify_one are reused from it).

Per tag:
  1. present (GET /api/tags, normalized) -> no pull. Otherwise a free-disk check on the drive of the Ollama model store:
     free bytes >= DISK_FACTOR x the library size + DISK_MARGIN_BYTES, else no pull and a failed row. Then
     `ollama pull <tag>` (PULL_ATTEMPTS attempts; ollama resumes partial blobs).
  2. pulled or present: the manifest on disk is read; every layer's blob (and the config blob) is checked for
     existence, size and sha256 against its digest. The row records the manifest's own sha256 (the ID `ollama list`
     shows the first 12 hex of), the config digest, and the model / params / template / license layer digests and
     sizes; the chat template source (chat_template_source.classify_manifest: a template layer -> ollama_library);
     the /api/show capabilities and template sha256; a 1-token chat call as the load check.
Never deletes anything: a digest mismatch is a failed row and nothing is removed. Idempotent: a re-run pulls nothing
that is present and re-verifies (hashing reads the blobs only).

Calls t2s_queue.advance() exactly once on exit (finally); sets no pause.

Usage (evo-x2, from C:\\apu\\ovn): python x2_model_pulls_scale.py --out results\\x2_model_pulls_scale.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if (_here.parent / "harness").is_dir():
    sys.path.insert(0, str(_here.parent / "harness"))

import chat_template_source as cts  # noqa: E402

# tag -> library size in bytes as listed on ollama.com on 2026-10-14 (results/ollama_library_evidence_20261014.json)
PULLS = {"llama3.3:70b": 43 * 10**9, "mistral-small3.2:24b": 15 * 10**9}
DISK_FACTOR = 1.1
DISK_MARGIN_BYTES = 100 * 10**9
PULL_ATTEMPTS = 2
PULL_TIMEOUT_S = 6 * 3600          # per attempt; x2_model_pulls' own default (1 h) is short for a 43 GB pull
HASH_CHUNK = 16 * 1024 * 1024
HASH_HEARTBEAT_S = 60
OUT_FILE = "x2_model_pulls_scale.jsonl"
JOB_ID = "x2_model_pulls_scale"
LAYER_KIND = {"application/vnd.ollama.image.model": "model", "application/vnd.ollama.image.params": "params",
              "application/vnd.ollama.image.template": "template", "application/vnd.ollama.image.license": "license",
              "application/vnd.ollama.image.system": "system", "application/vnd.ollama.image.projector": "projector"}


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


# ── pure parts (unit-tested) ────────────────────────────────────────────────────────────────────────

def required_free_bytes(tag: str) -> int:
    return int(PULLS[tag] * DISK_FACTOR) + DISK_MARGIN_BYTES


def blob_path(models_dir, digest: str) -> Path:
    return Path(models_dir) / "blobs" / digest.replace(":", "-")


def manifest_summary(tag: str, manifest: dict) -> dict:
    """Digests and sizes per layer kind (a kind with several layers keeps a list), the config digest, and the template
    source by chat_template_source.classify_manifest."""
    layers = {}
    for l in manifest.get("layers") or []:
        kind = LAYER_KIND.get(l.get("mediaType"), l.get("mediaType"))
        layers.setdefault(kind, []).append({"digest": l.get("digest"), "size": l.get("size")})
    cfg = manifest.get("config") or {}
    cls = cts.classify_manifest(tag, manifest)
    return {"config_digest": cfg.get("digest"), "config_size": cfg.get("size"),
            "layers": {k: (v[0] if len(v) == 1 else v) for k, v in layers.items()},
            "model_digest": (layers.get("model") or [{}])[0].get("digest"),
            "params_digest": (layers.get("params") or [{}])[0].get("digest"),
            "template_digest": (layers.get("template") or [{}])[0].get("digest"),
            "chat_template_source": cls["chat_template_source"],
            "chat_template_sha256": cls["template_layer_sha256"]}


def sha256_file(path, log=print, label="") -> str:
    h = hashlib.sha256()
    last = time.monotonic()
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(HASH_CHUNK)
            if not b:
                break
            h.update(b)
            done += len(b)
            if time.monotonic() - last >= HASH_HEARTBEAT_S:
                log(f"heartbeat: hashing {label} ({done / 1e9:.1f} GB read)")
                last = time.monotonic()
    return h.hexdigest()


def verify_blobs(models_dir, manifest: dict, hash_fn=None, log=print) -> dict:
    """Every layer blob and the config blob: exists, size equals the manifest size, sha256 equals the digest."""
    hash_fn = hash_fn or (lambda p, label: sha256_file(p, log, label))
    entries = list(manifest.get("layers") or [])
    if manifest.get("config"):
        entries.append(dict(manifest["config"], mediaType=manifest["config"].get("mediaType", "config")))
    checks = []
    for e in entries:
        d = e.get("digest") or ""
        p = blob_path(models_dir, d)
        c = {"kind": LAYER_KIND.get(e.get("mediaType"), e.get("mediaType")), "digest": d, "exists": p.exists()}
        if c["exists"]:
            c["size"] = p.stat().st_size
            c["size_ok"] = c["size"] == e.get("size")
            c["sha256_ok"] = d.startswith("sha256:") and hash_fn(p, d[:19]) == d.split(":", 1)[1]
        else:
            c.update(size=None, size_ok=False, sha256_ok=False)
        checks.append(c)
    return {"digests_ok": bool(checks) and all(c["exists"] and c["size_ok"] and c["sha256_ok"] for c in checks),
            "blob_checks": checks}


# ── job ─────────────────────────────────────────────────────────────────────────────────────────────

def _read_manifest(models_dir, tag):
    p = cts.manifest_path(models_dir, tag)
    raw = p.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), str(p)


def _show_info(tag, show_fn):
    try:
        s = show_fn(tag)
        tpl = s.get("template") or ""
        return {"api_show_capabilities": s.get("capabilities"),
                "api_show_template_sha256": hashlib.sha256(tpl.encode("utf-8")).hexdigest() if tpl else None,
                "api_show_details": s.get("details")}
    except Exception as e:
        return {"api_show_error": repr(e)[:200]}


def process_tag(tag, models_dir, present, emit, log, pull_fn, verify_load_fn, show_fn, free_fn, hash_fn=None):
    """One tag end to end; returns the result row (also emitted). Never deletes."""
    row = {"record": "model_pull_result", "model_tag": tag, "library_size_bytes": PULLS[tag], "ts_utc": utc_iso()}
    if any(cts.norm_tag(tag) == cts.norm_tag(p) for p in present):
        row.update(already_present=True, pulled=False, pull_attempts=0, pull_exit_code=None, pull_error=None)
        log(f"{tag}: already present, skipping pull")
    else:
        need = required_free_bytes(tag)
        free = free_fn(models_dir)
        row.update(already_present=False, free_bytes_before=free, required_free_bytes=need,
                   disk_rule=f"free >= {DISK_FACTOR} x {PULLS[tag]} + {DISK_MARGIN_BYTES}")
        if free is None or free < need:
            row.update(pulled=False, pull_attempts=0, pull_exit_code=None,
                       pull_error=f"not pulled: free bytes {free} < required {need}", digests_ok=False,
                       verify_ok=False)
            log(f"{tag}: {row['pull_error']}")
            emit(row)
            return row
        res = {}
        for attempt in range(1, PULL_ATTEMPTS + 1):
            log(f"{tag}: pulling (attempt {attempt})")
            res = pull_fn(tag)
            row["pull_attempts"] = attempt
            if res.get("pull_exit_code") == 0:
                break
        row.update(pulled=res.get("pull_exit_code") == 0, pull_exit_code=res.get("pull_exit_code"),
                   pull_error=res.get("pull_error"))
        row["free_bytes_after"] = free_fn(models_dir)
    try:
        manifest, msha, mpath = _read_manifest(models_dir, tag)
        row.update(manifest_sha256=msha, manifest_id12=msha[:12], manifest_path_tail="/".join(Path(mpath).parts[-4:]))
        row.update(manifest_summary(tag, manifest))
        row.update(verify_blobs(models_dir, manifest, hash_fn=hash_fn, log=log))
    except Exception as e:
        row.update(digests_ok=False, manifest_error=repr(e)[:300])
    row.update(_show_info(tag, show_fn))
    if row.get("digests_ok"):
        row.update(verify_load_fn(tag))
    else:
        row.update(verify_ok=False, verify_error="not attempted: digests not verified")
    log(f"{tag}: present={row['already_present']} pulled={row['pulled']} digests_ok={row.get('digests_ok')} "
        f"template={row.get('chat_template_source')} verify_ok={row.get('verify_ok')}")
    emit(row)
    return row


def run(out_path, log=print, hc=None, mp=None, show_fn=None, free_fn=None, models_dir=None, hash_fn=None):
    """Starts Ollama, processes every tag in PULLS, stops Ollama. Dependencies are injectable for tests."""
    if hc is None:
        import host_config as hc
    if mp is None:
        import x2_model_pulls as mp
        mp.PULL_TIMEOUT_S = PULL_TIMEOUT_S
    show_fn = show_fn or cts._show
    free_fn = free_fn or (lambda d: shutil.disk_usage(d).free)
    models_dir = models_dir or cts._models_dir()

    def emit(row):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    import socket
    free0 = None
    if models_dir:
        try:
            free0 = free_fn(models_dir)
        except Exception:
            free0 = None
    emit({"record": "run_start", "job_id": JOB_ID, "host": socket.gethostname(), "tags": list(PULLS),
          "disk_factor": DISK_FACTOR, "disk_margin_bytes": DISK_MARGIN_BYTES, "models_dir_set": bool(models_dir),
          "free_bytes_at_start": free0, "evidence": "results/ollama_library_evidence_20261014.json",
          "ts_utc": utc_iso()})
    log(f"free disk on the model store drive at start: {free0} bytes")
    if not models_dir:
        raise RuntimeError("Ollama model store directory unknown (OLLAMA_MODELS unset and no host entry)")
    hc.start_ollama_server()
    try:
        if not hc.wait_for_ollama_ready(timeout_s=60):
            raise RuntimeError("ollama did not become ready")
        present = mp.currently_present_tags()
        rows = [process_tag(tag, models_dir, present, emit, log, lambda t: mp.pull_one(t, out_path, log),
                            mp.verify_one, show_fn, free_fn, hash_fn) for tag in PULLS]
        all_ok = all(r.get("digests_ok") and r.get("verify_ok") for r in rows)
        emit({"record": "run_end", "all_models_verified": all_ok, "n_models": len(rows), "ts_utc": utc_iso()})
        return all_ok, rows
    finally:
        hc.stop_ollama_server()


def main(argv=None, advance=True):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out_path = Path(args.out) if args.out else _here / "results" / OUT_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {out_path}", flush=True)
    note = "completed"
    try:
        all_ok, rows = run(out_path)
        if not all_ok:
            failed = [r["model_tag"] for r in rows if not (r.get("digests_ok") and r.get("verify_ok"))]
            note = f"completed; models not verified: {failed}"
        print(f"done: {len(rows)} models, all_ok={all_ok}", flush=True)
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        print(note, flush=True)
        traceback.print_exc()
    finally:
        if advance:
            try:
                import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                print(f"queue advance failed: {e!r}", flush=True)
    return note


if __name__ == "__main__":
    main()
