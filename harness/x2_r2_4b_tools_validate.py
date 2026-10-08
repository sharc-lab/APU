"""Queue job x2_r2_4b_tools_validate (evo-x2, 2026-10-08): create the Ollama tag qwen3-4b-2507-tools, then run the R2
validation for that tag alone into results/x2_r2_validation_v2c.jsonl.

Why: in x2_r2_validation_v2b, qwen3-4b-2507 (an Ollama tag created from a bare GGUF: Modelfile "FROM <gguf>" only, no
TEMPLATE) wrote its tool calls as text inside its JSON answer, while qwen3:8b (library template) made native calls
(register R2-validation-v2b-runinfo). qwen3-4b-2507-tools holds the same weights with the library Qwen3 template, so
the comparison isolates the install path. The existing qwen3-4b-2507 tag is never modified (the outcome table uses it);
this job checks its manifest bytes are unchanged afterwards.

Step 1, create (idempotent):
  - source GGUF: x2_outcome_table.MODEL_MAP["qwen3-4b-2507"][1]; its sha256 must equal the model-layer digest of the
    qwen3-4b-2507 manifest (same blob, checked here, not assumed);
  - TEMPLATE: POST /api/show {"model": "qwen3:8b"} "template", verbatim; its sha256 must be LIBRARY_TEMPLATE_SHA256
    (the template blob of qwen3:8b, qwen3:14b and qwen3:32b on evo-x2, read 2026-10-08);
  - PARAMETER stop: every "stop" line of qwen3:8b's /api/show "parameters" (the template's turn delimiters
    <|im_start|> and <|im_end|>); the sampling parameters (temperature, top_k, top_p, repeat_penalty) are NOT copied,
    every R2 request sets temperature 0 and seed 42 itself;
  - Modelfile = "FROM <gguf>" + a TEMPLATE block (the template between triple quotes) + the stop lines, written next to --out and recorded verbatim
    in the result file ("r2t_create" record) with the template sha256;
  - `ollama create qwen3-4b-2507-tools -f <Modelfile>` only if the tag is absent or fails verification; verification
    (before and after): manifest model layer = the source digest, /api/show template sha256 = LIBRARY_TEMPLATE_SHA256,
    /api/show stop list = the copied stops.
Step 2, validate: x2_r2_agent.main(["--mode", "validation", "--call2-tools", "off", "--models", "qwen3-4b-2507-tools",
  "--out", <out>], advance=False): v2b's plan (arm b num_ctx 131072 3 seeds x 10 turns, positive control 8192 1 seed x
  15 turns, diagnostic arm b with tools on call 2), think false sent and the thinking_off gate checked.
Exit: t2s_queue.advance(note) exactly once (finally). The note says whether the v2c gates passed; failing them does not
stop the queue, the real run's preflight (validation_preflight_per_model) refuses the tag on its own.

Usage: py -3.12 x2_r2_4b_tools_validate.py --out C:\\apu\\ovn\\results\\x2_r2_validation_v2c.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if (_here.parent / "harness").is_dir():
    sys.path.insert(0, str(_here.parent / "harness"))

SOURCE_TAG = "qwen3-4b-2507"
TOOLS_TAG = "qwen3-4b-2507-tools"
TEMPLATE_FROM = "qwen3:8b"
LIBRARY_TEMPLATE_SHA256 = "ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4"
OLLAMA = "http://127.0.0.1:11434"
CREATE_TIMEOUT_S = 1800


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def parse_stop_params(parameters_text: str | None) -> list[str]:
    """The stop values of an /api/show "parameters" string ('stop   "<|im_start|>"' lines), in order."""
    out = []
    for line in (parameters_text or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] == "stop":
            out.append(parts[1].strip().strip('"'))
    return out


def build_modelfile(gguf: str, template: str, stops: list[str]) -> str:
    if '"""' in template:
        raise ValueError("template contains a triple quote; cannot be written as a Modelfile TEMPLATE block")
    if any('"' in s for s in stops):
        raise ValueError(f"a stop value contains a double quote: {stops}")
    lines = [f"FROM {gguf}", f'TEMPLATE """{template}"""'] + [f'PARAMETER stop "{s}"' for s in stops]
    return "\n".join(lines) + "\n"


def verify_tag(manifest: dict | None, show: dict | None, source_digest: str, stops: list[str]) -> dict:
    """Pure check of an existing qwen3-4b-2507-tools: same model blob as the source, the library template verbatim,
    the copied stop list. {"ok", "reasons", ...}."""
    reasons = []
    if manifest is None:
        return {"ok": False, "reasons": ["tag absent (no manifest)"]}
    model = [l.get("digest") for l in manifest.get("layers") or [] if l.get("mediaType", "").endswith(".model")]
    if model != [source_digest]:
        reasons.append(f"model layer {model} != source {source_digest}")
    tpl_sha = sha256_text(show.get("template") or "") if show else None
    if tpl_sha != LIBRARY_TEMPLATE_SHA256:
        reasons.append(f"template sha256 {tpl_sha} != library {LIBRARY_TEMPLATE_SHA256}")
    got_stops = parse_stop_params(show.get("parameters")) if show else None
    if got_stops != stops:
        reasons.append(f"stop params {got_stops} != {stops}")
    return {"ok": not reasons, "reasons": reasons, "model_layers": model, "template_sha256": tpl_sha,
            "stops": got_stops, "capabilities": (show or {}).get("capabilities")}


def _post(path, body, timeout=60):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _show_or_none(tag):
    try:
        return _post("/api/show", {"model": tag})
    except Exception:
        return None


def _read_manifest(models_dir, tag):
    import chat_template_source as cts
    p = cts.manifest_path(models_dir, tag)
    return (json.loads(p.read_text(encoding="utf-8")), p) if p.is_file() else (None, p)


def _run_hidden(argv, timeout):
    try:
        from proc_util import run_hidden as run
    except Exception:
        import subprocess
        run = subprocess.run
    return run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def ensure_tools_tag(emit, out_path: Path, log=print, hc=None, models_dir=None) -> dict:
    """Step 1. Returns the r2t_create record (also emitted). Raises on any check that fails."""
    if hc is None:
        import host_config as hc
    import x2_outcome_table as x2ot
    import chat_template_source as cts
    models_dir = models_dir or cts._models_dir()
    gguf = x2ot.MODEL_MAP[SOURCE_TAG][1]
    rec = {"record": "r2t_create", "ts_utc": utc_iso(), "tag": TOOLS_TAG, "source_tag": SOURCE_TAG,
           "template_from": TEMPLATE_FROM, "gguf": gguf, "models_dir": models_dir}
    src_man, src_path = _read_manifest(models_dir, SOURCE_TAG)
    if src_man is None:
        raise RuntimeError(f"source tag {SOURCE_TAG} has no manifest at {src_path}")
    src_bytes_sha_before = sha256_file(src_path)
    src_digest = [l["digest"] for l in src_man["layers"] if l.get("mediaType", "").endswith(".model")][0]
    t0 = time.monotonic()
    gguf_sha = sha256_file(gguf)
    rec.update({"source_model_digest": src_digest, "gguf_sha256": gguf_sha,
                "gguf_sha256_s": round(time.monotonic() - t0, 1),
                "gguf_matches_source_blob": src_digest == f"sha256:{gguf_sha}"})
    if not rec["gguf_matches_source_blob"]:
        raise RuntimeError(f"{gguf} sha256 {gguf_sha} != {SOURCE_TAG} model layer {src_digest}")
    lib = _post("/api/show", {"model": TEMPLATE_FROM})
    template = lib.get("template") or ""
    stops = parse_stop_params(lib.get("parameters"))
    rec.update({"template_sha256": sha256_text(template), "template_len": len(template), "stops": stops,
                "template_from_parameters": lib.get("parameters")})
    if rec["template_sha256"] != LIBRARY_TEMPLATE_SHA256:
        raise RuntimeError(f"{TEMPLATE_FROM} template sha256 {rec['template_sha256']} != {LIBRARY_TEMPLATE_SHA256}")
    if not stops:
        raise RuntimeError(f"{TEMPLATE_FROM} has no stop parameters: {lib.get('parameters')!r}")
    modelfile = build_modelfile(gguf, template, stops)
    mf_path = out_path.parent / (out_path.stem + ".Modelfile")
    # newline="\n": in text mode Windows turns "\n" into "\r\n", and Ollama keeps those CRs inside the TEMPLATE
    # (2026-10-08: the first run created a template with CRLF line ends, sha c9d0f733 vs library ae370d88; caught by
    # verify_after, so the job stopped before validating).
    mf_path.write_text(modelfile, encoding="utf-8", newline="\n")
    rec.update({"modelfile": modelfile, "modelfile_sha256": sha256_text(modelfile), "modelfile_path": str(mf_path)})
    man, _ = _read_manifest(models_dir, TOOLS_TAG)
    before = verify_tag(man, _show_or_none(TOOLS_TAG) if man else None, src_digest, stops)
    rec["verify_before"] = before
    if before["ok"]:
        rec["action"] = "already_present_verified"
        log(f"{TOOLS_TAG}: already present and verified, not recreated")
    else:
        exe = hc._resolve_ollama_exe_for_serve()
        log(f"{TOOLS_TAG}: creating ({'; '.join(before['reasons'])})")
        t1 = time.monotonic()
        res = _run_hidden([exe, "create", TOOLS_TAG, "-f", str(mf_path)], CREATE_TIMEOUT_S)
        rec.update({"action": "created", "create_returncode": res.returncode,
                    "create_elapsed_s": round(time.monotonic() - t1, 1),
                    "create_stdout_tail": (res.stdout or "")[-800:], "create_stderr_tail": (res.stderr or "")[-800:]})
        man, _ = _read_manifest(models_dir, TOOLS_TAG)
        rec["verify_after"] = verify_tag(man, _show_or_none(TOOLS_TAG) if man else None, src_digest, stops)
    src_bytes_sha_after = sha256_file(src_path)
    rec.update({"source_manifest_sha256_before": src_bytes_sha_before,
                "source_manifest_sha256_after": src_bytes_sha_after,
                "source_tag_untouched": src_bytes_sha_before == src_bytes_sha_after})
    try:
        rec["template_facts"] = cts.probe([SOURCE_TAG, TOOLS_TAG, TEMPLATE_FROM], [gguf], models_dir)
    except Exception as e:
        rec["template_facts"] = f"probe failed: {e!r}"[:200]
    emit(rec)
    final = rec.get("verify_after", before)
    if not final["ok"]:
        raise RuntimeError(f"{TOOLS_TAG} failed verification: {final['reasons']}")
    if not rec["source_tag_untouched"]:
        raise RuntimeError(f"{SOURCE_TAG} manifest changed during this job")
    return rec


def gates_summary(out_path: Path) -> str:
    import x2_r2_agent as ra
    rows = ra.read_rows(out_path)
    pf = ra.validation_preflight(rows, "off", [TOOLS_TAG])
    if pf["ok"]:
        return f"v2c gates passed for {TOOLS_TAG} (rules in use {pf['rules_in_use'].get(TOOLS_TAG)})"
    return (f"v2c gates FAILED for {TOOLS_TAG}, the real run's preflight will refuse it: "
            + "; ".join(pf["reasons"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{utc_iso()}] {msg}", flush=True)

    def emit(row):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    note = "stopped: unknown"
    import host_config as hc
    try:
        import socket
        if socket.gethostname().upper() != "EVO-X2":
            raise RuntimeError(f"x2_r2_4b_tools_validate runs on EVO-X2 only, this is {socket.gethostname()!r}")
        hc.start_ollama_server()
        try:
            if not hc.wait_for_ollama_ready(timeout_s=90):
                raise RuntimeError("ollama did not become ready")
            ensure_tools_tag(emit, out_path, log=log, hc=hc)
        finally:
            hc.stop_ollama_server()
        import x2_r2_agent as ra
        vnote = ra.main(["--mode", "validation", "--call2-tools", "off", "--models", TOOLS_TAG, "--out", str(out_path)],
                        advance=False)
        if str(vnote).startswith("completed"):
            note = f"completed; {TOOLS_TAG} created/verified; " + gates_summary(out_path)
        else:
            note = str(vnote)
        note = note[:600]
        log(note)
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        log(note)
        traceback.print_exc()
    finally:
        try:
            import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            log(f"queue advance failed: {e!r}")
    return note


if __name__ == "__main__":
    main()
