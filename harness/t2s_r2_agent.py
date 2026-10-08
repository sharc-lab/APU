"""evo-t2s queue entry point for the R2 agent-session harness (harness/x2_r2_agent.py). Host-neutral wrapper: it
does not reimplement anything in x2_r2_agent.py and never edits it. docs/T2S_WEEK_PLAN.md is the plan this serves.

What it adds on top of x2_r2_agent.main():
  * --runtime-config: cpu_default (stock Ollama; on evo-t2s Ollama drops the Arc iGPU by policy, runs on CPU, and
    sizes the default context at 4096) or igpu_enable (OLLAMA_IGPU_ENABLE=1 for this job's own server only, never
    persisted; measured default context 32768 for llama3.1:8b, register/FINDINGS "OLLAMA_IGPU_ENABLE=1").
  * a host check (EVO-T2S only, through harness/host_config.py, interactive_guard True there) and an output-name
    check (every output file stem starts with "t2s_", so no evo-t2s file can collide with an evo-x2 result name);
  * a capability check of the deployed x2_r2_agent.py before any Ollama call: x2_r2_agent.py as of 62c0f8b refuses
    to run on any host but EVO-X2 and has no flag for a server environment variable or for a tier/seed subset. The
    exact changes are listed in docs/T2S_WEEK_PLAN.md ("Changes needed in harness/x2_r2_agent.py"). Until they are
    on main and deployed, this wrapper refuses with a "stopped:" note naming what is missing (the queue marks the
    entry error and moves on; entries gated on it stay pending);
  * a post-run runtime check: the loaded context of every turn of the ollama_default arm, per model, against the
    configuration's expected default (recorded as a t2s_runtime_check row; a mismatch is appended to the exit note,
    it never deletes or rewrites a row);
  * exactly one t2s_queue.advance(note) on exit (finally), with x2_r2_agent.main(..., advance=False) so the inner
    harness never advances the queue itself.

Arguments the wrapper does not know are forwarded to x2_r2_agent.py verbatim (e.g. --plan strong --call2-tools off
--rules-from ... --require-validation-gates --per-model-refusal --client-trim margin=0.05).

--dry-run: no host check, no Ollama, no queue. Prints the forwarded command line, whether x2_r2_agent.py's own parser
accepts it (harness/argparse_probe.py), and which capabilities are still missing.

Usage on evo-t2s (deployed flat to C:\\apu\\ovn):
  python t2s_r2_agent.py --mode validation --runtime-config cpu_default \\
      --out results/t2s_r2_validation_cpu.jsonl --call2-tools off
"""
from __future__ import annotations

import argparse
import inspect
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if (_here.parent / "harness").is_dir():
    sys.path.insert(0, str(_here.parent / "harness"))

import argparse_probe  # noqa: E402

HOST_KEY = "EVO-T2S"
OUT_PREFIX = "t2s_"
MODES = ("validation", "real", "mitigation", "mechanism")
IGPU_ENV = {"OLLAMA_IGPU_ENABLE": "1"}
# Expected Ollama default context on evo-t2s per runtime config. 4096 and 32768 are the measured llama3.1:8b values
# (docs/FINDINGS.md, K1 v3 table and the OLLAMA_IGPU_ENABLE table); qwen3:14b has not been measured under
# OLLAMA_IGPU_ENABLE=1 on evo-t2s, so the check records it and flags a difference instead of assuming it.
RUNTIME_CONFIGS = {
    "cpu_default": {"server_env": {}, "expected_default_ctx": 4096, "runtime_label": "ollama_default"},
    "igpu_enable": {"server_env": dict(IGPU_ENV), "expected_default_ctx": 32768,
                    "runtime_label": "ollama_igpu_enable"},
}
# Flags this wrapper adds to the forwarded command line, which the x2_r2_agent.py of 62c0f8b does not have.
SERVER_ENV_FLAG, TIERS_FLAG, SEEDS_FLAG = "--server-env", "--tiers", "--seeds"
HOST_GUARD_LITERAL = "runs on EVO-X2 only"


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--runtime-config", choices=tuple(RUNTIME_CONFIGS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None)
    ap.add_argument("--tiers", default=None, help="comma-separated base arm ids, e.g. ollama_default")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def check_out_name(out) -> str | None:
    """None if the output file name is evo-t2s-prefixed, else the reason it is refused."""
    name = Path(out).name
    if not name.startswith(OUT_PREFIX):
        return f"output file {name!r} does not start with {OUT_PREFIX!r} (would risk an evo-x2 name collision)"
    return None


def forwarded_argv(args, passthrough) -> list[str]:
    cfg = RUNTIME_CONFIGS[args.runtime_config]
    fwd = ["--mode", args.mode, "--out", args.out]
    if args.models:
        fwd += ["--models", args.models]
    if args.tiers:
        fwd += [TIERS_FLAG, args.tiers]
    if args.seeds:
        fwd += [SEEDS_FLAG, args.seeds]
    for k, v in cfg["server_env"].items():
        fwd += [SERVER_ENV_FLAG, f"{k}={v}"]
    return fwd + list(passthrough)


def x2_capabilities(x2mod) -> dict:
    """What the importable x2_r2_agent supports: its parser's flags and --mode choices (read by parsing a minimal
    valid command line through argparse_probe, so nothing runs), and whether main() still carries the EVO-X2-only
    hostname guard."""
    res = argparse_probe.probe(x2mod.main, ["--mode", "validation", "--out", "probe.jsonl"])
    opts = res["options"]
    try:
        host_generic = HOST_GUARD_LITERAL not in inspect.getsource(x2mod.main)
    except (OSError, TypeError):
        host_generic = False
    return {"parsed": res["ok"], "flags": sorted(opts), "modes": list(opts.get("--mode") or ()),
            "host_generic": host_generic}


def missing_capabilities(caps: dict, fwd: list[str]) -> list[str]:
    missing = []
    if not caps["host_generic"]:
        missing.append(f"x2_r2_agent.main still refuses non-EVO-X2 hosts ({HOST_GUARD_LITERAL!r})")
    flags = set(caps["flags"])
    for tok in fwd:
        if tok.startswith("--") and tok.split("=", 1)[0] not in flags:
            missing.append(f"x2_r2_agent.py has no {tok.split('=', 1)[0]} flag")
    if "--mode" in fwd:
        mode = fwd[fwd.index("--mode") + 1]
        if mode not in caps["modes"]:
            missing.append(f"x2_r2_agent.py --mode has no {mode!r} choice (have {caps['modes']})")
    return sorted(set(missing))


def strip_unsupported(fwd: list[str], flags) -> list[str]:
    """fwd minus every --flag (and its value, if the next token is not a flag) the parser does not know, so the
    remaining argv can still be parse-checked against today's x2_r2_agent.py in a dry run."""
    out, i = [], 0
    while i < len(fwd):
        tok = fwd[i]
        if tok.startswith("--") and tok.split("=", 1)[0] not in flags:
            i += 1
            if "=" not in tok and i < len(fwd) and not fwd[i].startswith("--"):
                i += 1
            continue
        out.append(tok)
        i += 1
    return out


def runtime_check(rows, runtime_config) -> dict:
    """Loaded context per model on the ollama_default arm's turns versus the configuration's expected default."""
    expected = RUNTIME_CONFIGS[runtime_config]["expected_default_ctx"]
    seen = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and str(r.get("arm_id", "")).startswith("ollama_default"):
            if r.get("loaded_context") is not None:
                seen.setdefault(r.get("model_id"), set()).add(r["loaded_context"])
    per_model = {m: sorted(v) for m, v in seen.items()}
    mismatched = {m: v for m, v in per_model.items() if v != [expected]}
    return {"record": "t2s_runtime_check", "runtime_config": runtime_config, "expected_default_ctx": expected,
            "loaded_context_by_model": per_model, "mismatched": mismatched,
            "applicable": bool(per_model), "ok": not mismatched, "ts_utc": utc_iso()}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def dry_run(args, passthrough, x2mod) -> dict:
    fwd = forwarded_argv(args, passthrough)
    caps = x2_capabilities(x2mod)
    missing = missing_capabilities(caps, fwd)
    checkable = strip_unsupported(fwd, set(caps["flags"]) | {"--mode"})
    parsed_as = args.mode
    if args.mode not in caps["modes"]:
        # a mode today's parser lacks (mitigation before its merge): parse the rest of the line under "real", so a
        # typo in the remaining flags is still caught; the missing mode itself is in missing_capabilities.
        parsed_as = "real"
        checkable[checkable.index("--mode") + 1] = parsed_as
    parse = argparse_probe.probe(x2mod.main, checkable)
    return {"forwarded_argv": fwd, "out_name_problem": check_out_name(args.out),
            "x2_parse_ok": parse["ok"], "x2_parse_error": parse["error"], "x2_parsed_as_mode": parsed_as,
            "x2_argv_checked": checkable, "missing_capabilities": missing,
            "runtime_config": args.runtime_config, "server_env": RUNTIME_CONFIGS[args.runtime_config]["server_env"]}


def _import_x2():
    import x2_r2_agent as x2
    return x2


def main(argv=None, x2mod=None, hc=None, tq=None, hostname=None):
    args, passthrough = build_parser().parse_known_args(argv)
    if args.dry_run:
        rep = dry_run(args, passthrough, x2mod or _import_x2())
        print(json.dumps(rep, indent=1))
        return rep
    note = "completed"
    out_path = Path(args.out)
    try:
        if hc is None:
            import host_config as hc
        problem = check_out_name(args.out)
        if problem:
            raise RuntimeError(problem)
        host = (hostname or socket.gethostname()).upper()
        if host != HOST_KEY:
            raise RuntimeError(f"t2s_r2_agent runs on {HOST_KEY} only, this is {host!r}")
        host_cfg = hc.require_host(host)
        try:
            hc.enforce_or_record_interactive_session(host_cfg)
        except SystemExit as e:  # the queue halts on this note ("another interactive session")
            note = f"stopped: {e}"[:400]
            return note
        x2mod = x2mod or _import_x2()
        fwd = forwarded_argv(args, passthrough)
        missing = missing_capabilities(x2_capabilities(x2mod), fwd)
        if missing:
            note = ("stopped: deployed x2_r2_agent.py cannot run this T2S job yet (docs/T2S_WEEK_PLAN.md, changes "
                    "needed in harness/x2_r2_agent.py): " + "; ".join(missing))[:400]
            return note
        import t2s_week_preflight as pf
        pin = pf.apply_ollama_pin()  # OLLAMA_BIN = the side-by-side 0.34.4 when the operator pinned one
        print(f"[{utc_iso()}] ollama pin: {pin}", flush=True)
        hc.stop_ollama_server()  # so the server x2_r2_agent starts carries this job's env, not a leftover one
        note = x2mod.main(fwd, advance=False) or "completed"
        chk = runtime_check(read_rows(out_path), args.runtime_config)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(chk, default=str) + "\n")
        if chk["applicable"] and not chk["ok"]:
            note = f"{note}; runtime check: loaded context {chk['mismatched']} != {chk['expected_default_ctx']}"[:400]
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
    finally:
        print(f"[{utc_iso()}] t2s_r2_agent exit: {note}", flush=True)
        try:
            if tq is None:
                import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}", flush=True)
    return note


if __name__ == "__main__":
    main()
