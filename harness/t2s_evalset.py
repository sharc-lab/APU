"""evo-t2s queue entry point for the B3 v2 evalset harness (harness/x2_r2_evalset.py). Host-neutral wrapper in the
pattern of harness/t2s_r2_agent.py: it does not reimplement anything in x2_r2_evalset.py.

Pre-registration: docs/FINDINGS.md "PRE-REGISTRATION: Intel laptop task accuracy" (2026-10-22), staged for the
evo-t2s window 2026-10-26 to 2026-11-08 (queues/t2s_oct26_queue.json, entries t2s_o26_b3_*).

What it adds on top of x2_r2_evalset.main():
  * --runtime-config: cpu_default (stock Ollama; on evo-t2s it runs on the CPU and sizes the default window at 4096)
    or igpu_enable (OLLAMA_IGPU_ENABLE=1 for this job's own server only, passed as x2_r2_evalset --server-env, never
    persisted; measured default window 32768). Same table as t2s_r2_agent.RUNTIME_CONFIGS.
  * a host check (EVO-T2S only) and an output-name check (every output stem starts with "t2s_");
  * the operator's Ollama pin (t2s_week_preflight.apply_ollama_pin) and a server stop before the job, so the server
    x2_r2_evalset starts carries this job's env and binary;
  * an argument error inside x2_r2_evalset (its parser's SystemExit) becomes a "stopped:" note, never "completed";
  * a post-run runtime check: the loaded window of every ollama_default turn against the configuration's expected
    default (t2s_r2_agent.runtime_check; a t2s_runtime_check row; a mismatch is appended to the exit note);
  * exactly one t2s_queue.advance(note) on exit (finally), with x2_r2_evalset.main(..., advance=False).

Arguments the wrapper does not know are forwarded verbatim (--design v2 --mode ... --plan ... --rules-from ...).

--dry-run: no host check, no Ollama, no queue. Runs x2_r2_evalset's own argument checks on the forwarded line with
--check-plan (prints the plan; no server, no model call) and reports whether it was accepted.

Usage on evo-t2s (deployed flat to C:\\apu\\ovn):
  python t2s_evalset.py --runtime-config cpu_default --design v2 --mode validation --workload evalset \\
      --guardrails off --models qwen3:14b --out results/t2s_evalset_tasks_val_v2_cpu.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import t2s_r2_agent as w  # noqa: E402  (RUNTIME_CONFIGS, check_out_name, runtime_check, read_rows)

HOST_KEY = w.HOST_KEY
RUNTIME_CONFIGS = w.RUNTIME_CONFIGS


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-config", choices=tuple(RUNTIME_CONFIGS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def forwarded_argv(args, passthrough) -> list[str]:
    fwd = ["--out", args.out] + list(passthrough)
    for k, v in RUNTIME_CONFIGS[args.runtime_config]["server_env"].items():
        fwd += ["--server-env", f"{k}={v}"]
    return fwd


def _import_ev():
    import x2_r2_evalset as ev
    return ev


def dry_run(args, passthrough, ev) -> dict:
    fwd = forwarded_argv(args, passthrough)
    buf = io.StringIO()
    ok, err = True, None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            res = ev.main(fwd + ["--check-plan"], advance=False)
        ok = res == "checked"
        if not ok:
            err = f"x2_r2_evalset returned {res!r}"
    except SystemExit as e:
        ok, err = False, f"x2_r2_evalset rejected the arguments (exit {e.code})"
    except Exception as e:   # check_plan's own refusal (ValueError) and the like
        ok, err = False, repr(e)
    plan = None
    if ok:
        txt = "\n".join(ln for ln in buf.getvalue().splitlines() if not ln.lstrip().startswith("[filler trim"))
        try:
            plan = json.loads(txt).get("plan")
        except json.JSONDecodeError:
            plan = None
    return {"forwarded_argv": fwd, "out_name_problem": w.check_out_name(args.out), "parse_ok": ok,
            "parse_error": err, "plan": plan, "runtime_config": args.runtime_config,
            "server_env": RUNTIME_CONFIGS[args.runtime_config]["server_env"]}


def main(argv=None, ev=None, hc=None, tq=None, hostname=None):
    args, passthrough = build_parser().parse_known_args(argv)
    if args.dry_run:
        rep = dry_run(args, passthrough, ev or _import_ev())
        print(json.dumps(rep, indent=1, default=str))
        return rep
    note = "completed"
    out_path = Path(args.out)
    try:
        if hc is None:
            import host_config as hc
        problem = w.check_out_name(args.out)
        if problem:
            raise RuntimeError(problem)
        host = (hostname or socket.gethostname()).upper()
        if host != HOST_KEY:
            raise RuntimeError(f"t2s_evalset runs on {HOST_KEY} only, this is {host!r}")
        ev = ev or _import_ev()
        fwd = forwarded_argv(args, passthrough)
        import t2s_week_preflight as pf
        pin = pf.apply_ollama_pin()
        print(f"[{utc_iso()}] ollama pin: {pin}", flush=True)
        hc.stop_ollama_server()
        try:
            note = ev.main(fwd, advance=False) or "completed"
        except SystemExit as e:
            note = f"stopped: x2_r2_evalset.py rejected its arguments (exit {e.code}): {' '.join(fwd)}"[:400]
            return note
        chk = w.runtime_check(w.read_rows(out_path), args.runtime_config)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(chk, default=str) + "\n")
        if chk["applicable"] and not chk["ok"]:
            note = f"{note}; runtime check: loaded context {chk['mismatched']} != {chk['expected_default_ctx']}"[:400]
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
    finally:
        print(f"[{utc_iso()}] t2s_evalset exit: {note}", flush=True)
        try:
            if tq is None:
                import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}", flush=True)
    return note


if __name__ == "__main__":
    main()
