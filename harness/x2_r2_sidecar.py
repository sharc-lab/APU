"""Run a side copy of x2_r2_agent.py from its own directory, everything else from the deployed directory.

Why (2026-10-10, x2_r2_llama_val_32k, pre-registered in docs/FINDINGS.md): the operator rule is that
C:\\apu\\ovn\\x2_r2_agent.py is not replaced while x2_r2_real_v1b and x2_r2_mitigation_v1 still have to run on it, but
the confound-check job needs this branch's x2_r2_agent.py (--neg-ctx / --neg-sessions). So that copy, and any other
module whose deployed git blob differs from the branch, goes into a side directory (C:\\apu\\ovn\\r2v32k\\) next to
this launcher, and the queue entry runs this file:

    python r2v32k\\x2_r2_sidecar.py <x2_r2_agent argv>

It puts this file's directory first and its parent (C:\\apu\\ovn) second on sys.path, then runs the side copy of
x2_r2_agent.py as __main__ with the given argv. A module present in the side directory is imported from there; every
other module (t2s_queue, host_config, x2_r2_mechanism, ...) from the parent. The agent itself calls
t2s_queue.advance() on exit, exactly as when the queue runs x2_r2_agent.py directly.

  --sidecar-import-check   (first argument; the rest is the job's argv) imports the side copy as a module, checks it
                           and the modules it pulled in resolve to the expected directories, parses the argv with the
                           agent's own build_arg_parser(), prints the parsed args and, for --mode validation, the plan;
                           with --backend also the backend env, the server log path and whether the Ollama tray app
                           runs (read-only process listing); then exits 0. No server start, no model call, no queue
                           access.
"""
from __future__ import annotations

import importlib
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE / "x2_r2_agent.py"
CHECK_FLAG = "--sidecar-import-check"


def setup_path(here: Path = HERE) -> list[str]:
    """This directory first, its parent second (each once, ahead of everything else)."""
    front = [str(here), str(here.parent)]
    sys.path[:] = front + [p for p in sys.path if p not in front]
    return front


def import_check(argv: list[str], here: Path = HERE) -> dict:
    """Imports x2_r2_agent from `here` and parses argv; returns what was found (raises on a wrong location or a bad
    argv, argparse's SystemExit included)."""
    setup_path(here)
    sys.modules.pop("x2_r2_agent", None)
    ag = importlib.import_module("x2_r2_agent")
    agent_file = Path(ag.__file__).resolve()
    if agent_file.parent != here.resolve():
        raise RuntimeError(f"x2_r2_agent imported from {agent_file}, expected {here}")
    args = ag.build_arg_parser().parse_args(argv)
    out = {"agent_file": str(agent_file), "args": vars(args), "modules": {}}
    for name in ("t2s_r2_session_growth", "t2s_queue", "host_config", "x2_r2_client_trim", "x2_r2_mechanism"):
        mod = importlib.import_module(name)
        out["modules"][name] = str(Path(mod.__file__).resolve())
    if getattr(args, "backend", None):   # 2026-10-19, x2_r2_llama_val_32k_vulkan: the backend env and log, no server
        xb = importlib.import_module("x2_backend")
        for name in ("x2_backend", "x2_num_gpu"):
            out["modules"][name] = str(Path(importlib.import_module(name).__file__).resolve())
        out["backend_env"] = xb.BACKEND_ENVS[args.backend]
        out["server_log"] = xb.server_log_for(args.out)
        out["tray_app_running"] = xb.tray_app_running()
    if args.mode == "validation":
        plan, neg_arm = ag.validation_plan_neg(args.call2_tools, args.neg_ctx, args.neg_sessions)
        out["plan"] = [(a, list(s), t, ag.ARMS[a]["num_ctx"]) for a, s, t in plan]
        out["neg_arm"] = neg_arm
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == CHECK_FLAG:
        print(json.dumps(import_check(argv[1:]), indent=1, default=str))
        return 0
    setup_path()
    sys.argv = [str(AGENT)] + argv
    runpy.run_path(str(AGENT), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
