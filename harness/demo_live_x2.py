"""Live demo, the evo-x2 side (pre-registered in docs/EXPERIMENTS.md "Live demo: detector check and acceptance run").

Modes:
  --mode detector-check   (queued job) each frozen pick at num_ctx 4096 and at the runtime default (no num_ctx), one
                          Ollama server per configuration (demo_servers.LaneServer, own port, own log, OLLAMA_DEBUG=1);
                          every call's detector verdict recorded. Output results/demo_detector_check.jsonl.
  --mode acceptance       (queued job) the local parts of the 5-task run for each scenario (demo_run.SCENARIOS),
                          lanes sequential per task: Default local, then Ours (local when route_at_start says so); the
                          cloud parts are recorded as pending because evo-x2 holds no cloud key. The controller
                          completes them (demo/live/complete_run.py) into results/demo_live_run_<date>.jsonl.
  --mode agent            (live demo) line protocol on stdin/stdout, started by the controller over SSH: commands
                          {"cmd": "start", "scenario": s}, {"cmd": "local", "lane": l, "seed": n, "task_id": t,
                          "num_ctx": c}, {"cmd": "stop"}; every call and result is one JSON line on stdout. Servers are
                          stopped on "stop", on end of input and on any error.

Queued modes call t2s_queue.advance() exactly once on exit (finally) and set no pause. Nothing here reads a cloud key.
Picks come only from the frozen picks file (--picks), whose sha256 is recorded in run_start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
import demo_live as L  # noqa: E402  (sets the rest of sys.path)
import demo_run as DR  # noqa: E402
import demo_servers as S  # noqa: E402

PORTS = {4096: 11541, 8192: 11542, 32768: 11543, None: 11544}   # one per configuration (all base server env)
LOG_DIR_NAME = "demo_live_logs"


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_picks(path) -> list[dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    picks = d["picks"]
    if not picks or not all({"seed", "task_id"} <= set(p) for p in picks):
        raise ValueError("picks file has no picks or a pick without seed/task_id")
    return picks


class Emitter:
    def __init__(self, out=None, stream=None):
        self.out, self.stream = out, stream
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)

    _lock = __import__("threading").Lock()

    def __call__(self, row: dict):
        row = {"ts_utc": utc_iso(), **row}
        line = json.dumps(row, default=str)
        with self._lock:
            self._write(line)

    def _write(self, line):
        if self.out:
            with open(self.out, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        if self.stream:
            self.stream.write(line + "\n")
            self.stream.flush()


class Servers:
    """The configurations' servers, started on first use, all stopped by close()."""

    def __init__(self, log_dir: Path, emit, ports=PORTS, server_cls=S.LaneServer):
        self.log_dir, self.emit, self.ports, self.cls = Path(log_dir), emit, ports, server_cls
        self.running = {}

    def get(self, num_ctx):
        if num_ctx not in self.running:
            tag = "default" if num_ctx is None else str(num_ctx)
            srv = self.cls(self.ports[num_ctx], {}, str(self.log_dir / f"serve_{tag}.log"))
            rec = srv.start()
            self.emit({"record": "server_start", "num_ctx": num_ctx, **rec})
            self.running[num_ctx] = srv
        srv = self.running[num_ctx]
        if not srv.check():
            raise RuntimeError(f"stale-server guard: the listener on port {srv.port} is not the server started")
        return srv

    def close(self):
        for k in list(self.running):
            try:
                self.running.pop(k).stop()
                self.emit({"record": "server_stop", "num_ctx": k})
            except Exception as e:
                self.emit({"record": "server_stop_error", "num_ctx": k, "error": repr(e)[:300]})


def local_runner(servers: Servers, ratio: float, emit, model=L.MODEL):
    def run(lane, task, num_ctx):
        srv = servers.get(num_ctx)
        rt = L.LaneRuntime(srv.base, srv.log_path, ratio,
                           on_call=lambda c: emit({"record": "call", "lane": lane, "seed": task["seed"],
                                                   "task_id": task["task_id"], "num_ctx": num_ctx, **c}))
        return L.run_local(rt, task, num_ctx, model)
    return run


def strip_task(t: dict) -> dict:
    return {k: t[k] for k in ("seed", "task_id", "turn_idx", "projected_prompt", "history_turns", "history_check",
                              "task_type", "recorded")}


def provenance(args, picks_path) -> dict:
    return {"host": socket.gethostname(), "harness_blobs": {f: L.EV.git_blob_sha1(_here / f) for f in
                                                            ("demo_live.py", "demo_run.py", "demo_servers.py",
                                                             "demo_overflow.py", "demo_live_x2.py")},
            "imported": {m.__name__: {"file": str(Path(m.__file__).resolve()), "blob": L.EV.git_blob_sha1(m.__file__)}
                         for m in (L.EV, L.ag, L.E, L.DO, L.DO.mech)},
            "picks_file": str(picks_path), "picks_sha256": sha256_file(picks_path), "model": L.MODEL,
            "argv": sys.argv[1:]}


def detector_check(args, emit, results_dir: Path) -> str:
    picks = load_picks(args.picks)
    rows = L.read_jsonl(results_dir / L.B3_FILE)
    prof = load_profile()
    ratio = prof["calibration"]["ratio"]
    emit({"record": "run_start", "harness": "demo_detector_check", "arms": ["num_ctx_4096", "default"],
          **provenance(args, args.picks)})
    servers = Servers(log_root(args, results_dir) / "detector_check", emit)
    fired = {"num_ctx_4096": 0, "default": 0}
    try:
        tasks = [L.build_task(rows, p["seed"], p["task_id"], with_cloud=False) for p in picks]
        run = local_runner(servers, ratio, emit, args.model)
        for arm, ctx in (("num_ctx_4096", 4096), ("default", None)):
            for t in tasks:
                r = run(arm, t, ctx)
                fired[arm] += bool(r["flagged"])
                emit({"record": "detector_task", "arm": arm, "num_ctx": ctx, **strip_task(t),
                      **{k: r[k] for k in ("end_state_ok", "flagged", "calls", "n_calls", "seconds",
                                           "max_used_tokens", "max_own_tokens", "any_http_error")}})
    finally:
        servers.close()
    emit({"record": "run_end", "fired": fired, "n_tasks": len(picks)})
    return f"completed: fired at 4096 {fired['num_ctx_4096']}/{len(picks)}, at default {fired['default']}/{len(picks)}"


def acceptance(args, emit, results_dir: Path) -> str:
    picks = load_picks(args.picks)
    rows = L.read_jsonl(results_dir / L.B3_FILE)
    prof = load_profile()
    scenarios = args.scenarios.split(",")
    emit({"record": "run_start", "harness": "demo_live_run", "part": "local (evo-x2)", "scenarios": scenarios,
          "cloud": "pending: completed on the controller (no cloud key on evo-x2)", **provenance(args, args.picks)})
    servers = Servers(log_root(args, results_dir) / "acceptance", emit)
    t0 = time.monotonic()
    try:
        tasks = [L.build_task(rows, p["seed"], p["task_id"], with_cloud=False) for p in picks]
        run = local_runner(servers, prof["calibration"]["ratio"], emit, args.model)
        for sc in scenarios:
            for t in tasks:
                res = DR.run_task(sc, prof, t, run, lambda lane, task, model: {"pending": True, "model": model},
                                  lambda: float(args.budget_usd), emit=lambda ev: emit({"record": "event", **ev}))
                emit({"record": "task_result", **strip_task(t), **res})
    finally:
        servers.close()
    emit({"record": "run_end", "part": "local (evo-x2)", "wall_s": round(time.monotonic() - t0, 1)})
    return "completed: local parts recorded; cloud parts pending on the controller"


def agent(args, emit, results_dir: Path) -> str:
    rows = L.read_jsonl(results_dir / L.B3_FILE)
    prof = load_profile()
    servers = Servers(log_root(args, results_dir) / "agent", emit)
    run = local_runner(servers, prof["calibration"]["ratio"], emit, args.model)
    cache = {}
    emit({"record": "agent_ready", "host": socket.gethostname(), "model": args.model,
          **(provenance(args, args.picks) if args.picks else {})})
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            cmd = json.loads(line)
            if cmd.get("cmd") == "stop":
                break
            if cmd.get("cmd") == "local":
                key = (cmd["seed"], cmd["task_id"])
                if key not in cache:
                    cache[key] = L.build_task(rows, cmd["seed"], cmd["task_id"], with_cloud=False)
                try:
                    r = run(cmd["lane"], cache[key], cmd.get("num_ctx"))
                    emit({"record": "local_result", "lane": cmd["lane"], "seed": cmd["seed"],
                          "task_id": cmd["task_id"], "result": r})
                except Exception as e:
                    emit({"record": "local_error", "lane": cmd["lane"], "error": repr(e)[:300]})
    finally:
        servers.close()
    return "agent stopped"


def log_root(args, results_dir: Path) -> Path:
    return Path(args.log_dir) if args.log_dir else results_dir / LOG_DIR_NAME


def check_plan(args, results_dir: Path) -> int:
    """Dry check (no server start, no model call): the picks, each task's rebuilt history against the recorded
    transcript estimate, projected prompts, the scenarios' decisions and configurations, the imported files."""
    picks = load_picks(args.picks)
    rows = L.read_jsonl(results_dir / L.B3_FILE)
    prof = load_profile()
    out = {"provenance": provenance(args, args.picks), "tasks": [], "configurations": {}}
    ok = True
    for p in picks:
        t = L.build_task(rows, p["seed"], p["task_id"], with_cloud=False)
        hc = t["history_check"]
        ok = ok and hc["rebuilt_est"] == hc["recorded_est"] and t["turn_idx"] == p.get("turn_idx", t["turn_idx"])
        out["tasks"].append({**strip_task(t), "decisions": {sc: {k: DR.decide(sc, prof, t, 3.0).get(k) for k in
                                                                 ("route", "window", "setting_label", "model")}
                                                            for sc in DR.SCENARIOS}})
    for sc in DR.SCENARIOS:
        out["configurations"][sc] = DR.configurations(sc, prof)
    out["ok"] = ok
    print(json.dumps(out, indent=1, default=str))
    return 0 if ok else 1


def load_profile() -> dict:
    import route_at_start as R
    return R.load_profiles()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("detector-check", "acceptance", "agent", "check-plan"), required=True)
    ap.add_argument("--picks", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--scenarios", default="amd8192,intel")
    ap.add_argument("--budget-usd", default="3.0")
    ap.add_argument("--model", default=L.MODEL, help="local model (development only; the runs use qwen3:14b)")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--queued", action="store_true", help="call t2s_queue.advance() on exit")
    args = ap.parse_args(argv)
    results_dir = Path(args.results_dir) if args.results_dir else (L.REPO / "results")
    if args.mode == "check-plan":
        return check_plan(args, results_dir)
    if args.mode == "agent":
        agent(args, Emitter(stream=sys.stdout), results_dir)
        return 0
    if not args.picks or not args.out:
        ap.error("--picks and --out are required for queued modes")
    emit = Emitter(out=args.out)
    note = "completed"
    try:
        note = (detector_check if args.mode == "detector-check" else acceptance)(args, emit, results_dir)
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
        emit({"record": "error", "error": note, "traceback": traceback.format_exc()[-2000:]})
    finally:
        if args.queued:
            try:
                import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                print(f"queue advance failed: {e!r}")
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
