"""py -3.12 -m demo.live [--port 8800] [--mode live|laptop|replay] [--replay results/demo_live_run_<date>.jsonl]

The live run screen on 127.0.0.1 (headless; prints the URL). GET / is the page; POST /api/run?scenario=amd8192
starts one run (one at a time); GET /api/events?since=N polls its events; GET /api/meta has the scenarios, the
recorded headline (register row hybrid-route-at-start-measured, MEASURED, replayed runs) and the cloud budget.
"""
from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from demo.live import runner as RN  # noqa: E402

DR = RN.DR
STATIC = Path(__file__).resolve().parent / "static"
HEADLINE_ROW = "hybrid-route-at-start-measured"


def headline(register_text: str) -> dict:
    """The 8192 / claude-sonnet-5-5 cell of the headline row, every number parsed from the register text."""
    import route_at_start as R
    row = R.register_rows(register_text).get(HEADLINE_ROW)
    if row is None or row["status"] != "VERIFIED":
        return {"status": "missing", "row": HEADLINE_ROW}
    cell = row["value"].split("evo-x2 ollama_ctx_8192 (window 8192) vs claude-sonnet-5-5:")[1].split(" / ")[0]
    g = lambda p: re.search(p, cell)  # noqa: E731
    h, a, n = g(r"hybrid accuracy (\d+)/(\d+)"), g(r"all-cloud (\d+)/(\d+) \([\d.]+%\), USD ([\d.]+)/task"), \
        g(r"naive local (\d+)/(\d+)")
    return {"status": "VERIFIED", "row": HEADLINE_ROW, "label": "MEASURED (replay of recorded runs)",
            "window": 8192, "cloud_model": "claude-sonnet-5-5",
            "ours": {"correct": int(h[1]), "n": int(h[2]),
                     "usd_task": float(g(r"cloud USD ([\d.]+)/task")[1]),
                     "kept_local": int(g(r"kept local (\d+)/\d+")[1])},
            "cloud": {"correct": int(a[1]), "n": int(a[2]), "usd_task": float(a[3])},
            "default": {"correct": int(n[1]), "n": int(n[2]), "usd_task": 0.0}}


class App:
    def __init__(self, mode: str = "live", replay_file=None, agent_factory=None, cloud=None, picks=RN.PICKS,
                 idle_speedup: float = 1.0):
        self.mode, self.replay_file, self.picks = mode, replay_file, Path(picks)
        self.idle_speedup = idle_speedup
        self.agent_factory, self.cloud = agent_factory, cloud
        self.run = None
        self.meta = {"scenarios": DR.SCENARIOS, "mode": mode, "detector": DR.DETECTOR,
                     "headline": headline((REPO / "docs" / "NUMBERS_REGISTER.md").read_text(encoding="utf-8")),
                     "replay_source": Path(replay_file).name if replay_file else None}

    def start(self, scenario: str) -> dict:
        if self.run is not None and not self.run.done:
            return {"started": False, "reason": "a run is in progress"}
        self.run = RN.LiveRun(scenario, self.mode, picks_path=self.picks, agent_factory=self.agent_factory, cloud=self.cloud,
                              replay_file=Path(self.replay_file) if self.replay_file else None,
                              idle_speedup=self.idle_speedup if self.mode == "replay" else 1.0)
        self.run.start()
        return {"started": True, "scenario": scenario, "mode": self.mode}

    def handle(self, method: str, target: str):
        u = urlparse(target)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if method == "GET" and u.path in ("/", "/index.html"):
            return 200, "text/html; charset=utf-8", (STATIC / "index.html").read_bytes()
        if method == "GET" and u.path == "/api/meta":
            m = dict(self.meta)
            if self.cloud is not None:
                m["budget_left_usd"] = round(self.cloud.budget_left(), 4)
            return 200, "application/json", json.dumps(m).encode()
        if method == "GET" and u.path == "/api/events":
            evs = self.run.since(int(q.get("since", 0))) if self.run else []
            return 200, "application/json", json.dumps({"events": evs, "done": bool(self.run and self.run.done)},
                                                       default=str).encode()
        if method == "POST" and u.path == "/api/run":
            sc = q.get("scenario", "amd8192")
            if sc not in DR.SCENARIOS:
                return 400, "application/json", b'{"error": "unknown scenario"}'
            return 200, "application/json", json.dumps(self.start(sc)).encode()
        return 404, "text/plain", b"not found"


def make_server(port: int, app: App) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def _go(self, method):
            code, ctype, body = app.handle(method, self.path)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._go("GET")

        def do_POST(self):
            self._go("POST")

        def log_message(self, *a):
            pass
    return ThreadingHTTPServer(("127.0.0.1", port), H)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m demo.live", description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--mode", choices=("live", "laptop", "replay"), default="live")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--laptop-model", default="qwen3:4b-instruct")
    ap.add_argument("--picks", default=str(RN.PICKS))
    ap.add_argument("--idle-speedup", type=float, default=1.0,
                    help="replay only: play stretches where nothing changes at this speed, labelled on screen")
    ap.add_argument("--session-limit", type=float, default=3.0,
                    help="live cloud spend limit for this session in USD (default 3; above 3 needs "
                         "--operator-approval)")
    ap.add_argument("--operator-approval", default=None,
                    help="operator only: approve this session above USD 3, e.g. "
                         "'approve demo_live_session up to USD <n>: <reason>'; written to the demo ledger before the "
                         "first live call; never raises the USD 10 cap or skips the Console check")
    a = ap.parse_args(argv)
    cloud, factory = None, None
    if a.mode == "replay":
        if not a.replay:
            ap.error("--replay <run file> is required in replay mode")
    else:
        from demo.live import cloud_lane
        cloud = cloud_lane.CloudLane(session_limit=a.session_limit, operator_approval=a.operator_approval)
        argv_agent = RN.x2_agent_argv() if a.mode == "live" else RN.laptop_agent_argv(a.laptop_model)
        factory = lambda on_line: RN.Agent(argv_agent, on_line=on_line)  # noqa: E731
    app = App(a.mode, a.replay, factory, cloud, a.picks, a.idle_speedup)
    srv = make_server(a.port, app)
    print(f"live demo on http://127.0.0.1:{srv.server_address[1]}/  (mode {a.mode})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
