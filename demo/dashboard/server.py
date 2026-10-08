"""Stdlib HTTP server for the dashboard (no Flask/FastAPI: neither is installed, and nothing here needs them).

`App.handle(method, target)` is the whole application as a pure function of the request line, returning a
`Response`; `make_server()` wraps it in a ThreadingHTTPServer bound to 127.0.0.1. Tests drive `App` directly
(the test client) and one test also talks to a real server on an ephemeral 127.0.0.1 port. Nothing here opens a
browser or spawns a process.

Endpoints (all GET):
  /                         the single-page UI (static/index.html)
  /static/<name>            app.js, app.css (vendored, no CDN)
  /api/meta                 manifest, backend in use, hardware options, workload examples, defaults
  /api/frontier?hardware=   points and frontier per machine, stub cloud points separate
  /api/recommend?budget=&floor=&latency=&hardware=
  /api/scenario             the 4096 sessions available for the side-by-side, register rows cited
  /api/replay?session=&budget=&floor=&latency=          every step at once (polling clients)
  /api/replay/stream?session=&delay_ms=&...             the same steps as server-sent events
  /api/register[?id=]       register rows from the snapshot
  /api/source?file=&line=   one raw line of a whitelisted source file
  /api/source?file=&match=k:v,k:v   line numbers of rows matching every k:v
"""
from __future__ import annotations

import json
import mimetypes
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qs, urlsplit

from demo.dashboard import data

STATIC = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"index.html", "app.js", "app.css"}
DEFAULTS = {"budget_usd": 50.0, "quality_floor": 0.9, "latency_target_ms": 30000.0}


@dataclass
class Response:
    status: int
    content_type: str
    body: bytes = b""
    stream: Optional[Iterable[bytes]] = None
    headers: dict = field(default_factory=dict)

    def json(self):
        return json.loads(self.body)


def _json(obj, status=200) -> Response:
    return Response(status, "application/json; charset=utf-8",
                    json.dumps(obj, default=str, allow_nan=False).encode("utf-8"))


def _num(q, key, default):
    try:
        return float(q[key][0]) if key in q and q[key][0] != "" else float(default)
    except ValueError:
        raise ValueError(f"{key} must be a number")


def _list(q, key):
    if key not in q:
        return None
    return [x for v in q[key] for x in v.split(",") if x]


def _sse(event: str, obj) -> bytes:
    return f"event: {event}\ndata: {json.dumps(obj, default=str)}\n\n".encode("utf-8")


class App:
    def __init__(self, cache: data.Cache | None = None, backend: data.Backend | None = None):
        self.cache = cache or data.Cache()
        self.backend = backend or data.load_backend()

    # one place for every route
    def handle(self, method: str, target: str) -> Response:
        if method not in ("GET", "HEAD"):
            return _json({"error": "method not allowed"}, 405)
        u = urlsplit(target)
        q = parse_qs(u.query)
        path = u.path.rstrip("/") or "/"
        try:
            if path == "/":
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            route = {"/api/meta": self.meta, "/api/frontier": self.frontier, "/api/recommend": self.recommend,
                     "/api/scenario": self.scenario, "/api/replay": self.replay,
                     "/api/replay/stream": self.replay_stream, "/api/register": self.register,
                     "/api/source": self.source}.get(path)
            if route is None:
                return _json({"error": f"no route {path}"}, 404)
            return route(q)
        except KeyError as e:
            return _json({"error": f"unknown {e}"}, 404)
        except PermissionError as e:
            return _json({"error": f"not a whitelisted source: {e}"}, 403)
        except ValueError as e:
            return _json({"error": str(e)}, 400)

    def _static(self, name: str) -> Response:
        if name not in STATIC_FILES:
            return _json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        return Response(200, ctype, (STATIC / name).read_bytes())

    def meta(self, q) -> Response:
        c = self.cache
        return _json({"manifest": c.manifest, "backend": {"router": self.backend.router_source,
                                                          "pareto": self.backend.pareto_source,
                                                          "fake": self.backend.is_fake},
                      "cache_backend": c.manifest.get("backend"), "hardware": c.hardware,
                      "machines": data.machines(c), "workloads": c.workloads, "defaults": DEFAULTS,
                      "mitigation": c.mitigation})

    def frontier(self, q) -> Response:
        return _json(data.frontier_view(self.cache, self.backend, _list(q, "hardware")))

    def recommend(self, q) -> Response:
        hw = _list(q, "hardware") or data.machines(self.cache)
        return _json(data.recommendation(self.cache, self.backend, _num(q, "budget", DEFAULTS["budget_usd"]),
                                         _num(q, "floor", DEFAULTS["quality_floor"]),
                                         _num(q, "latency", DEFAULTS["latency_target_ms"]), hw))

    def scenario(self, q) -> Response:
        return _json(data.scenario_list(self.cache))

    def _replay(self, q) -> dict:
        key = (q.get("session") or [None])[0] or self.cache.r2_sessions["sessions"][0]["key"]
        return data.replay(self.cache, self.backend, key, _num(q, "budget", DEFAULTS["budget_usd"]),
                           _num(q, "floor", DEFAULTS["quality_floor"]),
                           _num(q, "latency", DEFAULTS["latency_target_ms"]))

    def replay(self, q) -> Response:
        return _json(self._replay(q))

    def replay_stream(self, q) -> Response:
        r = self._replay(q)
        delay = max(0.0, min(5000.0, _num(q, "delay_ms", 400))) / 1000.0
        head = {k: v for k, v in r.items() if k != "steps"}

        def gen():
            yield _sse("start", head)
            for i, s in enumerate(r["steps"]):
                if i and delay:
                    time.sleep(delay)
                yield _sse("step", s)
            yield _sse("done", {"n_steps": len(r["steps"])})

        return Response(200, "text/event-stream; charset=utf-8", stream=gen(),
                        headers={"Cache-Control": "no-cache"})

    def register(self, q) -> Response:
        if "id" in q:
            return _json(self.cache.register[q["id"][0]])
        return _json(self.cache.register)

    def source(self, q) -> Response:
        file = (q.get("file") or [""])[0]
        line = int(q["line"][0]) if "line" in q else None
        match = None
        if "match" in q:
            match = dict(kv.split(":", 1) for kv in q["match"][0].split(",") if ":" in kv)
        return _json(data.source(self.cache, file, line, match))


class Handler(BaseHTTPRequestHandler):
    app: App = None  # set by make_server
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        self._serve("GET")

    def do_HEAD(self):
        self._serve("HEAD")

    def do_POST(self):
        self._serve("POST")

    def _serve(self, method):
        r = self.app.handle(method, self.path)
        self.send_response(r.status)
        self.send_header("Content-Type", r.content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in r.headers.items():
            self.send_header(k, v)
        if r.stream is None:
            self.send_header("Content-Length", str(len(r.body)))
        self.end_headers()
        if method == "HEAD":
            return
        try:
            if r.stream is None:
                self.wfile.write(r.body)
            else:
                for chunk in r.stream:
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the viewer closed the tab mid-stream

    def log_message(self, fmt, *args):  # quiet: no console noise per request
        pass


def make_server(port: int = 8765, app: App | None = None) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app or App()})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    return srv
