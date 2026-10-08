"""py -3.12 -m demo.dashboard [--port 8765] [--rebuild-cache]

Serves the dashboard on 127.0.0.1 only, headless: it prints the URL and never opens a browser or a console.
Ctrl+C stops it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.dashboard import build_cache, server  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m demo.dashboard", description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="rebuild demo/dashboard/cache from the committed result files before serving")
    a = ap.parse_args(argv)
    if a.rebuild_cache:
        build_cache.build()
    srv = server.make_server(a.port)
    app = srv.RequestHandlerClass.app
    print(f"dashboard on http://127.0.0.1:{srv.server_address[1]}/  (router: {app.backend.router_source}; "
          f"pareto: {app.backend.pareto_source})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
