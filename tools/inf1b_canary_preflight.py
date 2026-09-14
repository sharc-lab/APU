"""INF-1b launch preflight (logic-only). Prefer tools/run_h1_canary_opening.py."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_h1_canary_opening import main

if __name__ == "__main__":
    raise SystemExit(main(["--logic-only"]))
