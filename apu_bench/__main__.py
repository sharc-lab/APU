"""py -3.12 -m apu_bench run   --profile <device.json> --subset b3v2-small --out <dir> [--runtime fake|live] [--execute]
   py -3.12 -m apu_bench audit --profile <device.json> --sessions <sessions.jsonl> --out <dir>
                               [--server-log WINDOW=PATH ...] [--cloud-model ID] [--calib R]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    from apu_bench import audit as A
    from apu_bench import bench as B
    from apu_bench import profile as P
    from apu_bench import taskset as TS
    ap = argparse.ArgumentParser(prog="apu_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="device profile + B3 v2 subset -> result.json, report.md")
    r.add_argument("--profile", required=True)
    r.add_argument("--subset", default="b3v2-small", choices=sorted(TS.SUBSETS))
    r.add_argument("--out", required=True)
    r.add_argument("--runtime", default="fake", choices=("fake", "live"))
    r.add_argument("--execute", action="store_true", help="live only: actually run x2_r2_evalset (on the device)")
    r.add_argument("--rules-from", default=None, help="live only: an existing validation file (skips that step)")
    a = sub.add_parser("audit", help="replay recorded sessions against a device profile")
    a.add_argument("--profile", required=True)
    a.add_argument("--sessions", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--server-log", action="append", default=[], metavar="WINDOW=PATH")
    a.add_argument("--cloud-model", default=None)
    a.add_argument("--calib", type=float, default=1.0, help="estimate mode: tokenizer calibration on chars/4")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        res = B.run(args.profile, args.subset, args.out, args.runtime, args.execute, args.rules_from,
                    argv=sys.argv if argv is None else argv)
        print(json.dumps(res.get("results", res), indent=1, default=str))
        return 0
    prof, _sha, _missing = P.load(args.profile)
    logs = {}
    for item in args.server_log:
        w, _, path = item.partition("=")
        logs[int(w)] = Path(path).read_text(encoding="utf-8", errors="replace")
    res = A.run_audit(A.load_requests(args.sessions), prof, logs, args.cloud_model, args.calib)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "audit.md").write_text(A.report_md(res) + "\n", encoding="utf-8")
    print(A.report_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
