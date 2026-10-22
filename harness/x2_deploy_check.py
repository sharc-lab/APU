"""Read-only checks on evo-x2 before a job of the 2026-10-22 queue fill is queued (docs/X2_QUEUE_PLAN.md). No server
start, no model call, no queue access.

    py -3.12 C:\\apu\\ovn\\qfill\\x2_deploy_check.py --verify <dir> [<dir> ...]
        run_provenance.verify_deployed_blobs on each directory's expected_blobs.json (raises on any mismatch)
    py -3.12 C:\\apu\\ovn\\qfill\\x2_deploy_check.py --b5 <side dir> [--results C:\\apu\\ovn\\results]
        policy_compare from <side dir>: its import check (router stack), the validation preflight for llama3.1:8b and
        qwen3:14b on the B5 validation files, B5's measured windows (b5_resolution rows of x2_policy_compare.jsonl),
        the policy configs under --window-cap 8192 / 4096 resolved from those windows, the f16 + flash-attention env,
        whether the Ollama tray app runs, and the flash-attention control's reference means from B5's rows
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

OVN = Path(r"C:\apu\ovn")


def verify(dirs) -> dict:
    sys.path.insert(0, str(OVN))
    import run_provenance as rp
    out = {}
    for d in dirs:
        r = rp.verify_deployed_blobs(d, Path(d) / "expected_blobs.json")
        out[str(d)] = {"problems": r.get("problems"), "files": {f["path"]: f["actual_git_blob_sha"][:8]
                                                               for f in r.get("files", [])}}
    return out


def b5_check(side: Path, results: Path) -> dict:
    sys.path[:] = [str(side), str(side.parent)] + [p for p in sys.path if p not in (str(side), str(side.parent))]
    for m in ("policy_compare", "x2_r2_agent"):
        sys.modules.pop(m, None)
    pc = importlib.import_module("policy_compare")
    ag = pc.ag
    out = {"policy_compare_file": pc.__file__, "harness_blob": pc.git_blob_sha1(pc.__file__),
           "agent_file": ag.__file__, "agent_blob": pc.git_blob_sha1(ag.__file__),
           "import_check": pc.import_check()}
    models = list(pc.BASE_MODELS)
    vfiles = [(n, ag.read_rows(results / n)) for n in pc.PLATFORMS["evo-x2"]["rules_from"]]
    pf = ag.validation_preflight_per_model(vfiles, pc.CALL2, models, neg_arm=ag.NEG_ARM)
    out["preflight"] = {k: pf[k] for k in ("ok", "models_ok", "refused", "rules_in_use")}
    b5 = ag.read_rows(results / pc.PLATFORMS["evo-x2"]["out"])
    wins = {r["model_id"]: r["windows"] for r in b5 if r.get("record") == "b5_resolution"}
    out["b5_windows"] = {m: {k: w.get(k) for k in ("D", "L", "G", "native")} for m, w in wins.items()}
    out["cap_configs"] = {cap: {m: {p: (c["num_ctx_sent"], c["effective_ctx"], c["env"], c["alias_of"],
                                        c.get("proxy_ctx"))
                                    for p, c in pc.resolve_policies(w, cap=cap).items()}
                                for m, w in wins.items()} for cap in (8192, 4096)}
    out["fa_env"] = pc.KV_ENV[pc.FA_ENV]
    out["fa_arm"] = pc.arm_id(pc.FA_POLICY)
    out["fa_recall_timeout_s"] = pc.FA_RECALL_TIMEOUT_S
    mets = pc.metrics_for_platform(b5)
    out["b5_reference_minutes"] = {m: {p: round(v["wall_min"], 1) for p, v in c["policies"].items()
                                       if p in ("a", "g", "h")} for m, c in mets.items()}
    import x2_num_keep as nk
    out["tray_app_running"] = nk.tray_app_running()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", nargs="+", default=None)
    ap.add_argument("--b5", default=None)
    ap.add_argument("--results", default=str(OVN / "results"))
    args = ap.parse_args(argv)
    out = {}
    if args.verify:
        out["verify"] = verify([Path(d) for d in args.verify])
    if args.b5:
        out["b5"] = b5_check(Path(args.b5), Path(args.results))
    if not out:
        ap.error("nothing to do (--verify and/or --b5)")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
