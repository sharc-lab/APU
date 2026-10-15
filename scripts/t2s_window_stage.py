"""Staged evo-t2s queue entries for the 2026-10-26 to 2026-11-08 window: B5 (policy comparison, harness/policy_compare.py
--platform evo-t2s) and B6 (qwen3:32b and llama3.3:70b R2 at num_ctx 4096 and at the iGPU-enabled default window,
harness/t2s_r2_agent.py --runtime-config igpu_enable). NOT loaded anywhere: the evo-t2s agent references the file and
loads what it decides to run (scripts/t2s_week_queue.py `load <file>` on evo-t2s appends entries as pending).

  py -3.12 scripts/t2s_window_stage.py build          # writes queues/t2s_window_20261026_staged.json
  py -3.12 scripts/t2s_window_stage.py dry-run        # checks every entry with the harness parsers; no machine, no SSH
  py -3.12 scripts/t2s_window_stage.py displacement   # what B5 and B6 displace in the 72 h week plan; recommended cut

HOURS ARE ESTIMATES (labelled per entry). B5: policy_compare.estimate_platform("evo-t2s") (evo-x2 session minutes at the
same window x the iGPU/X2 latency ratio). B6: evo-x2 llama3.1:8b R2 timings x the nominal parameter-count ratio of the
model to llama3.1:8b x the iGPU/X2 ratio; the iGPU-enabled default window of the large models is assumed 32768 (the
value measured for llama3.1:8b); nothing here is a measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "scripts"))

import t2s_week_queue as wq  # noqa: E402

STAGE_PATH = REPO / "queues" / "t2s_window_20261026_staged.json"
WINDOW = ("2026-10-26", "2026-11-08")
ID_PREFIX = "t2s_win_"
BUDGET_H = wq.BUDGET_H
B6_MODELS = ("qwen3:32b", "llama3.3:70b")
# nominal parameter counts (billions, model cards): the B6 time scaling input, an assumption
PARAMS_B = {"llama3.1:8b": 8.0, "qwen3:32b": 32.8, "llama3.3:70b": 70.6}
B5_RULES_FROM = "results/t2s_r2_validation_igpu_v1.jsonl"
DEPLOY_B5 = ["harness/policy_compare.py", "harness/x2_r2_agent.py", "harness/x2_r2_client_trim.py",
             "harness/x2_num_keep.py", "harness/prompt_token_check.py",
             "src/__init__.py -> C:\\apu\\ovn\\src", "src/dse/__init__.py, src/dse/router.py, src/dse/envelope_data.json "
             "-> C:\\apu\\ovn\\src\\dse", "src/cloud/__init__.py, src/cloud/client.py -> C:\\apu\\ovn\\src\\cloud",
             "analysis/envelope_model.py, analysis/make_failure_map.py -> C:\\apu\\ovn\\analysis"]
DEPLOY_B6 = ["harness/x2_r2_agent.py (this branch: think values for qwen3:32b / llama3.3:70b)",
             "harness/t2s_r2_agent.py", "Ollama tags qwen3:32b and llama3.3:70b in evo-t2s's store (not checked here)"]


def _slug(model):
    return model.replace(":", "_").replace(".", "")


def b6_estimates() -> dict:
    """{model: {"val": (h, how), "real_4096": (h, how), "real_igpu": (h, how)}} (labelled estimates)."""
    v = {k: x[0] for k, x in wq.ESTIMATE_INPUTS.items()}
    g = v["t2s_igpu_over_x2_ollama"]
    out = {}
    for m in B6_MODELS:
        r = PARAMS_B[m] / PARAMS_B["llama3.1:8b"]
        val = v["x2_val_v2_total_min"] / 2 * r * g / 60
        r4 = 3 * v["x2_real_4096_llama_min"] * r * g / 60
        rg = 3 * v["x2_real_32768_llama_min"] * r * g / 60
        base = f"param ratio {r:.2f} x iGPU/X2 {g}"
        out[m] = {"val": (val, f"x2_r2_validation_v2 {v['x2_val_v2_total_min']} min / 2 models x {base}"),
                  "real_4096": (r4, f"3 seeds x X2 llama 4096 session {v['x2_real_4096_llama_min']} min x {base}"),
                  "real_igpu": (rg, f"3 seeds x X2 llama 32768 session {v['x2_real_32768_llama_min']} min x {base} "
                                    f"(iGPU default window assumed 32768)")}
    return out


def b5_estimate() -> tuple[float, str]:
    import policy_compare as pc
    e = pc.estimate_platform("evo-t2s")
    return e["total_h"], f"policy_compare.estimate_platform('evo-t2s'): {e['assumptions']}"


def build_entries() -> list[dict]:
    py = wq.T2S_PY
    b5_h, b5_how = b5_estimate()
    items = [{
        "id": ID_PREFIX + "b5_policy_compare",
        "cmd": [py, "policy_compare.py", "--platform", "evo-t2s", "--out", "results/t2s_policy_compare.jsonl",
                "--rules-from", B5_RULES_FROM],
        "status": "pending", "est_hours": round(b5_h, 2), "est_basis": "scaled (ESTIMATE)",
        "note": f"B5 policy comparison (policies a-h, llama3.1:8b, OLLAMA_IGPU_ENABLE=1 base env), pre-registered in "
                f"docs/FINDINGS.md 2026-10-14. ESTIMATE {b5_h:.1f} h ({b5_how}). Releases via tq.advance().",
        "expected_out": "results/t2s_policy_compare.jsonl", "gate": {"requires_done": "t2s_wk_r2_val_igpu"},
        "deploy": DEPLOY_B5, "track": "B5"}]
    est = b6_estimates()
    for m in B6_MODELS:
        s = _slug(m)
        val_out = f"results/t2s_b6_validation_igpu_{s}_v1.jsonl"
        common = ["--models", m, "--call2-tools", "off"]
        e = est[m]
        items.append({"id": f"{ID_PREFIX}b6_val_igpu_{s}",
                      "cmd": [py, "t2s_r2_agent.py", "--mode", "validation", "--runtime-config", "igpu_enable",
                              "--out", val_out, *common],
                      "status": "pending", "est_hours": round(e["val"][0], 2), "est_basis": "scaled (ESTIMATE)",
                      "note": f"B6 R2 validation, {m}, iGPU enabled. ESTIMATE {e['val'][0]:.1f} h ({e['val'][1]}).",
                      "expected_out": val_out, "gate": {"requires_done": "t2s_wk_preflight"}, "deploy": DEPLOY_B6,
                      "track": "B6"})
        for tier, key, label in (("ollama_ctx_4096", "real_4096", "4096"), ("ollama_default", "real_igpu", "igpu")):
            out = f"results/t2s_b6_real_{label}_{s}_v1.jsonl"
            items.append({"id": f"{ID_PREFIX}b6_real_{label}_{s}",
                          "cmd": [py, "t2s_r2_agent.py", "--mode", "real", "--runtime-config", "igpu_enable",
                                  "--out", out, *common, "--tiers", tier, "--order", "seed_major", "--rules-from",
                                  val_out, "--require-validation-gates", "--per-model-refusal"],
                          "status": "pending", "est_hours": round(e[key][0], 2), "est_basis": "scaled (ESTIMATE)",
                          "note": f"B6 R2 real, {m}, {tier} under OLLAMA_IGPU_ENABLE=1, 3 seeds x 40 turns. ESTIMATE "
                                  f"{e[key][0]:.1f} h ({e[key][1]}).",
                          "expected_out": out, "gate": {"requires_done": f"{ID_PREFIX}b6_val_igpu_{s}"},
                          "deploy": DEPLOY_B6, "track": "B6"})
    return items


def check_staged(items, week_ids) -> dict:
    """Every staged entry with the harness's own parsers (no side effects): interpreter, flat script name, t2s_ output,
    no evo-x2 reference, gate present in the week queue or earlier in this file, and the argv accepted (t2s_r2_agent's
    dry run against x2_r2_agent's parser; policy_compare's parser and its evo-t2s estimate)."""
    import policy_compare as pc
    import t2s_r2_agent as w
    import x2_r2_agent as x2
    res, seen = [], set(week_ids)
    for it in items:
        problems = []
        cmd = it["cmd"]
        if not it["id"].startswith(ID_PREFIX):
            problems.append("id prefix")
        if cmd[0] != wq.T2S_PY:
            problems.append("interpreter is not evo-t2s's")
        if "/" in cmd[1] or "\\" in cmd[1] or not (REPO / "harness" / cmd[1]).exists():
            problems.append(f"script {cmd[1]!r} is not a flat harness/ name")
        joined = " ".join(cmd).lower()
        problems += [f"evo-x2 reference {b!r}" for b in ("ritz", "evo-x2", "100.118.33.76") if b in joined]
        argv = cmd[2:]
        for o in [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--out"]:
            if not Path(o).name.startswith("t2s_"):
                problems.append(f"output {o} not t2s_-prefixed")
            if (REPO / o).exists():
                problems.append(f"output {o} already exists")
        gate = (it.get("gate") or {}).get("requires_done")
        if gate and gate not in seen:
            problems.append(f"gate {gate!r} is neither in the week queue nor earlier in this file")
        info = {}
        if cmd[1] == "t2s_r2_agent.py":
            rep = w.dry_run(*w.build_parser().parse_known_args(argv), x2)
            info = {"x2_parse_ok": rep["x2_parse_ok"], "missing_capabilities": rep["missing_capabilities"]}
            if not rep["x2_parse_ok"]:
                problems.append(f"x2_r2_agent rejects: {rep['x2_parse_error']}")
            if rep["missing_capabilities"]:
                problems.append(f"missing capabilities {rep['missing_capabilities']}")
            for m in (rep["forwarded_argv"][rep["forwarded_argv"].index("--models") + 1]).split(","):
                if m not in x2.MODELS:
                    problems.append(f"x2_r2_agent.MODELS has no think value for {m}")
        elif cmd[1] == "policy_compare.py":
            a = pc.build_arg_parser().parse_args(argv)
            if a.platform != "evo-t2s":
                problems.append("policy_compare entry is not --platform evo-t2s")
            info = {"plan_models": list(pc.PLATFORMS["evo-t2s"]["models"]), "est_hours": it["est_hours"]}
        res.append({"id": it["id"], "ok": not problems, "problems": problems, **info})
        seen.add(it["id"])
    return {"ok": all(r["ok"] for r in res), "entries": res, "fresh_build_matches_file":
            (STAGE_PATH.exists() and json.loads(STAGE_PATH.read_text(encoding="utf-8")) == items)}


RECOMMENDED_CUT = ["t2s_wk_preflight", "t2s_wk_r2_val_igpu", "t2s_win_b5_policy_compare", "t2s_wk_r2_val_cpu",
                   "t2s_wk_r2_real_cpu", "t2s_wk_px2i", "t2s_wk_r2_mechanism_cpu", "t2s_win_b6_val_igpu_qwen3_32b",
                   "t2s_win_b6_real_4096_qwen3_32b"]
CUT_RATIONALE = {
    "t2s_wk_r2_real_igpu": "B5 policy (a) on evo-t2s is the same arm (ollama_default under OLLAMA_IGPU_ENABLE=1, call-2 "
                           "off, 40 turns) on seeds 20260901-03, i.e. 3 of this entry's 5 seeds",
    "t2s_wk_r2_mitigation_cpu": "B5 policy (c) runs the same client-side trimming on evo-t2s (on the iGPU platform)",
    "t2s_win_b6_real_igpu_qwen3_32b": "the iGPU-window tier of B6 does not fit; the 4096 tier carries the H-scale "
                                      "rule-loss half",
    "t2s_win_b6_val_igpu_llama33_70b": "llama3.3:70b alone is estimated above the remaining budget",
    "t2s_win_b6_real_4096_llama33_70b": "as above",
    "t2s_win_b6_real_igpu_llama33_70b": "as above",
}


def displacement(staged=None, week=None) -> dict:
    staged = staged if staged is not None else build_entries()
    week = week if week is not None else json.loads(wq.QUEUE_PATH.read_text(encoding="utf-8"))
    hours = {it["id"]: it["est_hours"] for it in week + staged}
    week_h = sum(it["est_hours"] for it in week)
    b5 = sum(it["est_hours"] for it in staged if it["track"] == "B5")
    b6 = sum(it["est_hours"] for it in staged if it["track"] == "B6")

    def beyond(order):
        cum, out = 0.0, []
        for i in order:
            cum += hours[i]
            if cum > BUDGET_H:
                out.append(i)
        return out
    first = [it["id"] for it in staged] + [it["id"] for it in week]
    cut_h = sum(hours[i] for i in RECOMMENDED_CUT)
    dropped = [i for i in hours if i not in RECOMMENDED_CUT]
    return {"label": "ESTIMATES (hours), not measurements", "budget_h": BUDGET_H, "window": WINDOW,
            "week_plan_h": round(week_h, 2), "b5_h": round(b5, 2), "b6_h": round(b6, 2),
            "b6_by_entry_h": {it["id"]: it["est_hours"] for it in staged if it["track"] == "B6"},
            "all_h": round(week_h + b5 + b6, 2),
            "if_b5_b6_first_displaced_week_entries": [i for i in beyond(first) if i.startswith("t2s_wk_")],
            "if_b5_b6_first_displaced_h": round(sum(hours[i] for i in beyond(first) if i.startswith("t2s_wk_")), 2),
            "if_appended_beyond_budget": [i for i in beyond([it["id"] for it in week] + [it["id"] for it in staged])],
            "recommended_cut": RECOMMENDED_CUT, "recommended_cut_h": round(cut_h, 2),
            "recommended_dropped": {i: {"h": hours[i], "why": CUT_RATIONALE.get(i, "")} for i in dropped}}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "dry-run"
    if cmd == "build":
        STAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STAGE_PATH.write_text(json.dumps(build_entries(), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {STAGE_PATH}")
        return 0
    if cmd == "dry-run":
        week_ids = [it["id"] for it in json.loads(wq.QUEUE_PATH.read_text(encoding="utf-8"))]
        rep = check_staged(json.loads(STAGE_PATH.read_text(encoding="utf-8")), week_ids)
        print(json.dumps(rep, indent=1))
        return 0 if rep["ok"] and rep["fresh_build_matches_file"] else 1
    if cmd == "displacement":
        print(json.dumps(displacement(), indent=1))
        return 0
    raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    sys.exit(main())
