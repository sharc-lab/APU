"""Route at session start, for the live demo (2026-10-22 operator brief, item 1).

Inputs:
  device profile   measured facts about one machine for one model: the window the runtime loads by default, the
                   larger measured local settings (the KV q8_0 window from the B5 window probes on evo-x2, the
                   integrated-GPU window on evo-t2s), each read from a register row (build_profiles below);
  context need     how many tokens this task will need: what is already in the conversation (system prompt, tool
                   schema, history, this request) plus a per-turn growth allowance for each expected turn,
                   calibrated on the B3 v2 history lengths in results/x2_evalset_tasks_real_v2.jsonl (calibrate_b3v2);
  cloud budget     USD left under the demo cap, against the projected cloud cost of the task (all-cloud-b3-cost).

Output: a dict with route "local" (and the chosen settings) or "cloud" (the whole task on the cloud from turn 1), the
numbers behind it, and a one-line reason in plain words.

Rule (fixed before the demo runs; docs/EXPERIMENTS.md "2026-10-23: Live demo: detector check and acceptance run"):
  1. need is passed in (route(..., need_tokens, ..., need_source)). Two definitions:
     - context_need: ratio * (est_now + expected_turns * growth_p90) + num_predict, where est_now is the harness's
       chars/4 estimate of the prompt so far (x2_r2_agent.prompt_tokens_est), ratio the B3 v2 tokenizer calibration
       (median of the sessions' token_calib_ratio) and growth_p90 the 90th percentile per-turn transcript growth of
       the B3 v2 default-window sessions (estimate units);
     - the live demo (operator decision of 2026-10-23, to match the headline row hybrid-route-at-start-measured): the
       projected task-turn prompt from the task definition (system prompt, tool schema, every user message so far,
       PROJ_NONFILLER_PER_TURN per turn, times PROJ_CALIB; harness/demo_live.projected_task_prompt), with margin 0.
  2. A local setting fits when need <= floor(window * (1 - margin)); margin MARGIN (as the client trim of R2's
     mitigation) by default, 0 for the headline rule.
  3. Local settings are tried in the profile's preference order (the measured faster setting first); the first that
     fits is chosen.
  4. If none fits: the cloud from turn 1, when the budget left covers the projected cloud cost; otherwise local at the
     largest measured window, flagged "will be cut off" (the demo shows that, it never hides it).
Only measured settings are offered; a setting with no measurement for that machine is not a candidate.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

try:                                    # imported as src.dse.route_at_start
    from .fits_locally import fits_locally
except ImportError:                     # imported as a top-level module (src/dse on sys.path: harness/demo_*.py)
    from fits_locally import fits_locally

MARGIN = 0.05
NUM_PREDICT = 384            # x2_r2_agent.MAX_TOKENS_PER_CALL, the answer cap every local call sends
CLOUD_MODEL = "claude-haiku-5-5"
MODEL = "qwen3:14b"
PROFILE_FILE = Path(__file__).resolve().parent / "route_profiles.json"


# ── calibration from the B3 v2 file (pure) ─────────────────────────────────────────────────────────────

def calibrate_b3v2(rows: list[dict], arm: str = "ollama_default_call2_notools") -> dict:
    """Tokenizer ratio and per-turn growth from the B3 v2 task real run (qwen3:14b). Growth is the difference of
    transcript_tokens_est between consecutive turns of one session, in the default-window arm (nothing cut)."""
    turns = [r for r in rows if r.get("record") == "r2a_turn" and r.get("arm_id") == arm]
    ratios = sorted({(r["attempt_id"], r["token_calib_ratio"]) for r in turns if r.get("token_calib_ratio")})
    by: dict = {}
    for r in turns:
        by.setdefault(r["attempt_id"], {})[r["turn_idx"]] = r["transcript_tokens_est"]
    growth = sorted(d[k] - d[k - 1] for d in by.values() for k in d if k - 1 in d)
    if not ratios or not growth:
        raise ValueError("no B3 v2 default-window turns to calibrate on")
    return {"ratio": round(statistics.median(v for _, v in ratios), 4),
            "growth_p90": growth[min(len(growth) - 1, int(0.9 * len(growth)))],
            "growth_median": statistics.median(growth), "n_growth": len(growth), "n_sessions": len(ratios),
            "source": "results/x2_evalset_tasks_real_v2.jsonl", "arm": arm}


# ── profiles from the register (pure) ──────────────────────────────────────────────────────────────────

def register_rows(text: str) -> dict:
    """{row id: value} from docs/NUMBERS_REGISTER.md table lines (value cells may contain ' | ')."""
    out = {}
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| ID") or line.startswith("|--"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split(" | ")]
        if len(parts) >= 9:
            out[parts[0]] = {"status": parts[1], "value": " | ".join(parts[2:len(parts) - 6])}
    return out


def _find(rows: dict, rid: str, pattern: str) -> str:
    row = rows.get(rid)
    if row is None or row["status"] != "VERIFIED":
        raise ValueError(f"register row {rid} missing or not VERIFIED")
    m = re.search(pattern, row["value"])
    if not m:
        raise ValueError(f"register row {rid}: pattern {pattern!r} not found")
    return m.group(1)


def build_profiles(register_text: str, b3_rows: list[dict]) -> dict:
    """The two measured device profiles for qwen3:14b and the cloud figures, every number from a register row."""
    rows = register_rows(register_text)
    x2_default = int(_find(rows, "b5-fa-control", r"qwen3:14b: .*?window (\d+)"))
    x2_q8 = int(_find(rows, "b5-kv-recall", r"evo-x2 qwen3:14b: .*?q8_0 ctx (\d+)"))
    x2_min_a = float(_find(rows, "b5-policy-metrics", r"evo-x2 qwen3:14b: a rule [\d.]+% hist [\d.]+% ([\d.]+) min"))
    x2_min_g = float(_find(rows, "b5-policy-metrics", r"evo-x2 qwen3:14b: .*?; g rule [\d.]+% hist [\d.]+% ([\d.]+) min"))
    t2s_default = int(_find(rows, "t2s-r2-real-cpu", r"qwen3:14b: loaded window \[(\d+)\]"))
    t2s_igpu = int(_find(rows, "t2s-r2-real-igpu", r"qwen3:14b: loaded window \[(\d+)\]"))
    _find(rows, "x2-device-detect-mechanism", r"(evo-x2 uses the iGPU via ROCm)")
    t2s_other = _find(rows, "T2S-vs-X2-default-ctx", r'"qwen3:8b": \{"t2s_default_ctx": (\d+)')
    clouds = {}
    for cm in ("claude-haiku-5-5", "claude-sonnet-5-5"):
        led = float(_find(rows, "all-cloud-b3-cost", cm + r": .*?ledger basis, all input at the input rate: \$([\d.]+)"))
        n = int(_find(rows, "all-cloud-b3-quality", cm + r": correct \d+/(\d+)"))
        cache = float(_find(rows, "all-cloud-b3-cost", cm + r": USD per task mean \$([\d.]+)"))
        secs = float(_find(rows, "all-cloud-b3-cost", cm + r": .*?seconds per task mean ([\d.]+)"))
        clouds[cm] = {"model": cm, "usd_per_task_ledger": round(led / n, 6), "usd_per_task_cache_priced": cache,
                      "seconds_per_task": secs, "rows": ["all-cloud-b3-cost", "all-cloud-b3-quality"]}
    cap_window = int(_find(rows, "hybrid-route-at-start-measured", r"evo-x2 ollama_ctx_(8192) \(window 8192\)"))
    return {
        "model": MODEL,
        "calibration": calibrate_b3v2(b3_rows),
        "cloud": clouds[CLOUD_MODEL],
        "clouds": clouds,
        "devices": {
            "evo-x2-8192": {
                "label": "AMD workstation, constrained memory (8192)",
                "default_window": cap_window,
                "cloud_model": "claude-sonnet-5-5",
                "settings": [
                    {"id": "default", "label": f"constrained memory ({cap_window:,})", "window": cap_window,
                     "server_env": {}, "why": "the simulated device window of the headline row",
                     "rows": ["hybrid-route-at-start-measured"]},
                ],
                "simulated": True,
                "rows": ["hybrid-route-at-start-measured", "all-cloud-b3-cost"],
            },
            "evo-x2": {
                "label": "AMD workstation",
                "default_window": x2_default,
                "settings": [
                    {"id": "kv_q8_0", "label": "compact memory (KV q8_0, flash attention)", "window": x2_q8,
                     "server_env": {"OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1"},
                     "why": f"measured faster: {x2_min_g} min against {x2_min_a} min per B5 session",
                     "rows": ["b5-kv-recall", "b5-policy-metrics", "b5-fa-control"]},
                    {"id": "default", "label": "runtime default", "window": x2_default, "server_env": {},
                     "why": "the window the runtime loads by default", "rows": ["b5-fa-control"]},
                ],
                "rows": ["b5-fa-control", "b5-kv-recall", "b5-policy-metrics", "x2-device-detect-mechanism"],
            },
            "evo-t2s": {
                "label": "Intel laptop",
                "default_window": t2s_default,
                "cloud_model": "claude-haiku-5-5",
                "settings": [
                    {"id": "igpu", "label": "Intel graphics-chip setting (OLLAMA_IGPU_ENABLE=1)", "window": t2s_igpu,
                     "server_env": {"OLLAMA_IGPU_ENABLE": "1"},
                     "why": "the window measured with the integrated GPU enabled", "rows": ["t2s-r2-real-igpu"]},
                    {"id": "default", "label": "runtime default", "window": t2s_default, "server_env": {},
                     "why": "the window the runtime loads by default (CPU)", "rows": ["t2s-r2-real-cpu"]},
                ],
                "not_measured": ["KV q8_0 window (B5 policies g and h are not run on evo-t2s)"],
                "note": f"default window {t2s_other} also measured for qwen3:8b (T2S-vs-X2-default-ctx)",
                "rows": ["t2s-r2-real-cpu", "t2s-r2-real-igpu", "T2S-vs-X2-default-ctx", "x2-device-detect-mechanism"],
            },
        },
    }


def load_profiles(path: Path = PROFILE_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── the decision (pure) ────────────────────────────────────────────────────────────────────────────────

def context_need(est_now: int, expected_turns: int, calib: dict, num_predict: int = NUM_PREDICT) -> int:
    return math.ceil(calib["ratio"] * (est_now + expected_turns * calib["growth_p90"])) + num_predict


def fits(need: int, window: int, margin: float = MARGIN, predictor=None) -> bool:
    """need <= floor(window * (1 - margin)), through the FitsLocallyPredictor interface (src/dse/fits_locally.py)."""
    return fits_locally(need, window, margin, predictor)["fits"]


def route(profile: dict, device: str, need_tokens: int, cloud_budget_usd: float, expected_turns: int = 1,
          margin: float = MARGIN, need_source: str = "context_need", predictor=None) -> dict:
    """The routing decision for one task on one device. profile: build_profiles / load_profiles output.
    need_tokens: context_need(...), or in the live demo the projected task-turn prompt of the headline row
    hybrid-route-at-start-measured (need_source names which); margin: MARGIN, or 0 for the headline's rule
    (local iff projected <= window). The cloud model and price are the device's (cloud_model in the profile).
    predictor: a src/dse/fits_locally.FitsLocallyPredictor (None: the projection baseline)."""
    dev = profile["devices"][device]
    cloud = profile["clouds"][dev.get("cloud_model", profile["cloud"]["model"])]
    need = int(need_tokens)
    cloud_cost = round(cloud["usd_per_task_ledger"] * max(expected_turns, 1), 6)
    base = {"device": device, "device_label": dev["label"], "need_tokens": need, "need_source": need_source,
            "margin": margin, "expected_turns": expected_turns, "default_window": dev["default_window"],
            "cloud_budget_usd": cloud_budget_usd, "cloud_cost_projected_usd": cloud_cost}
    for s in dev["settings"]:
        if fits(need, s["window"], margin, predictor):
            fits_default = fits(need, dev["default_window"], margin, predictor)
            extra = "" if s["id"] == "default" else (
                f"; {s['why']}" if fits_default else f"; the default {dev['default_window']:,} would cut it off")
            return {**base, "route": "local", "setting": s["id"], "setting_label": s["label"],
                    "window": s["window"], "server_env": s["server_env"],
                    "reason": f"Runs here: needs about {need:,} tokens of memory for the conversation, "
                              f"{s['label']} holds {s['window']:,}{extra}."}
    largest = max(dev["settings"], key=lambda s: s["window"])
    if cloud_budget_usd >= cloud_cost:
        return {**base, "route": "cloud", "setting": None, "model": cloud["model"], "window": None,
                "server_env": None,
                "reason": f"Cloud from the start: needs about {need:,} tokens, more than any measured setting on the "
                          f"{dev['label']} holds (largest {largest['window']:,})."}
    return {**base, "route": "local", "setting": largest["id"], "setting_label": largest["label"],
            "window": largest["window"], "server_env": largest["server_env"], "will_be_cut_off": True,
            "reason": f"Runs here but will be cut off: needs about {need:,} tokens, the largest measured setting "
                      f"holds {largest['window']:,}, and the cloud budget left (${cloud_budget_usd:.2f}) is below "
                      f"the projected ${cloud_cost:.4f}."}


def main(argv=None) -> int:
    """Rebuild src/dse/route_profiles.json from the register and the B3 v2 file (run from the repo root)."""
    repo = Path(__file__).resolve().parents[2]
    text = (repo / "docs" / "NUMBERS_REGISTER.md").read_text(encoding="utf-8")
    with open(repo / "results" / "x2_evalset_tasks_real_v2.jsonl", encoding="utf-8") as fh:
        b3 = [json.loads(line) for line in fh if line.strip()]
    prof = build_profiles(text, b3)
    PROFILE_FILE.write_text(json.dumps(prof, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: prof[k] for k in ("calibration", "cloud")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
