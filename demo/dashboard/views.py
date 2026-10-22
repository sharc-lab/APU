"""The four demo views (2026-10-28 freeze): silent failure, per-device recommendation, cost panel, live escalation.

Two halves:
  build_*(register, repo)   build time, called by build_cache.py: read docs/NUMBERS_REGISTER.md rows (already
                            parsed) and committed result files, write plain dicts into demo/dashboard/cache/.
  *_view(cache, ...)        request time, called by server.py: pure functions of the cached snapshot (the cost view
                            also looks for docs/OCT29_COST_SUMMARY.md and docs/IMPACT.md on disk, so a summary that
                            lands after the cache was built still shows).

Rules every view follows:
  - A number comes from a register row (src {"register": id}) or a result-file row (src {"file", "line"}); nothing is
    typed in here. A register value is parsed with a regex; when the row's wording changes and the regex no longer
    matches, the view says "not parsed" and shows the row verbatim instead of a number.
  - Every item carries a label: measured, projected, simulated, priced, not executed, or not measured.
  - No router savings figure unless a register row supports it (row b5-verdict is "refuted" on evo-x2).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

MEASURED = "measured"
PROJECTED = "projected"
SIMULATED = "simulated"
PRICED = "priced, not executed"
NOT_MEASURED = "not measured"
# A measured configuration whose attribution waits on the flash-attention control (operator 2026-10-21).
PENDING_FA = "pending flash-attention control"
# A figure that is being re-measured and is not shown (operator 2026-10-22: the v1 cloud redo was a broken protocol
# run, thinking consumed the output budget; see docs/FINDINGS.md, cloud redo diagnosis).
BEING_MEASURED = "being measured"
LABELS = (MEASURED, PROJECTED, SIMULATED, PRICED, NOT_MEASURED, PENDING_FA, BEING_MEASURED)
FIX_COST_LEAD = "Cost to fix a wrong answer: being measured"

CHECKPOINTS = (5, 10, 15, 20, 25, 30, 35, 40)  # analysis/r2_real_report.py CHECKPOINTS_FULL (8 intact counts)
T2S_FILES = {"cpu": "results/t2s_r2_real_cpu_v1.jsonl", "igpu": "results/t2s_r2_real_igpu_v1.jsonl"}
T2S_ROWS = {"cpu": "t2s-r2-real-cpu", "igpu": "t2s-r2-real-igpu"}
B3_FILE = "results/x2_evalset_tasks_real_v2.jsonl"
B5_FILE = "results/x2_policy_compare.jsonl"
COST_SUMMARY_DOC = "docs/OCT29_COST_SUMMARY.md"
IMPACT_DOC = "docs/IMPACT.md"
DEFAULT_MAX_TIMEOUT_SHARE = 0.05
CLOUD_REDO_FILE = "results/cloud_redo_v1.jsonl"
# Result files the views read at build time (added to the cache manifest with their sha256, and to the source viewer).
VIEW_SOURCES = [*T2S_FILES.values(), B3_FILE, B5_FILE, CLOUD_REDO_FILE]

SILENT_IDS = ["t2s-r2-real-cpu", "t2s-r2-real-igpu", "x2-evalset-tasks-real-v2-wrong-rate",
              "x2-evalset-tasks-real-v2-ci-units", "R2-v1b-gated-per-model", "R2-real-v1b-gated-kill"]
DEVICE_IDS = ["b5-policy-metrics", "b5-kv-recall", "b5-fa-control", "cost-cut-config-time", "R2-v1b-gated-per-model",
              "x2-evalset-tasks-real-v2-wrong-rate", "t2s-r2-real-cpu", "t2s-r2-real-igpu", "t2s-r2-mitigation-cpu",
              "R2-mitigation-v1-verdict", "x2-v3-usability", "H-scale-verdict", "x2-scale-real-default-tier",
              "num-keep-guardrail-test", "num-keep-qwen3-14b", "x2-evalset-tasks-real-v2-qwen3-8b-wrong-rate",
              "x2-evalset-tasks-real-v2-mistral-small32-wrong-rate", "T2S-vs-X2-default-ctx"]
COST_IDS = ["x2-evalset-tasks-real-v2-wrong-rate", "cost-cut-config-time", "impact-P3", "impact-total", "b5-verdict",
            "b5-policy-metrics", "impact-hour-value"]


# Clauses of register rows the demo must not show, even in the click-through source dialog (operator 2026-10-21: the
# cloud redo's fix rate is under diagnosis). The redaction is marked in place, so the row is visibly not verbatim.
REDACTIONS = [("cloud-redo-cost-per-wrong-task", re.compile(r"cloud correct \d+/\d+ \([\d.]+%\)"),
               "[cloud fix rate withheld in the demo: under diagnosis]")]
# Rows withheld whole (operator 2026-10-22): results/cloud_redo_v1.jsonl is a broken protocol run, so its row and the
# two impact rows built on its cost per wrong task are not shown until the corrected redo lands.
_V1_NOTE = ("the v1 cloud redo was a broken protocol run (thinking consumed the output budget; docs/FINDINGS.md, "
            "cloud redo diagnosis)")
WITHHELD_ROWS = {"cloud-redo-cost-per-wrong-task": f"[withheld in the demo: {_V1_NOTE}. Cost to fix a wrong answer: "
                                                   "being measured.]",
                 "impact-v2-P1": f"[withheld in the demo: it includes the cost per wrong task from the v1 cloud redo; "
                                 f"{_V1_NOTE}. Being measured.]",
                 "impact-ranking-v2": f"[withheld in the demo: its P1 row includes the v1 cloud redo cost; {_V1_NOTE}. "
                                      "Being measured.]"}


def redact_register(register: dict) -> dict:
    out = dict(register)
    for rid, pat, repl in REDACTIONS:
        if rid in out and pat.search(out[rid].get("value") or ""):
            out[rid] = dict(out[rid], value=pat.sub(repl, out[rid]["value"]), redacted=True)
    for rid, text in WITHHELD_ROWS.items():
        if rid in out:
            out[rid] = dict(out[rid], value=text, redacted=True)
    return out


def reg_src(rid: str) -> dict:
    return {"register": rid}


def _row(register: dict, rid: str) -> dict | None:
    return register.get(rid)


def _verified(register: dict, rid: str) -> bool:
    r = _row(register, rid)
    return bool(r) and r["status"] not in ("PENDING", "UNSUPPORTED")


def _ints(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", s or "")]


# ── register value parsers (pure; None or [] when the wording does not match) ─────────────────────────────

_T2S_RE = re.compile(r"(?P<model>[\w.:-]+): loaded window \[(?P<win>[\d, ]*)\], first over window turn "
                     r"\[(?P<fo>[^\]]*)\]; gated (?P<s>\d+)/(?P<n>\d+) silent, (?P<after>\d+) after window exceeded, "
                     r"(?P<base>\d+) baseline-type \(gate excludes (?P<excl>[^)]*)\); intact (?P<intact>[\d/]+)")


def parse_t2s_real(value: str) -> list[dict]:
    out = []
    for m in _T2S_RE.finditer(value or ""):
        out.append({"model": m["model"], "loaded_window": _ints(m["win"]), "first_over_window_turn": _ints(m["fo"]),
                    "gated_silent": int(m["s"]), "n": int(m["n"]), "silent_after_window": int(m["after"]),
                    "baseline_type": int(m["base"]), "gate_excludes": m["excl"], "intact": _ints(m["intact"])})
    return out


_GATED_RE = re.compile(r"(?P<tier>ollama_default|num_ctx_\d+) (?P<model>[\w.:-]+): (?P<s>\d+)/(?P<n>\d+) silent, "
                       r"(?P<after>\d+) after window exceeded, (?P<base>\d+) baseline-type; intact (?P<intact>[\d/]+)")


def parse_gated_per_model(value: str) -> list[dict]:
    return [{"tier": m["tier"], "model": m["model"], "gated_silent": int(m["s"]), "n": int(m["n"]),
             "silent_after_window": int(m["after"]), "baseline_type": int(m["base"]), "intact": _ints(m["intact"])}
            for m in _GATED_RE.finditer(value or "")]


_WRONG_RE = re.compile(r"(?P<tier>ollama_default|ollama_ctx_\d+(?:_numkeep)?): wrong (?P<w>\d+)/(?P<n>\d+) "
                       r"\((?P<pct>[\d.]+)%\) \(silent (?P<silent>\d+), surfaced (?P<surf>\d+)\), loaded context "
                       r"\[(?P<ctx>[\d, ]*)\]")
_EXCESS_RE = re.compile(r"(?P<tier>ollama_ctx_\d+(?:_numkeep)?) excess over default (?P<ex>[+-][\d.]+) points "
                        r"\(95% CI (?P<lo>[+-][\d.]+) to (?P<hi>[+-][\d.]+)")


def parse_wrong_rate(value: str) -> list[dict]:
    excess = {m["tier"]: {"excess_points": float(m["ex"]), "ci_lo": float(m["lo"]), "ci_hi": float(m["hi"])}
              for m in _EXCESS_RE.finditer(value or "")}
    out = []
    for m in _WRONG_RE.finditer(value or ""):
        out.append({"tier": m["tier"], "wrong": int(m["w"]), "n": int(m["n"]), "pct": float(m["pct"]),
                    "silent": int(m["silent"]), "surfaced": int(m["surf"]), "loaded_context": _ints(m["ctx"]),
                    **excess.get(m["tier"], {"excess_points": None, "ci_lo": None, "ci_hi": None})})
    return out


_B5_SEG_RE = re.compile(r"evo-x2 (?P<model>[\w.:-]+): (?P<body>.*?)frontier \[(?P<fr>[^\]]*)\]; complete (?P<c>\w+)")
_B5_POL_RE = re.compile(r"(?P<p>[a-i]) rule (?P<rule>[\d.]+)% hist (?P<hist>[\d.]+)% (?P<min>[\d.]+) min "
                        r"\$(?P<usd>[\d.]+)(?: \(=(?P<alias>[a-i])\))?")


def parse_b5_metrics(value: str) -> dict:
    out = {}
    for seg in _B5_SEG_RE.finditer(value or ""):
        pols = {m["p"]: {"rule_pct": float(m["rule"]), "hist_pct": float(m["hist"]), "minutes": float(m["min"]),
                         "usd": float(m["usd"]), "alias_of": m["alias"]} for m in _B5_POL_RE.finditer(seg["body"])}
        out[seg["model"]] = {"policies": pols, "frontier": re.findall(r"[a-i]", seg["fr"]),
                             "complete": seg["c"] == "True"}
    return out


_KV_SEG_RE = re.compile(r"^evo-x2 (?P<model>[\w.:-]+): (?P<body>.*)$")
_KV_RE = re.compile(r"(?P<prec>f16|q8_0|q4_0) ctx (?P<ctx>\d+) recall (?P<rec>[\d.]+)%")


def parse_kv_recall(value: str) -> dict:
    out = {}
    for part in (value or "").split(" / "):
        seg = _KV_SEG_RE.match(part.strip())
        if seg:
            out[seg["model"]] = {m["prec"]: {"ctx": int(m["ctx"]), "recall_pct": float(m["rec"])}
                                 for m in _KV_RE.finditer(seg["body"])}
    return out


def parse_mitigation(value: str) -> dict:
    """'num_ctx_4096 llama3.1:8b recovered' -> {('num_ctx_4096', 'llama3.1:8b'): 'recovered'}"""
    return {f"{m[1]}|{m[2]}": m[3] for m in
            re.finditer(r"(num_ctx_\d+) ([\w.:-]+) (recovered|not_recovered)", value or "")}


def parse_usability(value: str) -> dict:
    """'qwen3-14b/ollama_default 0/340 (0.0%)' -> {'qwen3-14b|ollama_default': {...}}"""
    return {f"{m[1]}|{m[2]}": {"timeouts": int(m[3]), "n": int(m[4]), "share": int(m[3]) / int(m[4])}
            for m in re.finditer(r"([\w.:-]+)/(llama_server|ollama_default) (\d+)/(\d+) \(", value or "")}


def parse_cloud_redo(value: str) -> dict | None:
    m = re.search(r"USD per wrong task mean \$(?P<mean>[\d.]+), median \$(?P<med>[\d.]+), total \$(?P<tot>[\d.]+)",
                  value or "")
    if not m:
        return None
    model = re.search(r"model ([\w.-]+)", value)
    tiers = {t[1]: {"n": int(t[2]), "mean_usd": float(t[3]), "median_usd": float(t[4])} for t in re.finditer(
        r"(ollama_default|ollama_ctx_\d+(?:_numkeep)?) n=(\d+) mean \$([\d.]+) median \$([\d.]+)", value)}
    mode = re.search(r"mode (executed|priced, not executed)", value)
    ok = re.search(r"cloud correct (\d+)/(\d+) \(([\d.]+)%\)", value)
    secs = re.search(r"mean ([\d.]+) s per task", value)
    executed = bool(mode) and mode[1] == "executed"
    return {"mean_usd": float(m["mean"]), "median_usd": float(m["med"]), "total_usd": float(m["tot"]),
            "model": model[1] if model else None, "per_tier": tiers, "executed": executed,
            "mode_priced_not_executed": not executed,
            "cloud_correct": int(ok[1]) if ok else None, "cloud_graded": int(ok[2]) if ok else None,
            "cloud_correct_pct": float(ok[3]) if ok else None, "seconds_mean": float(secs[1]) if secs else None}


def b5_verdict(register: dict) -> str | None:
    r = _row(register, "b5-verdict")
    if not r:
        return None
    m = re.match(r"\s*(\w+)", r["value"])
    return m[1] if m else None


FA_READ_FA = "the B5 speedup is flash attention, not KV precision"
FA_READ_KV = "the B5 speedup is KV precision, not flash attention"


def parse_fa_control(value: str) -> dict:
    """Row b5-fa-control per model: (i) f16 + flash attention minutes per session, window, valid flag, reading
    (harness/policy_compare.py fa_reading: flash attention / KV precision / both contribute / neither reading /
    not read ...)."""
    out = {}
    for part in (value or "").split(" / "):
        m = re.match(r"\s*([\w.:-]+): \(i\) f16\+FA (None|[\d.]+) min per session", part)
        if not m:
            continue
        win = re.search(r"window (\d+)", part)
        valid = re.search(r"valid (True|False)", part)
        rd = re.search(r"reading: ([^;()]+?)(?:\s*\(tolerance [^)]*\))?\s*$", part)
        out[m[1]] = {"minutes_i": None if m[2] == "None" else float(m[2]),
                     "window": int(win[1]) if win else None, "valid": bool(valid) and valid[1] == "True",
                     "reading": rd[1].strip() if rd else None}
    return out


def fa_control_for(register: dict, model: str) -> dict | None:
    """The control's result for one model, only when the row is VERIFIED, the model's run is valid and it states a
    reading (not 'not read ...'). None means the attribution is still pending."""
    if not _verified(register, "b5-fa-control"):
        return None
    r = parse_fa_control(register["b5-fa-control"]["value"]).get(model)
    if not r or not r["valid"] or not r["reading"] or r["reading"].startswith("not read"):
        return None
    return r


def fa_control_reading(register: dict) -> str | None:
    """The flash-attention control's reading, only when row b5-fa-control is VERIFIED and states one (harness
    policy_compare.fa_control_summary: 'the B5 speedup is flash attention, not KV precision' or the reverse). A VERIFIED
    row whose reading is 'not read (incomplete)' returns None: the attribution is still open. Per model, joined."""
    if not _verified(register, "b5-fa-control"):
        return None
    parts = []
    for model in parse_fa_control(register["b5-fa-control"]["value"]):
        r = fa_control_for(register, model)
        if r:
            parts.append(f"{model}: {r['reading']}")
    return "; ".join(parts) or None


def router_savings_supported(register: dict) -> bool:
    """A router savings figure may be shown only if register row b5-verdict exists, is VERIFIED and is not refuted."""
    v = b5_verdict(register)
    return bool(v) and _verified(register, "b5-verdict") and v.lower() not in ("refuted", "mixed")


# ── result-file readers (build time) ─────────────────────────────────────────────────────────────────────

def _rows_with_lines(path: Path):
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if line.strip():
                yield i, json.loads(line)


def t2s_session_evidence(repo: Path, rel: str) -> dict:
    """Per session of one evo-t2s R2 real file: loaded window, first over-window turn, calls, calls that surfaced an
    error (HTTP status other than 200 or an error field), turns over the loaded window, canary misses."""
    p = repo / rel
    if not p.exists():
        return {"file": rel, "present": False, "sessions": []}
    turns: dict[str, dict] = {}
    sessions = []
    for ln, r in _rows_with_lines(p):
        rec = r.get("record")
        if rec == "r2a_turn":
            t = turns.setdefault(r["attempt_id"], {"calls": 0, "surfaced": 0, "over": 0, "turns": 0})
            t["turns"] += 1
            t["over"] += 1 if r.get("over_loaded_window") else 0
            for c in r.get("calls") or []:
                t["calls"] += 1
                if c.get("error") or str(c.get("http_status")) != "200":
                    t["surfaced"] += 1
        elif rec == "r2a_session":
            sessions.append((ln, r))
    out = []
    for ln, s in sessions:
        t = turns.get(s["attempt_id"], {})
        out.append({"model": s["model_id"], "seed": s["seed"], "arm_id": s["arm_id"],
                    "loaded_window": s.get("loaded_context_values"), "first_over_window_turn":
                    s.get("first_over_window_turn"), "first_error_turn": s.get("first_error_turn"),
                    "canary_sys_misses": s.get("canary_sys_misses"), "canary_hist_misses": s.get("canary_hist_misses"),
                    "canary_checks": s.get("canary_checks"), "turns": t.get("turns", 0),
                    "turns_over_window": t.get("over", 0), "calls": t.get("calls", 0),
                    "calls_surfaced_error": t.get("surfaced", 0), "src": {"file": rel, "line": ln}})
    return {"file": rel, "present": True, "sessions": out,
            "totals": {"sessions": len(out), "calls": sum(s["calls"] for s in out),
                       "calls_surfaced_error": sum(s["calls_surfaced_error"] for s in out)}}


def b5_configs(repo: Path, rel: str = B5_FILE) -> dict:
    """From the B5 file: per model the configuration each policy ran (b5_resolution), the KV check per env
    (b5_kv_check: flash attention on the server config), and num_keep per seed (b5_num_keep)."""
    p = repo / rel
    out: dict = {}
    if not p.exists():
        return out
    for ln, r in _rows_with_lines(p):
        rec = r.get("record")
        if rec == "b5_resolution":
            out.setdefault(r["model_id"], {})["configs"] = r.get("configs") or {}
            out[r["model_id"]]["resolution_src"] = {"file": rel, "line": ln}
            out[r["model_id"]]["windows"] = r.get("windows") or {}
        elif rec == "b5_kv_check":
            chk = r.get("check") or {}
            out.setdefault(r["model_id"], {}).setdefault("kv_check", {})[r["env"]] = {
                "flash_attention": chk.get("config_flash_attention"), "observed": chk.get("observed"),
                "verified": chk.get("verified"), "src": {"file": rel, "line": ln}}
        elif rec == "b5_num_keep":
            out.setdefault(r["model_id"], {}).setdefault("num_keep", {})[str(r["seed"])] = {
                "num_keep": r.get("num_keep"), "src": {"file": rel, "line": ln}}
    return out


def b3_wrong_examples(repo: Path, rel: str = B3_FILE) -> dict:
    """Per tier, the source line of each wrong task turn and how many surfaced an HTTP error (file-level check of the
    register's 'surfaced' count)."""
    p = repo / rel
    if not p.exists():
        return {}
    out: dict = {}
    for ln, r in _rows_with_lines(p):
        if r.get("record") == "r2a_turn" and r.get("item_kind") == "task" and r.get("end_state_ok") is False \
                and r.get("mode") == "real":
            tier = r["arm_id"].replace("_call2_notools", "")
            d = out.setdefault(tier, {"wrong_rows": 0, "http_error_rows": 0, "first_line": ln})
            d["wrong_rows"] += 1
            d["http_error_rows"] += 1 if r.get("any_http_error") else 0
    return out


# ── view 1: silent failure ─────────────────────────────────────────────────────────────────────────────

def build_silent(register: dict, repo: Path = REPO) -> dict:
    pair = {}
    for dev in ("cpu", "igpu"):
        rid = T2S_ROWS[dev]
        row = _row(register, rid)
        parsed = parse_t2s_real(row["value"]) if row else []
        pair[dev] = {"register_id": rid, "status": row["status"] if row else "MISSING", "models": parsed,
                     "parsed": bool(parsed), "label": MEASURED if parsed else NOT_MEASURED,
                     "evidence": t2s_session_evidence(repo, T2S_FILES[dev])}
    wr = _row(register, "x2-evalset-tasks-real-v2-wrong-rate")
    tiers = parse_wrong_rate(wr["value"]) if wr else []
    gp = _row(register, "R2-v1b-gated-per-model")
    cells = parse_gated_per_model(gp["value"]) if gp else []
    return {
        "device_pair": {"machine": "evo-t2s", "arm": "ollama_default (no num_ctx sent)", **pair,
                        "label": MEASURED,
                        "caption": "Same machine, same models, same 40-turn sessions, Ollama default settings. The "
                                   "only difference is the device Ollama picked, which sets the window it loads."},
        "wrong_rate": {"register_id": "x2-evalset-tasks-real-v2-wrong-rate", "model": "qwen3:14b", "machine": "evo-x2",
                       "tiers": tiers, "parsed": bool(tiers), "label": MEASURED if tiers else NOT_MEASURED,
                       "ci_register_id": "x2-evalset-tasks-real-v2-ci-units",
                       "file_check": b3_wrong_examples(repo), "file": B3_FILE},
        "v1b_per_model": {"register_id": "R2-v1b-gated-per-model", "machine": "evo-x2", "cells": cells,
                          "parsed": bool(cells), "label": MEASURED if cells else NOT_MEASURED,
                          "checkpoints": list(CHECKPOINTS)},
        "register_ids": SILENT_IDS,
    }


def silent_view(cache) -> dict:
    s = dict(cache.silent)
    s["register"] = {i: cache.register[i] for i in SILENT_IDS if i in cache.register}
    return s


# ── view 2: per-device recommendation ──────────────────────────────────────────────────────────────────

DEVICES = [
    {"id": "evo-x2", "label": "evo-x2 (Strix Halo iGPU, 128 GB)", "machine": "evo-x2"},
    {"id": "evo-t2s-cpu", "label": "evo-t2s, Intel CPU (Ollama default device)", "machine": "evo-t2s"},
    {"id": "evo-t2s-igpu", "label": "evo-t2s, Intel iGPU", "machine": "evo-t2s"},
    {"id": "blade", "label": "Blade 14 (discrete GPU)", "machine": "blade"},
]
MODELS = ["llama3.1:8b", "qwen3:8b", "qwen3:14b", "qwen3:32b", "llama3.3:70b"]
WORKLOADS = [
    {"id": "r2_agent", "label": "Multi-turn agent session, 40 turns (R2)"},
    {"id": "b3_tasks", "label": "Agent task session, 30 graded task turns (B3)"},
    {"id": "single_shot", "label": "Single-shot calls up to long documents (outcome table)"},
]
# Outcome-table runtimes per device (results/*_outcome_table*.jsonl runtime field).
SINGLE_SHOT_RUNTIMES = {"evo-x2": ("ollama_default", "llama_server"), "evo-t2s-cpu": ("ollama_default",),
                        "evo-t2s-igpu": ("ollama_igpu_enable", "llama_server_vulkan"), "blade": ()}
POLICY_NAMES = {"a": "default", "b": "largest", "c": "trim", "d": "router", "e": "autosize", "f": "num_keep",
                "g": "kv_q8_0", "h": "kv_q4_0", "i": "f16 + flash attention"}  # harness/policy_compare.py POLICIES


def model_key(m: str | None) -> str:
    """'qwen3-14b', 'qwen3:14b' -> 'qwen3 14b'; 'llama3.1-8b-gguf' -> 'llama3.1 8b'."""
    s = (m or "").lower().replace("-gguf", "")
    return re.sub(r"[:\-_]", " ", s)


def build_device(register: dict, repo: Path = REPO, r2_sessions: dict | None = None) -> dict:
    def val(rid):
        r = _row(register, rid)
        return r["value"] if r else ""

    x2_default = {}
    for s in (r2_sessions or {}).get("sessions", []):
        adv = s.get("advertised_ctx")
        if adv:
            x2_default.setdefault(s["model"], {"ctx": adv["ctx"], "src": {"file": adv.get("file"),
                                                                          "line": adv.get("line")}})
    refused = {}
    for rid, model in (("x2-evalset-tasks-real-v2-wrong-rate", "llama3.1:8b"),
                       ("x2-evalset-tasks-real-v2-qwen3-8b-wrong-rate", "qwen3:8b")):
        m = re.search(re.escape(model) + r": refused \((.*?)\)(?: / |\s*\(bootstrap|$)", val(rid))
        if m:
            refused[model] = {"reason": m[1], "register_id": rid}
    return {
        "devices": DEVICES, "models": MODELS, "workloads": WORKLOADS,
        "b5_metrics": parse_b5_metrics(val("b5-policy-metrics")),
        "b5_configs": b5_configs(repo),
        "kv_recall": parse_kv_recall(val("b5-kv-recall")),
        "gated": parse_gated_per_model(val("R2-v1b-gated-per-model")),
        "wrong_rate": parse_wrong_rate(val("x2-evalset-tasks-real-v2-wrong-rate")),
        "b3_refused": refused,
        "t2s": {d: parse_t2s_real(val(T2S_ROWS[d])) for d in ("cpu", "igpu")},
        "t2s_mitigation": parse_mitigation(val("t2s-r2-mitigation-cpu")),
        "x2_mitigation": parse_mitigation(val("R2-mitigation-v1-verdict")),
        "usability": parse_usability(val("x2-v3-usability")),
        "x2_default_window": x2_default,
        "status": {rid: (register[rid]["status"] if rid in register else "MISSING") for rid in DEVICE_IDS},
        "fa_reading": fa_control_reading(register),
        "fa_control": {m: fa_control_for(register, m) for m in parse_fa_control(val("b5-fa-control"))},
    }


def _set(value, label, src=None, note=None) -> dict:
    return {"value": value, "label": label, "src": src, "note": note}


def _nm(note=None) -> dict:
    return _set("not measured", NOT_MEASURED, None, note)


def _out(text, label, src, value=None) -> dict:
    return {"text": text, "label": label, "src": src, "value": value}


def _empty_settings(note=None) -> dict:
    return {k: _nm(note) for k in ("window", "num_keep", "kv_precision", "flash_attention", "context_handling")}


def _rec_b5(dv: dict, model: str) -> dict | None:
    m, cfg = dv["b5_metrics"].get(model), dv["b5_configs"].get(model)
    if not m or not cfg or not m.get("frontier"):
        return None
    p = m["frontier"][0]
    pol, base = m["policies"].get(p), m["policies"].get("a")
    c = (cfg.get("configs") or {}).get(p) or {}
    env, wrapper = c.get("env"), c.get("wrapper")
    rsrc, msrc = cfg.get("resolution_src"), reg_src("b5-policy-metrics")
    kvc = (cfg.get("kv_check") or {}).get(env) or {}
    fc = (dv.get("fa_control") or {}).get(model)  # None until b5-fa-control is VERIFIED, valid and read for model
    kv_fa = env in ("q8_0", "q4_0") and bool(kvc.get("flash_attention"))
    pending = kv_fa and fc is None
    fa_state = ("PENDING" if dv["status"].get("b5-fa-control") != "VERIFIED" else "not read yet (run incomplete)")
    if kv_fa and fc and fc["reading"] == FA_READ_FA and fc.get("minutes_i") is not None:
        return _rec_fa_alone(dv, model, m, fc, p, pol, base)
    kv_lab = PENDING_FA if pending else MEASURED
    s = {"window": _set(c.get("effective_ctx"), MEASURED, rsrc,
                        "num_ctx sent" if c.get("num_ctx_sent") else "runtime default (no num_ctx sent)"),
         "kv_precision": (_set(env, kv_lab, kvc.get("src"), ("OLLAMA_KV_CACHE_TYPE, verified in the server log"
                               if kvc.get("verified") else "OLLAMA_KV_CACHE_TYPE") +
                               (f"; pending flash-attention control (b5-fa-control {fa_state})" if pending else ""))
                          if env in ("q8_0", "q4_0") else _set("f16 (runtime default)", MEASURED, rsrc)),
         "flash_attention": (_set("on (OLLAMA_FLASH_ATTENTION=1)", kv_lab, kvc.get("src"),
                                  f"not separated from KV precision: b5-fa-control {fa_state}" if pending
                                  else f"b5-fa-control: {fc['reading']}" if fc else None)
                             if kvc.get("flash_attention") else _set("not set (runtime default)", MEASURED, rsrc)),
         "num_keep": (_set("system prompt tokens, per seed " + ", ".join(
             str(v["num_keep"]) for v in (cfg.get("num_keep") or {}).values()), MEASURED,
             next(iter((cfg.get("num_keep") or {}).values()), {}).get("src")) if wrapper == "num_keep"
             else _set("not set", MEASURED, rsrc)),
         "context_handling": _set({None: "full history", "client_trim": "client-side trimming",
                                   "router": "router"}.get(wrapper, wrapper), MEASURED, rsrc)}
    out = [_out(f"rules held {pol['rule_pct']:.1f}%, history recalled {pol['hist_pct']:.1f}%", MEASURED, msrc),
           _out(f"{pol['minutes']:.1f} min per 40-turn session", MEASURED, msrc, pol["minutes"])]
    if base and p != "a":
        out.append(_out(f"default (a): {base['minutes']:.1f} min, rules {base['rule_pct']:.1f}%, history "
                        f"{base['hist_pct']:.1f}%; saves {base['minutes'] - pol['minutes']:.1f} min per session",
                        MEASURED, msrc, base["minutes"] - pol["minutes"]))
    kv = (dv["kv_recall"].get(model) or {}).get(env if env in ("q8_0", "q4_0") else "f16")
    out.append(_out(f"long-context recall probe at this precision: {kv['recall_pct']:.1f}% at ctx {kv['ctx']}",
                    MEASURED, reg_src("b5-kv-recall")) if kv else
               _out("long-context recall probe at this precision: not measured", NOT_MEASURED, reg_src("b5-kv-recall")))
    caveats = []
    if pending:
        caveats.append("Pending flash-attention control: flash attention was switched on together with the KV "
                       f"precision, and the control that separates them (b5-fa-control) is {fa_state}. Once it reads, "
                       "this view recommends whichever of the two it shows causes the speedup.")
    elif fc:
        caveats.append(f"Flash-attention control (b5-fa-control) for {model}: {fc['reading']}.")
    caveats.append("The router policy (d) did not make the frontier (row b5-verdict); no router saving is shown.")
    return {"status": PENDING_FA if pending else MEASURED,
            "basis": f"B5 policy comparison: frontier policy ({p}) {POLICY_NAMES.get(p, p)}, 3 seeds x 40 turns" +
                     ("; pending flash-attention control" if pending else ""),
            "settings": s, "outcome": out, "caveats": caveats,
            "register_ids": ["b5-policy-metrics", "b5-kv-recall", "b5-fa-control", "cost-cut-config-time"],
            "alternatives": [{"policy": k, "name": POLICY_NAMES.get(k, k), **v} for k, v in m["policies"].items()
                             if not v.get("alias_of")]}


def _rec_fa_alone(dv: dict, model: str, m: dict, fc: dict, p: str, pol: dict, base: dict | None) -> dict:
    """b5-fa-control read 'flash attention, not KV precision' for this model: recommend policy (i), f16 KV with flash
    attention, at the control's own window, with the control's measured minutes."""
    src = reg_src("b5-fa-control")
    s = {"window": _set(fc.get("window"), MEASURED, src),
         "kv_precision": _set("f16 (runtime default)", MEASURED, src),
         "flash_attention": _set("on (OLLAMA_FLASH_ATTENTION=1)", MEASURED, src, f"b5-fa-control: {fc['reading']}"),
         "num_keep": _set("not set", MEASURED, src),
         "context_handling": _set("full history", MEASURED, src)}
    out = [_out(f"(i) f16 + flash attention: {fc['minutes_i']:.1f} min per 40-turn session", MEASURED, src,
                fc["minutes_i"]),
           _out(f"({p}) {POLICY_NAMES.get(p, p)} with flash attention: {pol['minutes']:.1f} min", MEASURED,
                reg_src("b5-policy-metrics"))]
    if base:
        out.append(_out(f"default (a): {base['minutes']:.1f} min; saves {base['minutes'] - fc['minutes_i']:.1f} min "
                        f"per session", MEASURED, src, base["minutes"] - fc["minutes_i"]))
    return {"status": MEASURED, "basis": "Flash-attention control: flash attention alone gives the B5 speedup "
                                         "(within the pre-registered tolerance), so the KV cache stays f16",
            "settings": s, "outcome": out,
            "caveats": ["The router policy (d) did not make the frontier (row b5-verdict); no router saving is shown."],
            "register_ids": ["b5-fa-control", "b5-policy-metrics", "cost-cut-config-time"],
            "alternatives": [{"policy": k, "name": POLICY_NAMES.get(k, k), **v} for k, v in m["policies"].items()
                             if not v.get("alias_of")]}


def _tier_window(tier: str, default: dict | None):
    if tier == "ollama_default":
        return (default or {}).get("ctx"), (default or {}).get("src"), "runtime default (no num_ctx sent)"
    n = _ints(tier)
    return (n[0] if n else None), reg_src("R2-v1b-gated-per-model"), "num_ctx sent"


def _rec_gated(dv: dict, model: str) -> dict | None:
    cells = [c for c in dv["gated"] if c["model"] == model]
    if not cells:
        return None
    order = {"ollama_default": 0}
    best = min(cells, key=lambda c: (c["gated_silent"], order.get(c["tier"], 1), -(_ints(c["tier"]) or [0])[0]))
    win, wsrc, wnote = _tier_window(best["tier"], dv["x2_default_window"].get(model))
    src = reg_src("R2-v1b-gated-per-model")
    s = _empty_settings("not varied in this run (runtime default)")
    s["window"] = _set(win, MEASURED, wsrc, wnote)
    s["context_handling"] = _set("full history", MEASURED, src)
    out = [_out(f"{best['gated_silent']}/{best['n']} sessions silently lost a gated rule, {best['silent_after_window']} "
                f"after the window filled", MEASURED, src),
           _out(f"sessions with every gated rule intact at turn 40: {best['intact'][-1]}/{best['n']}", MEASURED, src)]
    return {"status": MEASURED, "basis": "R2 strengthened run (v1 + v1b), 5 seeds x 40 turns per window: the window "
                                         "with the fewest silent rule losses", "settings": s, "outcome": out,
            "caveats": [], "register_ids": ["R2-v1b-gated-per-model"],
            "alternatives": [{"tier": c["tier"], "gated_silent": c["gated_silent"], "n": c["n"],
                              "silent_after_window": c["silent_after_window"]} for c in cells]}


def _rec_b3(dv: dict, model: str) -> dict:
    tiers = dv["wrong_rate"] if model == "qwen3:14b" else []
    if not tiers:
        ref = dv["b3_refused"].get(model)
        note = (f"task run refused for {model}: {ref['reason']}" if ref else
                f"no B3 task run for {model} on evo-x2")
        return {"status": NOT_MEASURED, "basis": note, "settings": _empty_settings(), "outcome": [],
                "caveats": [], "register_ids": [ref["register_id"]] if ref else
                ["x2-evalset-tasks-real-v2-wrong-rate"], "alternatives": []}
    src = reg_src("x2-evalset-tasks-real-v2-wrong-rate")
    best = min(tiers, key=lambda t: (t["pct"], t["tier"] != "ollama_default"))
    win, wsrc, wnote = _tier_window(best["tier"], dv["x2_default_window"].get(model))
    if best["tier"] == "ollama_default" and best["loaded_context"]:
        win, wsrc = best["loaded_context"][0], src
    s = _empty_settings("not varied in this run (runtime default)")
    s["window"] = _set(win, MEASURED, wsrc, wnote)
    nk = next((t for t in tiers if t["tier"].endswith("_numkeep")), None)
    plain = next((t for t in tiers if nk and t["tier"] == nk["tier"].replace("_numkeep", "")), None)
    s["num_keep"] = _set("system prompt tokens" if best["tier"].endswith("_numkeep") else "not set", MEASURED, src)
    s["context_handling"] = _set("full history", MEASURED, src)
    out = [_out(f"wrong task results {best['wrong']}/{best['n']} ({best['pct']:.1f}%), {best['surfaced']} surfaced an "
                f"error", MEASURED, src)]
    if nk and plain:
        out.append(_out(f"num_keep at {plain['loaded_context'][0] if plain['loaded_context'] else '?'}: "
                        f"{nk['pct']:.1f}% wrong vs {plain['pct']:.1f}% without it; still above the best window's "
                        f"{best['pct']:.1f}%", MEASURED, src))
    return {"status": MEASURED, "basis": "B3 v2 task run, qwen3:14b, 5 seeds x 30 task turns per window: the window "
                                         "with the lowest wrong-task rate", "settings": s, "outcome": out,
            "caveats": ["KV precision and flash attention were not varied in the task run."],
            "register_ids": ["x2-evalset-tasks-real-v2-wrong-rate", "x2-evalset-tasks-real-v2-ci-units"],
            "alternatives": [{"tier": t["tier"], "pct": t["pct"], "wrong": t["wrong"], "n": t["n"]} for t in tiers]}


def _rec_t2s(dv: dict, dev: str, model: str) -> dict | None:
    rows = {m["model"]: m for m in dv["t2s"].get(dev, [])}
    m = rows.get(model)
    if not m:
        return None
    rid = T2S_ROWS[dev]
    src = reg_src(rid)
    win = m["loaded_window"][0] if m["loaded_window"] else None
    default_out = _out(f"Ollama default window {win}: {m['gated_silent']}/{m['n']} sessions silently lost a gated "
                       f"rule, {m['silent_after_window']} after the window filled", MEASURED, src)
    s = _empty_settings("not varied on evo-t2s")
    if m["silent_after_window"] == 0:
        s["window"] = _set(win, MEASURED, src, "runtime default (no num_ctx sent)")
        s["context_handling"] = _set("full history", MEASURED, src)
        return {"status": MEASURED, "basis": "evo-t2s R2 real run at the Ollama default, 5 seeds x 40 turns",
                "settings": s, "outcome": [default_out], "caveats": [] if m["baseline_type"] == 0 else
                [f"{m['baseline_type']} silent session(s) are baseline-type: the same loss happens inside the window "
                 "on this model's validation runs."], "register_ids": [rid], "alternatives": []}
    mit = dv["t2s_mitigation"].get(f"num_ctx_{win}|{model}") if dev == "cpu" else None
    msrc = reg_src("t2s-r2-mitigation-cpu")
    if mit == "recovered":
        s["window"] = _set(win, MEASURED, src, "runtime default on this device")
        s["context_handling"] = _set("client-side trimming that keeps the system prompt", MEASURED, msrc)
        return {"status": MEASURED, "basis": "evo-t2s CPU: the default window loses rules; client-side trimming at the "
                                             "same window recovered them (3 seeds x 40 turns)", "settings": s,
                "outcome": [default_out, _out("with client-side trimming: recovered (pre-registered rule)", MEASURED,
                                              msrc)],
                "caveats": [], "register_ids": [rid, "t2s-r2-mitigation-cpu"], "alternatives": []}
    outs = [default_out]
    if mit:
        outs.append(_out("with client-side trimming: not recovered (pre-registered rule)", MEASURED, msrc))
    return {"status": NOT_MEASURED, "basis": "no measured setting on this device holds the rules for this model; a "
                                             "larger window on the CPU was not run. Escalate long sessions or use "
                                             "another device.", "settings": s, "outcome": outs, "caveats": [],
            "register_ids": [rid] + (["t2s-r2-mitigation-cpu"] if mit else []), "alternatives": []}


def _rec_single_shot(points: list[dict], dev: str, model: str, share: float) -> dict:
    """points: decorated outcome-table points with excluded_reason already set for `share`."""
    rts = SINGLE_SHOT_RUNTIMES.get(dev, ())
    machine = next((d["machine"] for d in DEVICES if d["id"] == dev), None)
    cand = [p for p in points if p.get("machine") == machine and p.get("runtime") in rts
            and model_key(p.get("model")) == model_key(model) and not p.get("stub")]
    if not cand:
        return {"status": NOT_MEASURED, "basis": f"no outcome-table row for {model} on {dev}",
                "settings": _empty_settings(), "outcome": [], "caveats": [], "register_ids": ["x2-v3-usability"],
                "alternatives": [], "excluded": []}
    ok = [p for p in cand if not p.get("excluded_reason") and p.get("quality") is not None]
    excluded = [{"config_id": p["config_id"], "runtime": p["runtime"], "reason": p["excluded_reason"],
                 "timeout_latency_share": p.get("timeout_latency_share"), "src": p.get("src")}
                for p in cand if p.get("excluded_reason")]
    if not ok:
        return {"status": NOT_MEASURED, "basis": f"every measured config is above the {share:.0%} timeout floor",
                "settings": _empty_settings(), "outcome": [], "caveats": [], "register_ids": ["x2-v3-usability"],
                "alternatives": [], "excluded": excluded}
    best = max(ok, key=lambda p: (p["quality"], -(p.get("latency_p50_ms") or 1e18)))
    src = best.get("src")
    s = _empty_settings("not varied in the outcome tables")
    s["window"] = _set("runtime default", MEASURED, src, "the outcome table runs each context tier at the "
                                                          "runtime's own window handling")
    s["context_handling"] = _set(f"runtime {best['runtime']}", MEASURED, src)
    ts = best.get("timeout_latency_share")
    out = [_out(f"quality {best['quality']:.3f} (n {best.get('quality_n')})", MEASURED, src),
           _out(f"p50 latency {best['latency_p50_ms'] / 1000:.1f} s" if best.get("latency_p50_ms") else
                "p50 latency n/a", MEASURED, src),
           _out(f"timeout share {ts:.3f} (floor {share:.2f})" if ts is not None else "timeout share n/a", MEASURED, src)]
    caveats = [f"flags: {', '.join(best['flags'])}"] if best.get("flags") else []
    return {"status": MEASURED, "basis": f"single-shot outcome table: highest quality config with timeout share at or "
                                         f"below {share:.2f}", "settings": s, "outcome": out, "caveats": caveats,
            "register_ids": ["x2-v3-usability"],
            "alternatives": [{"config_id": p["config_id"], "quality": p["quality"],
                              "timeout_latency_share": p.get("timeout_latency_share")} for p in ok if p is not best],
            "excluded": excluded}


def device_recommend(dv: dict, device: str, model: str, workload: str, points: list[dict] | None = None,
                     max_timeout_share: float = DEFAULT_MAX_TIMEOUT_SHARE) -> dict:
    """The recommended setting for (device, model, workload) with its measured outcome, or 'not measured'."""
    if device not in {d["id"] for d in DEVICES} or workload not in {w["id"] for w in WORKLOADS}:
        raise KeyError(f"{device}/{workload}")
    rec = None
    if workload == "single_shot":
        rec = _rec_single_shot(points or [], device, model, max_timeout_share)
    elif device == "evo-x2" and workload == "r2_agent":
        rec = _rec_b5(dv, model) or _rec_gated(dv, model)
    elif device == "evo-x2" and workload == "b3_tasks":
        rec = _rec_b3(dv, model)
    elif device.startswith("evo-t2s") and workload == "r2_agent":
        rec = _rec_t2s(dv, device.rsplit("-", 1)[1], model)
    if rec is None:
        rec = {"status": NOT_MEASURED, "basis": f"no run of {model} for this workload on {device}",
               "settings": _empty_settings(), "outcome": [], "caveats": [], "register_ids": [], "alternatives": []}
        if device == "evo-x2" and workload == "r2_agent" and model in ("qwen3:32b", "llama3.3:70b"):
            rec["basis"] = (f"no setting comparison for {model}: the scale probe ran num_ctx 4096 and 32768 only, to "
                            "test whether rule loss holds at this size (rows H-scale-verdict, "
                            "x2-scale-real-default-tier)")
            rec["register_ids"] = ["H-scale-verdict", "x2-scale-real-default-tier"]
    rec.setdefault("excluded", [])
    return {"device": device, "model": model, "workload": workload, "max_timeout_share": max_timeout_share, **rec}


# ── view 3: cost panel ─────────────────────────────────────────────────────────────────────────────────

P1_LEAD = "Fixing a wrong answer in the cloud costs about USD {usd}; the cost is that the failure is silent."
# The figure in P1_LEAD is filled from register row cloud-redo-cost-per-wrong-task at build time, never typed here.
_ORDER = {MEASURED: 0, PRICED: 1, SIMULATED: 2, PROJECTED: 3, BEING_MEASURED: 4, NOT_MEASURED: 5}


def parse_config_time(value: str) -> list[dict]:
    pat = re.compile(r"evo-x2 (?P<model>[\w.:-]+): default \(a\) (?P<d>[\d.]+) min per session.*?fastest measured "
                     r"setting \((?P<p>\w)\) (?P<name>\w+) (?P<f>[\d.]+) min.*?saved (?P<s>[\d.]+) min per session "
                     r"\((?P<pct>\d+)%\)")
    return [{"model": m["model"], "default_min": float(m["d"]), "policy": m["p"], "setting": m["name"],
             "fast_min": float(m["f"]), "saved_min": float(m["s"]), "saved_pct": int(m["pct"])}
            for m in pat.finditer(value or "")]


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def build_cost(register: dict) -> dict:
    def val(rid):
        r = _row(register, rid)
        return r["value"] if r else ""

    items = []
    # 2026-10-22: no figure from the v1 cloud redo (results/cloud_redo_v1.jsonl, a broken protocol run). The lead says
    # the cost is being measured; the full-history redo (row cloud-redo-full-context) replaces it in the final pass.
    lead = {"text": FIX_COST_LEAD, "label": BEING_MEASURED, "src": None,
            "note": "The earlier cloud redo was a broken protocol run (thinking consumed the output budget); its "
                    "figures are not shown. A corrected redo is running."}
    tiers = parse_wrong_rate(val("x2-evalset-tasks-real-v2-wrong-rate"))
    if tiers:
        lo, hi = min(tiers, key=lambda t: t["pct"]), max(tiers, key=lambda t: t["pct"])
        items.append({"what": "How often the answer is wrong, and silent",
                      "text": f"qwen3:14b on evo-x2: {lo['pct']:.1f}% wrong at {lo['tier']} to {hi['pct']:.1f}% at "
                              f"{hi['tier']}; {sum(t['surfaced'] for t in tiers)} of "
                              f"{sum(t['wrong'] for t in tiers)} wrong results surfaced an error",
                      "label": MEASURED, "src": reg_src("x2-evalset-tasks-real-v2-wrong-rate")})
    for c in parse_config_time(val("cost-cut-config-time")):
        items.append({"what": f"Time saved by the measured setting, {c['model']}",
                      "text": f"{c['saved_min']:.1f} min per 40-turn session ({c['saved_pct']}%): default "
                              f"{c['default_min']:.1f} min vs {c['setting']} {c['fast_min']:.1f} min" +
                              (f" (b5-fa-control: {fa_control_reading(register)})" if fa_control_reading(register)
                               else " (flash attention and KV precision changed together; b5-fa-control has no "
                                    "reading yet)"),
                      "label": MEASURED, "src": reg_src("cost-cut-config-time")})
    m = re.search(r"per session: re-sent history ([\d,]+) tok, USD ([\d.]+) caching on, ([\d.]+) no caching",
                  val("impact-P3"))
    if m:
        items.append({"what": "Re-sending the history to the cloud, per session",
                      "text": f"{m[1]} tokens re-sent; USD {m[2]} with caching, USD {m[3]} without",
                      "label": PRICED, "src": reg_src("impact-P3")})
    # The P1 USD/day and the problem ranking both include the v1 redo cost, so they wait with it (2026-10-22).
    items.append({"what": "Wrong answers per day for one stated team (P1)",
                  "text": "being measured (it includes the cost to fix a wrong answer in the cloud)",
                  "label": BEING_MEASURED, "src": None})
    m = re.search(r"measured-only USD/day ([\d.]+)", val("impact-total"))
    if m:
        items.append({"what": "Today's cost for the reference team, measured parts only",
                      "text": f"USD {m[1]} per day (measured token counts and list prices at the assumed team "
                              f"volume; the register calls this measured-only)",
                      "label": PROJECTED, "src": reg_src("impact-total")})
    items.sort(key=lambda i: _ORDER.get(i["label"], 9))
    verdict = b5_verdict(register)
    router = {"shown": router_savings_supported(register), "verdict": verdict, "src": reg_src("b5-verdict"),
              "text": ("Router savings: not shown. Row b5-verdict is " + (verdict or "missing") +
                       " on evo-x2 (the router policy was dominated); no register row supports a savings figure.")
              if not router_savings_supported(register) else "Router savings: see row b5-verdict."}
    return {"lead": lead, "items": items, "router": router, "register_ids": COST_IDS,
            "docs": [COST_SUMMARY_DOC, IMPACT_DOC]}


def doc_summary(repo: Path, rel: str) -> dict:
    """Present or not; if present its sha256, H1, first paragraph after the H1 (quoted verbatim) and its ## headings
    with line numbers. Read at request time so a doc that lands after the cache build still shows."""
    import hashlib
    p = repo / rel
    if not p.exists():
        return {"path": rel, "present": False}
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), None)
    para, started, para_line = [], False, None
    for i, l in enumerate(lines, 1):
        if l.startswith("# ") and not started:
            started = True
            continue
        if not started:
            continue
        if l.startswith("#"):
            if para:
                break
            continue
        if l.strip():
            para_line = para_line or i
            para.append(l.strip())
        elif para:
            break
    heads = [{"line": i, "text": l.lstrip("#").strip()} for i, l in enumerate(lines, 1) if l.startswith("## ")]
    return {"path": rel, "present": True, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "title": title,
            "headline": " ".join(para), "headline_line": para_line, "sections": heads[:20]}


def cost_view(cache, repo: Path = REPO) -> dict:
    c = dict(cache.cost)
    c["docs"] = [doc_summary(repo, d) for d in c.get("docs", [COST_SUMMARY_DOC, IMPACT_DOC])]
    c["register"] = {i: cache.register[i] for i in COST_IDS if i in cache.register}
    return c


# ── story mode: five guided steps over the same numbers ────────────────────────────────────────────────

STORY_IDS = ["t2s-r2-real-cpu", "t2s-r2-real-igpu", "x2-evalset-tasks-real-v2-wrong-rate",
             "R2-qwen3-14b-4096-reconciliation", "H-scale-verdict", "H-scale-family-verdict", "R2-v1b-gated-per-model",
             "cloud-cost-per-real-task", "b5-fa-control", "b5-policy-metrics"]


def parse_reconciliation_rule2(value: str) -> dict:
    """Row R2-qwen3-14b-4096-reconciliation: rule2 (log_event called) per seed in the untrimmed v1 run and under
    client-side trimming (mitigation), plus the first trimmed failure's tool lists."""
    segs = {}
    for part in (value or "").split(" // "):
        m = re.match(r"\s*(v1b|v1|mitigation|num_keep):", part)
        if m:
            segs[m[1]] = [(int(a), int(b)) for a, b in re.findall(r"rule2 (\d+)/(\d+)", part)]
    first = re.search(r"first trimmed rule2 failure seed (\d+) turn (\d+): call 1 tools trimmed \[([^\]]*)\], "
                      r"untrimmed v1 \[([^\]]*)\]", value or "")
    return {"v1": segs.get("v1", []), "trimmed": segs.get("mitigation", []),
            "first_failure": {"seed": int(first[1]), "turn": int(first[2]), "trimmed_tools": first[3],
                              "untrimmed_tools": first[4]} if first else None}


def parse_scale(value: str) -> list[dict]:
    """H-scale rows: 'qwen3:32b: R4096 2/3 (...)' plus the per-model verdict wording."""
    out = []
    for part in (value or "").split(" / "):
        m = re.search(r"([\w.:-]+): R4096 (\d+)/(\d+)", part)
        if not m:
            continue
        v = re.search(r"verdict (\w+)(?: \(([^)]*)\))?", part)
        out.append({"model": m[1], "silent": int(m[2]), "n": int(m[3]), "verdict": v[1] if v else None,
                    "note": "resists rule loss" if "resists rule loss" in part else (v[2] if v else None)})
    return out


def parse_task_cost(value: str) -> dict:
    """Row cloud-cost-per-real-task, the long-task trace set (first segment): steps and median / p90 USD per task
    for each model and caching case."""
    seg = (value or "").split(" | ")[0]
    m = re.search(r"([\w-]+) \(n=(\d+) tasks, median (\d+) steps", seg)
    if not m:
        return {}
    models = {}
    for mm in re.finditer(r"(claude-[\w.-]+): none \$([\d.]+) / \$([\d.]+), realistic \$([\d.]+) / \$([\d.]+), "
                          r"best \$([\d.]+) / \$([\d.]+)", seg):
        models[mm[1]] = {"none": (float(mm[2]), float(mm[3])), "realistic": (float(mm[4]), float(mm[5])),
                         "best": (float(mm[6]), float(mm[7]))}
    return {"trace_set": m[1], "n_tasks": int(m[2]), "median_steps": int(m[3]), "models": models,
            "priced": "priced, not executed" in value}


def _item(text, label, rid=None, src=None):
    return {"text": text, "label": label, "src": src or (reg_src(rid) if rid else None)}


def build_story(register: dict, silent: dict, cost: dict, escalation: dict | None = None) -> dict:
    def val(rid):
        r = _row(register, rid)
        return r["value"] if r else ""

    steps = []
    # 1. the problem
    s1 = []
    pair = silent.get("device_pair", {})
    for dev, name in (("cpu", "Intel CPU"), ("igpu", "Intel iGPU")):
        d = pair.get(dev) or {}
        for m in d.get("models", []):
            s1.append(_item(f"{name}, {m['model']}: Ollama loads {', '.join(map(str, m['loaded_window']))} tokens; "
                            f"{m['gated_silent']}/{m['n']} sessions silently lost a rule, {m['silent_after_window']} "
                            f"after the window filled", MEASURED, d.get("register_id")))
        tot = (d.get("evidence") or {}).get("totals")
        if tot:
            first = ((d.get("evidence") or {}).get("sessions") or [{}])[0].get("src")
            s1.append(_item(f"{name}: {tot['calls_surfaced_error']} of {tot['calls']} calls surfaced an error",
                            MEASURED, src=first))
    tiers = {t["tier"]: t for t in (silent.get("wrong_rate") or {}).get("tiers", [])}
    if "ollama_default" in tiers and "ollama_ctx_4096" in tiers:
        a, b = tiers["ollama_default"], tiers["ollama_ctx_4096"]
        s1.append(_item(f"qwen3:14b task results: {a['pct']:.1f}% wrong at the default window, {b['pct']:.1f}% at "
                        f"4096; {b['surfaced']} of {b['wrong']} wrong results surfaced an error", MEASURED,
                        "x2-evalset-tasks-real-v2-wrong-rate"))
    steps.append({"n": 1, "title": "The problem: a default nobody chose, a failure nobody sees", "minutes": 1,
                  "items": s1, "panel": "silent"})
    # 2. why it is hard
    s2 = []
    nk = tiers.get("ollama_ctx_4096_numkeep")
    if nk and "ollama_ctx_4096" in tiers:
        s2.append(_item(f"The one-line fix, num_keep (keep the system prompt) at 4096: {nk['pct']:.1f}% wrong, down "
                        f"from {tiers['ollama_ctx_4096']['pct']:.1f}% but still above {tiers['ollama_default']['pct']:.1f}%"
                        f" at the default window", MEASURED, "x2-evalset-tasks-real-v2-wrong-rate"))
    rc = parse_reconciliation_rule2(val("R2-qwen3-14b-4096-reconciliation"))
    if rc["trimmed"] and rc["v1"]:
        s2.append(_item("Client-side trimming has a side effect: qwen3:14b called the logging tool in "
                        f"{'/'.join(str(a) for a, _ in rc['trimmed'])} of {rc['trimmed'][0][1]} turns per seed with "
                        f"trimming, {'/'.join(str(a) for a, _ in rc['v1'])} of {rc['v1'][0][1]} without. Trimming "
                        "removes the earlier turns that showed the two-call pattern the model imitates "
                        "(interpretation in docs/FINDINGS.md)", MEASURED, "R2-qwen3-14b-4096-reconciliation"))
    small = {c["model"]: c for c in parse_gated_per_model(val("R2-v1b-gated-per-model")) if c["tier"] == "num_ctx_4096"}
    if small:
        s2.append(_item("Rule loss at 4096 depends on the model: " + ", ".join(
            f"{m} {c['gated_silent']}/{c['n']}" for m, c in sorted(small.items())) + " sessions", MEASURED,
            "R2-v1b-gated-per-model"))
    scale = parse_scale(val("H-scale-verdict")) + parse_scale(val("H-scale-family-verdict"))
    if scale:
        s2.append(_item("Larger models at 4096: " + ", ".join(
            f"{s['model']} {s['silent']}/{s['n']}" + (" (resists rule loss)" if s["note"] and "resists" in s["note"]
                                                       else "") for s in scale) + " sessions", MEASURED,
            "H-scale-verdict"))
    steps.append({"n": 2, "title": "Why it is hard: the obvious fixes do not hold, and it depends on the model",
                  "minutes": 1, "items": s2, "panel": None})
    # 3. the tool
    steps.append({"n": 3, "title": "The tool: pick a device and a model, get the measured setting", "minutes": 1,
                  "items": [_item("Each setting carries its label: measured, pending flash-attention control, or not "
                                  "measured. Where we have no data, it says so. Try evo-t2s Intel CPU next.", None)],
                  "panel": "device", "defaults": {"device": "evo-x2", "model": "qwen3:14b",
                                                   "workload": "r2_agent"}})
    # 4. the money
    s4 = []
    tc = parse_task_cost(val("cloud-cost-per-real-task"))
    son = (tc.get("models") or {}).get("claude-sonnet-5-5")
    hai = (tc.get("models") or {}).get("claude-haiku-5-5")
    if son:
        s4.append(_item(f"One long agent task ({tc['trace_set']}, median {tc['median_steps']} steps) sent whole to "
                        f"the cloud: USD {son['realistic'][0]:.2f} median, USD {son['realistic'][1]:.2f} p90 on "
                        f"claude-sonnet-5-5 at an assumed 70% cache hit rate" +
                        (f"; USD {hai['realistic'][0]:.2f} on claude-haiku-5-5" if hai else ""),
                        PRICED, "cloud-cost-per-real-task"))
    lead = (cost or {}).get("lead")
    if lead:
        s4.append(_item(lead["text"], lead["label"], src=lead["src"]))
    s4.append(_item("Takeaway: sending every long task to the cloud is not cheap; escalating only the step that went "
                    "wrong needs to know which step went wrong, and the failure is silent.", None))
    steps.append({"n": 4, "title": "The money: a whole task in the cloud is not cheap, and the failure is silent",
                  "minutes": 1, "items": s4, "panel": None})
    # 5. one live escalation
    s5 = []
    esc = escalation or {}
    if esc.get("available"):
        loc = esc["local"]
        diff = ", ".join(loc.get("end_state_diff") or []) or "end state"
        got = next((c["arguments"] for c in loc.get("tool_calls") or [] if c.get("name") ==
                    (esc.get("expected_calls") or [{}])[0].get("name")), None)
        want = (esc.get("expected_calls") or [{}])[0].get("arguments")
        key = diff.rsplit("/", 1)[-1]
        if isinstance(got, dict) and isinstance(want, dict) and key in got and key in want:
            s5.append(_item(f"The local model ({esc['local_model']}, {loc.get('loaded_context')} window) set {key} = "
                            f"{got[key]}; the standing instruction gives {want[key]}. HTTP error: "
                            f"{'yes' if loc.get('any_http_error') else 'no'}.", MEASURED, src=loc.get("src")))
        s5.append(_item(FIX_COST_LEAD, BEING_MEASURED))
    s5.append(_item("One click sends that turn to the cloud through the capped client (one call per demo run); "
                    "without a network or key nothing is sent and the cost shows as being measured.", None))
    steps.append({"n": 5, "title": "One live escalation: send the wrong task to the cloud", "minutes": 1,
                  "items": s5, "panel": "escalate"})
    return {"steps": steps, "register_ids": STORY_IDS, "total_minutes": sum(s["minutes"] for s in steps)}


def story_view(cache) -> dict:
    s = dict(cache.story)
    s["register"] = {i: cache.register[i] for i in STORY_IDS if i in cache.register}
    return s
