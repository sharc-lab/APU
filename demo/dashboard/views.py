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
LABELS = (MEASURED, PROJECTED, SIMULATED, PRICED, NOT_MEASURED)

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
COST_IDS = ["cloud-redo-cost-per-wrong-task", "x2-evalset-tasks-real-v2-wrong-rate", "cost-cut-config-time",
            "impact-v2-P1", "impact-ranking-v2", "impact-P3", "impact-total", "b5-verdict", "b5-policy-metrics",
            "impact-hour-value"]


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
    return {"mean_usd": float(m["mean"]), "median_usd": float(m["med"]), "total_usd": float(m["tot"]),
            "model": model[1] if model else None, "per_tier": tiers,
            "mode_priced_not_executed": "priced, not executed" in value}


def b5_verdict(register: dict) -> str | None:
    r = _row(register, "b5-verdict")
    if not r:
        return None
    m = re.match(r"\s*(\w+)", r["value"])
    return m[1] if m else None


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
