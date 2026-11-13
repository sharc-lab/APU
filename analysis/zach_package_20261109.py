"""Zach's data package of 2026-11-09 (his ISPASS part), read from ~/zach_runs/data_20261109 (the user's home), outside
the repo (his data, not committed). One function per README item, for the register rows zach-*-20261109. Credit:
every run, score file and sealed prediction here is Zach's; this module only reads them, and recomputes the T4-8B
counts and the PITCH-LL rates from his raw rows to check his README.

Pure: no network, no model call, no subprocess. Every path is checked against MANIFEST.csv's sha256 by
manifest_check() (all 672 matched on 2026-11-12).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

DATA = Path("~/zach_runs/data_20261109").expanduser()
PREFIX = "rithwik_data_20261109/"
CREDIT = "Zach's run and scoring (data package 2026-11-09)"


def p(rel: str) -> Path:
    return DATA / rel


def jload(rel):
    return json.loads(p(rel).read_text(encoding="utf-8"))


def jrows(rel):
    return [json.loads(x) for x in p(rel).read_text(encoding="utf-8").splitlines() if x.strip()]


def text(rel):
    return p(rel).read_text(encoding="utf-8")


def manifest_check() -> dict:
    rows = list(csv.DictReader(open(p("MANIFEST.csv"), encoding="utf-8-sig")))
    bad, missing = [], []
    for r in rows:
        rel = r["path_in_zip"].replace("\\", "/")
        rel = rel[len(PREFIX):] if rel.startswith(PREFIX) else rel
        f = p(rel)
        if not f.exists():
            missing.append(rel)
        elif hashlib.sha256(f.read_bytes()).hexdigest().lower() != r["sha256"].strip().lower():
            bad.append(rel)
    return {"n": len(rows), "ok": len(rows) - len(bad) - len(missing), "mismatch": bad, "missing": missing,
            "status": {r["item"]: sorted({x["sealed_or_exploratory"] for x in rows if x["item"] == r["item"]})
                       for r in rows}}


def _pct(a, b):
    return f"{a}/{b} ({100 * a / b:.1f}%)"


# ── 1. T4-8B ────────────────────────────────────────────────────────────────────────────────────────

T4_ROWS = "1_t4_8b/t2s/window_20261106_195839_cu_t4_8b/ct_t4_8b_20261106_195839.jsonl"


def t4_recompute() -> dict:
    """From the raw r2a_turn rows: task turns and wrong (end_state_ok False) per arm, and recall error over is_recall
    turns (recall_ok False), as tools/bridge1_secondary.py defines it."""
    out = {}
    for r in jrows(T4_ROWS):
        if r.get("record") != "r2a_turn" or r.get("item_kind") != "task":
            continue
        a = r["arm_id"].replace("_call2_notools", "")
        d = out.setdefault(a, {"turns": 0, "wrong": 0, "recall": 0, "recall_wrong": 0, "sessions": set()})
        d["turns"] += 1
        d["wrong"] += r.get("end_state_ok") is not True
        d["sessions"].add(r["attempt_id"])
        if r.get("is_recall"):
            d["recall"] += 1
            d["recall_wrong"] += r.get("recall_ok") is not True
    for d in out.values():
        d["sessions"] = len(d["sessions"])
    return out


def t4_text() -> str:
    s, c = jload("1_t4_8b/scores/bridge1_scores.json"), jload("1_t4_8b/scores/cluster_ci.json")
    rc = t4_recompute()
    D, P = s["arms"]["bridge1_D"], s["arms"]["bridge1_P"]
    rD, rP = rc["bridge1_D"], rc["bridge1_P"]
    match = (rD["wrong"], rD["turns"], rP["wrong"], rP["turns"]) == (D["wrong"], D["task_turns"], P["wrong"],
                                                                     P["task_turns"])
    pe = c["paired_error_rate"]
    return (f"SEALED (S0, S0a, S0g, S0h; cluster CI EXPLORATORY), {CREDIT}: T4-8B, qwen3:8b on evo-t2s, Ollama "
            f"0.40.14, 8 seeds. Task error D (default, loaded {D['loaded_context']}) {_pct(D['wrong'], D['task_turns'])}"
            f" vs P (loaded {P['loaded_context']}) {_pct(P['wrong'], P['task_turns'])}; D silent {s['D_silent_failures']}"
            f"/{s['D_failures']}; paired D-P +{pe['mean']:.3f}, 95% CI t [{pe['ci95_t'][0]:+.3f}, "
            f"{pe['ci95_t'][1]:+.3f}], cluster bootstrap [{pe['ci95_bootstrap'][0]:+.3f}, {pe['ci95_bootstrap'][1]:+.3f}]"
            f"; sign {pe['sign_test']['pos']}/{pe['sign_test']['neg']}, p = {pe['sign_test']['p_two_sided']:.4f}; "
            f"verdicts {s['verdicts']}. Recall row MISS as sealed: recall error D {rD['recall_wrong']}/{rD['recall']} "
            f"({rD['recall_wrong'] / rD['recall']:.2f}) vs P {rP['recall_wrong']}/{rP['recall']} "
            f"({rP['recall_wrong'] / rP['recall']:.2f}). RECOMPUTED from the raw rows ({T4_ROWS}): D "
            f"{rD['wrong']}/{rD['turns']}, P {rP['wrong']}/{rP['turns']} over {rD['sessions']} + {rP['sessions']} sessions"
            f": {'matches' if match else 'DOES NOT MATCH'} the README (193/240, 60/240)")


# ── 2. PITCH-LL ─────────────────────────────────────────────────────────────────────────────────────

PITCH_RUN1 = "2_pitch_ll/rows/pitch_20261107_081906.jsonl"      # C, D (its A, B rows are void)
PITCH_RUN2 = "2_pitch_ll/rows/pitch2_20261107_141829.jsonl"     # A, B, E
PITCH_SPEND = ("2_pitch_ll/rows/pitch_20261107_081906.spend.jsonl", "2_pitch_ll/rows/pitch2_20261107_141829.spend.jsonl",
               "2_pitch_ll/rows/pitch_smoke_20261107_081906.spend.jsonl",
               "2_pitch_ll/rows/pitch2_smoke_20261107_140052.spend.jsonl")
FLAG_REASONS = {"echo", "echo_truncated", "verifier", "verifier_claim_without_action", "turn_escalated"}


def pitch_recompute() -> dict:
    """Per arm from the raw rows (SCORES.md definitions): wrong = end_state_ok False over task turns; silent-wrong =
    wrong, no raised error (any_http_error), and no escalation flag on that turn (echo, verifier, or the cloud re-run
    of an escalated turn; routing before the call is not a flag); local share = turns whose final answer is local;
    cloud USD = the pitch_call cost_usd of the arm."""
    out = {}
    for f, arms in ((PITCH_RUN1, "CD"), (PITCH_RUN2, "ABE")):
        rows = jrows(f)
        calls = {}
        for c in (r for r in rows if r.get("record") == "pitch_call" and r["arm"] in arms):
            k = (c["arm"], c["seed"], c["turn"])
            calls.setdefault(k, []).append(c)
        for r in (x for x in rows if x.get("record") == "r2a_turn" and x.get("item_kind") == "task"):
            a = r["arm_id"].replace("pitch_", "").replace("_call2_notools", "")
            if a not in arms:
                continue
            cs = calls.get((a, r["seed"], r["turn_idx"]), [])
            flagged = any(c["reason"] in FLAG_REASONS for c in cs)
            final_local = (cs[-1]["responder"] == "local") if cs else True
            d = out.setdefault(a, {"turns": 0, "wrong": 0, "silent": 0, "local": 0, "cloud_turns": 0, "usd": 0.0})
            wrong = r.get("end_state_ok") is not True
            d["turns"] += 1
            d["wrong"] += wrong
            d["silent"] += wrong and not r.get("any_http_error") and not flagged
            d["local"] += final_local
            d["cloud_turns"] += not final_local
        for c in (r for r in rows if r.get("record") == "pitch_call" and r["arm"] in arms):
            out.setdefault(c["arm"], {}).setdefault("usd", 0.0)
            out[c["arm"]]["usd"] += float(c.get("cost_usd") or 0.0)
    total = 0.0
    for f in PITCH_SPEND:
        for r in jrows(f):
            total += float(r.get("cost_usd") or r.get("usd") or 0.0)
    return {"arms": out, "spend_total": total}


def pitch_text() -> str:
    s = jload("2_pitch_ll/scores/s0j_s0m_scores.json")
    t, rc = s["table"], pitch_recompute()
    cells, mism = [], []
    for a in "ABCDE":
        x, y = t[a], rc["arms"][a]
        cells.append(f"{a} wrong {_pct(x['wrong'], x['turns'])}, silent-wrong {_pct(x['silent_wrong'], x['turns'])}, "
                     f"local {100 * x['local_share']:.1f}%, cloud turns {x['cloud_turns']}, USD {x['cloud_usd']:.4f}")
        for k, a1, b1 in (("wrong", x["wrong"], y["wrong"]), ("silent-wrong", x["silent_wrong"], y["silent"]),
                          ("cloud turns", x["cloud_turns"], y["cloud_turns"])):
            if a1 != b1:
                mism.append(f"{a} {k}: README {a1}, recomputed {b1}")
        if abs(x["cloud_usd"] - y["usd"]) > 1e-6:
            mism.append(f"{a} USD: score file {x['cloud_usd']:.6f}, recomputed {y['usd']:.6f}")
    verd = {k: v["verdict"] for k, v in s["rows"].items()}
    return (f"SEALED (S0j, S0m), {CREDIT}: PITCH-LL, qwen3:8b on evo-t2s, Ollama 0.40.14, 8 seeds x 30 task turns "
            f"per arm; A LiteLLM failover, B LiteLLM declared pre-check, C hand threshold, D Zach's mask + echo + "
            f"verifier, E D + RECONFIGURE / " + " / ".join(cells) + f" / verdicts {verd} (E-VS-D MISS kept as sealed) "
            f"/ total spend in the 4 spend files USD {rc['spend_total']:.4f} (README $1.8533) / RECOMPUTED from the "
            f"raw rows (C, D from {PITCH_RUN1}; A, B, E from {PITCH_RUN2}): "
            + ("all counts and USD match the score file" if not mism else "MISMATCH: " + "; ".join(mism)))


# ── 3. CL1 ──────────────────────────────────────────────────────────────────────────────────────────

def cl1_text() -> str:
    md = text("3_cl1/scores/SCORES.md")
    lines = [ln.strip("- ").strip() for ln in md.splitlines() if ln.startswith("- P-S0k-")]
    return (f"SEALED (S0k), {CREDIT}: CL1 cloud arm, claude-haiku-5-5, rows 3_cl1/rows/cl1_20261106.jsonl / "
            + " / ".join(lines))


# ── 4a to 4f ────────────────────────────────────────────────────────────────────────────────────────

def x1_text() -> str:
    s = jload("4_x1/scores/score_s0f.json")
    pp = s["per_precision"]
    br = ", ".join(f"{k} ({v['bracket'][0]:,}, {v['bracket'][1]:,}] ({v['n_points']} points)" for k, v in pp.items())
    return (f"SEALED (S0f), {CREDIT}: X1 allocator and onset on the XPS, 9 cells; verdicts {s['verdicts']}; onset "
            f"brackets {br}")


def e1_text() -> str:
    rows = [ln for ln in text("4_e1/scores/E1_RESULTS.md").splitlines() if ln.startswith("| ") and "---" not in ln
            and not ln.startswith("| model")]
    cells = []
    for ln in rows:
        c = [x.strip() for x in ln.strip("|").split("|")]
        cells.append(f"{c[0]}: L_E1 {c[3]} ({c[4]})")
    return f"SEALED (feeds S-2), {CREDIT}: E1 largest OK prompt L per model, windows 083557 + 110326 / " + " / ".join(cells)


def e2_text() -> str:
    s = jload("4_e2/scores/s2_e2_scores.json")
    rows = s["rows"]
    hit = sum(1 for r in rows if r.get("verdict") == "HIT")
    scored = [r for r in rows if r.get("verdict") in ("HIT", "MISS")]
    pend = [r["id"] for r in rows if r.get("verdict") not in ("HIT", "MISS")]
    return (f"SEALED (S2), {CREDIT}: E2 bisection, {len(rows)} rows in the score file: {hit} HIT, "
            f"{len(scored) - hit} MISS, {len(pend)} not scored ({', '.join(pend)}); README '5 of 12 HIT' counts the "
            f"unscored row in the 12; the 4 session rows are pending (README); the INTERFERED 2cff068e cell is left out")


def c256_text() -> str:
    s = jload("4_c256/scores/s0i_scores.json")
    rows = "; ".join(f"{k}: {v['verdict']}" + (f" ({v['observed']})" if v.get("observed") else "")
                     for k, v in s["rows"].items())
    pts = {}
    for pt in s["points"]:
        d = pts.setdefault((pt["kv"], pt["n"]), [0, 0])
        d[0] += pt["cls"] == "PAGING_ONSET"
        d[1] += 1
    edge = ", ".join(f"{kv} {n:,} paged {a}/{b}" for (kv, n), (a, b) in sorted(pts.items()) if n >= 79000)
    return (f"SEALED (S0i), {CREDIT}: c256 onset (chunked prefill), run_ids 974f61a9, fae19c82, 582df6d2, cc19e964, "
            f"{len(s['points'])} points / {rows} / at the edge: {edge} / README: i8 onset ~79-80K vs X1 45K (MOVE HIT); "
            f"u4 censored >= 80K")


def t1_text() -> str:
    md = text("4_t1/scores/t2s_weekend_SCORES.md")
    sec = md.split("## T1 paging control", 1)[1].split("\n## ", 1)[0]
    tab = [ln for ln in sec.splitlines() if ln.startswith("| ") and "---" not in ln]
    hits = [ln.strip("- ").strip() for ln in sec.splitlines() if ln.startswith("- P-S0h-T1")]
    return (f"SEALED (S0h), {CREDIT}: T1 64 GB paging control on evo-t2s (run f421dd48) / " + " / ".join(tab) + " / "
            + " / ".join(hits) + " / README: the XPS ran the same sizes in 327 / 1,102 / 2,913 s (c2d87690)")


def w3_text() -> str:
    md = text("4_w3/scores/SCORES_P50.md")
    rows = [ln for ln in md.splitlines() if ln.startswith("| P-S0l-") and "-50 " in ln]
    return (f"SEALED (S0l), {CREDIT}: share sweep w3 at 50%, run_ids 55ed739f, 65b31663, 1d755d81 / " + " / ".join(rows)
            + " / the 74% cells are a separate PENDING row")


# ── 5. QUAL1 ────────────────────────────────────────────────────────────────────────────────────────

def qual1_text() -> str:
    q, b = jload("5_qual1/scores/qual1_scores.json"), jload("5_qual1/scores/qual1_bfcl_scores.json")
    r = q["P-S2-QUAL-RULER"]
    other = {k: v for k, v in q.items() if k != "P-S2-QUAL-RULER" and isinstance(v, dict) and "delta" in v}
    oth = "; ".join(f"{k}: {100 * v['delta']:+.1f} pp [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}] "
                    f"(n={v.get('n')})" for k, v in other.items())
    return (f"SEALED (S2), {CREDIT}: RULER, 4-bit (u4) KV vs reference {100 * r['delta']:+.1f} pp, 95% CI "
            f"[{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:+.1f}] (n={r['n']} paired items): at the edge of the 5 pp "
            f"margin, a 5 pp loss NOT RULED OUT / {oth} / BFCL multi-turn i8 {100 * b['acc_ref']:.0f}% vs u4 "
            f"{100 * b['acc_cmp']:.0f}%, {100 * b['delta']:+.1f} pp [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}] "
            f"(n={b['n']}): {b['verdict']} (power), kept as sealed")


# ── 6. CONTRACT-1 ───────────────────────────────────────────────────────────────────────────────────

def contract1_text() -> str:
    a1 = text("6_contract1/scores/contract1_a1_20261105/SCORES.md")
    lms = text("6_contract1/scores/contract1_lms_20261109/SCORES.md")
    oll = next(ln for ln in a1.splitlines() if ln.startswith("| ollama |"))
    lm = next(ln for ln in lms.splitlines() if ln.startswith("| P-CT1-LMS"))
    stacks = [ln for ln in a1.splitlines() if ln.startswith("| llama-server-") or ln.startswith("| openvino")]
    return (f"SEALED (S0, S0a), {CREDIT}: CONTRACT-1 overflow contract per stack / {oll} / " + " / ".join(stacks) +
            f" / LM Studio: {lm} (MISS kept as sealed: loud HTTP 400, not silent) / Foundry Local is a separate PENDING row")


# ── 7. OS reserve ───────────────────────────────────────────────────────────────────────────────────

def os_reserve_text() -> str:
    s = jload("7_os_reserve/fit/os_reserve_fit.json")
    r = s["R_prefill_57"]
    by = {k.split("=", 1)[1]: v["median"] for k, v in s["R_prefill_by"].items() if k.startswith("model=")}
    return (f"EXPLORATORY (no sealed row), {CREDIT}: OS reserve R_prefill median {r['median']:,} MiB (IQR {r['q1']:,} to "
            f"{r['q3']:,}; n={r['n']}), by model {min(by.values()):,} to {max(by.values()):,} MiB; {s['n_probes']} probes, "
            f"{s['n_fit']} in the fit; README: subtracting R cuts the static predictor's error 2-11x")


# ── PENDING rows ────────────────────────────────────────────────────────────────────────────────────

def foundry_pending(_repo=None) -> str:
    return ("PENDING (Zach's README 6*): Foundry Local 0.13 (92502061) NOT SCORED; the autoscore marks it NEEDS_MANUAL "
            "(its p5 LOUD_REJECT is an HTTP 499 client disconnect after 1,588 s, not a stack outcome)")


def w3_74_pending(_repo=None) -> str:
    return ("PENDING (Zach's README 4f): the w3 74% cells (74b800db, 4f887d62, 32353275) are in the package but not yet "
            "scored")


ITEMS = {"1": t4_text, "2": pitch_text, "3": cl1_text, "4a": x1_text, "4b": e1_text, "4c": e2_text, "4d": c256_text,
         "4e": t1_text, "4f": w3_text, "5": qual1_text, "6": contract1_text, "7": os_reserve_text}

if __name__ == "__main__":
    print(json.dumps({k: v for k, v in manifest_check().items() if k != "status"}, indent=1))
    for k, f in ITEMS.items():
        print(f"\n[{k}] {f()}")
