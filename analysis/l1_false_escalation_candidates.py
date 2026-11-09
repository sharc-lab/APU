"""Post-run check: candidate decision rules that cut false escalations attributed to L1. Offline, USD 0, no network,
no model call, writes nothing (prints its tables).

Question (operator, 2026-11-09): the live 8192 run (results/x2_post_run_check_live_v1.jsonl, held-out seed 20260905)
escalated 11 of 21 tasks that were right locally; the operator attributes this to L1 (detector v2) firing on every
turn after the window overflows. This script (1) breaks the 11 down by layer, (2) evaluates candidate rules.

Data:
  replay: results/post_run_check_tuned_v1.jsonl prt_task rows (B3 v2 qwen3:14b, windows 4096, 8192, default; L1 is the
    L1 PROXY over_loaded_window, L2 is the L2 PROXY prompt_eval_count below the chars/4 estimate, L3 tuned
    CONFIG_TUNED, L1 errors real). Tuning seeds 20260901-04, held-out seed 20260905.
  live: results/x2_post_run_check_local_v1.jsonl via analysis/post_run_check_l2_example.live_recompute (real L1 =
    detector v2, real L2 render check, tuned L3, the controller's own decide_turn), with L2 as RECORDED (earliest
    example turn, the rule the run used) and L2 under Amendment 2026-11-03 (latest example turn, the rule any future
    run uses). Cloud outcomes of escalated tasks from results/x2_post_run_check_live_v1.jsonl (measured).

Candidates (fixed from the tuning seeds and the operator's list before any held-out replay figure was computed;
"L1" = real L1 live / L1 PROXY replay, "L2" = real L2 live / L2 PROXY replay, L1e = L1 call errors):
  C0 current: L1e or L1 or L2 or L3.
  C1 L1 onset only: L1 counts only on the first turn of the session on which it fires; L1e, L2, L3 unchanged.
  C2 L1 needs agreement: L1 counts only if L2 or L3 also fires. Equivalent to dropping L1 as a trigger.
  C3 L1 ignored when the render is citable: L1 counts only if a prompt-token check at or before the turn did not
     match (live only; the replay has no prompt-token check: not computable).
  C4 context-loss onset only: L1 and L2 each count only on the first turn of the session on which each fires.
  C5 L1 and L2 must agree: context loss escalates only if both L1 and L2 fire; L1e and L3 unchanged.
  C6 L3 only (plus L1e): L1 and L2 dropped (the floor).
"""
from __future__ import annotations

import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import post_run_check_replay as R  # noqa: E402

TUNED_FILE = REPO / "results" / "post_run_check_tuned_v1.jsonl"
LIVE_FILE = REPO / "results" / "x2_post_run_check_live_v1.jsonl"
LOCAL_FILE = REPO / "results" / "x2_post_run_check_local_v1.jsonl"
HOLDOUT = 20260905
TUNE_SEEDS = (20260901, 20260902, 20260903, 20260904)
MAP = {"L1_proxy": "L1", "L2_proxy": "L2", "L1_errors": "L1e", "L3": "L3", "L1": "L1", "L2": "L2"}
CANDS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
NAMES = {"C0": "current (any FAIL)", "C1": "L1 onset only", "C2": "L1 needs L2 or L3 (= L1 dropped)",
         "C3": "L1 ignored when render citable", "C4": "L1 and L2 onset only", "C5": "L1 and L2 must agree",
         "C6": "L3 + L1e only"}


# -- pure decision rules ---------------------------------------------------------------------------------------------

def annotate(session: list[dict]) -> list[dict]:
    """session: one (tier, seed) session's tasks with 'f' (set of layers fired). Adds onset flags."""
    out, seen = [], {"L1": False, "L2": False}
    for t in sorted(session, key=lambda x: x["turn_idx"]):
        t = dict(t)
        for k in ("L1", "L2"):
            t[f"{k}_onset"] = k in t["f"] and not seen[k]
            seen[k] = seen[k] or k in t["f"]
        out.append(t)
    return out


def escalate(t: dict, cand: str):
    f = t["f"]
    base = "L1e" in f or "L3" in f
    l1, l2 = "L1" in f, "L2" in f
    if cand == "C0":
        return base or l1 or l2
    if cand == "C1":
        return base or l2 or (l1 and t["L1_onset"])
    if cand == "C2":
        return base or l2 or (l1 and ("L2" in f or "L3" in f))
    if cand == "C3":
        if t.get("uncitable") is None:
            return None
        return base or l2 or (l1 and t["uncitable"])
    if cand == "C4":
        return base or (l1 and t["L1_onset"]) or (l2 and t["L2_onset"])
    if cand == "C5":
        return base or (l1 and l2)
    if cand == "C6":
        return base
    raise ValueError(cand)


def score(tasks: list[dict], cand: str, ref: list[dict] | None = None) -> dict | None:
    """ref: the task rows the CURRENT rule (C0) is applied to for 'lost' (default: the same rows)."""
    dec = [escalate(t, cand) for t in tasks]
    if any(d is None for d in dec):
        return None
    cur = [escalate(t, "C0") for t in (ref or tasks)]
    wrong = [i for i, t in enumerate(tasks) if not t["local_ok"]]
    right = [i for i, t in enumerate(tasks) if t["local_ok"]]
    return {"n": len(tasks), "right": len(right), "wrong": len(wrong),
            "false_esc": sum(dec[i] for i in right), "caught": sum(dec[i] for i in wrong),
            "lost": sum(1 for i in wrong if cur[i] and not dec[i]),
            "fe_removed": sum(1 for i in right if cur[i] and not dec[i]), "esc": dec}


# -- replay ----------------------------------------------------------------------------------------------------------

def replay_tasks() -> list[dict]:
    R._paths(REPO)
    from src.dse import l3_invariants as L3
    rows = [json.loads(x) for x in open(TUNED_FILE, encoding="utf-8")]
    if not any(r.get("record") == "run_end" for r in rows):
        raise SystemExit("post_run_check_tuned_v1.jsonl has no run_end")
    tasks = []
    for r in rows:
        if r.get("record") != "prt_task":
            continue
        f = {MAP[x] for x in r["other_fired"]}
        if L3.select(r["l3_all"], L3.CONFIG_TUNED):
            f.add("L3")
        tasks.append({"tier": r["tier"], "seed": r["seed"], "turn_idx": r["turn_idx"], "task_id": r["task_id"],
                      "local_ok": r["local_ok"], "f": f, "uncitable": None})
    out = []
    for tier in R.TIERS:
        for s in sorted({t["seed"] for t in tasks}):
            out += annotate([t for t in tasks if t["tier"] == tier and t["seed"] == s])
    return out


def replay_accuracy(tasks: list[dict], esc: list[bool], src: dict, tier_mean: dict) -> dict:
    g = [R.grade(e, t["local_ok"], (t["tier"], t["seed"], t["task_id"]), src, tier_mean) for t, e in zip(tasks, esc)]
    return {"correct": sum(x["ok"] for x in g), "n": len(g),
            "imputed": sum(x["src"].startswith("imputed") for x in g)}


# -- live ------------------------------------------------------------------------------------------------------------

def live_tasks() -> tuple[dict, dict]:
    import post_run_check_l2_example as X
    with contextlib.redirect_stdout(io.StringIO()):
        d = X.live_recompute(REPO)
    if d["status"] != "VERIFIED":
        raise SystemExit("live recompute guard failed: " + "; ".join(d["mismatches"][:5]))
    live = {r["turn_idx"]: r for r in (json.loads(x) for x in open(LIVE_FILE, encoding="utf-8"))
            if r.get("record") == "prc_live_task"}
    loc = [json.loads(x) for x in open(LOCAL_FILE, encoding="utf-8")]
    checks = [r for r in loc if r.get("record") == "prc_l2_prompt_check"]
    turns = {r["turn_idx"]: r for r in loc if r.get("record") == "r2a_turn" and r.get("item_kind") == "task"}
    sides = {}
    for side in ("before", "after"):
        ts = []
        for t in d["tasks"]:
            lv = live[t["turn_idx"]]
            if side == "before" and (t["fired_before"] != lv["fired"] or t["before"] != lv["decision"]):
                raise SystemExit(f"recomputed decision differs from the live file at turn {t['turn_idx']}")
            unc = any(not c.get("match") for c in checks if c["seed"] == t["seed"] and c["turn_idx"] <= t["turn_idx"])
            ts.append({"tier": "ollama_ctx_8192", "seed": t["seed"], "turn_idx": t["turn_idx"],
                       "task_id": t["task_id"], "local_ok": t["local_ok"], "f": {MAP[x] for x in t[f"fired_{side}"]},
                       "uncitable": unc, "cloud": lv.get("cloud") or {}, "row": turns[t["turn_idx"]],
                       "example": t[f"example_{side}"]})
        sides[side] = annotate(ts)
    return sides, d


def live_accuracy(tasks: list[dict], esc: list[bool]) -> dict:
    ok, unmeasured = 0, 0
    for t, e in zip(tasks, esc):
        if not e:
            ok += t["local_ok"]
        elif t["cloud"].get("outcome") == "cloud" or "cloud_end_state_ok" in t["cloud"]:
            ok += bool(t["cloud"].get("cloud_end_state_ok"))
        else:
            unmeasured += 1
    return {"correct": ok, "n": len(tasks), "unmeasured": unmeasured}


# -- printing --------------------------------------------------------------------------------------------------------

def line(label: str, s: dict | None, acc: dict | None) -> str:
    if s is None:
        return f"  {label:<38} not computable (no prompt-token check in this data)"
    a = ""
    if acc:
        a = f"  final {acc['correct']:.2f}/{acc['n']}" + (f" (imputed {acc['imputed']})" if acc.get("imputed") else "")
        if acc.get("unmeasured"):
            a += f" (unmeasured {acc['unmeasured']})"
    return (f"  {label:<38} false esc {s['false_esc']:>3}/{s['right']:<3} caught {s['caught']:>3}/{s['wrong']:<3} "
            f"lost vs C0 {s['lost']:>3}  false esc removed vs C0 {s['fe_removed']:>3}{a}")


def main() -> None:
    out = []
    p = out.append
    sides, d = live_tasks()
    rec = sides["before"]
    aft = {t["turn_idx"]: t for t in sides["after"]}

    p("== 1. Live 8192 (seed 20260905): the false escalations by layer ==")
    fe = [t for t in rec if t["local_ok"] and escalate(t, "C0")]
    p(f"false escalations under the recorded decisions: {len(fe)}/{sum(t['local_ok'] for t in rec)}")
    for t in fe:
        a = aft[t["turn_idx"]]
        wo = sorted(t["f"] - {"L1"})
        wo_a = sorted(a["f"] - {"L1"})
        p(f"  turn {t['turn_idx']:>2} {t['task_id']:<20} fired {sorted(t['f'])}; without L1 (L2 recorded): "
          f"{'ESCALATE via ' + '+'.join(wo) if wo else 'accept'}; L2 latest-example rule fires: {'L2' in a['f']} "
          f"(example turn {a['example']}); without L1 (L2 latest): {'ESCALATE via ' + '+'.join(wo_a) if wo_a else 'accept'}")
    n_l1_alone = sum(1 for t in fe if t["f"] == {"L1"})
    n_wo = sum(1 for t in fe if t["f"] - {"L1"})
    n_wo_a = sum(1 for t in fe if aft[t["turn_idx"]]["f"] - {"L1"})
    p(f"  L1 fired on {sum('L1' in t['f'] for t in fe)}/{len(fe)}, L1 alone on {n_l1_alone}/{len(fe)}; still escalated "
      f"without L1: {n_wo}/{len(fe)} with L2 as recorded (earliest example), {n_wo_a}/{len(fe)} with L2 latest example")
    l1_turns = [t["turn_idx"] for t in rec if "L1" in t["f"]]
    l2_turns = [t["turn_idx"] for t in rec if "L2" in t["f"]]
    olw = [t["turn_idx"] for t in rec if t["row"].get("over_loaded_window")]
    p(f"  L1 fires on turns {l1_turns}; L2 (recorded) on {l2_turns}; over_loaded_window on {olw}")
    p(f"  L2 latest-example fails on calls: {d['l2_fail_calls_after']}/{d['n_calls']} (recorded: "
      f"{d['l2_fail_calls_before']}/{d['n_calls']}); prompt-token checks matched {d['prompt_checks_matched']}/"
      f"{d['n_prompt_checks']}")

    p("\n== 2. Evidence excerpts (live) ==")
    for ti in (12, 21, 30):
        t = next(x for x in rec if x["turn_idx"] == ti)
        r = t["row"]
        p(f"  turn {ti} {t['task_id']} local_ok={t['local_ok']}: cut_v2_flagged={r.get('cut_v2_flagged')}, "
          f"cut_v2_no_log_calls={r.get('cut_v2_no_log_calls')}, cut_v2_evidence={r.get('cut_v2_evidence')}; "
          f"over_loaded_window={r.get('over_loaded_window')} (final prompt {r.get('final_call_prompt_tokens_calibrated')}"
          f" calibrated tokens vs loaded window {r.get('loaded_context')}); fired {sorted(t['f'])}; L2 example turn "
          f"recorded {t['example']}, latest {aft[ti]['example']}")

    rt = replay_tasks()
    with contextlib.redirect_stdout(io.StringIO()):
        src = R.cloud_sources(REPO)
    tm = {}
    for tier in R.TIERS:
        tr = [v for (t, _s, _k), v in src["redo"].items() if t == tier]
        tm[tier] = {"usd": statistics.fmean(v["usd"] for v in tr), "upper": statistics.fmean(v["upper"] for v in tr)}

    def replay_block(sel, title):
        p(f"\n== {title} ==")
        for c in CANDS:
            p(f"{c} {NAMES[c]}")
            for tier in R.TIERS:
                ts = [t for t in rt if t["tier"] == tier and sel(t)]
                s = score(ts, c)
                acc = replay_accuracy(ts, s["esc"], src, tm[tier]) if s else None
                p(line(f"replay {tier}", s, acc))

    replay_block(lambda t: t["seed"] in TUNE_SEEDS, "3. TUNING seeds 20260901-04, replay (L1/L2 are PROXIES), n=120 per window")
    replay_block(lambda t: t["seed"] == HOLDOUT, "4. HELD-OUT seed 20260905, replay (L1/L2 are PROXIES), n=30 per window")

    p("\n== 5. HELD-OUT seed 20260905, live 8192 (real L1 and L2), n=30; final accuracy measured (escalation sets "
      "are subsets of the live run's, every escalated task has a recorded cloud outcome) ==")
    for c in CANDS:
        p(f"{c} {NAMES[c]}")
        for side, lab in (("before", "live 8192, L2 as recorded (earliest)"), ("after", "live 8192, L2 latest example")):
            ts = sides[side]
            s = score(ts, c, ref=rec)
            if s:
                extra = [t["turn_idx"] for t, e, r0 in zip(ts, s["esc"], rec) if e and not escalate(r0, "C0")]
                if extra:
                    p(f"  NOTE: escalates turns not escalated live: {extra}")
            p(line(lab, s, live_accuracy(ts, s["esc"]) if s else None))
            if s and c != "C0":
                lost = [t["turn_idx"] for t, e, r0 in zip(ts, s["esc"], rec)
                        if escalate(r0, "C0") and not e and not t["local_ok"]]
                kept = [t["turn_idx"] for t, e, r0 in zip(ts, s["esc"], rec)
                        if escalate(r0, "C0") and not e and t["local_ok"]]
                p(f"    vs C0 as run: wrong tasks no longer escalated: turns {lost}; right tasks no longer "
                  f"escalated: turns {kept}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
