"""The two candidate pick rules for the 5-task live run (operator to choose one; the chosen rule's output is frozen to
demo/live/picks_v1.json before any run, and never changed after the first run).

Both read only results/x2_evalset_tasks_real_v2.jsonl (qwen3:14b, 5 seeds, completed real sessions, task turns) and
never look at any Ours, cloud or demo result.

  rule_a (operator message 1): 3 tasks that were wrong at num_ctx 4096 in >= 3 of 5 seeds and right at the default
         window in >= 4 of 5 seeds, plus 2 tasks right in every recorded tier and seed; ties broken by task id.
         A task id occurs once per seed (at a seed-dependent position), so a pick needs a seed: the demo runs the task
         at its real position in the session of the lowest seed where its recorded outcome matches its class (wrong
         at 4096 and right at the default; right everywhere).
  rule_b (operator message 2, the hash rule proposed on 2026-10-22): task turns 6 to 12, not recall and not canary
         turns, from the recorded num_ctx 4096 sessions: 3 that were wrong and 2 that were right, each class ordered
         by sha256("<seed>:<turn>") ascending.

    py -3.12 -m demo.live.picks            # prints both candidate lists; writes nothing
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "harness") not in sys.path:
    sys.path.insert(0, str(REPO / "harness"))

A4096, ADEF = "ollama_ctx_4096_call2_notools", "ollama_default_call2_notools"


def task_turns(rows: list[dict]) -> list[dict]:
    import x2_r2_agent as ag
    att = {s["attempt_id"] for s in ag.completed_sessions(rows) if s.get("mode") == "real"
           and s["model_id"] == "qwen3:14b"}
    return [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in att
            and r.get("item_kind") == "task" and r.get("end_state_ok") is not None]


def rule_a(rows: list[dict]) -> dict:
    t = task_turns(rows)
    seeds = sorted({r["seed"] for r in t})
    arms = sorted({r["arm_id"] for r in t})
    by = {}
    for r in t:
        by.setdefault(r["task_id"], {})[(r["arm_id"], r["seed"])] = r
    fail, allpass = [], []
    for tid in sorted(by):
        v = by[tid]
        w4 = [s for s in seeds if (A4096, s) in v and v[(A4096, s)]["end_state_ok"] is False]
        rd = [s for s in seeds if (ADEF, s) in v and v[(ADEF, s)]["end_state_ok"] is True]
        if len(w4) >= 3 and len(rd) >= 4:
            both = [s for s in seeds if s in w4 and s in rd]
            fail.append({"task_id": tid, "wrong_at_4096_seeds": len(w4), "right_at_default_seeds": len(rd),
                         "seed": both[0] if both else None, "class": "wrong at 4096, right at default"})
        if len(v) == len(arms) * len(seeds) and all(r["end_state_ok"] is True for r in v.values()):
            allpass.append({"task_id": tid, "seed": seeds[0], "class": "right in every tier and seed"})
    for p in fail + allpass:
        if p["seed"] is not None:
            p["turn_idx"] = by[p["task_id"]][(ADEF, p["seed"])]["turn_idx"]
    return {"rule": "a", "qualifying_fail": fail, "qualifying_pass": allpass,
            "picks": fail[:3] + allpass[:2], "enough": len(fail) >= 3 and len(allpass) >= 2}


def rule_b(rows: list[dict]) -> dict:
    t = [r for r in task_turns(rows) if r["arm_id"] == A4096 and not r["is_recall"] and not r["canary_check"]
         and 6 <= r["turn_idx"] <= 12]
    h = lambda r: hashlib.sha256(f"{r['seed']}:{r['turn_idx']}".encode()).hexdigest()  # noqa: E731
    wrong = sorted((r for r in t if r["end_state_ok"] is False), key=h)
    right = sorted((r for r in t if r["end_state_ok"] is True), key=h)
    pick = lambda r, c: {"seed": r["seed"], "task_id": r["task_id"], "turn_idx": r["turn_idx"], "class": c}  # noqa
    return {"rule": "b", "n_wrong": len(wrong), "n_right": len(right),
            "picks": [pick(r, "wrong at 4096") for r in wrong[:3]] + [pick(r, "right at 4096") for r in right[:2]],
            "enough": len(wrong) >= 3 and len(right) >= 2}


RULE_V1 = (
    "Operator rule of 2026-10-23 (frozen): 3 task turns that were wrong at num_ctx 4096 in at least 3 of 5 seeds AND "
    "right at the default window in at least 4 of 5 seeds, plus 2 task turns that were right in at least 4 of 5 seeds "
    "in EVERY recorded arm; one pick per task id; ties broken by task id. A task occurs once per seed, at a "
    "seed-dependent position: the pick is that task in the session of the lowest seed whose own recorded outcomes "
    "match the class (wrong at 4096 and right at the default; right in every arm), preferring a turn position 6 to 12; "
    "the 6 to 12 limit is dropped for a class only if it leaves fewer picks than needed. If fewer tasks qualify than "
    "needed, the class is filled with the tasks that failed (resp. succeeded) most often, and the counts are reported. "
    "Source: results/x2_evalset_tasks_real_v2.jsonl (qwen3:14b, completed real sessions, 5 seeds). History rebuilt "
    "from the recorded default-window session (REPLAYED); the task turn runs LIVE.")


def rule_v1(rows: list[dict]) -> dict:
    t = task_turns(rows)
    seeds = sorted({r["seed"] for r in t})
    arms = sorted({r["arm_id"] for r in t})
    by = {}
    for r in t:
        by.setdefault(r["task_id"], {})[(r["arm_id"], r["seed"])] = r
    ok = lambda v, a, s: (a, s) in v and v[(a, s)]["end_state_ok"] is True  # noqa: E731
    bad = lambda v, a, s: (a, s) in v and v[(a, s)]["end_state_ok"] is False  # noqa: E731
    fail_q, pass_q, fail_score, pass_score = [], [], {}, {}
    for tid in sorted(by):
        v = by[tid]
        w4 = sum(bad(v, A4096, s) for s in seeds)
        rd = sum(ok(v, ADEF, s) for s in seeds)
        per_arm = {a: sum(ok(v, a, s) for s in seeds) for a in arms}
        fail_score[tid] = (w4, rd)
        pass_score[tid] = min(per_arm.values())
        if w4 >= 3 and rd >= 4:
            fail_q.append(tid)
        if all(n >= 4 for n in per_arm.values()):
            pass_q.append(tid)

    def seed_turns(tid, match):
        v = by[tid]
        return [(s, v[(ADEF, s)]["turn_idx"]) for s in seeds if (ADEF, s) in v and match(v, s)]

    m_fail = lambda v, s: bad(v, A4096, s) and ok(v, ADEF, s)  # noqa: E731
    m_pass = lambda v, s: all(ok(v, a, s) for a in arms)  # noqa: E731

    def choose(qual, match, need, cls):
        cands = [(tid, st) for tid in qual for st in [seed_turns(tid, match)] if st]
        inside = [(tid, [x for x in st if 6 <= x[1] <= 12]) for tid, st in cands]
        inside = [(tid, st) for tid, st in inside if st]
        limit = len(inside) >= need
        use = inside if limit else cands
        picks = [{"seed": st[0][0], "task_id": tid, "turn_idx": st[0][1], "class": cls,
                  "turn_limit_6_12": limit} for tid, st in use[:need]]
        return picks, {"qualifying": len(qual), "with_matching_seed": len(cands), "inside_6_12": len(inside),
                       "turn_limit_applied": limit}

    fp, fstat = choose(fail_q, m_fail, 3, "wrong at 4096, right at default")
    pp, pstat = choose(pass_q, m_pass, 2, "right in every arm")
    filled = {}
    if len(fp) < 3:   # fill with the tasks that failed most often (reported, never silent)
        extra = sorted((tid for tid in by if tid not in {p["task_id"] for p in fp}),
                       key=lambda tid: (-fail_score[tid][0], -fail_score[tid][1], tid))
        filled["fail"] = extra[:3 - len(fp)]
    if len(pp) < 2:
        extra = sorted((tid for tid in by if tid not in {p["task_id"] for p in pp}),
                       key=lambda tid: (-pass_score[tid], tid))
        filled["pass"] = extra[:2 - len(pp)]
    for p in fp + pp:
        p["recorded"] = {a: [by[p["task_id"]][(a, s)]["end_state_ok"] for s in seeds if (a, s) in by[p["task_id"]]]
                         for a in arms}
    return {"rule": RULE_V1, "fail": fstat, "pass": pstat, "filled_beyond_rule": filled,
            "picks": fp + pp, "enough": len(fp) == 3 and len(pp) == 2, "seeds": seeds, "arms": arms}


def freeze(out: Path, rows: list[dict]) -> dict:
    import hashlib as _h
    if out.exists():
        raise SystemExit(f"{out} exists; frozen picks are never rewritten")
    r = rule_v1(rows)
    if r["filled_beyond_rule"]:
        raise SystemExit(f"fewer tasks qualify than needed ({r['fail']}, {r['pass']}); report to the operator first")
    src = REPO / "results" / "x2_evalset_tasks_real_v2.jsonl"
    doc = {"frozen": "2026-10-23", "rule": r["rule"], "source": "results/x2_evalset_tasks_real_v2.jsonl",
           "source_sha256": _h.sha256(src.read_bytes()).hexdigest(), "counts": {"fail": r["fail"], "pass": r["pass"]},
           "picks": [{k: p[k] for k in ("seed", "task_id", "turn_idx", "class", "turn_limit_6_12", "recorded")}
                     for p in r["picks"]]}
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def main(argv=None) -> int:
    import argparse
    from demo_live import read_jsonl
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default=None, help="write the rule_v1 picks to this file (refuses to overwrite)")
    a = ap.parse_args(argv)
    rows = read_jsonl(REPO / "results" / "x2_evalset_tasks_real_v2.jsonl")
    if a.freeze:
        print(json.dumps(freeze(Path(a.freeze), rows), indent=1))
        return 0
    print(json.dumps({"rule_v1": rule_v1(rows), "rule_a": rule_a(rows), "rule_b": rule_b(rows)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
