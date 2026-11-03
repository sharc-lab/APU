"""Post-run check, docs/EXPERIMENTS.md "Amendment 2026-11-03: L2 tool-call example from the most recent completed turn".
Offline, no network, no model call, writes nothing.

Rows:
  post-run-check-live-8192-l2-recent-example: the live 8192 local phase (results/x2_post_run_check_local_v1.jsonl,
    seed 20260905, the only real L2 data for the held-out seed) recomputed from its recorded prc_l2_call rows. Per
    call, example_turn is recomputed on the client-side history rebuilt from the run's own r2a_turn rows
    (scripts/cloud_redo.rebuild) with the earliest rule (must reproduce every recorded example_turn, example_kept and
    l2_fail, and every turn's first-call message count, else nothing is reported) and with the latest rule;
    example_kept = example_turn not in the recorded dropped_turns; the rest of each row as recorded. The decisions use
    scripts/post_run_check_cloud.decide_turn (the controller's own code) on the original and on the recomputed rows.
  post-run-check-l2-recent-example: the replay's held-out figures (results/post_run_check_tuned_v1.jsonl, held-out
    seed, tuned L3, union decision), before and after. The replay's L2 is a PROXY that does not read the example turn,
    so before and after are the same decisions by construction.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import post_run_check_replay as R  # noqa: E402
import post_run_check_tuning as PRT  # noqa: E402

LIVE_LOCAL_FILE = PRT.LIVE_LOCAL_FILE
BEFORE, AFTER = "earliest", "latest"


# ── pure ────────────────────────────────────────────────────────────────────────────────────────

def example_kept(example_turn, call: dict):
    """As post_run_check_live.l2_row: None without an example turn or without a render; else whether the example
    turn's user message is in the render, which for these renders is `turn not in dropped_turns` (analyse_render's
    dropped_turns is the same whole user message substring test)."""
    if example_turn is None or not call.get("render_ok"):
        return None
    return example_turn not in (call.get("dropped_turns") or [])


def recomputed_call(call: dict, example_turn) -> dict:
    kept = example_kept(example_turn, call)
    fail = (not call.get("render_ok") or not call.get("system_kept") or not call.get("current_user_message_kept")
            or kept is False)
    return {**call, "example_turn": example_turn, "example_kept": kept, "l2_fail": bool(fail)}


def summary(tasks: list[dict], key: str) -> dict:
    wrong = [t for t in tasks if not t["local_ok"]]
    right = [t for t in tasks if t["local_ok"]]
    esc = lambda t: t[key] == "escalate"  # noqa: E731
    return {"n": len(tasks), "wrong": len(wrong), "right": len(right), "caught": sum(map(esc, wrong)),
            "false_esc": sum(map(esc, right)), "kept_local": sum(1 for t in tasks if not esc(t)),
            "recall": sum(map(esc, wrong)) / len(wrong) if wrong else None,
            "false_esc_rate": sum(map(esc, right)) / len(right) if right else None}


# ── live 8192, real L2 ──────────────────────────────────────────────────────────────────────────

def live_recompute(repo: Path = REPO, source: str = LIVE_LOCAL_FILE) -> dict:
    repo = Path(repo)
    mod = R._load("scripts/post_run_check_cloud.py", "post_run_check_cloud", repo)
    import post_run_check_live as PRCL
    rows = mod.CR.load_rows(repo / source)
    if not any(r.get("record") == "run_end" for r in rows):
        raise ValueError(f"{source} has no run_end (local phase unfinished)")
    turns, l2_calls, checks = mod.local_rows(rows)
    cache, mismatches, new_calls, ex = {}, [], [], {}
    with contextlib.redirect_stdout(io.StringIO()):
        for r in turns:
            es = mod.CR.build_session_for(r, cache)
            prior = [x for x in mod.CR.session_rows(rows, r["attempt_id"]) if x["turn_idx"] < r["turn_idx"]]
            msgs = mod.CR.rebuild(es, prior, r["turn_idx"])["messages"]
            old_t, _ = PRCL.example_user_content(msgs, BEFORE)
            new_t, _ = PRCL.example_user_content(msgs, AFTER)
            ex[(r["seed"], r["turn_idx"])] = (old_t, new_t)
            mine = [c for c in l2_calls if c["seed"] == r["seed"] and c["turn_idx"] == r["turn_idx"]]
            if mine and min(c["n_messages_sent"] for c in mine) != len(msgs):
                mismatches.append(f"turn {r['turn_idx']}: rebuilt {len(msgs)} messages, first call sent "
                                  f"{min(c['n_messages_sent'] for c in mine)}")
            for c in mine:
                re_old = recomputed_call(c, old_t)
                for k in ("example_turn", "example_kept", "l2_fail"):
                    if re_old[k] != c.get(k):
                        mismatches.append(f"turn {r['turn_idx']} {k}: recorded {c.get(k)!r}, recomputed {re_old[k]!r}")
                new_calls.append(recomputed_call(c, new_t))
        if mismatches:
            return {"status": "GUARD_FAILED", "mismatches": mismatches}
        tasks = []
        for r in turns:
            es = mod.CR.build_session_for(r, cache)
            b = mod.decide_turn(r, es, l2_calls, checks)
            a = mod.decide_turn(r, es, new_calls, checks)
            o, n = ex[(r["seed"], r["turn_idx"])]
            tasks.append({"seed": r["seed"], "turn_idx": r["turn_idx"], "task_id": r["task_id"],
                          "local_ok": r["end_state_ok"] is True, "before": b["decision"], "after": a["decision"],
                          "fired_before": b["fired"], "fired_after": a["fired"], "example_before": o,
                          "example_after": n})
    l2_after = sum(1 for c in new_calls if c["l2_fail"])
    return {"status": "VERIFIED", "tier": mod.TIER, "seed": turns[0]["seed"], "tasks": tasks,
            "before": summary(tasks, "before"), "after": summary(tasks, "after"),
            "n_calls": len(l2_calls), "l2_fail_calls_before": sum(1 for c in l2_calls if c.get("l2_fail")),
            "l2_fail_calls_after": l2_after, "n_prompt_checks": len(checks),
            "prompt_checks_matched": sum(1 for c in checks if c.get("match"))}


def live_cost(repo: Path, d: dict) -> dict:
    """ESTIMATE: the before and after escalation sets priced as row post-run-check-live-8192-estimate-rolling does
    (post_run_check_tuning.estimate_rolling, rolling cache breakpoint, and the recorded placement)."""
    import rolling_cache_reprice as rr
    with contextlib.redirect_stdout(io.StringIO()):
        rd = rr.redo_rolling(repo)
        src = R.cloud_sources(Path(repo))
    uppers = {k: v["upper"] for k, v in src["redo"].items()}
    secs = {k: v["seconds"] for k, v in src["redo"].items()}
    out = {"model": rd["model"]}
    for side in ("before", "after"):
        esc = [t for t in d["tasks"] if t[side] == "escalate"]
        out[side] = {col: PRT.estimate_rolling(esc, d["tier"], rd["redo"], rd["rb"], uppers, secs, col=col)
                     for col in ("usd_new", "usd_recorded")}
    return out


def false_esc_layers(tasks: list[dict], side: str) -> dict:
    """Per layer: on how many false escalations (right locally, escalated) it fired, and fired alone."""
    fe = [t for t in tasks if t["local_ok"] and t[side] == "escalate"]
    names = sorted({x for t in fe for x in t[f"fired_{side}"]})
    return {n: {"fired": sum(1 for t in fe if n in t[f"fired_{side}"]),
                "alone": sum(1 for t in fe if t[f"fired_{side}"] == [n])} for n in names}


def _lay(d: dict) -> str:
    return ", ".join(f"{k} on {v['fired']} (alone on {v['alone']})" for k, v in d.items()) or "none"


def _s(p: dict) -> str:
    return (f"recall {R._pc(p['caught'], p['wrong'])}, false escalations {R._pc(p['false_esc'], p['right'])}, kept "
            f"local {R._pc(p['kept_local'], p['n'])}")


def _met(p: dict) -> str:
    return (f"recall >= 80% {'MET' if p['recall'] is not None and p['recall'] >= R.TARGET_RECALL else 'MISSED'}, "
            f"false escalations <= 10% "
            f"{'MET' if p['false_esc_rate'] is not None and p['false_esc_rate'] <= R.TARGET_FALSE_ESC else 'MISSED'}")


def live_text(d: dict, c: dict) -> str:
    if d["status"] != "VERIFIED":
        raise ValueError("guard failed: " + "; ".join(d["mismatches"][:5]))
    b, a = d["before"], d["after"]
    changed = [t for t in d["tasks"] if t["before"] != t["after"]]
    ch = "; ".join(f"turn {t['turn_idx']} {t['task_id']} ({'right' if t['local_ok'] else 'wrong'} locally): "
                   f"{t['before']} -> {t['after']} (fired before {'+'.join(t['fired_before']) or 'none'}, after "
                   f"{'+'.join(t['fired_after']) or 'none'})" for t in changed)
    ex = sorted({t["example_after"] for t in d["tasks"] if t["example_after"] is not None})
    cb, ca = c["before"]["usd_new"], c["after"]["usd_new"]
    return (f"Live post-run check at evo-x2 num_ctx {d['tier'].split('_')[-1]}, held-out seed {d['seed']}, real L2 "
            f"(render check) recomputed offline from the {d['n_calls']} recorded prc_l2_call rows of {LIVE_LOCAL_FILE} "
            f"(Amendment 2026-11-03; rule fixed before this computation, nothing tuned on this run; the earliest rule "
            f"on the rebuilt history reproduces every recorded example_turn, example_kept and l2_fail and every "
            f"turn's message count); decision = L1 (detector v2) + L1 errors + L2 + tuned L3 "
            f"(scripts/post_run_check_cloud.decide_turn). BEFORE (example = earliest completed turn with log_event plus "
            f"a task tool, as recorded): {_s(b)}; {_met(b)}; L2 failed on {d['l2_fail_calls_before']} of "
            f"{d['n_calls']} calls. AFTER (example = most recent such turn): {_s(a)}; {_met(a)}; L2 failed on "
            f"{d['l2_fail_calls_after']} of {d['n_calls']} calls (example turns used after: "
            f"{', '.join(map(str, ex)) or 'none'}). Turns whose decision changed: {ch or 'none'}. Layers firing on the "
            f"false escalations: before {_lay(false_esc_layers(d['tasks'], 'before'))}; after "
            f"{_lay(false_esc_layers(d['tasks'], 'after'))}. Prompt-token checks "
            f"as recorded: {d['prompt_checks_matched']} of {d['n_prompt_checks']} matched. Final accuracy NOT "
            f"measured (the cloud phase has not run). Cloud cost ESTIMATE, not measured, on {c['model']} as row "
            f"post-run-check-live-8192-estimate-rolling (rolling cache breakpoint; measured redo USD where a redo row "
            f"exists, tier mean otherwise): before {cb['n_escalated']} escalations, billed point USD "
            f"{cb['point_billed']:.4f} (USD {cb['point_billed'] / b['n']:.5f}/task); after {ca['n_escalated']} "
            f"escalations, billed point USD {ca['point_billed']:.4f} (USD {ca['point_billed'] / a['n']:.5f}/task); at "
            f"the recorded placement: before USD {c['before']['usd_recorded']['point_billed']:.4f}, after USD "
            f"{c['after']['usd_recorded']['point_billed']:.4f}")


# ── replay, held-out seed, L2 PROXY ─────────────────────────────────────────────────────────────

def replay_heldout(repo: Path = REPO) -> dict:
    d = PRT.load(Path(repo))
    if d["end"] is None:
        raise ValueError(f"{PRT.OUT} has no run_end")
    return {t: d["cells"][(t, "holdout", "tuned")]["union"] for t in R.TIERS}


def replay_text(cells: dict) -> str:
    parts = []
    for tier, p in cells.items():
        m = R.targets(p)
        parts.append(f"{tier}: {PRT._p(p)}; targets " + PRT._mm(m))
    return (f"HELD-OUT seed {PRT.HOLDOUT} replay (30 task turns per window, {PRT.OUT}), L1 errors + L1 PROXY + L2 PROXY "
            f"+ tuned L3, Amendment 2026-11-03 (L2 tool-call example = most recent completed turn with log_event plus "
            f"a task tool instead of the earliest): BEFORE and AFTER are the same decisions and the same figures, "
            f"because the replay's L2 is a PROXY (server prompt_eval_count below the client's chars/4 estimate; no "
            f"render kept) that does not read the example turn; the change is evaluated on real L2 data in row "
            f"post-run-check-live-8192-l2-recent-example. Before = after: " + " / ".join(parts))


if __name__ == "__main__":
    R._paths(REPO)
    d = live_recompute()
    print(live_text(d, live_cost(REPO, d)))
    print(replay_text(replay_heldout()))
