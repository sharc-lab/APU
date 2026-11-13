"""Token-estimate audit (operator question, 2026-11-12): where a chars/4 prompt-token estimate feeds a fit, route or
escalation decision, and what replacing it with the server's recorded count would change. Replay of recorded files
only: no network, no model call, no subprocess, no cloud spend.

Part B1, the 90-job 8192 set (rows pitch1-standard-hybrid-8192 and pitch1-frontier): B3 v2 qwen3:14b,
ollama_ctx_8192_call2_notools, seeds 20260901-03, results/x2_evalset_tasks_real_v2.jsonl. Per local call the file has
prompt_tokens_est (raw chars/4, x2_r2_agent.prompt_tokens_est) and prompt_eval_count_info_only (Ollama's count).

Censoring (why the server count cannot simply replace the estimate): Ollama cuts a prompt that exceeds num_ctx before
it evaluates it, so at num_ctx 8192 prompt_eval_count is at most 8191 whatever was sent. A server count is therefore
the size of the prompt only when the prompt was not cut. Classification per call (DERIVED, stated in the output):
  whole  prompt_eval_count / raw estimate >= REF_MIN, the smallest ratio on the uncut default-window calls of the same
         three seeds (ollama_default_call2_notools, loaded 40960, never near the window): the server count is taken as
         the size of the prompt that was sent, so "server says fits" (count <= window).
  cut    ratio below REF_MIN: the server evaluated less than was sent, so the prompt's true size is NOT in the file.
         Strict reading: unknown. Inferred reading: over the window (Ollama only cuts a prompt larger than num_ctx).
Every decision is reported under both readings; the replay is run under both.

Part B2, the long-task files: per call prompt_tokens_est (raw chars/4) and prompt_eval_count; calibrated estimate =
raw x x2_r2_evalset.PROJ_CALIB (1.20). Same censoring at each task's loaded window, so a call counts as uncut only
before the server's count first reaches 98% of the loaded window (numbers_register.longtask_tokens_rows' rule).

Run: py -3.12 analysis/token_estimate_audit.py   (prints the tables)
"""
from __future__ import annotations

import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
B3 = "results/x2_evalset_tasks_real_v2.jsonl"
ARM_8192 = "ollama_ctx_8192_call2_notools"
ARM_DEFAULT = "ollama_default_call2_notools"
SEEDS = (20260901, 20260902, 20260903)
LONG_FILES = ["results/x2_longtask_tokens_v1_qwen3_32b.jsonl", "results/x2_longtask_tokens_v1_llama33_70b.jsonl",
              "results/x2_longtask_toolfix_v1_llama33_70b.jsonl", "results/x2_longtask_scale_v1_qwen3_32b.jsonl",
              "results/x2_longtask_scale_v1_llama33_70b.jsonl"]
HIT_FRAC = 0.98


def _paths():
    for sub in ("analysis", "src", "harness", "scripts", ""):
        p = str(REPO / sub) if sub else str(REPO)
        if p not in sys.path:
            sys.path.insert(0, p)


def _rows(rel):
    with open(REPO / rel, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _q(xs):
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return (xs[0], xs[0], xs[0])
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return (q[1], q[0], q[2])


def _fq(t, nd=3):
    return "n/a" if t is None else f"{t[0]:.{nd}f} [IQR {t[1]:.{nd}f} to {t[2]:.{nd}f}]"


# ── B1 ─────────────────────────────────────────────────────────────────────────────────────────────

def ref_min_ratio(rows) -> tuple[float, float, int]:
    """(min, max, n) of server/raw over the uncut default-window calls of SEEDS (the 'whole prompt' range)."""
    r = [c["prompt_eval_count_info_only"] / c["prompt_tokens_est"]
         for t in rows if t.get("record") == "r2a_turn" and t.get("arm_id") == ARM_DEFAULT and t["seed"] in SEEDS
         for c in t["calls"] if c.get("prompt_eval_count_info_only") and c.get("prompt_tokens_est")]
    return min(r), max(r), len(r)


def call_state(c, ref_min) -> str:
    p, e = c.get("prompt_eval_count_info_only"), c.get("prompt_tokens_est")
    if not p or not e:
        return "missing"
    return "whole" if p / e >= ref_min else "cut"


def server_over(c, window, ref_min, reading):
    """True / False / None (unknown). reading: 'strict' (cut -> None) or 'inferred' (cut -> True)."""
    s = call_state(c, ref_min)
    if s == "whole":
        return c["prompt_eval_count_info_only"] > window
    if s == "cut":
        return None if reading == "strict" else True
    return None


def any3(vals):
    """Task-level OR over calls with unknowns: True if any True; None if no True but any None; else False."""
    vals = list(vals)
    if any(v is True for v in vals):
        return True
    if any(v is None for v in vals):
        return None
    return False


def flip_table(pairs):
    """pairs: [(estimate_over: bool, server_over: bool|None)] -> counts."""
    out = {"n": len(pairs), "agree_fits": 0, "agree_over": 0, "est_fits_server_over": 0, "est_over_server_fits": 0,
           "unknown_est_fits": 0, "unknown_est_over": 0}
    for e, s in pairs:
        if s is None:
            out["unknown_est_over" if e else "unknown_est_fits"] += 1
        elif e and s:
            out["agree_over"] += 1
        elif not e and not s:
            out["agree_fits"] += 1
        elif e:
            out["est_over_server_fits"] += 1
        else:
            out["est_fits_server_over"] += 1
    return out


def ft_text(name, t):
    return (f"  {name}: n={t['n']}; agree fits {t['agree_fits']}, agree over {t['agree_over']}; FLIP est fits / "
            f"server over {t['est_fits_server_over']}; FLIP est over / server fits {t['est_over_server_fits']}; "
            f"unknown (server count cut) with est over {t['unknown_est_over']}, with est fits {t['unknown_est_fits']}")


def b1(out):
    _paths()
    import hybrid_full_pipeline as hfp
    import impact_model as im
    import pitch1_standard_hybrid as p1
    from src.dse import l3_invariants as L3
    from src.dse.fits_locally import SessionGrowthPredictor
    rows = _rows(B3)
    lo, hi, nref = ref_min_ratio(rows)
    out.append("=" * 100)
    out.append("PART B1: the 90-job 8192 set (B3 v2 qwen3:14b, ollama_ctx_8192, seeds 20260901-03)")
    out.append(f"MEASURED reference: server/raw-chars4 on uncut default-window calls of the same seeds: min {lo:.4f}, "
               f"max {hi:.4f}, n={nref} calls. REF_MIN = {lo:.4f} (DERIVED threshold for 'whole').")
    with contextlib.redirect_stdout(io.StringIO()):
        ci = p1.cloud_inputs(REPO)
    cloud = ci["cloud"]
    local, first, ceiling = hfp.config_sessions(REPO, "f16 8192")
    _rel, _arm, tier = hfp.CONFIGS["f16 8192"]
    tasks = sorted(k for k in local if k in cloud)
    esc = p1.escalations_with_seconds(REPO, tier, ci)
    out.append(f"jobs n={len(tasks)}, window (loaded) {ceiling}")

    # per-call census
    calls = [(k, i, c) for k in tasks for i, c in enumerate(local[k]["calls"])]
    st = {s: sum(call_state(c, lo) == s for _k, _i, c in calls) for s in ("whole", "cut", "missing")}
    maxpec = max(c["prompt_eval_count_info_only"] or 0 for _k, _i, c in calls)
    out.append(f"MEASURED: local calls n={len(calls)}; max server count {maxpec} (window {ceiling}: Ollama cut prompts, "
               f"so no count can exceed it); DERIVED state: whole {st['whole']}, cut {st['cut']}, missing {st['missing']}")
    rw = [c["prompt_eval_count_info_only"] / c["prompt_tokens_est"] for _k, _i, c in calls
          if call_state(c, lo) == "whole"]
    rc = [c["prompt_eval_count_info_only"] / round(c["prompt_tokens_est"] * (local[k].get("token_calib_ratio") or 1))
          for k, _i, c in calls if call_state(c, lo) == "whole"]
    out.append(f"  whole calls: server / raw chars4 { _fq(_q(rw)) }; server / (raw x session ratio) { _fq(_q(rc)) }")
    cr = sorted(c["prompt_eval_count_info_only"] / c["prompt_tokens_est"] for _k, _i, c in calls
                if call_state(c, lo) == "cut")
    near = sum(0.95 <= x < lo for x in cr)
    out.append(f"  cut calls: server / raw chars4 {_fq(_q(cr))}, max {cr[-1]:.4f}; within [0.95, REF_MIN) {near} "
               f"(the calls whose whole/cut label is closest to the threshold)")
    ex = sorted(((k, i, c) for k, i, c in calls if call_state(c, lo) == "whole"
                 and p1.call_estimate(c, local[k], "tokenizer") > ceiling), key=lambda x: -x[2]["prompt_tokens_est"])
    for k, i, c in ex[:3]:
        out.append(f"  example est over / server whole and under: {k} turn {local[k]['turn_idx']} call {i}: raw "
                   f"{c['prompt_tokens_est']}, x session ratio {p1.call_estimate(c, local[k], 'tokenizer')}, server "
                   f"{c['prompt_eval_count_info_only']}")

    # decision D1: standard hybrid prompt-length check, chars4 and 'tokenizer' (raw x session ratio), every/first
    out.append("")
    out.append("D1 standard hybrid prompt-length check (analysis/pitch1_standard_hybrid.over_window), per task and per call")
    d1 = {}
    for est in ("chars4", "tokenizer"):
        for every in (True, False):
            for reading in ("strict", "inferred"):
                tp, cp = [], []
                for k in tasks:
                    cs = local[k]["calls"] if every else local[k]["calls"][:1]
                    e_calls = [p1.call_estimate(c, local[k], est) > ceiling for c in cs]
                    s_calls = [server_over(c, ceiling, lo, reading) for c in cs]
                    cp += list(zip(e_calls, s_calls))
                    tp.append((any(e_calls), any3(s_calls)))
                d1[(est, every, reading)] = tp
                if reading == "strict":
                    out.append(ft_text(f"{est} {'every-call' if every else 'first-call'} per TASK (strict)",
                                       flip_table(tp)))
                    out.append(ft_text(f"{est} {'every-call' if every else 'first-call'} per CALL (strict)",
                                       flip_table(cp)))

    # D2: L1 PROXY over_loaded_window (final call, raw x session ratio vs loaded)
    out.append("")
    out.append("D2 L1 PROXY over_loaded_window (src/dse/post_run_check.l1_proxy_record; harness final call raw x session "
               "ratio > loaded), per task")
    d2 = {}
    for reading in ("strict", "inferred"):
        d2[reading] = {k: server_over(local[k]["calls"][-1], ceiling, lo, reading) for k in tasks}
        if reading == "strict":
            out.append(ft_text("final call (strict)", flip_table([(bool(local[k]["over_loaded_window"]), d2[reading][k])
                                                                   for k in tasks])))

    # D2b: L2 PROXY (server < raw chars4): fires on how many tasks
    l2 = [k for k in tasks if any(c.get("prompt_eval_count_info_only") and c.get("prompt_tokens_est")
                                  and c["prompt_eval_count_info_only"] < c["prompt_tokens_est"]
                                  for c in local[k]["calls"])]
    out.append(f"D2b L2 PROXY (server count < raw chars4 estimate) fires on {len(l2)}/{len(tasks)} tasks; it compares "
               f"the server count with chars/4, so it is correct only while chars/4 under-counts a whole prompt "
               f"(ratio >= 1 on every whole call here: min whole ratio {min(rw):.4f})")

    # D3: pipeline route at start (SessionGrowthPredictor k=3 on turn_prompt of the first 3 turns)
    out.append("")
    out.append("D3 pipeline route at start (hybrid_full_pipeline.run_pipeline -> fits_locally.SessionGrowthPredictor, "
               "first 3 turns' final_call_prompt_tokens_calibrated), per task")
    rows8 = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("arm_id") == ARM_8192 and r["seed"] in SEEDS:
            rows8.setdefault(r["seed"], {})[r["turn_idx"]] = r
    first_srv, first_unknown = {}, []
    for s, ft in first.items():
        vals = []
        for ti, _p in ft:
            c = rows8[s][ti]["calls"][-1]
            if call_state(c, lo) == "whole":
                vals.append((ti, c["prompt_eval_count_info_only"]))
            else:
                first_unknown.append((s, ti))
                vals.append((ti, _p))
        first_srv[s] = vals
    for s in sorted(first):
        out.append(f"  seed {s}: first turns estimate {first[s]} vs server {first_srv[s]}")
    pred = SessionGrowthPredictor(k=hfp.K)
    d3 = []
    for k in tasks:
        row = local[k]
        idx = {i for i, _p in first[k[0]]}
        if row["turn_idx"] in idx or row["turn_idx"] <= max(idx):
            continue
        e = not pred.predict({"first_turns": first[k[0]], "target_turn": row["turn_idx"]}, {"window": ceiling})["fits"]
        sv = not pred.predict({"first_turns": first_srv[k[0]], "target_turn": row["turn_idx"]},
                              {"window": ceiling})["fits"]
        d3.append((e, None if k[0] in {u[0] for u in first_unknown} else sv))
    out.append(f"  first-turn server counts unknown (cut): {first_unknown or 'none'}")
    out.append(ft_text("route at start per TASK (tasks after turn 3)", flip_table(d3)))

    # D4: product route_at_start / impact_model projection (task-definition est x 1.20 + 330/turn) vs window
    out.append("")
    out.append("D4 route_at_start projection (harness/demo_live.projected_task_prompt == impact_model."
               "projected_task_prompts; est x PROJ_CALIB + PROJ_NONFILLER_PER_TURN per turn) vs 8192, margin 0, per "
               "task. NOT the same prompt: compared with the prompts actually sent at that task turn (any call over)")
    proj = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for s in SEEDS:
            for tid, (_i, p) in im.projected_task_prompts(REPO, s).items():
                proj[(s, tid)] = p
    for reading in ("strict",):
        d4 = [(proj[k] > ceiling, any3(server_over(c, ceiling, lo, reading) for c in local[k]["calls"]))
              for k in tasks if k in proj]
        out.append(ft_text(f"projection vs task-turn calls ({reading})", flip_table(d4)))

    # ── replays
    out.append("")
    out.append("REPLAY (existing functions, recorded outcomes and prices; DERIVED)")
    allc = p1.reference_arms(tasks, local, cloud)["all_cloud"]

    def std_with(decision: dict):
        fake = {}
        for k in tasks:
            r = dict(local[k])
            r["calls"] = [{"prompt_tokens_est": ceiling + 1 if decision[k] else 0}]
            fake[k] = r
        return p1.standard_hybrid(tasks, fake, ceiling, cloud, esc, "chars4", False)

    base_std = p1.standard_hybrid(tasks, local, ceiling, cloud, esc, "chars4", True)
    base_tok = p1.standard_hybrid(tasks, local, ceiling, cloud, esc, "tokenizer", True)
    budget = hfp.load_task_budget(REPO)
    chk = p1._load_check(REPO, L3.CONFIG_TUNED)
    with contextlib.redirect_stdout(io.StringIO()):
        base_pipe = hfp.run_pipeline(tasks, local, first, ceiling, cloud, esc, chk, budget)
    base_pipe["sent"] = base_pipe["routed"] + base_pipe["escalated"]
    base_pipe["usd_job"] = base_pipe["usd"] / base_pipe["n"]

    def line(name, r):
        return (f"  {name}: correct {r['correct']}/{r['n']}, cloud USD {r['usd'] / r['n']:.5f}/job, sent "
                f"{r.get('sent', r.get('routed', 0) + r.get('escalated', 0))}")

    out.append(line("standard hybrid chars4 every-call AS RECORDED (register pitch1-standard-hybrid-8192)", base_std))
    out.append(line("standard hybrid 'tokenizer' (raw x session ratio) every-call AS RECORDED", base_tok))
    out.append(line("pipeline L3 tuned AS RECORDED", base_pipe))
    fr0 = p1.frontier(base_pipe, base_std, allc)
    out.append(f"  frontier AS RECORDED: mix p {fr0['p']:.4f}, mix USD {fr0['mix_usd_job']:.5f}/job, TF {fr0['TF']}")
    for reading in ("inferred", "strict-keep-estimate"):
        if reading == "inferred":
            dec = {k: v[1] for k, v in zip(tasks, d1[("chars4", True, "inferred")])}
        else:
            dec = {k: (v[1] if v[1] is not None else v[0]) for k, v in zip(tasks, d1[("chars4", True, "strict")])}
        r = std_with(dec)
        out.append(line(f"standard hybrid every-call with the SERVER decision ({reading})", r))
        rows_srv = {}
        for k in tasks:
            rr = dict(local[k])
            sv = d2["inferred" if reading == "inferred" else "strict"][k]
            rr["over_loaded_window"] = sv if sv is not None else bool(local[k]["over_loaded_window"])
            rows_srv[k] = rr
        with contextlib.redirect_stdout(io.StringIO()):
            pr = hfp.run_pipeline(tasks, rows_srv, first_srv, ceiling, cloud, esc, chk, budget)
        pr["sent"] = pr["routed"] + pr["escalated"]
        pr["usd_job"] = pr["usd"] / pr["n"]
        out.append(line(f"pipeline with SERVER L1-proxy and route-at-start inputs ({reading})", pr)
                   + f" (routed {pr['routed']}, escalated {pr['escalated']})")
        fr = p1.frontier(pr, r, allc)
        out.append(f"  frontier ({reading}): mix p {fr['p']}, mix USD "
                   f"{fr['mix_usd_job'] if fr['mix_usd_job'] is None else round(fr['mix_usd_job'], 5)}/job, TF {fr['TF']}")
        changed = [k for k in tasks if r["per_task"][k] != base_std["per_task"][k]]
        out.append(f"  standard hybrid tasks whose route changed ({reading}): {len(changed)}/{len(tasks)}")
        for k in changed:
            kc = local[k]["calls"]
            out.append(f"    {k} turn {local[k]['turn_idx']}: {base_std['per_task'][k]} -> {r['per_task'][k]}; calls "
                       f"(raw est, server, state): "
                       + ", ".join(f"({c['prompt_tokens_est']}, {c['prompt_eval_count_info_only']}, "
                                   f"{call_state(c, lo)})" for c in kc)
                       + f"; local ok {local[k]['end_state_ok']}, cloud ok {cloud[k]['ok']}")
        pchanged = [k for k in tasks if pr["per_task"][k] != base_pipe["per_task"][k]]
        out.append(f"  pipeline tasks whose route changed ({reading}): {len(pchanged)}/{len(tasks)} "
                   + "; ".join(f"{k}: {base_pipe['per_task'][k]} -> {pr['per_task'][k]}" for k in pchanged))
        for k in pchanged:
            out.append(f"    {k} turn {local[k]['turn_idx']}: recorded over_loaded_window "
                       f"{local[k]['over_loaded_window']} (final call est {local[k]['final_call_prompt_tokens_calibrated']}, "
                       f"server {local[k]['calls'][-1]['prompt_eval_count_info_only']}, "
                       f"{call_state(local[k]['calls'][-1], lo)}); local ok {local[k]['end_state_ok']}; cloud ok "
                       f"{cloud[k]['ok']}; escalation ok {esc[k]['ok']} ({esc[k]['src']})")


# ── B2 ─────────────────────────────────────────────────────────────────────────────────────────────

def b2(out):
    _paths()
    import x2_r2_evalset as EV
    calib = EV.PROJ_CALIB
    out.append("")
    out.append("=" * 100)
    out.append(f"PART B2: long-task files; calibrated estimate = raw chars/4 x PROJ_CALIB ({calib}); server = "
               f"prompt_eval_count (Ollama). 'uncut' = calls before the server count first reaches {HIT_FRAC:.0%} of "
               f"the loaded window")
    for rel in LONG_FILES:
        rows = _rows(rel)
        ends = [r for r in rows if r.get("record") == "run_end"]
        att = ends[-1].get("attempt_id") if ends else None
        tasks = [r for r in rows if r.get("record") == "longtask_task" and (att is None or r.get("attempt_id") == att)]
        groups = {}
        for t in tasks:
            groups.setdefault((t["model_id"], t["arm"], t["loaded_context"]), []).append(t)
        for (model, arm, loaded), ts in sorted(groups.items()):
            all_r, uncut_r = [], []
            flips = {"n_tasks": len(ts), "est_over": 0, "srv_over_window": 0, "srv_98": 0, "est_over_srv98_not": 0,
                     "est_not_srv98": 0, "est98_vs_srv98_flip": 0}
            call_pairs = {"n": 0, "est_over_srv98_not": 0, "est_not_srv98": 0}
            l2_false = [0, 0]
            for t in ts:
                est = [(p or 0) * calib for p in t["prompt_tokens_est"]]
                pec = t["prompt_eval_count"]
                hit = next((i for i, c in enumerate(pec) if c and c >= HIT_FRAC * loaded), None)
                for i, (e, c, raw) in enumerate(zip(est, pec, t["prompt_tokens_est"])):
                    if c and e:
                        all_r.append(c / e)
                        if hit is None or i < hit:
                            uncut_r.append(c / e)
                            l2_false[1] += 1
                            l2_false[0] += c < raw
                    if c is not None and e:
                        call_pairs["n"] += 1
                        eo, so = e > loaded, c >= HIT_FRAC * loaded
                        call_pairs["est_over_srv98_not"] += eo and not so
                        call_pairs["est_not_srv98"] += so and not eo
                eo = any(e > loaded for e in est)
                e98 = any(e >= HIT_FRAC * loaded for e in est)
                so = any(c and c > loaded for c in pec)
                s98 = hit is not None
                flips["est_over"] += eo
                flips["srv_over_window"] += so
                flips["srv_98"] += s98
                flips["est_over_srv98_not"] += eo and not s98
                flips["est_not_srv98"] += s98 and not eo
                flips["est98_vs_srv98_flip"] += e98 != s98
            out.append("")
            out.append(f"{rel} | {model} | arm {arm} | loaded {loaded}")
            out.append(f"  server / calibrated estimate, all calls: {_fq(_q(all_r))}, n={len(all_r)} calls")
            out.append(f"  server / calibrated estimate, uncut calls: {_fq(_q(uncut_r))}, n={len(uncut_r)} calls")
            out.append(f"  per TASK (n={flips['n_tasks']}): est x{calib} > window {flips['est_over']}; server > window "
                       f"{flips['srv_over_window']}; server >= {HIT_FRAC:.0%} window {flips['srv_98']}; FLIP est over / "
                       f"server not at {HIT_FRAC:.0%} {flips['est_over_srv98_not']}; FLIP server at {HIT_FRAC:.0%} / est "
                       f"not over {flips['est_not_srv98']}; est >= {HIT_FRAC:.0%} vs server >= {HIT_FRAC:.0%} disagree "
                       f"{flips['est98_vs_srv98_flip']}")
            out.append(f"  per CALL (n={call_pairs['n']}): FLIP est over / server below {HIT_FRAC:.0%} "
                       f"{call_pairs['est_over_srv98_not']}; FLIP server at {HIT_FRAC:.0%} / est not over "
                       f"{call_pairs['est_not_srv98']}")
            out.append(f"  L2-proxy condition (server < raw chars4) on uncut calls: {l2_false[0]}/{l2_false[1]}")
            for t in ts:
                est = [round((p or 0) * calib) for p in t["prompt_tokens_est"]]
                pec = t["prompt_eval_count"]
                if not est:
                    continue
                hit = next((i for i, c in enumerate(pec) if c and c >= HIT_FRAC * loaded), None)
                fe = next((i for i, e in enumerate(est) if e > loaded), None)
                if hit is not None or fe is not None or rel.endswith("tokens_v1_qwen3_32b.jsonl"):
                    j = hit if hit is not None else max(range(len(pec)), key=lambda i: pec[i] or 0)
                    out.append(f"    task {t['task_id']} seed {t['seed']}: calls {len(est)}; first call est > window "
                               f"{fe}; first call server >= {HIT_FRAC:.0%} {hit}; at call {j}: raw "
                               f"{t['prompt_tokens_est'][j]}, est x{calib} {est[j]}, server {pec[j]}; max est "
                               f"{max(est)}, max server {max(c or 0 for c in pec)}")


def main():
    out = ["token_estimate_audit (analysis/token_estimate_audit.py): replay of recorded files, no call, no spend"]
    b1(out)
    b2(out)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
