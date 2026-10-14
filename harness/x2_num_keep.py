"""num_keep guardrail flag test (x2_r2_agent.py --mode real --num-keep system; queue id x2_num_keep_guardrail_test,
results/x2_num_keep_guardrail_test.jsonl). Pre-registered in docs/FINDINGS.md ("PRE-REGISTRATION: num_keep guardrail
flag test", 2026-10-13).

Question: at num_ctx 4096, does sending Ollama's options.num_keep equal to the system prompt's token length stop the
rule loss that llama.cpp's context shift (n_keep 5) and Ollama's token-level cut (keep 4) cause (register row
R2-mechanism-lowlevel)?

What this module adds to the real run, and nothing else:
  1. measure_system_tokens: S, the system prompt's token length with the model's own tokenizer, by render and count
     (harness/prompt_token_check.py): Ollama renders [system] and [system, first user message] with
     _debug_render_only, with and without the tool definitions; the rendered system block is the common prefix of the
     two renders (it must contain the whole system prompt, else ValueError), tokenized with llama-tokenize on the
     model's GGUF blob (BOS included). S = max over the with-tools and without-tools variants.
  2. NumKeepRuntime: wraps x2_r2_mechanism.MechanismRuntime (OLLAMA_DEBUG log slice per call, no fresh-reload prompt
     checks). begin_session measures S and sets base.options_extra = {"num_keep": S}, so every /api/chat body of the
     session (the render-only request and the real call, both calls of every turn) carries options.num_keep = S
     (x2_r2_agent.native_chat_body). After each call it attaches resp["num_keep"]: S, the value sent, and what the
     server log shows for that call (context-shift n_keep, task-line n_keep, token-cut keep, a --keep start argument).
  3. verdict / report: the pre-registered mechanical verdict ("one-line fix" / "not a one-line fix" / "invalid
     (num_keep not applied)" / "inconclusive (...)") from the result rows. analysis/numbers_register.py's
     compute_num_keep_guardrail_test calls it.

Expected from Ollama 0.34.4's source (x2_r2_mechanism docstring): llama-server is started with --keep <num_keep> and
the token-level cut keeps num_keep tokens (+1 for BOS); whether the per-request value reaches both is what the log
observation answers, so nothing here assumes it.
"""
from __future__ import annotations

import os
import re

RULE_THRESHOLD = 0.90
EXPECTED_SEEDS = (20260901, 20260902, 20260903)
EXPECTED_TURNS = 40
EXPECTED_ARM = "ollama_ctx_4096_call2_notools"
EXPECTED_MODEL = "llama3.1:8b"
_KEEP_ARG = re.compile(r"--keep[= ](\d+)")


# ── measuring S (render and count) ──────────────────────────────────────────────────────────────────

def common_prefix(a: str, b: str) -> str:
    return os.path.commonprefix([a or "", b or ""])


def system_block(render_sys: str | None, render_sys_user: str | None, system_prompt: str) -> str:
    """The rendered system block: the common prefix of the render of [system] and of [system, first user message].
    Raises ValueError when a render is missing or the prefix does not contain the whole system prompt."""
    if render_sys is None or render_sys_user is None:
        raise ValueError("render-only request returned no prompt")
    pre = common_prefix(render_sys, render_sys_user)
    if system_prompt not in pre:
        raise ValueError("the common prefix of the two renders does not contain the whole system prompt")
    return pre


def measure_system_tokens(render_fn, count_fn, model, system_prompt, first_user, num_ctx, think, tools) -> dict:
    """S for one session. render_fn(model, messages, num_ctx, tools, think) -> (text|None, err); count_fn(text) ->
    token count (int). Variants: with the tool definitions (call 1) and without (call 2 in the v1 protocol).
    Returns {"system_prompt_tokens": S, "variants": {...}, "method": ...}; raises on any failure."""
    sysm = {"role": "system", "content": system_prompt}
    userm = {"role": "user", "content": first_user}
    variants = {}
    for name, t in (("with_tools", tools), ("without_tools", None)):
        r1, e1 = render_fn(model, [sysm], num_ctx, t, think)
        r2, e2 = render_fn(model, [sysm, userm], num_ctx, t, think)
        if r1 is None or r2 is None:
            raise ValueError(f"render-only failed ({name}): {e1 or e2}")
        block = system_block(r1, r2, system_prompt)
        variants[name] = {"tokens": int(count_fn(block)), "block_chars": len(block),
                          "render_sys_chars": len(r1), "render_sys_user_chars": len(r2)}
    s = max(v["tokens"] for v in variants.values())
    if s <= 0:
        raise ValueError(f"system prompt token count {s} is not positive")
    return {"system_prompt_tokens": s, "variants": variants,
            "method": "render_tokenize: common prefix of render([system]) and render([system, user1]), llama-tokenize "
                      "with BOS, max over with/without tools"}


def tokenizer_count_fn(model, models_dir=None, exe=None, run=None):
    """count_fn for measure_system_tokens on a real host: prompt_token_check.count_tokens on the model's GGUF blob."""
    import prompt_token_check as ptc
    blob = ptc.resolve_model_blob(models_dir or ptc.default_models_dir(), model)
    return lambda text: ptc.count_tokens(text, blob, exe or ptc.TOKENIZE_EXE, run)["n_tokens"]


# ── observing what the server did ───────────────────────────────────────────────────────────────────

def observed_from_call_row(row: dict) -> dict:
    """From one r2m_call row: the n_keep of each context shift, of each task line, the keep of a token-level cut, and
    any --keep argument in the call's log excerpt (llama-server start line)."""
    keep_args = sorted({int(m) for line in row.get("log_excerpt") or [] for m in _KEEP_ARG.findall(line)})
    tc = row.get("token_cut") or None
    return {"shift_n_keep": [s["n_keep"] for s in row.get("context_shifts") or []],
            "task_n_keep": [p["n_keep"] for p in row.get("new_prompt") or []],
            "token_cut_keep": tc["keep"] if tc else None,
            "keep_args": keep_args,
            "log_slice_complete": row.get("log_slice_complete"),
            "message_level_truncation": row.get("message_level_truncation")}


class NumKeepRuntime:
    """Wraps a MechanismRuntime (mech) whose inner runtime is `base` (x2_r2_agent.OllamaRuntime, options_extra
    attribute). begin_session(model, arm, seed) measures S with measure(model, arm, seed) -> measure_system_tokens
    dict and sets base.options_extra; chat() annotates the response with resp["num_keep"]."""

    def __init__(self, mech, base, measure, emit):
        self.inner, self.base, self.measure, self.emit = mech, base, measure, emit
        self.s = None
        self.session = None

    def loaded_context(self, model):
        return self.inner.loaded_context(model)

    def unload(self, model):
        return self.inner.unload(model)

    def recover(self):
        return self.inner.recover()

    def begin_session(self, model, arm, seed):
        self.base.options_extra = None        # the measuring renders go without num_keep (S is not known yet)
        m = self.measure(model, arm, seed)
        self.s = int(m["system_prompt_tokens"])
        self.base.options_extra = {"num_keep": self.s}
        self.session = {"model_id": model, "arm_id": arm, "seed": seed}
        self.emit({"record": "num_keep_session", **self.session, "num_keep": self.s, **m})
        return self.s

    def end_session(self):
        self.base.options_extra = None
        self.s = None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        if self.s is None:
            raise RuntimeError("NumKeepRuntime.chat before begin_session")
        sent = (self.base.options_extra or {}).get("num_keep")
        self.inner.call_meta = {"num_keep_sent": sent, "system_prompt_tokens": self.s}
        try:
            resp = (self.inner.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                    else self.inner.chat(model, messages, num_ctx, tools, think))
        finally:
            self.inner.call_meta = None
        row = self.inner.session_calls[-1] if self.inner.session_calls else {}
        resp = dict(resp)
        resp["num_keep"] = {"sent": sent, "system_prompt_tokens": self.s, "observed": observed_from_call_row(row)}
        return resp


# ── verdict (pure, on result rows) ──────────────────────────────────────────────────────────────────

def _cell(rows, model=EXPECTED_MODEL, arm=EXPECTED_ARM):
    import x2_r2_agent as ag
    sessions = [s for s in ag.completed_sessions(rows) if s["model_id"] == model and s["arm_id"] == arm
                and s.get("mode") == "real"]
    att = {s["attempt_id"] for s in sessions}
    turns = sorted((r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in att),
                   key=lambda r: (r["seed"], r["turn_idx"]))
    return sessions, turns


def rules_in_use_of(rows, model=EXPECTED_MODEL):
    start = next((r for r in rows if r.get("record") == "run_start"), {}) or {}
    riu = start.get("rules_in_use")
    if isinstance(riu, dict):
        return list(riu.get(model) or [])
    return list(riu or [])


def verdict(rows, model=EXPECTED_MODEL, arm=EXPECTED_ARM, seeds=EXPECTED_SEEDS, n_turns=EXPECTED_TURNS,
            threshold=RULE_THRESHOLD) -> dict:
    """The pre-registered verdict (docs/FINDINGS.md 2026-10-13), computed from the job's rows only."""
    sessions, turns = _cell(rows, model, arm)
    s_by_seed = {}
    for r in rows:
        if r.get("record") == "num_keep_session" and r.get("model_id") == model and r.get("arm_id") == arm:
            s_by_seed[r["seed"]] = r.get("num_keep")
    incomplete = []
    if not any(r.get("record") == "run_end" for r in rows):
        incomplete.append("no run_end")
    got = sorted(s["seed"] for s in sessions)
    if got != sorted(seeds):
        incomplete.append(f"sessions completed for seeds {got}, expected {sorted(seeds)}")
    for s in sessions:
        n = sum(1 for t in turns if t["attempt_id"] == s["attempt_id"])
        if n != n_turns:
            incomplete.append(f"seed {s['seed']}: {n} turns, expected {n_turns}")
    calls = [(t, c) for t in turns for c in t.get("calls") or []]
    not_sent, below, slices_bad = [], [], 0
    shifts = cuts = 0
    shift_keeps, task_keeps, cut_keeps, keep_args = set(), set(), set(), set()
    for t, c in calls:
        nk = c.get("num_keep") or {}
        s = s_by_seed.get(t["seed"])
        if not isinstance(s, int) or s <= 0 or nk.get("sent") != s or nk.get("system_prompt_tokens") != s:
            not_sent.append((t["seed"], t["turn_idx"]))
            continue
        o = nk.get("observed") or {}
        if o.get("log_slice_complete") is False:
            slices_bad += 1
        for v in o.get("shift_n_keep") or []:
            shifts += 1
            shift_keeps.add(v)
            if v < s:
                below.append(f"seed {t['seed']} turn {t['turn_idx']}: shift n_keep {v} < S {s}")
        task_keeps.update(o.get("task_n_keep") or [])
        keep_args.update(o.get("keep_args") or [])
        if o.get("token_cut_keep") is not None:
            cuts += 1
            cut_keeps.add(o["token_cut_keep"])
            if o["token_cut_keep"] < s:
                below.append(f"seed {t['seed']} turn {t['turn_idx']}: token-cut keep {o['token_cut_keep']} < S {s}")
    if not_sent:
        incomplete.append(f"{len(not_sent)} calls without num_keep = S (first {not_sent[0]})")
    if slices_bad:
        incomplete.append(f"{slices_bad} calls with an incomplete log slice")
    riu = rules_in_use_of(rows, model)
    rates = {}
    for rule in riu:
        vals = [t.get(rule) for t in turns if t.get(rule) is not None]
        first = {}
        for t in turns:
            if t.get(rule) is False and t["seed"] not in first:
                first[t["seed"]] = t["turn_idx"]
        rates[rule] = {"rate": (sum(1 for v in vals if v) / len(vals)) if vals else None, "n": len(vals),
                       "first_failure_per_seed": {sd: first.get(sd) for sd in sorted(seeds)}}
    if not riu:
        incomplete.append("no rules in use in run_start")
    if below:
        v = "invalid (num_keep not applied)"
    elif incomplete:
        v = "inconclusive (incomplete: " + "; ".join(incomplete) + ")"
    elif shifts == 0:
        v = "inconclusive (flag effect not observed)"
    else:
        low = [r for r, d in rates.items() if d["rate"] is None or d["rate"] < threshold]
        v = "one-line fix" if not low else "not a one-line fix"
    canary_c = sum(1 for t in turns if t.get("canary_sys_ok") is False)
    canary_h = sum(1 for t in turns if t.get("canary_hist_ok") is False)
    checks = sum(1 for t in turns if t.get("canary_check"))
    return {"verdict": v, "valid": not below and not incomplete and shifts > 0,
            "problems": below + incomplete, "system_prompt_tokens_per_seed": dict(sorted(s_by_seed.items())),
            "rules_in_use": riu, "rule_rates": rates, "n_turns": len(turns), "n_calls": len(calls),
            "context_shift_events": shifts, "shift_n_keep_values": sorted(shift_keeps),
            "task_n_keep_values": sorted(task_keeps), "token_cuts": cuts, "token_cut_keep_values": sorted(cut_keeps),
            "keep_arg_values": sorted(keep_args),
            "message_drop_calls": sum(1 for _, c in calls
                                      if ((c.get("num_keep") or {}).get("observed") or {}).get("message_level_truncation")),
            "canary_sys_misses": canary_c, "canary_hist_misses": canary_h, "canary_checks": checks,
            "error_turns": sum(1 for t in turns if t.get("any_http_error"))}


def tray_app_running(run=None) -> bool:
    """True if Ollama's tray app ("ollama app.exe") is running. Read-only process listing (hidden child process)."""
    if run is None:
        from proc_util import run_hidden as run
    try:
        p = run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-Process 'ollama app' -ErrorAction SilentlyContinue | Measure-Object).Count"],
                capture_output=True, text=True, timeout=30)
        return (p.stdout or "").strip() not in ("", "0")
    except Exception:
        return False


def run_num_keep(runtime, server_log, plan, models, out_path, emit, log, order, rules_in_use, skip_done=(),
                 measure=None):
    """The --num-keep system run's body (x2_r2_agent.main --mode real --num-keep system). The caller owns the Ollama
    server (started with OLLAMA_DEBUG=1, logging to server_log) and writes run_end. Returns the report dict."""
    import json
    from pathlib import Path
    import x2_r2_agent as ag
    import x2_r2_mechanism as mech
    import t2s_r2_session_growth as r2
    head = mech.read_log(server_log, 0)
    if not mech.debug_enabled(head):
        raise RuntimeError(f"OLLAMA_DEBUG not active: no OLLAMA_DEBUG:DEBUG / level=DEBUG in {server_log}")
    out_path = Path(out_path)
    mrt = mech.MechanismRuntime(runtime, server_log, emit, out_path.parent / (out_path.stem + "_logs"), checker=None)
    counters = {}

    def default_measure(model, arm, seed):
        turns = next(t for a, ss, t in plan if a == arm)
        ags = ag.build_agent_session(seed, turns)
        first_user = ag.user_message(ags.spec.turns[0], ags)
        if model not in counters:
            counters[model] = tokenizer_count_fn(model)

        def render(m, msgs, num_ctx, tools, think):
            return runtime.render_only(m, msgs, num_ctx, tools, think)
        return measure_system_tokens(render, counters[model], model, ags.system_prompt, first_user,
                                     ag.ARMS[arm]["num_ctx"], ag.MODELS.get(model), r2.ollama_tools_payload())
    nkr = NumKeepRuntime(mrt, runtime, measure or default_measure, emit)

    def _start(m, a, s):
        mrt.begin_session(m, a, s, mode="real", call2_mode=ag.ARMS[a]["call2_mode"])
        nkr.begin_session(m, a, s)

    def _end():
        mrt.end_session()
        nkr.end_session()
    ag.run_plan(nkr, plan, models, "real", out_path, log=log, emit=emit, skip_done=skip_done, order=order,
                on_session_start=_start, on_session_end=_end)
    rows = ag.read_rows(out_path)
    rep = {"rules_in_use": rules_in_use, "num_keep": "system", "mechanism": mech.report(rows)}
    Path(str(out_path) + ".report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return rep
