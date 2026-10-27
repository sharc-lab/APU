"""KV-precision evidence and f16 vs quantized-KV output agreement on evo-x2 qwen3:14b, from committed B5 files only.

Rows: kv-precision-evidence, kv-output-agreement (analysis/numbers_register.py).

Sources: results/x2_policy_compare.jsonl (B5: policies a default, g kv_q8_0, h kv_q4_0) and
results/x2_policy_compare_fa_control.jsonl (policy i, f16 KV + flash attention). No new run.

What the files hold and what they do not:
  - per turn: turn_wall_s (session minutes = sum / 60, as policy_compare.session_metrics), loaded_context (the
    window Ollama reported loaded), and per call prompt_eval_count / prompt_eval_duration_s and completion_tokens /
    eval_duration_s (Ollama's own timings; prefill and decode tokens/s are their ratios);
  - per env, at the window probe: the /api/ps "size" in bytes of the loaded model (weights + KV + compute buffers)
    and harness/policy_compare.kv_check's parse of the Ollama server log (every KV type llama.cpp printed, and
    whether OLLAMA_KV_CACHE_TYPE was in the server-config line). The server logs themselves were not kept for B5 on
    evo-x2, so the llama_kv_cache size line is not in any committed file.
  - KV buffer size is therefore derived, not read: with the same window, model and runtime, only the KV cache changes
    size between envs, at 16 / 8.5 / 4.5 bits per element for f16 / q8_0 / q4_0 (ggml block layouts: q8_0 34 bytes
    per 32 values, q4_0 18 bytes per 32 values). Two independent size differences (f16 - q8_0 = X * 7.5/16 and
    q8_0 - q4_0 = X * 4/16) each give the f16 KV size X; agreement of the two is the check that q4_0 is really 4.5
    bits and q8_0 really 8.5 (a q4_0 that fell back to f16 or q8_0 would break it).

Output agreement: decoding is greedy (x2_r2_agent options temperature 0, seed 42). In a multi-turn session a call's
prompt is the whole transcript so far, so once one call's output differs every later prompt differs. A call is
comparable only when its prompt is identical on both sides; the prompt is not hashed in the files, so this module
rebuilds a prompt key: sha256 over the seed and every earlier call's output (content plus native tool calls' name and
arguments, and the detection method), plus the turn and call index. Everything else in the prompt (system prompt,
tools, user messages, simulated tool results) is a deterministic function of the seed and those outputs in
x2_r2_agent. Tool-call ids are left out (Ollama assigns them at random; the qwen3 chat template does not render them).
First-divergence token index: the recorded output text retokenized with the Qwen3 tokenizer (Qwen/Qwen3-8B, the same
vocabulary as qwen3:14b; the repo's offline_token_calibration uses it the same way) when it is in the local Hugging
Face cache, else the character index (labelled). The server's emitted token ids were not recorded.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
from pathlib import Path

B5_FILE = "results/x2_policy_compare.jsonl"
FA_FILE = "results/x2_policy_compare_fa_control.jsonl"
MODEL = "qwen3:14b"
POLICIES = {"a": ("f16 (default)", "base"), "i": ("f16+FA", "f16"), "g": ("q8_0", "q8_0"), "h": ("q4_0", "q4_0")}
BITS = {"f16": 16.0, "q8_0": 8.5, "q4_0": 4.5}
MIB = 1024 * 1024
AGREEMENT_PAIRS = (("a", "g"), ("i", "g"), ("a", "i"), ("a", "h"), ("g", "h"))
TOKENIZER_REPO = "Qwen/Qwen3-8B"


def read_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def policy_of(arm_id: str) -> str:
    return arm_id.split("_")[1]


def sessions(rows: list[dict], model: str = MODEL, policies=POLICIES) -> dict:
    """{(policy, seed): [turn rows sorted by turn_idx]} for the model's r2a_turn rows."""
    out: dict = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("model_id") == model:
            p = policy_of(r["arm_id"])
            if p in policies:
                out.setdefault((p, r["seed"]), []).append(r)
    return {k: sorted(v, key=lambda t: t["turn_idx"]) for k, v in out.items()}


def p90(xs: list[float]) -> float | None:
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    return st.quantiles(xs, n=10, method="inclusive")[8]


def session_evidence(turns: list[dict]) -> dict:
    """Minutes, loaded window, and Ollama-timed prefill / decode tokens/s (median, p90 over calls) of one session."""
    calls = [c for t in turns for c in t.get("calls") or []]
    pf = [c["prompt_eval_count_info_only"] / c["prompt_eval_duration_s"] for c in calls
          if c.get("prompt_eval_duration_s") and c.get("prompt_eval_count_info_only")]
    dc = [c["completion_tokens"] / c["eval_duration_s"] for c in calls
          if c.get("eval_duration_s") and c.get("completion_tokens")]
    return {"n_turns": len(turns), "n_calls": len(calls),
            "minutes": sum(t.get("turn_wall_s") or 0.0 for t in turns) / 60.0,
            "loaded_context": sorted({t.get("loaded_context") for t in turns}, key=str),
            "prefill_median": st.median(pf) if pf else None, "prefill_p90": p90(pf),
            "decode_median": st.median(dc) if dc else None, "decode_p90": p90(dc),
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
            "length_stops": sum(1 for c in calls if c.get("done_reason") == "length"),
            "call_errors": sum(1 for c in calls if c.get("outcome") != "ok")}


def load_records(b5_rows: list[dict], fa_rows: list[dict], model: str = MODEL) -> dict:
    """Per env (base, f16, q8_0, q4_0): the window probe's loaded_context and /api/ps size, and the server-log KV
    check (observed types, env in the config line, flash-attention line)."""
    out: dict = {}
    for r in b5_rows:
        if r.get("record") == "b5_resolution" and r.get("model_id") == model:
            pts = r["windows"]["points"]
            for env in ("base", "q8_0", "q4_0"):
                p = pts.get(env)
                p = p[-1] if isinstance(p, list) else p
                if p:
                    out.setdefault(env, {}).update(loaded_context=p.get("loaded_context"), size=p.get("size"),
                                                   size_vram=p.get("size_vram"), num_ctx_sent=p.get("num_ctx_sent"))
    for r in fa_rows:
        if r.get("record") == "b5_fa_resolution" and r.get("model_id") == model:
            p = r["points"][-1]
            out.setdefault("f16", {}).update(loaded_context=p.get("loaded_context"), size=p.get("size"),
                                             size_vram=p.get("size_vram"), num_ctx_sent=p.get("num_ctx_sent"))
        if r.get("record") == "b5_base_fa_check" and r.get("model_id") == model:
            out.setdefault("base", {})["fa_log_lines"] = r["check"].get("flash_attention_lines")
            out["base"]["base_probe_size"] = r["probe"].get("size")
    for r in b5_rows + fa_rows:
        if r.get("record") == "b5_kv_check" and r.get("model_id") == model:
            c = r["check"]
            out.setdefault(r["env"], {}).update(kv_observed=c.get("observed"), kv_env_in_config=c.get("config_has_env"),
                                               kv_fa_in_config=c.get("config_flash_attention"),
                                               kv_valid=c.get("valid"), kv_verified=c.get("verified"))
    return out


def kv_size_inference(size_f16: int, size_q8: int, size_q4: int) -> dict:
    """The f16 KV size X (MiB) from each of the two size differences, and the implied per-type KV buffers."""
    x_a = (size_f16 - size_q8) * BITS["f16"] / (BITS["f16"] - BITS["q8_0"]) / MIB
    x_b = (size_q8 - size_q4) * BITS["f16"] / (BITS["q8_0"] - BITS["q4_0"]) / MIB
    x = x_b
    rel = abs(x_a - x_b) / max(abs(x_a), abs(x_b), 1e-9)
    return {"x_from_f16_q8": x_a, "x_from_q8_q4": x_b, "rel_diff": rel,
            "kv_mib": {k: x * b / BITS["f16"] for k, b in BITS.items()},
            "q4_over_f16": BITS["q4_0"] / BITS["f16"],
            "consistent": rel < 0.01}


def equal_means_explanation(per_seed: dict, ev: dict) -> dict:
    """Why (g) and (h) print the same mean minutes: per-seed minutes, the unrounded means, cap/limit indicators, and
    how many calls produced identical output (a copied value would make them all identical)."""
    g = {s: ev[("g", s)]["minutes"] for s in sorted(per_seed) if ("g", s) in ev}
    h = {s: ev[("h", s)]["minutes"] for s in sorted(per_seed) if ("h", s) in ev}
    mg, mh = st.mean(g.values()), st.mean(h.values())
    return {"g": g, "h": h, "mean_g": mg, "mean_h": mh, "diff_s": (mh - mg) * 60.0,
            "seed_diff_min": {s: h[s] - g[s] for s in g if s in h},
            "length_stops": sum(ev[(p, s)]["length_stops"] for p in "gh" for s in g),
            "turns": sorted({ev[(p, s)]["n_turns"] for p in "gh" for s in g}),
            "completion_tokens": {p: [ev[(p, s)]["completion_tokens"] for s in g] for p in "gh"}}


# ── output agreement ───────────────────────────────────────────────────────────────────────────────

def _call_output(c: dict) -> dict:
    tcs = [{"name": (t.get("function") or {}).get("name"), "arguments": (t.get("function") or {}).get("arguments")}
           for t in (c.get("native_tool_calls") or [])]
    return {"content": c.get("content") or "", "native_tool_calls": tcs,
            "method": c.get("tool_detection_method")}


def _tool_calls(c: dict) -> list:
    return [{"name": t.get("name"), "arguments": t.get("arguments")} for t in (c.get("tool_calls_parsed") or [])]


def _render(c: dict) -> str:
    o = _call_output(c)
    return o["content"] + ("\n" + json.dumps(o["native_tool_calls"], sort_keys=True) if o["native_tool_calls"] else "")


def _h(prev: str, obj) -> str:
    return hashlib.sha256((prev + json.dumps(obj, sort_keys=True, default=str)).encode("utf-8")).hexdigest()


def prompt_keys(turns: list[dict], seed) -> list[list[str]]:
    """Per turn, per call: the rebuilt prompt key (see module docstring)."""
    state = _h("", {"seed": seed})
    out = []
    for t in turns:
        state = _h(state, {"turn": t["turn_idx"]})
        keys = []
        for j, c in enumerate(t.get("calls") or []):
            keys.append(_h(state, {"call": j}))
            state = _h(state, _call_output(c))
        out.append(keys)
    return out


def first_divergence(a: str, b: str, tokenize=None) -> int | None:
    """Index of the first differing token (tokenize given) or character; None when equal."""
    if a == b:
        return None
    sa, sb = (tokenize(a), tokenize(b)) if tokenize else (a, b)
    for i, (x, y) in enumerate(zip(sa, sb)):
        if x != y:
            return i
    return min(len(sa), len(sb))


def compare_sessions(ta: list[dict], tb: list[dict], seed, tokenize=None) -> dict:
    """Turn-by-turn agreement over the turns whose call-1 prompt key matches on both sides."""
    ka, kb = prompt_keys(ta, seed), prompt_keys(tb, seed)
    by_b = {t["turn_idx"]: (t, k) for t, k in zip(tb, kb)}
    turns = []
    for t, k in zip(ta, ka):
        if t["turn_idx"] not in by_b:
            continue
        u, ku = by_b[t["turn_idx"]]
        if not k or not ku or k[0] != ku[0]:
            continue
        ca, cb = t.get("calls") or [], u.get("calls") or []
        call_eq = [_call_output(x) == _call_output(y) for x, y in zip(ca, cb)]
        exact = len(ca) == len(cb) and all(call_eq)
        div_call = next((j for j, e in enumerate(call_eq) if not e), None if len(ca) == len(cb) else len(call_eq))
        tok = None
        if div_call is not None and div_call < len(call_eq):
            tok = first_divergence(_render(ca[div_call]), _render(cb[div_call]), tokenize)
        tools_eq = [x for c in ca for x in _tool_calls(c)] == [x for c in cb for x in _tool_calls(c)]
        turns.append({"turn_idx": t["turn_idx"], "exact": exact, "divergent_call": div_call,
                      "first_divergence": tok, "tool_calls_agree": tools_eq,
                      "final_text_agree": (t.get("final_text") or "") == (u.get("final_text") or "")})
    first_div = next((x for x in turns if not x["exact"]), None)
    return {"n_turns": len(ta), "comparable": len(turns), "exact": sum(x["exact"] for x in turns),
            "tool_agree": sum(x["tool_calls_agree"] for x in turns),
            "first_divergent_turn": first_div["turn_idx"] if first_div else None,
            "first_divergent_call": first_div["divergent_call"] if first_div else None,
            "first_divergence_index": first_div["first_divergence"] if first_div else None,
            "turns": turns}


def load_tokenizer():
    """The Qwen3 tokenizer from the local Hugging Face cache only (no download); None if absent."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from tokenizers import Tokenizer
        p = try_to_load_from_cache(TOKENIZER_REPO, "tokenizer.json")
        if not isinstance(p, str):
            return None
        tk = Tokenizer.from_file(p)
        return lambda s: tk.encode(s, add_special_tokens=False).ids
    except Exception:
        return None


def agreement(ses: dict, pairs=AGREEMENT_PAIRS, tokenize=None) -> dict:
    out = {}
    seeds = sorted({s for _, s in ses})
    for pa, pb in pairs:
        per = {s: compare_sessions(ses[(pa, s)], ses[(pb, s)], s, tokenize) for s in seeds
               if (pa, s) in ses and (pb, s) in ses}
        out[(pa, pb)] = {"per_seed": per, "comparable": sum(v["comparable"] for v in per.values()),
                         "exact": sum(v["exact"] for v in per.values()),
                         "tool_agree": sum(v["tool_agree"] for v in per.values()),
                         "n_turns": sum(v["n_turns"] for v in per.values())}
    return out


def evidence(repo) -> dict:
    repo = Path(repo)
    b5, fa = read_rows(repo / B5_FILE), read_rows(repo / FA_FILE)
    ses = sessions(b5 + fa)
    ev = {k: session_evidence(v) for k, v in ses.items()}
    loads = load_records(b5, fa)
    kv = kv_size_inference(loads["base"]["size"], loads["q8_0"]["size"], loads["q4_0"]["size"])
    seeds = sorted({s for _, s in ses})
    return {"sessions": ses, "ev": ev, "loads": loads, "kv": kv, "seeds": seeds,
            "gh": equal_means_explanation(dict.fromkeys(seeds), ev),
            "run_end": {f: any(r.get("record") == "run_end" for r in rows) for f, rows in ((B5_FILE, b5), (FA_FILE, fa))}}
