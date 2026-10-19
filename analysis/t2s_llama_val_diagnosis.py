"""Row-level diagnosis of the failed llama3.1:8b baseline gate in the two evo-t2s R2 validations
(results/t2s_r2_validation_cpu_v1.jsonl, Intel CPU; results/t2s_r2_validation_igpu_v1.jsonl, Intel iGPU), harness
harness/x2_r2_agent.py in call-2 mode "off" (tools withheld on a turn's second call). Pure: reads the committed JSONL
files and re-scores nothing for a gate; the gates stand as the harness computed them (docs/FINDINGS.md, dated section
2026-10-19).

Unit: one failed rule on one baseline turn. The baseline arm in mode "off" is ollama_ctx_131072_call2_notools (the same
arm as the negative control), completed sessions only, exactly the turns harness baseline_table counts. Every failed
rule gets exactly one class, with the same four classes as the evalset v1 diagnosis (analysis/evalset_val_diagnosis.py,
whose text-call detector `text_calls` is reused unchanged), first match wins:

  scoring_problem    the harness, not the model: rule 2 failed although a native log_event entry is in the raw
                     native_tool_calls; rule 5 failed although the answer field ends with the session code, or ends
                     with it followed only by punctuation or spaces
  tool_call_as_text  the call was written into the reply content instead of made natively: rule 2, a log_event call
                     with arguments in some call's content (the whole content a text call, or a log_event entry with
                     arguments in the answer object's optional "tool_calls" field, which the harness by design treats
                     as a report and never as a call); rules 1 and 5, the final content is a text tool call (no answer)
  wrong_value        something was given but it is the wrong thing: rule 2, a native call to an unknown tool name;
                     rule 5, a different SC- code, the session code inside the answer but not at its end, or the code
                     outside the answer field; rule 1, an answer object of the wrong shape; rules 3 and 4, always
  omitted            nothing was given: rule 2, no log_event call anywhere (subtype `claimed_in_answer_report` when
                     the answer's "tool_calls" field names log_event without arguments, a claim of a call that never
                     happened; else `other_tool_only` or `no_call`); rule 5, no session code in the answer (or empty
                     content, or no answer field)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "harness", REPO / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import evalset_val_diagnosis as D  # noqa: E402  (stdlib only; reused: text_calls, read_rows)

FILES = {"cpu": "results/t2s_r2_validation_cpu_v1.jsonl", "igpu": "results/t2s_r2_validation_igpu_v1.jsonl"}
BACKEND_LABEL = {"cpu": "Intel CPU", "igpu": "Intel iGPU"}
BASELINE_ARM = "ollama_ctx_131072_call2_notools"   # NEG_ARM + MODE_SUFFIX["off"]
MODELS = ("llama3.1:8b", "qwen3:14b")
RULES = ("rule1_json_keys", "rule2_log_event_called", "rule3_no_zebra", "rule4_metres", "rule5_session_code")
CLASSES = ("tool_call_as_text", "omitted", "wrong_value", "scoring_problem")
KNOWN_TOOLS = ("log_event", "lookup_fact")
_SC_RE = re.compile(r"SC-[A-Z0-9]{6,}")
_TAIL_PUNCT = re.compile(r"[\s.,;:!?\"')\]]*")


def _strip_fence(text):
    s = (text or "").strip()
    m = re.fullmatch(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```", s, flags=re.DOTALL)
    return m.group(1).strip() if m else s


def parse_json_strict(text):
    """x2_r2_agent.parse_json_strict re-implemented (no network stack import): json.loads of the whole content after
    stripping at most one code fence; None on any failure. Checked against the harness in the tests."""
    try:
        return json.loads(_strip_fence(text))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def baseline_turns(rows):
    """The turns harness baseline_table counts for mode "off": completed sessions of the baseline arm."""
    done = {r["attempt_id"] for r in rows if r.get("record") == "r2a_session" and r.get("arm_id") == BASELINE_ARM}
    return [r for r in rows if r.get("record") == "r2a_turn" and r.get("attempt_id") in done]


def _native_names(row):
    out = []
    for c in row["calls"]:
        for x in c.get("native_tool_calls") or []:
            fn = x.get("function") if isinstance(x, dict) else None
            out.append(fn.get("name") if isinstance(fn, dict) else None)
    return out


def _report_entries(content):
    """The answer object's optional "tool_calls" field as [(name, has_arguments)], [] if absent."""
    obj = parse_json_strict(content)
    if not isinstance(obj, dict) or "answer" not in obj:
        return []
    rep = obj.get("tool_calls")
    out = []
    if isinstance(rep, dict):
        if isinstance(rep.get("name"), str):
            out.append((rep["name"], bool(rep.get("arguments") or rep.get("parameters"))))
        else:
            out += [(k, v not in (None, "", {}, [])) for k, v in rep.items()]
    elif isinstance(rep, list):
        for x in rep:
            if isinstance(x, str):
                out.append((x, False))
            elif isinstance(x, dict) and isinstance(x.get("name"), str):
                out.append((x["name"], bool(x.get("arguments") or x.get("parameters"))))
            elif isinstance(x, dict) and isinstance(x.get("function"), dict):
                fn = x["function"]
                out.append((fn.get("name"), bool(fn.get("arguments"))))
            elif isinstance(x, dict):
                out += [(k, v not in (None, "", {}, [])) for k, v in x.items()]
    elif isinstance(rep, str):
        out.append((rep, False))
    return out


def classify_rule2(row):
    names = _native_names(row)
    if "log_event" in names:
        return "scoring_problem", "native log_event present in native_tool_calls but rule 2 scored false"
    claimed = False
    for i, c in enumerate(row["calls"], 1):
        content = c.get("content") or ""
        if any(tc.get("name") == "log_event" for tc in D.text_calls(content)):
            return "tool_call_as_text", f"call {i} content is a text log_event call"
        for name, has_args in _report_entries(content):
            if name == "log_event" and has_args:
                return "tool_call_as_text", f"log_event with arguments in the answer's tool_calls field (call {i})"
            if name == "log_event":
                claimed = True
    unknown = [n for n in names if n not in KNOWN_TOOLS]
    if unknown:
        return "wrong_value", f"native call to unknown tool {unknown[0]!r}"
    if claimed:
        return "omitted", "claimed_in_answer_report: log_event named in the answer's tool_calls without arguments"
    if names:
        return "omitted", "other_tool_only: native " + "/".join(sorted(set(n for n in names if n))) + ", no log_event"
    return "omitted", "no_call: no tool call at all"


def classify_rule5(row):
    text = row.get("final_text") or ""
    code = row["session_code"]
    if not text.strip():
        return "omitted", "final content empty"
    if D.text_calls(text):
        return "tool_call_as_text", "final answer is a tool call written as text"
    obj = parse_json_strict(text)
    ans = obj.get("answer") if isinstance(obj, dict) else None
    if isinstance(ans, str):
        if ans.rstrip().endswith(code):
            return "scoring_problem", "answer ends with the session code but rule 5 scored false"
        if code in ans:
            tail = ans[ans.rfind(code) + len(code):]
            if _TAIL_PUNCT.fullmatch(tail):
                return "scoring_problem", f"session code at the end followed only by {tail!r}"
            return "wrong_value", "session code inside the answer but not at its end"
        other = [c for c in _SC_RE.findall(ans) if c != code]
        if other:
            return "wrong_value", f"different session code {other[0]}"
        if code in text:
            return "wrong_value", "session code outside the answer field"
        return "omitted", "no session code in the answer"
    if code in text:
        return "wrong_value", "session code present but no string answer field"
    return "omitted", "no string answer field"


def classify_rule(row, rule):
    if rule == "rule2_log_event_called":
        return classify_rule2(row)
    if rule == "rule5_session_code":
        return classify_rule5(row)
    text = row.get("final_text") or ""
    if rule == "rule1_json_keys":
        if not text.strip():
            return "omitted", "final content empty"
        if D.text_calls(text):
            return "tool_call_as_text", "final answer is a tool call written as text"
        return "wrong_value", "answer object of the wrong shape"
    return "wrong_value", f"{rule} violated in the final answer"


def failures(rows, model):
    """[(row, rule, class, detail)] for every failed rule on every baseline turn of the model."""
    out = []
    for r in baseline_turns(rows):
        if r["model_id"] != model:
            continue
        for rule in RULES:
            if r.get(rule) is False:
                out.append((r, rule) + classify_rule(r, rule))
    return out


def _subtype(detail):
    return detail.split(":", 1)[0] if ":" in detail and detail.split(":", 1)[0].isidentifier() else detail


def rule_rates(rows, model):
    """{rule: (n_fail, n)} on the baseline turns, None-valued rules excluded from n (harness _rate)."""
    mt = [r for r in baseline_turns(rows) if r["model_id"] == model]
    out = {}
    for rule in RULES:
        vals = [r.get(rule) for r in mt if r.get(rule) is not None]
        out[rule] = (sum(1 for v in vals if v is False), len(vals))
    return out


def run_meta(rows):
    """What the file itself records about the model build and sampling."""
    rs = next((r for r in rows if r.get("record") == "run_start"), {})
    rt = next((r for r in rows if r.get("record") == "runtime"), {})
    blob = json.dumps(rows)
    return {
        "hw_id": rs.get("hw_id"),
        "ollama_version": rt.get("ollama_version"),
        "digest_recorded": "digest" in blob.lower(),
        "temperature_recorded": '"temperature"' in blob,
        "seeds": sorted({r["seed"] for r in rows if r.get("record") == "r2a_session"
                         and r.get("arm_id") == BASELINE_ARM}),
        "call2_mode": rs.get("call2_mode"),
        "max_tokens_per_call": rs.get("max_tokens_per_call"),
    }


def rule2_by_task(rows, model):
    """{turn task: (rule 2 passed, turns)}: lookup_fact turns need two calls (log_event and lookup_fact), log_event
    turns need one."""
    out = {}
    for r in baseline_turns(rows):
        if r["model_id"] == model and r.get("rule2_log_event_called") is not None:
            ok, n = out.get(r["tool_task"], (0, 0))
            out[r["tool_task"]] = (ok + bool(r["rule2_log_event_called"]), n + 1)
    return dict(sorted(out.items()))


def canary_misses(rows, model):
    return sum((r.get("canary_sys_ok") is False) + (r.get("canary_hist_ok") is False)
               for r in baseline_turns(rows) if r["model_id"] == model)


def cross_backend(rows_a, rows_b, model):
    """Baseline turns of the model matched by (seed, turn): identical final text, identical native call names."""
    key = lambda r: (r["seed"], r["turn_idx"])  # noqa: E731
    a = {key(r): r for r in baseline_turns(rows_a) if r["model_id"] == model}
    b = {key(r): r for r in baseline_turns(rows_b) if r["model_id"] == model}
    common = sorted(set(a) & set(b))
    return {"n": len(common),
            "same_final_text": sum(1 for k in common if a[k].get("final_text") == b[k].get("final_text")),
            "same_native_calls": sum(1 for k in common if _native_names(a[k]) == _native_names(b[k])),
            "same_system_prompt": sum(1 for k in common if [c.get("system_prompt_sha256") for c in a[k]["calls"]]
                                      == [c.get("system_prompt_sha256") for c in b[k]["calls"]])}


def summary(repo=REPO):
    out = {}
    for be, fn in FILES.items():
        rows = D.read_rows(repo / fn)
        e = {"meta": run_meta(rows)}
        for m in MODELS:
            f = failures(rows, m)
            e[m] = {
                "rates": rule_rates(rows, m),
                "failed_rules": len(f),
                "classes": dict(Counter(x[2] for x in f)),
                "by_rule": {rule: dict(Counter(x[2] for x in f if x[1] == rule)) for rule in RULES
                            if any(x[1] == rule for x in f)},
                "subtypes": dict(Counter(f"{x[1].split('_')[0]}:{x[2]}/{_subtype(x[3])}" for x in f)),
                "canary_misses": canary_misses(rows, m),
                "rule2_by_task": rule2_by_task(rows, m),
            }
        out[be] = e
    ra, rb = (D.read_rows(repo / FILES[k]) for k in ("cpu", "igpu"))
    out["cross"] = {m: cross_backend(ra, rb, m) for m in MODELS}
    return out


def _fmt(d):
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))) or "none"


def dominant_class(s, model="llama3.1:8b"):
    tot = Counter()
    for be in FILES:
        tot.update(s[be][model]["classes"])
    return tot.most_common(1)[0][0] if tot else None


def register_value(repo=REPO):
    s = summary(repo)
    m = "llama3.1:8b"
    parts = []
    for be in FILES:
        e, meta = s[be][m], s[be]["meta"]
        rates = "; ".join(f"{r.split('_')[0]} {e['rates'][r][1] - e['rates'][r][0]}/{e['rates'][r][1]}"
                          for r in RULES)
        parts.append(
            f"{BACKEND_LABEL[be]} ({FILES[be]}): baseline rules passed {rates}; canary misses {e['canary_misses']}; "
            f"failed rules {e['failed_rules']} ({_fmt(e['classes'])}); by rule "
            + "; ".join(f"{r.split('_')[0]} {_fmt(c)}" for r, c in e["by_rule"].items())
            + f"; subtypes {_fmt(e['subtypes'])}; rule2 passed by turn task "
            + ", ".join(f"{k} {ok}/{n}" for k, (ok, n) in e["rule2_by_task"].items())
            + f"; hw_id {meta['hw_id']}, Ollama {meta['ollama_version']}, model digest "
            f"{'recorded' if meta['digest_recorded'] else 'not recorded'}, temperature "
            f"{'recorded' if meta['temperature_recorded'] else 'not recorded'} per row, session seeds "
            f"{'/'.join(map(str, meta['seeds']))}, call2 mode {meta['call2_mode']}")
    q = s["cross"][m]
    parts.append(f"CPU vs iGPU same (seed, turn): identical final text {q['same_final_text']}/{q['n']}, identical "
                 f"native call names {q['same_native_calls']}/{q['n']}, identical system prompts "
                 f"{q['same_system_prompt']}/{q['n']}")
    qw = "; ".join(f"{BACKEND_LABEL[be]} {s[be]['qwen3:14b']['failed_rules']} ({_fmt(s[be]['qwen3:14b']['classes'])})"
                   for be in FILES)
    parts.append(f"qwen3:14b failed rules {qw}")
    parts.append(f"dominant class llama3.1:8b both backends: {dominant_class(s)}")
    n = sum(s[be][m]["rates"]["rule1_json_keys"][1] for be in FILES)
    return {"value": " / ".join(parts), "n": n}


def main():
    s = summary()
    print(json.dumps(s, indent=1, default=str))
    for be, fn in FILES.items():
        rows = D.read_rows(REPO / fn)
        print(f"\n===== {be}: llama3.1:8b failed rules")
        for r, rule, cls, det in failures(rows, "llama3.1:8b"):
            print(f"{r['seed']} t{r['turn_idx']} {r['tool_task']} {rule}: {cls} | {det} | {r['final_text'][:90]!r}")


if __name__ == "__main__":
    main()
