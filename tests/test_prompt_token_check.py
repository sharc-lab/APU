"""harness/prompt_token_check.py and its use in harness/x2_r2_mechanism.py: blob resolution from a fake Ollama
manifest tree, llama-tokenize output parsing, the 1% comparison, per-tier match rate and citable flag, the checker
with a fake runtime and a fake tokenize process, and the mechanism wrapper/report/register wiring. No network, no
process launches."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import prompt_token_check as ptc  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
import x2_r2_mechanism as mech  # noqa: E402

# Shape of llama-tokenize (llama.cpp b10970) --ids --show-count stdout, per its --help text ("only print the token
# IDs, in a Python-parseable list form like [1, 2, ...]", "print the total number of tokens") and tokenize.cpp.
TOK_OUT = "[128000, 128006, 9125, 128007, 271]\nTotal number of tokens: 5\n"
DIGEST = "667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"


def _store(tmp_path, ref_parts, layers, blob=True):
    d = tmp_path / "models"
    mp = d / "manifests" / Path(*ref_parts)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps({"schemaVersion": 2, "layers": layers}), encoding="utf-8")
    (d / "blobs").mkdir(exist_ok=True)
    if blob:
        (d / "blobs" / f"sha256-{DIGEST}").write_bytes(b"GGUF")
    return d


MODEL_LAYER = {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{DIGEST}", "size": 4}
TEMPLATE_LAYER = {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:" + "a" * 64, "size": 1}


def test_manifest_path_forms(tmp_path):
    lib = tmp_path / "manifests" / "registry.ollama.ai" / "library"
    assert ptc.manifest_path(tmp_path, "llama3.1:8b") == lib / "llama3.1" / "8b"
    assert ptc.manifest_path(tmp_path, "qwen3-4b-2507") == lib / "qwen3-4b-2507" / "latest"
    assert ptc.manifest_path(tmp_path, "me/m:q4") == tmp_path / "manifests" / "registry.ollama.ai" / "me" / "m" / "q4"
    assert ptc.manifest_path(tmp_path, "hf.co/me/m:q4") == tmp_path / "manifests" / "hf.co" / "me" / "m" / "q4"


def test_resolve_model_blob(tmp_path):
    d = _store(tmp_path, ["registry.ollama.ai", "library", "llama3.1", "8b"], [MODEL_LAYER, TEMPLATE_LAYER])
    assert ptc.resolve_model_blob(d, "llama3.1:8b") == d / "blobs" / f"sha256-{DIGEST}"
    with pytest.raises(FileNotFoundError):
        ptc.resolve_model_blob(d, "qwen3:8b")


def test_resolve_model_blob_refuses_bad_manifest_or_missing_blob(tmp_path):
    d = _store(tmp_path, ["registry.ollama.ai", "library", "x", "latest"], [TEMPLATE_LAYER])
    with pytest.raises(ValueError):
        ptc.resolve_model_blob(d, "x")
    d2 = _store(tmp_path / "b", ["registry.ollama.ai", "library", "x", "latest"], [MODEL_LAYER], blob=False)
    with pytest.raises(FileNotFoundError):
        ptc.resolve_model_blob(d2, "x")


def test_parse_tokenize_output():
    r = ptc.parse_tokenize_output(TOK_OUT)
    assert r["n_tokens"] == 5 and r["n_ids"] == 5 and r["n_shown"] == 5 and r["ids_head"] == [128000, 128006, 9125,
                                                                                               128007]
    assert r["double_bos"] is False
    assert ptc.parse_tokenize_output("[128000, 128000, 5]\n")["double_bos"] is True
    assert ptc.parse_tokenize_output("Total number of tokens: 42\n")["n_tokens"] == 42
    with pytest.raises(ValueError):
        ptc.parse_tokenize_output("[1, 2, 3]\nTotal number of tokens: 4\n")
    with pytest.raises(ValueError):
        ptc.parse_tokenize_output("error: failed to load model\n")


def test_count_tokens_uses_temp_file_stdin_and_flags(tmp_path):
    seen = {}
    text = 'line one\nliteral \\n stays; json {"a": "b"} \u00e9\n'

    def run(cmd, stdin=None, capture_output=None, timeout=None):
        seen["cmd"], seen["data"] = cmd, stdin.read()
        return types.SimpleNamespace(returncode=0, stdout=TOK_OUT.encode(), stderr=b"")

    r = ptc.count_tokens(text, "C:/m/blob", exe="C:/bin/llama-tokenize.exe", run=run, tmp_dir=tmp_path)
    assert seen["cmd"] == ["C:/bin/llama-tokenize.exe", "-m", "C:/m/blob", "--stdin", "--ids", "--show-count",
                           "--no-escape", "--log-disable"]
    assert seen["data"] == text.encode("utf-8")  # exact bytes, no newline translation, not on the command line
    assert text not in " ".join(seen["cmd"])
    assert r["n_tokens"] == 5 and list(tmp_path.iterdir()) == []  # temp file removed


def test_count_tokens_fails_loudly(tmp_path):
    def run(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"failed to load model")
    with pytest.raises(RuntimeError):
        ptc.count_tokens("x", "blob", run=run, tmp_dir=tmp_path)


def test_compare_one_percent():
    assert ptc.compare(1010, 1000) == {"diff": 10, "rel_diff": 0.01, "match": True}
    assert ptc.compare(1011, 1000)["match"] is False
    assert ptc.compare(989, 1000)["match"] is False
    assert ptc.compare(None, 1000)["match"] is False and ptc.compare(5, 0)["match"] is False


def _chk(tier, turn, match, post=True, err=None):
    return {"record": "r2m_prompt_check", "arm_id": tier, "seed": 1, "turn_idx": turn, "call_idx": 1,
            "post_overflow": post, "match": match, "rel_diff": 0.001 if match else 0.2, "error": err}


def test_tier_summary_match_rate_and_citable():
    rows = ([_chk("a", t, True, post=t > 1) for t in range(1, 7)] + [_chk("b", t, t != 3) for t in range(1, 7)]
            + [_chk("c", t, True) for t in range(1, 4)] + [_chk("d", t, True) for t in range(1, 5)]
            + [_chk("d", 5, False, err="tokenize: boom")])
    s = ptc.tier_summary(rows)
    assert s["a"]["mechanism_citable"] and s["a"]["match_rate"] == 1.0 and s["a"]["statement"] is None
    assert s["a"]["n_pre_overflow"] == 1 and s["a"]["n_post_overflow"] == 5
    assert not s["b"]["mechanism_citable"] and s["b"]["match_rate"] == pytest.approx(5 / 6)
    assert s["b"]["statement"] == ptc.NOT_CITABLE
    assert not s["c"]["mechanism_citable"] and "only 3 checks" in s["c"]["statement"]  # fewer than MIN_CHECKS
    assert not s["d"]["mechanism_citable"] and s["d"]["n_errors"] == 1


class FakeRT:
    def __init__(self, pec=100, unload_ok=True):
        self.pec, self.unload_ok, self.posts, self.unloads = pec, unload_ok, [], []

    def unload(self, model):
        self.unloads.append(model)
        return self.unload_ok

    def _post(self, path, body):
        self.posts.append((path, body))
        return 200, {"prompt_eval_count": self.pec, "load_duration": 123}, None, 0.5


def _checker(tmp_path, rt, n_tokens=100, rc=0):
    d = _store(tmp_path, ["registry.ollama.ai", "library", "llama3.1", "8b"], [MODEL_LAYER])

    def run(cmd, stdin=None, **kw):
        ids = ", ".join(["7"] * n_tokens)
        return types.SimpleNamespace(returncode=rc, stdout=f"[{ids}]\nTotal number of tokens: {n_tokens}\n".encode(),
                                     stderr=b"x")
    return ptc.PromptChecker(rt, ag.native_chat_body, models_dir=d, run=run, tmp_dir=tmp_path)


def test_checker_unloads_then_sends_same_body_with_num_predict_1(tmp_path):
    rt = FakeRT(pec=100)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    r = _checker(tmp_path, rt, n_tokens=100).check("llama3.1:8b", msgs, 4096, None, False, None, "RENDER")
    assert rt.unloads == ["llama3.1:8b"] and r["unloaded_before_check"]
    path, body = rt.posts[0]
    want = ag.native_chat_body("llama3.1:8b", msgs, 4096, None, False)
    want["options"]["num_predict"] = 1
    assert path == "/api/chat" and body == want
    assert r["render_tokens"] == 100 and r["check_prompt_eval_count"] == 100 and r["match"] and r["error"] is None


def test_checker_errors_are_rows_not_skips(tmp_path):
    msgs = [{"role": "user", "content": "u"}]
    r = _checker(tmp_path, FakeRT(unload_ok=False)).check("llama3.1:8b", msgs, 4096, None, False, None, "R")
    assert r["match"] is False and "still listed" in r["error"]
    rt = FakeRT()
    r = _checker(tmp_path / "x", rt, rc=1).check("llama3.1:8b", msgs, 4096, None, False, None, "R")
    assert r["match"] is False and "tokenize" in r["error"]
    r = _checker(tmp_path / "y", FakeRT()).check("llama3.1:8b", msgs, 4096, None, False, None, None)
    assert r["match"] is False and "no prompt" in r["error"]


def test_check_schedule():
    pre = [k for k in range(80) if mech.check_due(False, k, sum(1 for j in range(k) if j % 10 == 0))]
    assert pre == [0, 10, 20, 30, 40]
    done, trunc = 0, []
    for k in range(80):
        if mech.check_due(True, k, done):
            trunc.append(k)
            done += 1
    assert trunc == [0, 1, 2, 4, 8, 12, 16, 20]


# ── wired into the mechanism wrapper, report and register ──────────────────────────────────────────

class FakeChecker:
    def __init__(self, bad_tier=None):
        self.bad_tier, self.calls, self.arm = bad_tier, [], None

    def check(self, model, messages, num_ctx, tools, think, extra, rendered):
        self.calls.append(len(messages))
        bad = self.arm == self.bad_tier
        rt, pec = (1300 if bad else 1001), 1000
        return {"render_tokens": rt, "check_prompt_eval_count": pec, "error": None, **ptc.compare(rt, pec)}


def _run_tier(tmp_path, checker, arm, turns=12):
    from test_x2_r2_mechanism import FakeLoggingOllama
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = tmp_path / f"{arm}.log"
    rows = []
    mrt = mech.MechanismRuntime(FakeLoggingOllama(log, keep_msgs=9), log, rows.append, tmp_path / "logs",
                                sleep=lambda s: None, checker=checker)
    checker.arm = arm
    mrt.begin_session("llama3.1:8b", arm, ag.SEEDS[0], call2_mode="off")
    ag.run_session(mrt, "llama3.1:8b", arm, ag.SEEDS[0], turns, rows.append, lambda m: None, "mechanism")
    mrt.end_session()
    return rows


def test_wrapper_report_and_register(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ck = FakeChecker(bad_tier="ollama_ctx_8192_call2_notools")
    rows = (_run_tier(tmp_path, ck, "ollama_ctx_4096_call2_notools")
            + _run_tier(tmp_path, ck, "ollama_ctx_8192_call2_notools"))
    checks = [r for r in rows if r["record"] == "r2m_prompt_check"]
    a = [r for r in checks if r["arm_id"] == "ollama_ctx_4096_call2_notools"]
    assert sum(1 for r in a if not r["post_overflow"]) == 1 and sum(1 for r in a if r["post_overflow"]) >= 5
    assert all(r["turn_idx"] and r["real_call_prompt_eval_count"] == 100 for r in a)
    rep = mech.report(rows)
    pt = rep["verdict"]["per_tier"]
    assert pt["ollama_ctx_4096_call2_notools"]["mechanism_citable"] is True
    assert pt["ollama_ctx_4096_call2_notools"]["hypothesis"] == "confirmed"
    assert pt["ollama_ctx_8192_call2_notools"]["mechanism_citable"] is False
    assert pt["ollama_ctx_8192_call2_notools"]["match_rate"] == 0.0
    assert rep["verdict"]["mechanism_citable"] is False
    assert any(ptc.NOT_CITABLE in s for s in rep["verdict"]["not_citable_statements"])
    sess = [r for r in rows if r["record"] == "r2m_session"]
    assert sess[0]["prompt_check"]["mechanism_citable"] and not sess[1]["prompt_check"]["mechanism_citable"]

    # register: confirmed/refuted only over citable tiers
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    (repo / "results" / "x2_r2_mechanism.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    real_repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(real_repo / "analysis"))
    import numbers_register as nr
    monkeypatch.setattr(nr, "_r2_agent_module", lambda: ag)
    (repo / "harness").mkdir()
    v = nr.compute_r2_mechanism_verdict(repo)["value"]
    assert v.startswith("verdict confirmed over citable tiers [ollama_ctx_4096]")
    assert "not citable (render does not show what the model saw): [ollama_ctx_8192]" in v
    assert "tier verdict not citable" in v

    ck2 = FakeChecker(bad_tier="ollama_ctx_4096_call2_notools")
    rows2 = _run_tier(tmp_path / "z", ck2, "ollama_ctx_4096_call2_notools")
    (repo / "results" / "x2_r2_mechanism.jsonl").write_text("\n".join(json.dumps(r) for r in rows2),
                                                           encoding="utf-8")
    v2 = nr.compute_r2_mechanism_verdict(repo)["value"]
    assert v2.startswith("verdict not citable") and "confirmed" not in v2.split(";")[0]


def test_old_rows_without_checks_are_not_citable(tmp_path):
    from test_x2_r2_mechanism import _run_mech
    rows, _, _ = _run_mech(tmp_path, keep_msgs=9)
    rep = mech.report(rows)
    assert rep["verdict"]["mechanism_citable"] is False
    assert all(not p["mechanism_citable"] for p in rep["verdict"]["per_tier"].values())
