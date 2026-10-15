"""llama3.3:70b library pull and the B2 model family (pre-registered in docs/FINDINGS.md 2026-10-14):
harness/x2_model_pulls_scale.py (disk check, digest verification, never deletes, idempotent), harness/x2_family_smoke.py
(native valid tool calls, >= 4 of 5), x2_r2_scale.py's family options (smoke preflight, validation file, template source
per row, anchored queue insertion), and the register rows. Fakes only: no server, no network, no queue."""
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "tests"))

import x2_r2_agent as ag  # noqa: E402
import x2_r2_scale as sc  # noqa: E402
import x2_family_smoke as fs  # noqa: E402
import x2_model_pulls_scale as mps  # noqa: E402
from test_x2_r2_agent import FakeRuntime, good_policy, _native  # noqa: E402

TAG = "mistral-small3.2:24b"


# ── pull job ────────────────────────────────────────────────────────────────────────────────────────

def _store(tmp_path, tag, blobs: dict, corrupt=None):
    """A fake Ollama store: blobs {mediaType: bytes}; manifest with correct digests (corrupt: mediaType whose file
    content is changed after the digest is computed)."""
    md = tmp_path / "models"
    (md / "blobs").mkdir(parents=True, exist_ok=True)
    layers = []
    for mt, data in blobs.items():
        d = "sha256:" + hashlib.sha256(data).hexdigest()
        (md / "blobs" / d.replace(":", "-")).write_bytes(data if mt != corrupt else data + b"x")
        layers.append({"mediaType": mt, "digest": d, "size": len(data)})
    cfg = b'{"model_format":"gguf"}'
    cd = "sha256:" + hashlib.sha256(cfg).hexdigest()
    (md / "blobs" / cd.replace(":", "-")).write_bytes(cfg)
    man = {"schemaVersion": 2, "config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                                          "digest": cd, "size": len(cfg)}, "layers": layers}
    import chat_template_source as cts
    mp = cts.manifest_path(md, tag)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(man), encoding="utf-8")
    return md, man


LIB = {"application/vnd.ollama.image.model": b"GGUF" + b"\0" * 64,
       "application/vnd.ollama.image.template": b"{{ .Prompt }} library",
       "application/vnd.ollama.image.params": b'{"stop":["</s>"]}'}


def test_required_free_bytes_and_tags():
    assert set(mps.PULLS) == {"llama3.3:70b", TAG}
    assert mps.required_free_bytes("llama3.3:70b") == int(43e9 * 1.1) + 100 * 10**9


def test_manifest_summary_library_template(tmp_path):
    _, man = _store(tmp_path, TAG, LIB)
    s = mps.manifest_summary(TAG, man)
    assert s["chat_template_source"] == "ollama_library" and s["template_digest"].startswith("sha256:")
    assert s["model_digest"] and s["params_digest"] and s["config_digest"]
    assert s["chat_template_sha256"] == s["template_digest"].split(":", 1)[1]


def test_verify_blobs_ok_corrupt_missing(tmp_path):
    md, man = _store(tmp_path, TAG, LIB)
    assert mps.verify_blobs(md, man, log=lambda m: None)["digests_ok"] is True
    md2, man2 = _store(tmp_path / "b", TAG, LIB, corrupt="application/vnd.ollama.image.model")
    v = mps.verify_blobs(md2, man2, log=lambda m: None)
    assert v["digests_ok"] is False and any(not c["size_ok"] for c in v["blob_checks"])
    blob = mps.blob_path(md, man["layers"][0]["digest"])
    blob.unlink()
    assert mps.verify_blobs(md, man, log=lambda m: None)["digests_ok"] is False


def _process(tmp_path, present, free, pull_results, tags_in_store=True):
    md = tmp_path / "models"
    if tags_in_store:
        md, _ = _store(tmp_path, TAG, LIB)
    rows, pulls = [], []

    def pull(t):
        pulls.append(t)
        return pull_results[len(pulls) - 1]
    r = mps.process_tag(TAG, md, present, rows.append, lambda m: None, pull,
                        lambda t: {"verify_ok": True, "verify_error": None},
                        lambda t: {"capabilities": ["completion", "tools"], "template": "x"}, lambda d: free)
    return r, rows, pulls


def test_process_tag_already_present_no_pull_still_verified(tmp_path):
    r, rows, pulls = _process(tmp_path, {TAG}, 0, [])
    assert pulls == [] and r["already_present"] and r["digests_ok"] and r["verify_ok"]
    assert r["api_show_capabilities"] == ["completion", "tools"] and rows == [r]


def test_process_tag_disk_check_refuses_pull(tmp_path):
    r, rows, pulls = _process(tmp_path, set(), mps.required_free_bytes(TAG) - 1, [], tags_in_store=False)
    assert pulls == [] and r["pulled"] is False and "not pulled" in r["pull_error"] and r["verify_ok"] is False


def test_process_tag_pull_retry_then_verify(tmp_path):
    r, rows, pulls = _process(tmp_path, {"other:1b"}, 10**13,
                              [{"pull_exit_code": 1, "pull_error": "net"}, {"pull_exit_code": 0, "pull_error": None}])
    assert pulls == [TAG, TAG] and r["pull_attempts"] == 2 and r["pulled"] and r["digests_ok"]
    assert r["free_bytes_before"] == 10**13 and r["required_free_bytes"] == mps.required_free_bytes(TAG)


def test_process_tag_digest_mismatch_fails_and_deletes_nothing(tmp_path):
    md, man = _store(tmp_path, TAG, LIB, corrupt="application/vnd.ollama.image.model")
    before = sorted(p.name for p in (md / "blobs").iterdir())
    r = mps.process_tag(TAG, md, {TAG}, lambda row: None, lambda m: None, None,
                        lambda t: pytest.fail("load check must not run"), lambda t: {}, lambda d: 0)
    assert r["digests_ok"] is False and r["verify_ok"] is False
    assert sorted(p.name for p in (md / "blobs").iterdir()) == before


def test_run_with_fakes_logs_free_disk_and_stops_server(tmp_path):
    md, _ = _store(tmp_path, TAG, LIB)
    _store(tmp_path, "llama3.3:70b", LIB)
    ev = []
    hc = types.SimpleNamespace(start_ollama_server=lambda: ev.append("start"),
                               stop_ollama_server=lambda: ev.append("stop"), wait_for_ollama_ready=lambda timeout_s: True)
    mp = types.SimpleNamespace(currently_present_tags=lambda: {TAG, "llama3.3:70b"},
                               pull_one=lambda *a: pytest.fail("no pull"),
                               verify_one=lambda t: {"verify_ok": True, "verify_error": None})
    out = tmp_path / "pulls.jsonl"
    logs = []
    ok, rows = mps.run(out, log=logs.append, hc=hc, mp=mp, show_fn=lambda t: {}, free_fn=lambda d: 12345,
                       models_dir=md)
    assert ok and ev == ["start", "stop"] and len(rows) == 2
    recs = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert recs[0]["record"] == "run_start" and recs[0]["free_bytes_at_start"] == 12345
    assert any("12345 bytes" in m for m in logs) and recs[-1]["all_models_verified"] is True


def test_pull_main_advances_once(monkeypatch, tmp_path):
    notes = []
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
    monkeypatch.setattr(mps, "run", lambda out: (False, [{"model_tag": TAG, "digests_ok": False}]))
    note = mps.main(["--out", str(tmp_path / "o.jsonl")])
    assert notes == [note] and "not verified" in note


# ── smoke ───────────────────────────────────────────────────────────────────────────────────────────

def _smoke_rows(policy_mod=None, turns=5):
    ags = ag.build_agent_session(fs.SMOKE_SEED, turns)
    base = good_policy(ags)
    rt = FakeRuntime(policy_mod(base) if policy_mod else base, loaded=32768)
    rows = []
    ag.run_session(rt, TAG, fs.SMOKE_ARM, fs.SMOKE_SEED, turns, rows.append, lambda m: None, fs.SMOKE_MODE)
    return rows, rt


def test_smoke_all_native_valid_passes_and_think_omitted():
    rows, rt = _smoke_rows()
    s = fs.smoke_summary(rows, TAG)
    assert s["n_calls"] == 5 and s["n_native_valid"] == 5 and s["passed"]
    assert all(c["think"] is None for c in rt.calls) and ag.MODELS[TAG] is None
    assert rt.calls[0]["num_ctx"] == 32768 and rt.calls[0]["tools"] == ag.r2.ollama_tools_payload()


def _bad_on(turns_bad, kind):
    def mod(base):
        def policy(call_no, messages, turn_idx):
            if call_no == 1 and turn_idx in turns_bad:
                if kind == "text":
                    return json.dumps({"name": "log_event", "arguments": {"event": "x"}}), None
                return "", [_native("log_event", wrong="x")]
            return base(call_no, messages, turn_idx)
        return policy
    return mod


def test_smoke_one_bad_still_passes_two_bad_fails():
    s1 = fs.smoke_summary(_smoke_rows(_bad_on({2}, "invalid"))[0], TAG)
    assert s1["n_native_valid"] == 4 and s1["passed"] and s1["n_native"] == 5
    s2 = fs.smoke_summary(_smoke_rows(_bad_on({1, 3}, "text"))[0], TAG)
    assert s2["n_native_valid"] == 3 and not s2["passed"]
    assert [p["detection_method"] for p in s2["per_call"]][:3] == ["text_fallback", "native", "text_fallback"]


def test_smoke_ok_reads_result_row():
    assert fs.smoke_ok([], TAG)[0] is False and "no smoke result" in fs.smoke_ok([], TAG)[1][0]
    rows = [{"record": "family_smoke_result", "model_id": TAG, "passed": False, "n_native_valid": 2, "n_calls": 5,
             "pass_min": 4}]
    ok, why = fs.smoke_ok(rows, TAG)
    assert not ok and "2/5" in why[0] and why[0].startswith(f"{TAG}: smoke test")
    assert fs.smoke_ok(rows + [dict(rows[0], passed=True)], TAG) == (True, [])


def _fakes(monkeypatch, rt, notes, started):
    import socket
    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=lambda: started.append(1), stop_ollama_server=lambda: None,
        wait_for_ollama_ready=lambda timeout_s=0: True,
        require_host=lambda h: {"hw_id": "evo-x2", "interactive_guard": False},
        enforce_or_record_interactive_session=lambda cfg: {"logged_in": False}))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
    monkeypatch.setattr(ag, "OllamaRuntime", lambda: rt)


class SmokeFake(FakeRuntime):
    def __init__(self, avail):
        ags = ag.build_agent_session(fs.SMOKE_SEED, fs.SMOKE_TURNS)
        super().__init__(good_policy(ags), loaded=32768)
        self.avail = avail

    def available_models(self):
        return self.avail

    def version(self):
        return "fake"


def test_smoke_job_end_to_end(tmp_path, monkeypatch):
    notes, started = [], []
    _fakes(monkeypatch, SmokeFake([TAG]), notes, started)
    out = tmp_path / sc.SMOKE_FILE
    note = fs.main(["--out", str(out)])
    assert note == f"completed; smoke {TAG} pass 5/5" and notes == [note] and started == [1]
    rows = ag.read_rows(out)
    res = [r for r in rows if r["record"] == "family_smoke_result"]
    assert len(res) == 1 and res[0]["passed"] and fs.smoke_ok(rows, TAG)[0]
    assert all("chat_template_source" in r for r in rows if r.get("model_id"))


def test_smoke_job_missing_model(tmp_path, monkeypatch):
    notes, started = [], []
    _fakes(monkeypatch, SmokeFake([]), notes, started)
    out = tmp_path / sc.SMOKE_FILE
    note = fs.main(["--out", str(out)])
    assert "fail 0/0" in note and len(notes) == 1
    assert not fs.smoke_ok(ag.read_rows(out), TAG)[0]


# ── family options in x2_r2_scale ───────────────────────────────────────────────────────────────────

def test_family_real_refuses_on_failed_smoke_and_starts_no_server(tmp_path, monkeypatch):
    notes, started = [], []
    _fakes(monkeypatch, SmokeFake([TAG]), notes, started)
    (tmp_path / sc.SMOKE_FILE).write_text(json.dumps({"record": "family_smoke_result", "model_id": TAG,
                                                      "passed": False, "n_native_valid": 1, "n_calls": 5,
                                                      "pass_min": 4}) + "\n", encoding="utf-8")
    out = tmp_path / sc.FAMILY_FILES["real"]
    note = sc.main(["--mode", "real", "--models", TAG, "--validation-file", sc.FAMILY_FILES["validation"],
                    "--smoke-file", sc.SMOKE_FILE, "--out", str(out)])
    assert started == [] and notes == [note] and "and the smoke test" in note
    rows = ag.read_rows(out)
    ref = [r for r in rows if r["record"] == "refused_model"]
    assert len(ref) == 1 and any("smoke test failed" in x for x in ref[0]["reasons"])
    assert any("no validation rows" in x for x in ref[0]["reasons"]) and ref[0]["refusal_category"] == "missing"
    assert rows[0]["validation_file"] == sc.FAMILY_FILES["validation"] and rows[0]["smoke_file"] == sc.SMOKE_FILE
    assert rows[0]["default_tier"][TAG]["include"] is False


def test_refusal_category_with_smoke():
    assert sc.refusal_category_with_smoke(TAG, [f"{TAG}: smoke test: no smoke result in the smoke file"], {}, {}) \
        == "smoke_missing"
    assert sc.refusal_category_with_smoke(TAG, [f"{TAG}: smoke test failed (1/5)"], {}, {}) == "smoke_failed"


def test_family_real_items_have_no_default_tier():
    inc = {TAG: sc.default_tier_decision(None, TAG)["include"]}
    items = sc.real_items([TAG], inc)
    assert len(items) == 6 and {a for _, a, _, _ in items} == {sc.arm(sc.TIER_4096), sc.arm(sc.TIER_32768)}
    assert TAG not in sc.SCALE_MODELS


def test_template_fields_from_manifest(tmp_path, monkeypatch):
    md, _ = _store(tmp_path, TAG, LIB)
    monkeypatch.setenv("OLLAMA_MODELS", str(md))
    sc._TEMPLATE_CACHE.clear()
    t = sc.template_fields(TAG)
    assert t["chat_template_source"] == "ollama_library" and len(t["chat_template_sha256"]) == 64
    out = tmp_path / "rows.jsonl"
    emit = sc.row_emitter(out)
    emit({"record": "r2a_turn", "model_id": TAG})
    emit({"record": "run_start"})
    rows = ag.read_rows(out)
    assert rows[0]["chat_template_source"] == "ollama_library" and "chat_template_source" not in rows[1]
    sc._TEMPLATE_CACHE.clear()


def _q(*ids):
    return [{"id": i, "status": "pending"} for i in ids]


def test_insert_b2_anchored_positions():
    q = _q("x2_window_override_methods", "steward_x", "x2_scale_timing_probe", "x2_r2_scale_validation",
           "x2_r2_scale_real", "x2_policy_compare")
    out = [i["id"] for i in sc.insert_b2_anchored(q, sc.queue_entries_b2())]
    assert out == ["x2_window_override_methods", "steward_x", "x2_model_pulls_scale", "x2_family_smoke_mistral_small32",
                   "x2_scale_timing_probe", "x2_r2_scale_validation", "x2_r2_scale_real",
                   "x2_r2_family_mistral_small32_validation", "x2_r2_family_mistral_small32_real", "x2_policy_compare"]
    with pytest.raises(ValueError):
        sc.insert_b2_anchored(_q("x2_scale_timing_probe"), sc.queue_entries_b2())
    with pytest.raises(ValueError):
        sc.insert_b2_anchored(sc.insert_b2_anchored(q, sc.queue_entries_b2()), sc.queue_entries_b2())


def test_queue_entries_b2_commands():
    e = {x["id"]: x for x in sc.queue_entries_b2()}
    assert e["x2_model_pulls_scale"]["cmd"][1] == "x2_model_pulls_scale.py"
    assert e["x2_family_smoke_mistral_small32"]["cmd"][1] == "x2_family_smoke.py"
    real = e["x2_r2_family_mistral_small32_real"]["cmd"]
    assert real[real.index("--smoke-file") + 1] == sc.SMOKE_FILE
    assert real[real.index("--validation-file") + 1] == sc.FAMILY_FILES["validation"]
    assert all("tq.advance()" in x["note"] for x in e.values())
    sc.build_arg_parser().parse_args(real[2:])
    sc.build_arg_parser().parse_args(e["x2_r2_family_mistral_small32_validation"]["cmd"][2:])


def test_queue_insert_b2_pauses_and_always_clears(monkeypatch):
    log = []
    state = {"items": _q("x2_scale_timing_probe", "x2_r2_scale_real")}
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(
        set_pause=lambda reason: log.append(("pause", reason)), clear_pause=lambda: log.append(("clear",)),
        read_queue=lambda: state["items"], write_queue=lambda items: state.update(items=items)))
    sc.queue_insert_b2(log=lambda m: None)
    assert log[0][0] == "pause" and "finally" in log[0][1] and log[-1] == ("clear",) and len(state["items"]) == 6
    log.clear()
    with pytest.raises(ValueError):
        sc.queue_insert_b2(log=lambda m: None)
    assert log[-1] == ("clear",)


# ── register rows ───────────────────────────────────────────────────────────────────────────────────

def test_register_rows_pending_then_compute(tmp_path):
    sys.path.insert(0, str(REPO))
    import analysis.numbers_register as nr
    entries = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    for cid in ("x2-model-pulls-scale", "x2-family-smoke", "x2-family-validation", "H-scale-family-verdict"):
        assert entries[cid]["pending_ok"] and nr._pending(entries[cid], repo=tmp_path)
    assert "mistral-small3.2:24b" not in sc.SCALE_MODELS
    (tmp_path / "results").mkdir()
    md, man = _store(tmp_path, TAG, LIB)
    row = mps.process_tag(TAG, md, {TAG}, lambda r: None, lambda m: None, None,
                          lambda t: {"verify_ok": True}, lambda t: {}, lambda d: 0)
    (tmp_path / nr.X2_MODEL_PULLS_SCALE).write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")
    v = nr.compute_x2_model_pulls_scale(tmp_path)["value"]
    assert v.startswith(f"{TAG}: already present; digests verified") and "template source ollama_library" in v
    rows, _ = _smoke_rows(_bad_on({4}, "text"))
    rows.append(dict(fs.smoke_summary(rows, TAG), record="family_smoke_result"))
    (tmp_path / nr.X2_FAMILY_SMOKE).write_text("".join(json.dumps(r, default=str) + "\n" for r in rows),
                                               encoding="utf-8")
    s = nr.compute_x2_family_smoke(tmp_path)["value"]
    assert s.startswith(f"{TAG}: 4/5 native valid tool calls (pass needs 4): pass") and "t4 text_fallback" in s
