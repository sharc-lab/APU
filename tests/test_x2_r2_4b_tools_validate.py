"""Tests for harness/x2_r2_4b_tools_validate.py and harness/chat_template_source.py. Fakes only; no network, no
process launches."""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import chat_template_source as cts  # noqa: E402
import x2_r2_4b_tools_validate as job  # noqa: E402
import x2_outcome_table as x2ot  # noqa: E402

LIB_PARAMS = ('repeat_penalty                 1\nstop                           "<|im_start|>"\n'
              'stop                           "<|im_end|>"\ntemperature                    0.6')


def test_parse_stop_params():
    assert job.parse_stop_params(LIB_PARAMS) == ["<|im_start|>", "<|im_end|>"]
    assert job.parse_stop_params(None) == []


def test_build_modelfile_exact():
    mf = job.build_modelfile(r"C:\m\x.gguf", "\n{{ .System }}", ["<|im_start|>", "<|im_end|>"])
    assert mf == ('FROM C:\\m\\x.gguf\nTEMPLATE """\n{{ .System }}"""\nPARAMETER stop "<|im_start|>"\n'
                  'PARAMETER stop "<|im_end|>"\n')
    with pytest.raises(ValueError):
        job.build_modelfile("x", 'a """ b', [])


def _manifest(model_digest, template_digest=None):
    layers = [{"mediaType": "application/vnd.ollama.image.model", "digest": model_digest}]
    if template_digest:
        layers.append({"mediaType": "application/vnd.ollama.image.template", "digest": template_digest})
    return {"layers": layers}


class _Store:
    """A fake Ollama store on disk + a fake /api/show + a fake `ollama create`."""

    def __init__(self, tmp_path, template, gguf_bytes=b"GGUF-fake-weights", tools_present=None):
        self.dir = tmp_path / "models"
        self.gguf = tmp_path / "qwen3-4b.gguf"
        self.gguf.write_bytes(gguf_bytes)
        self.digest = "sha256:" + hashlib.sha256(gguf_bytes).hexdigest()
        self.template = template
        self.created = []
        self._write("qwen3-4b-2507", _manifest(self.digest))
        self.shows = {"qwen3:8b": {"template": template, "parameters": LIB_PARAMS}}
        if tools_present:
            self._write("qwen3-4b-2507-tools", _manifest(*tools_present[0]))
            self.shows["qwen3-4b-2507-tools"] = tools_present[1]

    def _write(self, tag, man):
        p = cts.manifest_path(self.dir, tag)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(man), encoding="utf-8")

    def post(self, path, body, timeout=60):
        assert path == "/api/show"
        if body["model"] not in self.shows:
            raise RuntimeError("404")
        return self.shows[body["model"]]

    def run(self, argv, timeout):
        self.created.append(argv)
        mf = Path(argv[-1]).read_text(encoding="utf-8")
        assert mf.startswith(f"FROM {self.gguf}\n")
        self._write("qwen3-4b-2507-tools", _manifest(self.digest, "sha256:" + hashlib.sha256(
            self.template.encode()).hexdigest()))
        self.shows["qwen3-4b-2507-tools"] = {"template": self.template, "parameters": LIB_PARAMS,
                                             "capabilities": ["completion", "tools", "thinking"]}
        return types.SimpleNamespace(returncode=0, stdout="success", stderr="")


@pytest.fixture
def lib_template(monkeypatch):
    t = "\n{{- range .Messages }}<|im_start|>{{ .Role }}{{ end }}"
    monkeypatch.setattr(job, "LIBRARY_TEMPLATE_SHA256", hashlib.sha256(t.encode()).hexdigest())
    return t


def _wire(monkeypatch, store):
    monkeypatch.setattr(job, "_post", store.post)
    monkeypatch.setattr(job, "_run_hidden", store.run)
    monkeypatch.setitem(x2ot.MODEL_MAP, "qwen3-4b-2507", ("qwen3-4b-2507", str(store.gguf)))
    monkeypatch.setattr(cts, "gguf_template_sha256", lambda p: None)
    return types.SimpleNamespace(_resolve_ollama_exe_for_serve=lambda: "ollama.exe")


def test_creates_tag_and_records_modelfile(tmp_path, monkeypatch, lib_template):
    store = _Store(tmp_path, lib_template)
    hc = _wire(monkeypatch, store)
    rows = []
    rec = job.ensure_tools_tag(rows.append, tmp_path / "v2c.jsonl", log=lambda m: None, hc=hc, models_dir=store.dir)
    assert len(store.created) == 1 and store.created[0][:3] == ["ollama.exe", "create", "qwen3-4b-2507-tools"]
    assert rec["action"] == "created" and rec["verify_after"]["ok"] and rec["source_tag_untouched"]
    assert rec["gguf_matches_source_blob"] and rec["template_sha256"] == job.LIBRARY_TEMPLATE_SHA256
    assert rec["stops"] == ["<|im_start|>", "<|im_end|>"]
    assert rec["modelfile"] == Path(rec["modelfile_path"]).read_text(encoding="utf-8")
    assert f'TEMPLATE """{lib_template}"""' in rec["modelfile"]
    # 2026-10-08: the Modelfile on disk must keep LF line ends (CRLF would end up inside the created TEMPLATE).
    on_disk = Path(rec["modelfile_path"]).read_bytes()
    assert b"\r" not in on_disk and on_disk == rec["modelfile"].encode("utf-8")
    assert rows == [rec]


def test_idempotent_when_already_verified(tmp_path, monkeypatch, lib_template):
    tpl_digest = "sha256:" + hashlib.sha256(lib_template.encode()).hexdigest()
    gguf_bytes = b"GGUF-fake-weights"
    digest = "sha256:" + hashlib.sha256(gguf_bytes).hexdigest()
    store = _Store(tmp_path, lib_template, gguf_bytes,
                   tools_present=((digest, tpl_digest), {"template": lib_template, "parameters": LIB_PARAMS}))
    rec = job.ensure_tools_tag(lambda r: None, tmp_path / "v2c.jsonl", log=lambda m: None,
                               hc=_wire(monkeypatch, store), models_dir=store.dir)
    assert store.created == [] and rec["action"] == "already_present_verified"


def test_recreates_when_existing_tag_is_bare(tmp_path, monkeypatch, lib_template):
    gguf_bytes = b"GGUF-fake-weights"
    digest = "sha256:" + hashlib.sha256(gguf_bytes).hexdigest()
    store = _Store(tmp_path, lib_template, gguf_bytes,
                   tools_present=((digest, None), {"template": "jinja from gguf", "parameters": None}))
    rec = job.ensure_tools_tag(lambda r: None, tmp_path / "v2c.jsonl", log=lambda m: None,
                               hc=_wire(monkeypatch, store), models_dir=store.dir)
    assert len(store.created) == 1 and not rec["verify_before"]["ok"] and rec["verify_after"]["ok"]


def test_refuses_when_gguf_is_not_the_source_blob(tmp_path, monkeypatch, lib_template):
    store = _Store(tmp_path, lib_template)
    store.gguf.write_bytes(b"different weights")
    with pytest.raises(RuntimeError, match="sha256"):
        job.ensure_tools_tag(lambda r: None, tmp_path / "v2c.jsonl", log=lambda m: None,
                             hc=_wire(monkeypatch, store), models_dir=store.dir)
    assert store.created == []


def test_refuses_when_library_template_changed(tmp_path, monkeypatch, lib_template):
    store = _Store(tmp_path, lib_template + " ")
    with pytest.raises(RuntimeError, match="template sha256"):
        job.ensure_tools_tag(lambda r: None, tmp_path / "v2c.jsonl", log=lambda m: None,
                             hc=_wire(monkeypatch, store), models_dir=store.dir)


def test_main_creates_validates_and_advances_once(tmp_path, monkeypatch):
    import socket
    notes, calls = [], []
    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=lambda: calls.append("start"), stop_ollama_server=lambda: calls.append("stop"),
        wait_for_ollama_ready=lambda timeout_s=0: True,
        require_host=lambda h: {"hw_id": "evo-x2", "interactive_guard": False},
        enforce_or_record_interactive_session=lambda cfg: {}))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
    monkeypatch.setattr(job, "ensure_tools_tag", lambda emit, out, log=None, hc=None: calls.append("create"))
    import x2_r2_agent as ra

    def fake_main(argv, advance=True):
        calls.append(("validate", tuple(argv), advance))
        return "completed"
    monkeypatch.setattr(ra, "main", fake_main)
    monkeypatch.setattr(job, "gates_summary", lambda out: "v2c gates passed for qwen3-4b-2507-tools")
    out = tmp_path / "x2_r2_validation_v2c.jsonl"
    note = job.main(["--out", str(out)])
    assert calls[:3] == ["start", "create", "stop"]
    assert calls[3] == ("validate", ("--mode", "validation", "--call2-tools", "off", "--models",
                                     "qwen3-4b-2507-tools", "--out", str(out)), False)
    assert notes == [note] and note.startswith("completed;") and "gates passed" in note


def test_main_advances_with_stopped_note_on_create_failure(tmp_path, monkeypatch):
    import socket
    notes = []
    monkeypatch.setattr(socket, "gethostname", lambda: "EVO-X2")
    monkeypatch.setitem(sys.modules, "host_config", types.SimpleNamespace(
        start_ollama_server=lambda: None, stop_ollama_server=lambda: None, wait_for_ollama_ready=lambda timeout_s=0: True,
        require_host=lambda h: {"hw_id": "evo-x2", "interactive_guard": False},
        enforce_or_record_interactive_session=lambda cfg: {}))
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))

    def boom(*a, **k):
        raise RuntimeError("verification failed")
    monkeypatch.setattr(job, "ensure_tools_tag", boom)
    job.main(["--out", str(tmp_path / "o.jsonl")])
    assert len(notes) == 1 and notes[0].startswith("stopped:")


def test_gates_summary_on_failing_file(tmp_path):
    import x2_r2_agent as ra
    p = tmp_path / "v2c.jsonl"
    p.write_text(json.dumps({"record": "run_start", "mode": "validation", "call2_mode": "off"}) + "\n", encoding="utf-8")
    s = job.gates_summary(p)
    assert s.startswith("v2c gates FAILED") and "STOP" not in s and ra.TASK_TOOL_THRESHOLD == 0.9


# ── chat_template_source ───────────────────────────────────────────────────────────────────────────

class TestChatTemplateSource:
    def test_classify(self):
        assert cts.classify_manifest("qwen3:8b", _manifest("sha256:m", "sha256:t"))["chat_template_source"] == \
            "ollama_library"
        bare = cts.classify_manifest("qwen3-4b-2507", _manifest("sha256:m"))
        assert bare["chat_template_source"] == "bare_gguf_ollama_create" and bare["template_layer_sha256"] is None
        tools = cts.classify_manifest("qwen3-4b-2507-tools:latest", _manifest("sha256:m", "sha256:t"))
        assert tools["chat_template_source"] == "ollama_create_library_template"
        assert tools["template_layer_sha256"] == "t"

    def test_ollama_tag_template_from_disk(self, tmp_path, monkeypatch):
        cts.ollama_tag_template.cache_clear()
        p = cts.manifest_path(tmp_path, "qwen3-30b-a3b-2507")
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(_manifest("sha256:abc")), encoding="utf-8")
        monkeypatch.setattr(cts, "gguf_template_sha256", lambda path: "gguf-sha" if path.endswith("sha256-abc") else None)
        got = cts.ollama_tag_template("qwen3-30b-a3b-2507", str(tmp_path))
        assert got["chat_template_source"] == "bare_gguf_ollama_create" and got["chat_template_sha256"] == "gguf-sha"
        missing = cts.ollama_tag_template("nope:1", str(tmp_path))
        assert missing["chat_template_source"] is None and missing["chat_template_evidence"].startswith("unavailable")
        cts.ollama_tag_template.cache_clear()

    def test_source_for_row_recorded_and_derived(self):
        mm = {"qwen3-4b-2507": ("qwen3-4b-2507", "C:/g4.gguf"), "qwen3-8b": ("qwen3:8b", "C:/g8.gguf")}
        facts = cts.facts_from_probe([
            {"record": "ollama_tag_template", "tag": "qwen3-4b-2507:latest", "chat_template_source": "bare_gguf_ollama_create",
             "chat_template_sha256": "s4"},
            {"record": "ollama_tag_template", "tag": "qwen3:8b", "chat_template_source": "ollama_library",
             "chat_template_sha256": "s8"},
            {"record": "gguf_template", "gguf": "C:/g4.gguf", "chat_template_sha256": "g4"}])
        r = cts.source_for_row({"model_id": "qwen3-4b-2507", "config": "ollama_default"}, mm, facts)
        assert (r["chat_template_source"], r["chat_template_sha256"], r["derived"]) == ("bare_gguf_ollama_create", "s4", True)
        r = cts.source_for_row({"model_id": "qwen3-4b-2507", "config": "llama_server"}, mm, facts)
        assert (r["chat_template_source"], r["chat_template_sha256"]) == ("gguf_embedded_llama_server", "g4")
        r = cts.source_for_row({"model_id": "qwen3-8b", "config": "ollama_default",
                                "chat_template_source": "ollama_library", "chat_template_sha256": "x"}, mm, facts)
        assert r == {"chat_template_source": "ollama_library", "chat_template_sha256": "x", "derived": False}
