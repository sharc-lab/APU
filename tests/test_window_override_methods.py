"""harness/window_override_methods.py (operator plan v2 Part A5): log parsing on Ollama 0.34.4-shaped lines, the server
environment (inherited OLLAMA_* removed, own port), the fixed method sequence against a fake Ollama (request option,
Modelfile tag created and deleted, OLLAMA_CONTEXT_LENGTH on the job's own server, restarts), the summary, the refusals,
and main() with fakes (queue advance exactly once with --queue, never without). No network, no process launches."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import window_override_methods as wom  # noqa: E402

INFERENCE_LINE = ('time=2026-10-09T00:07:08.572-07:00 level=INFO source=types.go:42 msg="inference compute" id=0 '
                  'filter_id=0 library=CUDA compute=8.9 name=CUDA0 description="NVIDIA GeForce RTX 4070 Laptop GPU" '
                  'libdirs=ollama,cuda_v13 driver=13.3 pci_id=0000:01:00.0 type=discrete total="8.0 GiB" '
                  'available="6.9 GiB"')
VRAM_LINE = ('time=2026-10-01T14:09:50.185-07:00 level=INFO source=routes.go:2058 msg="vram-based default context" '
             'total_vram="36.2 GiB" default_num_ctx=32768')
CONFIG_LINE = ('time=2026-10-13T10:00:00.000Z level=INFO source=routes.go:1500 msg="server config" '
               'env="map[OLLAMA_CONTEXT_LENGTH:6144 OLLAMA_DEBUG:DEBUG OLLAMA_HOST:http://127.0.0.1:11500]"')


def test_parse_server_log_reads_config_gpu_and_default_lines():
    p = wom.parse_server_log("\n".join([CONFIG_LINE, INFERENCE_LINE, VRAM_LINE]))
    assert p["config_ollama_context_length"] == "6144" and p["debug_on"]
    assert p["vram_based_default"] == [{"total_vram": "36.2 GiB", "default_num_ctx": 32768}]
    g = p["inference_compute"][0]
    assert g["library"] == "CUDA" and g["type"] == "discrete" and g["total"] == "8.0 GiB" and g["available"] == "6.9 GiB"
    assert g["description"] == "NVIDIA GeForce RTX 4070 Laptop GPU"


def test_parse_ctx_priority_and_values():
    text = ("llama_context: n_ctx         = 12288\n"
            "slot update_slots: id  0 | task 0 | new prompt, n_ctx_slot = 12288, n_keep = 5, task.n_tokens = 20\n"
            'msg="starting llama-server" cmd="llama-server.exe --ctx-size 12288 --keep 4"')
    c = wom.parse_ctx(text)
    assert c["log_ctx"] == 12288 and c["log_ctx_source"] == "n_ctx_slot"
    assert c["values"]["ctx_size_arg"] == [12288] and c["values"]["llama_context_n_ctx"] == [12288]
    assert wom.parse_ctx("nothing here")["log_ctx"] is None


def test_child_env_strips_inherited_ollama_vars_and_sets_own_port():
    env, stripped = wom.child_env({"PATH": "x", "OLLAMA_CONTEXT_LENGTH": "999", "ollama_host": "y"}, 11500,
                                  r"C:\models", {"OLLAMA_CONTEXT_LENGTH": "6144"})
    assert stripped == ["OLLAMA_CONTEXT_LENGTH", "ollama_host"]
    assert env == {"PATH": "x", "OLLAMA_HOST": "127.0.0.1:11500", "OLLAMA_MODELS": r"C:\models", "OLLAMA_DEBUG": "1",
                   "OLLAMA_CONTEXT_LENGTH": "6144"}
    env2, _ = wom.child_env({"OLLAMA_CONTEXT_LENGTH": "999"}, 11500, "m", None)
    assert "OLLAMA_CONTEXT_LENGTH" not in env2


def test_values_are_distinct_and_tag_is_new_namespace():
    vals = {wom.V_REQUEST, wom.V_MODELFILE, wom.V_ENV}
    assert len(vals) == 3 and not vals & {4096, 8192, 32768, 131072}
    assert wom.DERIVED_TAG != wom.BASE_MODEL and wom.DERIVED_TAG.startswith("apu-wom-")
    assert wom.modelfile_text() == "FROM llama3.1:8b\nPARAMETER num_ctx 10240\n"
    assert wom.PORT != 11434


# ── fake Ollama ─────────────────────────────────────────────────────────────────────────────────────

class FakeOllama:
    """One fake 'store' shared by every server instance: tags survive restarts; the loaded context follows Ollama's
    precedence as modelled here (request num_ctx > Modelfile num_ctx > OLLAMA_CONTEXT_LENGTH > default)."""

    def __init__(self, default_ctx=131072, version="0.34.4", tags=("llama3.1:8b",)):
        self.default_ctx, self.ver = default_ctx, version
        self.tags_ = {t: {"digest": "sha-" + t} for t in tags}
        self.params = {}
        self.loaded = {}
        self.server = None


class FakeServer:
    def __init__(self, fo: FakeOllama, log_dir: Path):
        self.fo, self.log_dir = fo, log_dir
        self.exe, self.port = "ollama.exe", 11500
        self.proc = None
        self.n = 0
        self.extra = None
        self.log_path = None
        fo.server = self

    def start(self, extra=None, label="plain"):
        self.n += 1
        self.extra = dict(extra or {})
        self.proc = object()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"serve_{self.n}_{label}.log"
        self.log_path.write_text(f'msg="server config" env="map[OLLAMA_CONTEXT_LENGTH:'
                                 f'{self.extra.get("OLLAMA_CONTEXT_LENGTH", "")} OLLAMA_DEBUG:DEBUG]"\n'
                                 + INFERENCE_LINE + "\n", encoding="utf-8")
        self.fo.loaded = {}
        return {"instance": self.n, "label": label, "pid": 1000 + self.n, "env_extra": self.extra,
                "inherited_ollama_env_removed": [], "log_file": self.log_path.name}

    def stop(self, wait_s=20, sleep=None):
        self.proc = None
        self.fo.loaded = {}
        return {"pid": 1000 + self.n, "killed": True, "port_free": True}

    def log_size(self):
        return self.log_path.stat().st_size

    def read_log(self, start=0, end=None):
        data = self.log_path.read_bytes()[start:end]
        return data.decode()

    def cli_env(self):
        return {"OLLAMA_HOST": "127.0.0.1:11500"}

    def _w(self, line):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class FakeApi:
    def __init__(self, fo: FakeOllama):
        self.fo = fo
        self.chats = []

    def ready(self, **k):
        return True

    def version(self):
        return self.fo.ver

    def tags(self):
        return list(self.fo.tags_)

    def tag_digest(self, model):
        return (self.fo.tags_.get(model) or {}).get("digest")

    def show(self, model):
        return {"parameters": f"num_ctx {self.fo.params[model]}" if model in self.fo.params else ""}

    def ps(self):
        return {"models": [{"name": m, "context_length": c, "size": 1, "size_vram": 1} for m, c in self.fo.loaded.items()]}

    def post(self, path, body, timeout=0):
        if path == "/api/generate" and body.get("keep_alive") == 0:
            self.fo.loaded.pop(body["model"], None)
            return 200, {}, None, 0.0
        assert path == "/api/chat"
        self.chats.append(body)
        m = body["model"]
        if m not in self.fo.tags_:
            return 404, None, "model not found", 0.0
        srv = self.fo.server
        ctx = (body["options"].get("num_ctx") or self.fo.params.get(m)
               or int(srv.extra.get("OLLAMA_CONTEXT_LENGTH") or 0) or self.fo.default_ctx)
        self.fo.loaded = {m: ctx}
        srv._w(f"new prompt, n_ctx_slot = {ctx}, n_keep = 5, task.n_tokens = 20")
        srv._w('[GIN] 2026/10/13 - 10:00:01 | 200 |  1.0s | 127.0.0.1 | POST     "/api/chat"')
        return 200, {"prompt_eval_count": 20}, None, 0.01


def fake_run_factory(fo: FakeOllama, calls: list, fail_create=False):
    def run(argv, **k):
        calls.append(argv)
        if argv[1] == "create":
            if fail_create:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")
            mf = Path(argv[4]).read_text(encoding="utf-8")
            assert mf == wom.modelfile_text() and k["env"]["OLLAMA_HOST"] == "127.0.0.1:11500"
            fo.tags_[argv[2]] = {"digest": "sha-new"}
            fo.params[argv[2]] = int(mf.split("num_ctx ")[1])
            return types.SimpleNamespace(returncode=0, stdout="success", stderr="")
        if argv[1] == "rm":
            fo.tags_.pop(argv[2], None)
            fo.params.pop(argv[2], None)
            return types.SimpleNamespace(returncode=0, stdout="deleted", stderr="")
        raise AssertionError(argv)
    return run


def _run(tmp_path, **kw):
    fo = FakeOllama(**{k: v for k, v in kw.items() if k in ("default_ctx", "version", "tags")})
    srv, api, calls, rows = FakeServer(fo, tmp_path / "logs"), FakeApi(fo), [], []
    run = fake_run_factory(fo, calls, fail_create=kw.get("fail_create", False))
    summ = None
    err = None
    try:
        summ = wom.run_methods(srv, api, rows.append, log=lambda m: None, run=run, sleep=lambda s: None,
                               tmp_dir=str(tmp_path))
    except Exception as e:
        err = e
    return fo, rows, calls, summ, err


def test_sequence_measures_all_three_methods_and_cleans_up(tmp_path):
    fo, rows, calls, _, err = _run(tmp_path)
    assert err is None
    rows.append({"record": "run_end"})
    s = wom.summarize(rows)
    assert s["valid"], s["problems"]
    assert s["baseline"]["ps"] == 131072
    assert s["request"]["applied"] is True and s["request"]["persists_across_restart"] is False
    assert s["request"]["after_restart"]["ps"] == 131072
    m = s["modelfile"]
    assert m["applied"] and m["persists_across_restart"] and m["base_tag_unchanged"] and m["tag_created"] and m["tag_deleted"]
    e = s["env"]
    assert e["applied"] and e["persists_restart_same_env"] is True and e["persists_restart_without_env"] is False
    assert all(v is True for v in s["log_agrees_with_ps"].values())
    assert s["inference_compute"][0]["library"] == "CUDA"
    assert [c[1] for c in calls] == ["create", "rm"]
    assert wom.DERIVED_TAG not in fo.tags_ and "llama3.1:8b" in fo.tags_
    servers = [r for r in rows if r["record"] == "wom_server"]
    assert [sv["env_extra"] for sv in servers] == [{}, {}, {}, {"OLLAMA_CONTEXT_LENGTH": "6144"},
                                                   {"OLLAMA_CONTEXT_LENGTH": "6144"}, {}]
    assert rows[-2]["record"] == "wom_stop"                      # the own server is stopped at the end
    deletes = [r for r in rows if r.get("record") == "wom_tag" and r["action"] == "delete"]
    assert deletes[0]["base_digest_unchanged"] is True and deletes[0]["tag_listed_after"] is False


def test_refuses_an_existing_derived_tag_and_never_touches_it(tmp_path):
    fo, rows, calls, _, err = _run(tmp_path, tags=("llama3.1:8b", wom.DERIVED_TAG))
    assert isinstance(err, RuntimeError) and "already exists" in str(err)
    assert calls == [] and wom.DERIVED_TAG in fo.tags_


def test_failed_create_stops_without_deleting(tmp_path):
    fo, rows, calls, _, err = _run(tmp_path, fail_create=True)
    assert isinstance(err, RuntimeError) and [c[1] for c in calls] == ["create"]


def test_wrong_version_refuses(tmp_path):
    _, rows, calls, _, err = _run(tmp_path, version="0.33.2")
    assert isinstance(err, RuntimeError) and "0.34.4" in str(err) and calls == []


def test_finally_deletes_the_tag_when_the_run_stops_after_creating_it(tmp_path, monkeypatch):
    real = wom.measure

    def boom(api, server, method, phase, *a, **k):
        if method == "modelfile" and phase == "after_restart":
            raise RuntimeError("server died")
        return real(api, server, method, phase, *a, **k)
    monkeypatch.setattr(wom, "measure", boom)
    fo, rows, calls, _, err = _run(tmp_path)
    assert isinstance(err, RuntimeError)
    assert [c[1] for c in calls] == ["create", "rm"] and wom.DERIVED_TAG not in fo.tags_
    assert any(r.get("cleanup_in_finally") for r in rows if r.get("record") == "wom_tag")


def test_summary_invalid_when_baseline_equals_an_override(tmp_path):
    _, rows, _, _, _ = _run(tmp_path, default_ctx=wom.V_ENV)
    rows.append({"record": "run_end"})
    s = wom.summarize(rows)
    assert not s["valid"] and any("equals an override" in p for p in s["problems"])


# ── main ────────────────────────────────────────────────────────────────────────────────────────────

def _deps(tmp_path, fo=None, **over):
    fo = fo or FakeOllama()
    calls = []
    d = {"hostname": "EVO-X2", "host_defaults": lambda h: {"hw_id": "evo-x2", "ollama_exe": "ollama.exe",
                                                          "models_dir": r"C:\m"},
         "server": FakeServer(fo, tmp_path / "logs"), "api": FakeApi(fo),
         "default_api": types.SimpleNamespace(ps=lambda: {"error": "refused", "status": None}),
         "gpu": lambda: {"video_controller": {"Name": "fake"}}, "port_in_use": lambda p: False,
         "run": fake_run_factory(fo, calls), "sleep": lambda s: None, "interactive": lambda cfg: {}}
    d.update(over)
    return d, fo, calls


def test_main_queue_advances_once_and_writes_every_record(tmp_path, monkeypatch):
    notes = []
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
    deps, fo, calls = _deps(tmp_path)
    out = tmp_path / "x2_window_override_methods_v1.jsonl"
    note = wom.main(["--out", str(out), "--queue"], deps=deps)
    assert note == "completed" and notes == ["completed"]
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    recs = [r["record"] for r in rows]
    assert recs[0] == "run_start" and recs[1] == "gpu_readonly" and recs[-1] == "run_end"
    assert recs.count("wom_step") == 9 and recs.count("wom_server") == 6
    assert wom.summarize(rows)["valid"]


def test_main_without_queue_never_advances(tmp_path, monkeypatch):
    notes = []
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
    deps, _, _ = _deps(tmp_path, hostname="RITZLAPTOP",
                       host_defaults=lambda h: {"hw_id": "blade", "ollama_exe": "o.exe", "models_dir": "m"})
    assert wom.main(["--out", str(tmp_path / "blade_window_override_methods_v1.jsonl")], deps=deps) == "completed"
    assert notes == []


@pytest.mark.parametrize("over,needle", [
    ({"port_in_use": lambda p: True}, "port 11500 is in use"),
    ({"default_api": types.SimpleNamespace(ps=lambda: {"models": [{"name": "qwen3:8b"}]})}, "holds a model"),
    ({"hostname": "SOMEWHERE"}, "unknown host"),
])
def test_main_refusals_advance_with_a_stopped_note(tmp_path, monkeypatch, over, needle):
    notes = []
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))
    deps, fo, calls = _deps(tmp_path, **over)
    note = wom.main(["--out", str(tmp_path / "o.jsonl"), "--queue"], deps=deps)
    assert note.startswith("stopped:") and needle in note and notes == [note] and calls == []


def test_main_interactive_guard_on_t2s_halts(tmp_path, monkeypatch):
    notes = []
    monkeypatch.setitem(sys.modules, "t2s_queue", types.SimpleNamespace(advance=notes.append))

    def guard(cfg):
        raise SystemExit("another interactive session is logged in: x")
    deps, _, calls = _deps(tmp_path, hostname="EVO-T2S", interactive=guard)
    note = wom.main(["--out", str(tmp_path / "t2s_window_override_methods_v1.jsonl"), "--queue"], deps=deps)
    assert "another interactive session" in note and notes == [note] and calls == []


def test_gpu_memory_readonly_runs_only_read_queries():
    seen = []

    def run(argv, **k):
        seen.append(argv)
        return types.SimpleNamespace(returncode=0, stdout='{"Name": "GPU"}' if "Win32_VideoController" in " ".join(argv)
                                     else "123")
    out = wom.gpu_memory_readonly(run=run)
    assert out["video_controller"] == {"Name": "GPU"} and out["visible_ram_bytes"] == "123"
    joined = " ".join(" ".join(a) for a in seen)
    for verb in ("Set-", "New-Item", "Remove-", "reg add", "bcdedit", "Stop-"):
        assert verb not in joined


def test_register_row_computes_per_host(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    import numbers_register as nr
    entry = next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == "window-override-methods")
    assert entry["pending_ok"] and entry["data_files"] == [nr.WOM_FILE_X2]
    _, rows, _, _, err = _run(tmp_path / "run")
    assert err is None
    rows.append({"record": "run_end"})
    f = tmp_path / nr.WOM_FILE_X2
    f.parent.mkdir(parents=True)
    f.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    r = nr.compute_window_override_methods(tmp_path)
    assert r["value"].startswith("evo-x2: baseline 131072 (log 131072); request num_ctx 12288: loaded 12288 (log 12288)")
    assert "blade: not run yet" in r["value"] and "evo-t2s: not run yet" in r["value"] and r["n"] == "1 hosts"
    assert "valid True" in r["value"]
