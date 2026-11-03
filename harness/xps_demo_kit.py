"""XPS demo kit, local step (USD 0): the B3 v2 demo comparison on an external reviewer's own laptop.

The machine: Zach's Dell XPS (16 GB, Intel, Qwen3-4B). It is NOT configured in this repo (harness/host_config.py
knows evo-x2 and evo-t2s only) and nothing here reaches it. Zach runs this file on the XPS from a clone of the repo;
the operator runs the cloud step (scripts/xps_demo_cloud.py) on the controller from the file this writes. Step-by-step
commands: docs/XPS_DEMO_KIT.md.

Jobs: the B3 v2 tasks (harness/x2_r2_evalset.build_session, workload evalset, guardrails off, design v2), 30 task turns
per session x seeds 20260901, 20260902, 20260903 = 90 jobs per arm, each turn run with x2_r2_evalset.run_turn unchanged
(tool loop, tools on every call, MAX_TOOL_ROUNDS tool-bearing calls then one final call, max_tokens 384, temperature 0,
thinking off) and graded by evalset_env.check_task (end_state_ok).

Arms (local parts only here; every cloud part is the cloud step's):
  a  local_default    the runtime at its own defaults: no window set by the client (num_ctx None), no override.
                      Refused while an override is detected unless --allow-nondefault "<reason>" (recorded).
  b  standard_hybrid  Ollama-style local first + LiteLLM-style cloud fallback + prompt-length check (standard_route):
                      before EVERY local call, if the estimated prompt (x2_r2_agent.prompt_tokens_est, the harness's
                      chars/4 estimate of messages + tools) exceeds the runtime's window, the task goes to the cloud;
                      else local; a hard error (HTTP status other than 200, a connection error, a non-ok outcome)
                      also sends it to the cloud. A silent truncation is not an error, so it stays local.
                      Here a diverted task is DEFERRED: no local call is made for it, the local transcript and
                      environment carry on without it, and the cloud step redoes it from the pre-task state with the
                      full client-side history (scripts/cloud_redo.rebuild). Declared difference from a live client:
                      the fallback is per task, not per call (a call tripped mid-turn redoes the whole task).
  c  full_pipeline    route at start + post-run check + rollback/redo, as the repo's pipeline
                      (analysis/hybrid_full_pipeline.run_pipeline): the first K = 3 turns run locally; every later turn
                      is routed to the cloud at its start when src/dse/fits_locally.SessionGrowthPredictor (k = 3)
                      projects it past the window (deferred, as above). Every local task gets the post-run check
                      (src/dse/post_run_check.check_record with l3_invariants.CONFIG_TUNED: L1 recorded call errors,
                      L1 PROXY over_loaded_window, L2 PROXY prompt_eval_count below the estimate sent, L3 task-rule
                      invariants; on llama-server also the real L1 response class, xruntime_llamacpp.call_class). A
                      FAIL is recorded as "escalate": the cloud step rolls back to the pre-task snapshot and redoes the
                      task on the cloud (scripts/cloud_redo.rollback_env + live_turn), exactly as
                      scripts/post_run_check_cloud.py. The pipeline may set its own window (--pipeline-window); default:
                      the runtime's default window.
  d  all_cloud        no local part (cloud step only).

Declared approximation (b and c, same as the replay rows): the local session continues on its own transcript and
environment; a task sent to the cloud does not change what later local turns see.

Runtime record and default assertion (run_start "runtime_config"): runtime (ollama / llamacpp / openvino), its version,
its default window and where that was read, the model, and every override found: process, user and machine
environment variables with the prefixes in OVERRIDE_ENV_PREFIXES (OLLAMA_* except the address and path variables in
BENIGN_ENV, GGML_*, LLAMA_ARG_*, OV_* / OPENVINO_* / OVMS_*, SYCL_*, ZE_*, device selectors), an Ollama Modelfile
parameter (num_ctx and the rest of /api/show "parameters"), llama-server launch flags other than the model / address
flags (read from the listening process's command line), and OpenVINO Model Server config keys that set a device,
precision, KV cache or batching (OVMS_CONFIG_OVERRIDE_KEYS). Arm a refuses while any override is present, or while the
settings could not be read (llama-server command line not readable), unless --allow-nondefault is given.

Statistics: 1 warm-up call (excluded) before the measured sessions; each session starts with the model unloaded where
the runtime allows it (Ollama). Output: one JSONL per invocation (run_start, r2a_turn, xps_deferred, r2a_session,
run_end); its format is in docs/XPS_DEMO_KIT.md. No cloud call, no API key, no network beyond the local runtime's URL.

Subprocesses (version commands, the llama-server command line) go through harness/proc_util.py (no visible window).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in (_here, _here.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import x2_r2_evalset as EV  # noqa: E402
import x2_r2_agent as ag  # noqa: E402
from proc_util import run_hidden  # noqa: E402

HARNESS = "xps_demo_kit"
KIT_VERSION = "xps_demo_kit_v1"
SEEDS = (20260901, 20260902, 20260903)
DESIGN = "v2"
WORKLOAD = "evalset"
GUARDRAILS = False
ARMS = {"a": "local_default", "b": "standard_hybrid", "c": "full_pipeline"}
CLOUD_ARM_D = "all_cloud"
RUNTIMES = ("ollama", "llamacpp", "openvino")
DEFAULT_URL = {"ollama": "http://127.0.0.1:11434", "llamacpp": "http://127.0.0.1:8080",
               "openvino": "http://127.0.0.1:8000"}
CHAT_PATH = {"llamacpp": "/v1/chat/completions", "openvino": "/v3/chat/completions"}
ROUTE_K = 3
DEFER_RECORD = "xps_deferred"

OVERRIDE_ENV_PREFIXES = ("OLLAMA_", "GGML_", "LLAMA_ARG_", "LLAMA_", "OV_", "OPENVINO_", "OVMS_", "SYCL_", "ZE_",
                         "ONEAPI_DEVICE_SELECTOR", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "IPEX_LLM_")
BENIGN_ENV = ("OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_ORIGINS")
LLAMACPP_ALLOWED_FLAGS = ("-m", "--model", "--host", "--port", "--jinja", "-a", "--alias", "--log-file",
                          "--log-disable", "--metrics", "--no-webui", "-hf", "--hf-repo", "--api-key")
LLAMACPP_FLAGS_WITH_VALUE = ("-m", "--model", "--host", "--port", "-a", "--alias", "--log-file", "-hf", "--hf-repo",
                             "--api-key")
OVMS_CONFIG_OVERRIDE_KEYS = ("KV_CACHE_PRECISION", "INFERENCE_PRECISION_HINT", "KV_CACHE_GROUP_SIZE",
                             "DYNAMIC_QUANTIZATION_GROUP_SIZE", "cache_size", "max_num_batched_tokens", "max_num_seqs",
                             "enable_prefix_caching", "dynamic_split_fuse", "NUM_STREAMS", "PERFORMANCE_HINT",
                             "plugin_config", "device", "target_device", "MAX_PROMPT_LEN", "max_prompt_len",
                             "GPU_ENABLE_SDPA_OPTIMIZATION", "ENABLE_MMAP")


class Defer(Exception):
    """A task the local step hands to the cloud step (arm b: over the window or a hard error; arm c: routed)."""

    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail or {}


# ── runtime configuration and the default-settings assertion (pure parts) ──────────────────────────────────────

def env_overrides(envs: dict[str, dict]) -> tuple[list[dict], dict]:
    """envs: {source: {name: value}} (process, user, machine). (overrides, benign) where an override is any variable
    with an OVERRIDE_ENV_PREFIXES prefix that is not in BENIGN_ENV."""
    over, benign = [], {}
    for source, env in envs.items():
        for k, v in sorted((env or {}).items()):
            ku = k.upper()
            if not any(ku.startswith(p) for p in OVERRIDE_ENV_PREFIXES):
                continue
            if ku in BENIGN_ENV:
                benign[f"{source}:{ku}"] = v
            else:
                over.append({"source": f"env:{source}", "key": ku, "value": str(v)[:200]})
    return over, benign


def ollama_modelfile_overrides(show: dict | None) -> list[dict]:
    """Every PARAMETER line of /api/show "parameters" (num_ctx among them) is a setting baked into the model."""
    out = []
    for line in str((show or {}).get("parameters") or "").splitlines():
        parts = line.split(None, 1)
        if parts:
            out.append({"source": "ollama_modelfile", "key": parts[0], "value": (parts[1] if len(parts) > 1 else "")})
    return out


def llamacpp_flag_overrides(cmdline: str | None) -> list[dict] | None:
    """Launch flags other than LLAMACPP_ALLOWED_FLAGS; None when the command line could not be read."""
    if not cmdline:
        return None
    toks = [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', cmdline)]
    out, i = [], 1
    while i < len(toks):
        t = toks[i]
        if t.startswith("-"):
            name = t.split("=", 1)[0]
            takes = name in LLAMACPP_FLAGS_WITH_VALUE
            if name not in LLAMACPP_ALLOWED_FLAGS:
                val = t.split("=", 1)[1] if "=" in t else (toks[i + 1] if i + 1 < len(toks)
                                                            and not toks[i + 1].startswith("-") else "")
                out.append({"source": "llamacpp_cmdline", "key": name, "value": val})
                if "=" not in t and val:
                    i += 1
            elif takes and "=" not in t:
                i += 1
        i += 1
    return out


def ovms_config_overrides(text: str | None) -> list[dict]:
    out = []
    for k in OVMS_CONFIG_OVERRIDE_KEYS:
        m = re.search(rf'["\']?\b{re.escape(k)}\b["\']?\s*[:=]\s*("?[^,\n}}]*)', text or "")
        if m:
            out.append({"source": "ovms_config", "key": k, "value": m.group(1).strip().strip('"')[:120]})
    return out


def assert_default(runtime_config: dict, arm: str, allow_nondefault: str | None) -> None:
    """Arm a only: SystemExit while an override is present or the settings were not readable, unless allowed."""
    if arm != "a":
        return
    problems = [f"{o['source']} {o['key']}={o['value']}" for o in runtime_config.get("overrides") or []]
    problems += [f"not verifiable: {u}" for u in runtime_config.get("unverified") or []]
    if problems and not (allow_nondefault and allow_nondefault.strip()):
        raise SystemExit("refused (arm a, local default): the runtime is not at its own defaults:\n  "
                         + "\n  ".join(problems)
                         + "\nRemove the override(s) and restart the runtime, or rerun with "
                           "--allow-nondefault \"<reason>\" (recorded in run_start). Nothing was run.")


# ── runtime probes (I/O) ─────────────────────────────────────────────────────────────────────────────────────────

def _get_json(url: str, timeout: float = 30.0, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def registry_env() -> dict[str, dict]:
    """User and machine environment from the Windows registry (what a tray-started runtime inherits)."""
    out = {"user": {}, "machine": {}}
    try:
        import winreg  # noqa: PLC0415
    except ImportError:
        return out
    keys = {"user": (winreg.HKEY_CURRENT_USER, r"Environment"),
            "machine": (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")}
    for name, (hive, path) in keys.items():
        try:
            with winreg.OpenKey(hive, path) as k:
                i = 0
                while True:
                    try:
                        n, v, _t = winreg.EnumValue(k, i)
                    except OSError:
                        break
                    out[name][n] = v
                    i += 1
        except OSError:
            pass
    return out


def run_version_cmd(argv: list[str] | None) -> str | None:
    if not argv:
        return None
    try:
        r = run_hidden(argv, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return ((r.stdout or "") + (r.stderr or "")).strip()[:600] or None
    except Exception as e:  # noqa: BLE001 - recorded, never fatal
        return f"version command failed: {type(e).__name__}: {e}"[:300]


def listener_cmdline(port: int) -> str | None:
    """Command line of the process listening on a local TCP port (Windows: netstat + CIM, hidden); None if unread."""
    if os.name != "nt":
        return None
    try:
        r = run_hidden(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=30,
                       stdin=subprocess.DEVNULL)
        pids = []
        for line in (r.stdout or "").splitlines():
            f = line.split()
            if len(f) >= 5 and f[0].upper() == "TCP" and f[3].upper() == "LISTENING" and f[1].endswith(f":{port}"):
                pids.append(int(f[4]))
        if not pids:
            return None
        r = run_hidden(["powershell", "-NoProfile", "-Command",
                        f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pids[0]}').CommandLine"],
                       capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return (r.stdout or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def openvino_packages() -> dict:
    from importlib import metadata  # noqa: PLC0415
    out = {}
    for p in ("openvino", "openvino-genai", "openvino-tokenizers", "optimum-intel"):
        try:
            out[p] = metadata.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    return out


def probe_runtime(runtime: str, base_url: str, model: str, runtime_exe: str | None = None,
                  config_file: str | None = None, declared_default_window: int | None = None,
                  environ: dict | None = None, reg_env: dict | None = None, get_json=None, cmdline_fn=None,
                  version_fn=None) -> dict:
    """The runtime_config record (no model call). Injectable I/O for tests."""
    if runtime not in RUNTIMES:
        raise ValueError(f"runtime must be one of {RUNTIMES}")
    gj = get_json or _get_json
    envs = {"process": dict(os.environ if environ is None else environ)}
    envs.update(registry_env() if reg_env is None else reg_env)
    overrides, benign = env_overrides(envs)
    unverified = []
    cfg = {"runtime": runtime, "base_url": base_url, "model": model, "version": None, "version_source": None,
           "default_window": None, "default_window_source": None, "runtime_exe": runtime_exe}
    if runtime == "ollama":
        v = gj(f"{base_url}/api/version")
        cfg["version"], cfg["version_source"] = (v or {}).get("version"), "GET /api/version"
        show = gj(f"{base_url}/api/show", body={"model": model}) or {}
        cfg["model_details"] = show.get("details")
        cfg["model_info_context_length"] = next((val for k, val in (show.get("model_info") or {}).items()
                                                 if k.endswith(".context_length")), None)
        cfg["modelfile_parameters"] = show.get("parameters")
        overrides += ollama_modelfile_overrides(show)
        cfg["default_window_source"] = "GET /api/ps context_length after the warm-up call (num_ctx not sent)"
    elif runtime == "llamacpp":
        props = gj(f"{base_url}/props") or {}
        dgs = props.get("default_generation_settings") or {}
        cfg["version"] = props.get("build_info")
        cfg["version_source"] = "GET /props build_info"
        cfg["default_window"] = dgs.get("n_ctx") or props.get("n_ctx")
        cfg["default_window_source"] = "GET /props default_generation_settings.n_ctx"
        cfg["props_model_path"] = props.get("model_path")
        port = int(re.search(r":(\d+)", base_url.split("//", 1)[-1]).group(1)) if re.search(
            r":(\d+)", base_url.split("//", 1)[-1]) else 80
        cmd = (cmdline_fn or listener_cmdline)(port)
        cfg["server_cmdline"] = cmd
        flags = llamacpp_flag_overrides(cmd)
        if flags is None:
            unverified.append("llama-server command line could not be read (launch flags unknown)")
        else:
            overrides += flags
    else:   # openvino (OpenVINO Model Server, OpenAI-compatible /v3)
        cfg["openvino_packages"] = openvino_packages()
        cfg["ovms_config_endpoint"] = gj(f"{base_url}/v1/config")
        text = None
        if config_file:
            try:
                text = Path(config_file).read_text(encoding="utf-8", errors="replace")
                cfg["config_file"] = {"path": str(config_file), "text": text[:20000],
                                      "sha1": ag.git_blob_sha1(config_file)}
            except OSError as e:
                unverified.append(f"config file unreadable: {e}")
        else:
            unverified.append("no --runtime-config-file given (device / precision / KV settings not read)")
        overrides += ovms_config_overrides(text)
        cfg["default_window"] = declared_default_window
        cfg["default_window_source"] = "declared by the operator (--declared-default-window); OVMS does not report it"
        if declared_default_window is None:
            unverified.append("default window not declared (--declared-default-window)")
    if runtime_exe:
        vtxt = (version_fn or run_version_cmd)([runtime_exe, "--version"])
        cfg["exe_version_output"] = vtxt
        if cfg["version"] is None and vtxt:
            cfg["version"], cfg["version_source"] = vtxt.splitlines()[0][:200], f"{runtime_exe} --version"
    cfg.update({"overrides": overrides, "benign_env": benign, "unverified": unverified,
                "overrides_present": bool(overrides)})
    return cfg


def host_record() -> dict:
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "processor": platform.processor(),
            "machine": platform.machine(), "python": sys.version.split()[0]}


# ── runtimes (the interface x2_r2_evalset.run_turn uses) ───────────────────────────────────────────────────────

class OllamaKitRuntime(ag.OllamaRuntime):
    """x2_r2_agent.OllamaRuntime at any base URL; recover() only waits (Ollama on the XPS is not ours to restart)."""

    def __init__(self, base="http://127.0.0.1:11434"):
        import t2s_k1_ollama as k1  # noqa: PLC0415
        self.base = base.rstrip("/")
        hostport = self.base.split("//", 1)[-1]
        host, _, port = hostport.partition(":")
        self.client = k1.OllamaClient(host=host, port=int(port or 11434), timeout=ag.CALL_TIMEOUT_S)
        self.n_ctx = None

    def recover(self):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if _get_json(f"{self.base}/api/version", timeout=5):
                return True
            time.sleep(2)
        return False


class OpenAICompatRuntime:
    """llama-server (/v1/chat/completions) or OpenVINO Model Server (/v3/chat/completions), started by the operator.
    Request body and response mapping are harness/xruntime_llamacpp.chat_body / map_response (the model name is sent as
    the alias, which OVMS requires). Each call keeps its llama-server L1 fields for the post-run check."""

    def __init__(self, base, runtime, model, n_ctx=None, post=None):
        import xruntime_llamacpp as XR  # noqa: PLC0415
        self.XR, self.base, self.runtime, self.model = XR, base.rstrip("/"), runtime, model
        self.path, self.n_ctx, self.post = CHAT_PATH[runtime], n_ctx, post or XR._post
        self._events = []

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        if extra:
            raise ValueError("call-2 forced modes are not part of this plan")
        status, data, err, dt = self.post(self.base + self.path, self.XR.chat_body(messages, tools, think,
                                                                                 alias=self.model))
        resp = self.XR.map_response(status, data, err, dt)
        raw = resp.get("raw") or {}
        ev = {"http_status": status, "finish_reason": raw.get("done_reason"),
              "prompt_tokens": raw.get("prompt_eval_count"), "completion_tokens": raw.get("eval_count"),
              "n_ctx": self.n_ctx, "error": (err or "")[:300] or None}
        ev["class"] = (self.XR.call_class(status, ev["finish_reason"], ev["prompt_tokens"], ev["completion_tokens"],
                                          self.n_ctx, self.XR.classify_log(None), err)
                       if self.runtime == "llamacpp" else None)
        self._events.append(ev)
        return resp

    def pop_turn_events(self):
        ev, self._events = self._events, []
        return ev

    def recover(self):
        return False

    def loaded_context(self, model):
        return self.n_ctx

    def unload(self, model, wait_s=30):
        return None   # not possible through this API; recorded as model_unloaded_before_session None


# ── arm wrappers ─────────────────────────────────────────────────────────────────────────────────────────────

def standard_route(prompt_est: int, window: int | None, resp: dict | None = None) -> str:
    """The standard hybrid's decision for one call. Before the call (resp None): "cloud_over_window" if the estimated
    prompt exceeds the window, else "local". After it: "cloud_hard_error" on a hard error, else "local"."""
    if resp is None:
        return "cloud_over_window" if window is not None and prompt_est > window else "local"
    hard = (resp.get("outcome") != "ok" or resp.get("status") not in (200,) or bool(resp.get("error")))
    return "cloud_hard_error" if hard else "local"


class StandardHybridRuntime:
    """Arm b: the prompt-length check before every local call and the hard-error fallback after it (Defer)."""

    def __init__(self, inner, window):
        self.inner, self.window = inner, window

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        est = ag.prompt_tokens_est(messages, tools)
        if standard_route(est, self.window) != "local":
            raise Defer("over_window", {"prompt_tokens_est": est, "window": self.window})
        resp = (self.inner.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                else self.inner.chat(model, messages, num_ctx, tools, think))
        if standard_route(est, self.window, resp) != "local":
            raise Defer("hard_error", {"prompt_tokens_est": est, "window": self.window, "status": resp.get("status"),
                                       "outcome": resp.get("outcome"), "error": str(resp.get("error"))[:300]})
        return resp


def route_at_start(first_turns: list[tuple[int, int]], turn_idx: int, window: int | None, k: int = ROUTE_K) -> dict:
    """Route at start (fit-predictor-v1's rule, as hybrid_full_pipeline.run_pipeline): the first k turns run locally;
    a later turn routes when SessionGrowthPredictor projects it past the window."""
    if window is None or turn_idx <= k or len(first_turns) < k:
        return {"route": False, "why": "first k turns" if turn_idx <= k else "no window"}
    from src.dse.fits_locally import SessionGrowthPredictor  # noqa: PLC0415
    p = SessionGrowthPredictor(k=k).predict({"first_turns": first_turns[:k], "target_turn": turn_idx},
                                            {"window": window})
    return {"route": not p["fits"], **p}


def _strip_grader(row: dict) -> dict:
    """hybrid_full_pipeline.strip_grader: the check never reads the grader's fields."""
    a = str(_here.parent / "analysis")
    if a not in sys.path:
        sys.path.insert(0, a)
    import hybrid_full_pipeline as hfp  # noqa: PLC0415
    return hfp.strip_grader(row)


def post_run_check(row: dict, es, runtime_name: str, l1_events: list[dict] | None = None) -> dict:
    """The post-run check on one local task row: check_record (L1 call errors, L1 PROXY, L2 PROXY, L3 CONFIG_TUNED) on
    the row without the grader's fields, plus on llama-server the real L1 response class of every call."""
    from src.dse import l3_invariants as L3  # noqa: PLC0415
    from src.dse import post_run_check as PRC  # noqa: PLC0415
    rec = _strip_grader(row)
    user = {t.idx: t.user_text for t in es.turns}[row["turn_idx"]]
    d = PRC.check_record(rec, es.system_prompt, user, EV.tools_payload(es.design), L3.CONFIG_TUNED)
    fired = list(d["fired"])
    l1 = None
    if runtime_name == "llamacpp" and l1_events is not None:
        bad = [e.get("class") for e in l1_events if e.get("class") in ("flagged_reject", "silent_cut", "ctx_stop")]
        l1 = {"layer": "L1", "fail": bool(bad), "evidence": bad, "label": "llama-server response class (no log)"}
        if bad:
            fired.append("L1")
    return {"decision": "escalate" if fired else "accept", "fired": fired, "check_seconds": d["check_seconds"],
            "l3_config": "CONFIG_TUNED", "l1_llamacpp": l1,
            "layers": {k: {kk: vv for kk, vv in v.items() if kk in ("fail", "label", "evidence", "failed", "calls")}
                       for k, v in d["layers"].items()}}


# ── the local session driver ─────────────────────────────────────────────────────────────────────────────────

def _calib(row: dict) -> float | None:
    c1 = (row.get("calls") or [{}])[0]
    pec, est = c1.get("prompt_eval_count_info_only"), c1.get("prompt_tokens_est")
    if pec and est:
        r = pec / est
        if ag.CALIB_RATIO_BOUNDS[0] <= r <= ag.CALIB_RATIO_BOUNDS[1]:
            return r
    return None


def run_arm_session(runtime, runtime_name: str, model: str, arm: str, seed: int, window: int | None, emit, log,
                    num_ctx: int | None = None, think=False, max_turns: int | None = None, es=None) -> dict:
    """One B3 v2 session of one arm on the local runtime. Emits r2a_turn / xps_deferred rows and one r2a_session (or
    r2a_session_incomplete). Returns the session summary."""
    es = es or EV.build_session(seed, max_turns, WORKLOAD, GUARDRAILS, model, None, DESIGN)
    arm_id = ARMS[arm]
    attempt = uuid.uuid4().hex[:12]
    base = {"mode": "real", "model_id": model, "arm_id": arm_id, "arm": arm, "seed": seed, "attempt_id": attempt,
            "num_ctx_requested": num_ctx, "think": think, "call2_mode": "off", "harness": HARNESS,
            "workload": WORKLOAD, "guardrails": GUARDRAILS, "design": DESIGN, "turn_protocol": EV.TURN_PROTOCOL[DESIGN],
            "session_code": es.session_code, "runtime": runtime_name, "window": window}
    rt = StandardHybridRuntime(runtime, window) if arm == "b" else runtime
    unloaded = runtime.unload(model)
    messages = [{"role": "system", "content": es.system_prompt}]
    env = copy.deepcopy(es.env0)
    rows, deferred, first_turns, calib = [], [], [], None
    for turn, item in zip(es.turns, es.items):
        t0 = time.monotonic()
        task_id = item.task.task_id if item.task else None
        dbase = {**base, "record": DEFER_RECORD, "turn_idx": turn.idx, "task_id": task_id, "item_kind": item.kind,
                 "evalset_item": item.item_id, "task_type": item.task.ttype if item.task else None}
        if arm == "c":
            r = route_at_start(first_turns, turn.idx, window)
            if r["route"]:
                emit({**dbase, "reason": "route_at_start", "route": r, "local_seconds": time.monotonic() - t0,
                      "ts_utc": ag.utc_iso()})
                deferred.append(turn.idx)
                log(f"  {arm_id} seed={seed} turn {turn.idx} {task_id}: routed at start "
                    f"(projected {r.get('projected_tokens')} > {r.get('ceiling_tokens')})")
                continue
        n_msgs = len(messages)
        try:
            row, env = EV.run_turn(rt, model, num_ctx, think, es, turn, item, env, messages, log, "off")
        except Defer as d:
            del messages[n_msgs:]   # the local transcript carries on without the diverted task
            pop = getattr(runtime, "pop_turn_events", None)
            emit({**dbase, "reason": d.reason, "detail": d.detail, "local_seconds": time.monotonic() - t0,
                  "l1_events": pop() if callable(pop) else None, "ts_utc": ag.utc_iso()})
            deferred.append(turn.idx)
            log(f"  {arm_id} seed={seed} turn {turn.idx} {task_id}: to the cloud ({d.reason})")
            continue
        except ag.SessionIncomplete as e:
            emit(ag.incomplete_row(base, rows, e, len(es.turns)))
            log(f"  {arm_id} seed={seed}: session incomplete at turn {turn.idx} ({e.error!r}); next session")
            return {"incomplete": True, "n_turns": len(rows), "deferred": deferred}
        loaded = runtime.loaded_context(model)
        if turn.idx == 1:
            calib = _calib(row)
        ratio = calib if calib is not None else 1.0
        row.update(base)
        row.update({"record": "r2a_turn", "ts_utc": ag.utc_iso(), "turn_wall_s": time.monotonic() - t0,
                    "loaded_context": loaded, "token_calib_ratio": calib,
                    "transcript_tokens_calibrated": round(row["transcript_tokens_est"] * ratio),
                    "final_call_prompt_tokens_calibrated": round(row["calls"][-1]["prompt_tokens_est"] * ratio),
                    "model_unloaded_before_session": unloaded})
        row["over_loaded_window"] = loaded is not None and row["final_call_prompt_tokens_calibrated"] > loaded
        pop = getattr(runtime, "pop_turn_events", None)
        events = pop() if callable(pop) else None
        if events is not None:
            row["l1_events"] = events
        if len(first_turns) < ROUTE_K:
            first_turns.append((turn.idx, row["final_call_prompt_tokens_calibrated"]))
        if arm == "c" and item.kind == "task":
            chk = post_run_check(row, es, runtime_name, events)
            row["prc_check"] = chk
            row["local_seconds_with_check"] = row["turn_wall_s"] + chk["check_seconds"]
        emit(row)
        rows.append(row)
        log(f"  {arm_id} seed={seed} turn {turn.idx}/{len(es.turns)} {task_id} end_state={row['end_state_ok']} "
            f"ctx~{row['final_call_prompt_tokens_calibrated']}/{loaded} {row['turn_wall_s']:.1f}s"
            + (f" check={row['prc_check']['decision']}" if "prc_check" in row else ""))
    summary = EV.summarize(rows)
    summary.update(base)
    summary.update({"record": "r2a_session", "ts_utc": ag.utc_iso(), "n_turns": len(rows),
                    "n_deferred": len(deferred), "deferred_turns": deferred,
                    "n_local_task_correct": sum(1 for r in rows if r.get("end_state_ok") is True)})
    emit(summary)
    return summary


# ── warm-up and the window ───────────────────────────────────────────────────────────────────────────────────

def warmup(runtime, model: str, think=False) -> dict:
    t0 = time.monotonic()
    resp = runtime.chat(model, [{"role": "user", "content": "Reply with the single word OK."}], None, None, think)
    return {"record": "warmup", "excluded_from_stats": True, "outcome": resp.get("outcome"),
            "status": resp.get("status"), "error": resp.get("error"), "seconds": time.monotonic() - t0,
            "ts_utc": ag.utc_iso()}


def resolve_window(cfg: dict, runtime, model: str) -> int | None:
    """The runtime's default window: Ollama's loaded context after the warm-up (num_ctx not sent), else the probed /
    declared value."""
    if cfg["runtime"] == "ollama":
        w = runtime.loaded_context(model)
        cfg["default_window"] = w
        return w
    return cfg.get("default_window")


def make_runtime(runtime_name: str, base_url: str, model: str, n_ctx: int | None):
    if runtime_name == "ollama":
        return OllamaKitRuntime(base_url)
    return OpenAICompatRuntime(base_url, runtime_name, model, n_ctx=n_ctx)


def run(args, runtime=None, cfg=None, log=print, emit_path: Path | None = None) -> dict:
    """The local step. runtime / cfg injectable for tests (cfg = probe_runtime's record)."""
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"refused: unknown arm(s) {bad}; the local step runs a, b, c (d is cloud only)")
    if cfg is None:
        cfg = probe_runtime(args.runtime, args.base_url, args.model, args.runtime_exe, args.runtime_config_file,
                            args.declared_default_window)
    for a in arms:
        assert_default(cfg, a, args.allow_nondefault)
    out = Path(emit_path or args.out)
    if out.exists():
        raise SystemExit(f"refused: {out} exists (never overwritten); pass a new --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    runtime = runtime or make_runtime(args.runtime, args.base_url, args.model, cfg.get("default_window"))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        def emit(row):
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()

        w = warmup(runtime, args.model)
        window = resolve_window(cfg, runtime, args.model)
        pipe_window = args.pipeline_window or window
        emit({"record": "run_start", "harness": HARNESS, "kit_version": KIT_VERSION, "ts_utc": ag.utc_iso(),
              "mode": "executed", "host": host_record(), "runtime_config": cfg, "arms": arms,
              "arm_names": {a: ARMS[a] for a in arms}, "seeds": list(seeds), "design": DESIGN, "workload": WORKLOAD,
              "guardrails": GUARDRAILS, "max_turns": args.max_turns, "max_tokens": ag.MAX_TOKENS_PER_CALL,
              "temperature": 0, "think": False, "window_default": window, "window_pipeline": pipe_window,
              "route_k": ROUTE_K, "allow_nondefault": args.allow_nondefault,
              "estimate_fn": "x2_r2_agent.prompt_tokens_est (chars/4 of messages + tools)",
              "kit_git_blob": ag.git_blob_sha1(__file__), "evalset_git_blob": ag.git_blob_sha1(EV.__file__),
              "operator_note": args.note})
        emit(w)
        if w["outcome"] != "ok":
            emit({"record": "run_end", "ts_utc": ag.utc_iso(), "stopped": f"warm-up failed: {w.get('error')}"})
            raise SystemExit(f"refused: the warm-up call failed ({w.get('status')} {w.get('error')}); is the runtime "
                             f"up at {args.base_url} with model {args.model}?")
        n = {"sessions": 0, "incomplete": 0}
        for a in arms:
            win = pipe_window if a == "c" else window
            num_ctx = pipe_window if (a == "c" and args.runtime == "ollama" and args.pipeline_window) else None
            for s in seeds:
                res = run_arm_session(runtime, args.runtime, args.model, a, s, win, emit, log, num_ctx=num_ctx,
                                      max_turns=args.max_turns)
                n["sessions"] += 1
                n["incomplete"] += bool(res.get("incomplete"))
        emit({"record": "run_end", "ts_utc": ag.utc_iso(), "stopped": None, **n,
              "runtime_config_end": {"loaded_context": runtime.loaded_context(args.model)}})
    log(f"wrote {out}")
    return {"out": str(out), **n}


# ── offline self-test (fake runtime, no network) ─────────────────────────────────────────────────────────────

class FakeRuntime:
    """Answers every call with no tool call and a fixed JSON answer; reports a fixed loaded window."""

    def __init__(self, window=40960, error_after=None):
        self.window, self.n, self.error_after = window, 0, error_after

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        self.n += 1
        if self.error_after is not None and self.n > self.error_after:
            return {"outcome": "http_error", "status": 400, "error": "context length exceeded", "duration_s": 0.0}
        return {"outcome": "ok", "status": 200, "message": json.dumps({"answer": "done", "source": "tool results"}),
                "tool_calls": None, "prompt_eval_count": ag.prompt_tokens_est(messages, tools), "eval_count": 8,
                "duration_s": 0.0, "raw": {"message": {"content": "x"}, "done_reason": "stop"}}

    def loaded_context(self, model):
        return self.window

    def unload(self, model, wait_s=30):
        return True

    def recover(self):
        return False


def selftest(tmp: Path) -> int:
    ns = argparse.Namespace(arms="a,b,c", seeds=[SEEDS[0]], max_turns=5, runtime="ollama", base_url="fake",
                            model="qwen3:4b", allow_nondefault=None, pipeline_window=None, out=str(tmp / "st.jsonl"),
                            note="selftest", runtime_exe=None, runtime_config_file=None, declared_default_window=None)
    cfg = {"runtime": "ollama", "overrides": [], "unverified": [], "default_window": None}
    res = run(ns, runtime=FakeRuntime(window=2000), cfg=cfg, log=lambda m: None)
    rows = [json.loads(x) for x in open(res["out"], encoding="utf-8")]
    kinds = {r["record"] for r in rows}
    ok = {"run_start", "warmup", "r2a_turn", "r2a_session", "run_end"} <= kinds and DEFER_RECORD in kinds
    print("SELFTEST OK" if ok else f"SELFTEST FAILED: records {sorted(kinds)}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "run"):
        p = sub.add_parser(name)
        p.add_argument("--runtime", choices=RUNTIMES, required=True)
        p.add_argument("--base-url", default=None)
        p.add_argument("--model", required=True, help="the runtime's name for Qwen3-4B, e.g. qwen3:4b")
        p.add_argument("--runtime-exe", default=None, help="path of the runtime executable (version recorded)")
        p.add_argument("--runtime-config-file", default=None, help="OpenVINO Model Server config (scanned)")
        p.add_argument("--declared-default-window", type=int, default=None,
                       help="openvino only: the runtime's default window, as the operator read it")
        if name == "run":
            p.add_argument("--arms", default="a,b,c")
            p.add_argument("--seeds", type=int, nargs="*", default=None)
            p.add_argument("--max-turns", type=int, default=None, help="testing only; leave unset for the 90 jobs")
            p.add_argument("--pipeline-window", type=int, default=None,
                           help="arm c's window (Ollama: sent as num_ctx; other runtimes: the window the server "
                                "was restarted with); default the runtime's default window")
            p.add_argument("--allow-nondefault", default=None,
                           help="arm a only: run although an override was found; the reason is recorded")
            p.add_argument("--out", required=True)
            p.add_argument("--note", default=None)
    st = sub.add_parser("selftest")
    st.add_argument("--tmp", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        import tempfile  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as d:
            return selftest(Path(a.tmp or d))
    a.base_url = a.base_url or DEFAULT_URL[a.runtime]
    if a.cmd == "probe":
        cfg = probe_runtime(a.runtime, a.base_url, a.model, a.runtime_exe, a.runtime_config_file,
                            a.declared_default_window)
        print(json.dumps(cfg, indent=1, sort_keys=True, default=str))
        print("overrides present: " + ("YES (arm a will refuse without --allow-nondefault)" if
                                       cfg["overrides"] or cfg["unverified"] else "no"))
        return 0
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
