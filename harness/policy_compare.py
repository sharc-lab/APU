"""B5 policy comparison (pre-registered in docs/FINDINGS.md, 2026-10-14, "PRE-REGISTRATION: B5 policy comparison").
Read that section first; this docstring is the code map, the pre-registration is the contract.

The same R2 sessions (harness/x2_r2_agent.py: session plan, two-call turn, scoring, canaries; call-2 mode "off";
seeds 20260901-03; 40 turns) are run under eight context policies, one arm per policy:

  (a) default     runtime default: no num_ctx sent, the platform's base server env
  (b) largest     num_ctx = L, the largest window that starts (window probe, base env)
  (c) trim        num_ctx = D (the default window, measured) sent explicitly + client-side trimming
                  (harness/x2_r2_client_trim.ClientTrimRuntime, margin 0.05, render and count), as --mode mitigation
  (d) router      src/dse/router.Router per call at window D, cloud in stub mode; a cloud step is executed by the same
                  local model with the full history at window L (the full-history proxy, ASSUMPTION A-B5-cloud) and
                  costed at the shared price table by analysis/impact_model.escalated_cost (caching on, main case)
  (e) autosize    num_ctx = G, the largest window fully in GPU memory (K1 rule: size_vram >= 99% of size)
  (f) num_keep    no num_ctx (as (a)) + options.num_keep = S, S measured per session by
                  harness/x2_num_keep.measure_system_tokens (render and count), as --num-keep system
  (g) kv_q8_0     the job's own server with OLLAMA_KV_CACHE_TYPE=q8_0 and OLLAMA_FLASH_ATTENTION=1; num_ctx = L(q8_0)
  (h) kv_q4_0     the same with q4_0; num_ctx = L(q4_0)
  plus a long-context recall probe (needles at 85% of the window) at the largest window each KV precision admits
  (f16 with flash attention, q8_0, q4_0): 1 warm-up + 3 measured calls per precision and model.

Windows are measured by the job at start, per model and per server env (probe_windows): D from a load with no num_ctx,
L and G from a descending ladder of num_ctx loads (one-token generation each). Policies that resolve to the identical
configuration on a platform (same effective window, same server env, no client wrapper; only (a), (b), (e) can) share
sessions: the later one is recorded as an alias, not rerun.

Platforms (--platform): evo-x2 (queue x2_policy_compare), evo-t2s (staged, base env OLLAMA_IGPU_ENABLE=1), blade
(Blade night 3, harness/blade_queue.py, the Blade's own LocalOllama server). One output file per platform, prefixed
x2_ / t2s_ / blade_. Metrics, the Pareto frontier and the pre-registered verdict are pure functions over the rows
(metrics_for_platform, pareto_frontier, b5_verdict); analysis/numbers_register.py's B5 rows call them.

On evo-x2 / evo-t2s the job calls t2s_queue.advance() exactly once on exit (finally) and sets no pause; on the Blade it
returns an exit code to harness/blade_queue.py (0 done / refused cleanly, 2 error). --dry-run prints the plan with
estimated hours (labelled estimates, from result files) and touches nothing. --queue-insert is the one read-modify-write
of evo-x2's queue (under t2s_queue.set_pause with a reason naming its release, clear_pause() in a finally block).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics as st
import sys
import time
import uuid
from pathlib import Path

_here = Path(__file__).resolve().parent
# Three layouts: the repo (harness/ next to src/ and analysis/); flat C:\apu\ovn; a side directory such as
# C:\apu\ovn\b5 holding this file and this branch's x2_r2_agent.py (the deployed C:\apu\ovn\x2_r2_agent.py is not
# replaced while other queued jobs use it), with every other module imported from the parent directory. The router
# stack (src/dse, src/cloud, analysis/envelope_model.py) is looked up under REPO: the repo root, or this file's own
# directory when deployed.
if (_here.parent / "harness").is_dir():
    REPO = _here.parent
    _path_front = [str(_here), str(_here.parent / "harness")]
else:
    REPO = _here
    _path_front = [str(_here)] + ([str(_here.parent)] if (_here.parent / "host_config.py").exists() else [])
sys.path[:] = _path_front + [p for p in sys.path if p not in _path_front]

import x2_r2_agent as ag  # noqa: E402

# ── configuration (fixed by the pre-registration) ───────────────────────────────────────────────────

CALL2 = "off"
SFX = ag.MODE_SUFFIX[CALL2]
MODE = "b5"
SEEDS = tuple(ag.SEEDS)                 # 20260901, 20260902, 20260903
TURNS = 40
BASE_MODELS = ("llama3.1:8b", "qwen3:14b")
B1_MODELS = ("qwen3:32b", "qwen3-30b-a3b-2507", "llama3.3:70b")
B1_THINK = {"qwen3:32b": False, "qwen3-30b-a3b-2507": False, "llama3.3:70b": None}
for _m, _t in B1_THINK.items():
    ag.MODELS.setdefault(_m, _t)
B1_MODEL_CAP_H = 24.0                   # a B1 model is added only if its projected B5 hours on evo-x2 fit this cap

POLICIES = {
    "a": {"name": "default", "wrapper": None, "kv": None},
    "b": {"name": "largest", "wrapper": None, "kv": None},
    "c": {"name": "trim", "wrapper": "client_trim", "kv": None},
    "d": {"name": "router", "wrapper": "router", "kv": None},
    "e": {"name": "autosize", "wrapper": None, "kv": None},
    "f": {"name": "num_keep", "wrapper": "num_keep", "kv": None},
    "g": {"name": "kv_q8_0", "wrapper": None, "kv": "q8_0"},
    "h": {"name": "kv_q4_0", "wrapper": None, "kv": "q4_0"},
}
FIXED_POLICIES = ("a", "b", "c", "e", "f", "g", "h")   # every policy but the router
ROUTER_POLICY = "d"
RUN_ORDER = ("a", "c", "f", "d", "e", "g", "h", "b")   # within a seed and model; the largest window last
ALIAS_ORDER = ("a", "b", "e")                            # the only policies that can resolve to the same configuration
KV_ENV = {"q8_0": {"OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1"},
          "q4_0": {"OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_FLASH_ATTENTION": "1"},
          "f16": {"OLLAMA_KV_CACHE_TYPE": "f16", "OLLAMA_FLASH_ATTENTION": "1"}}
RECALL_PRECISIONS = ("f16", "q8_0", "q4_0")

LADDER = (262144, 131072, 65536, 40960, 32768, 16384, 8192, 4096)
FULL_GPU_SHARE = 0.99                   # K1's fits rule (harness/blade_k1.py)
CLIENT_TRIM_MARGIN = 0.05
ROUTER_QUALITY_FLOOR = 0.9
ROUTER_BUDGET_USD = 5.0                 # per session; the router's own affordability check (never binds for R2 sizes)

# ASSUMPTION A-B5-cloud-latency (labelled in every report): the wall time of a cloud step.
CLOUD_TTFT_S_ASSUMED = 1.0
CLOUD_DECODE_TOK_S_ASSUMED = 50.0
# Cloud costing: the shared price table (operator decision A, 2026-10-14: results/cloud_prices_anthropic_20261014.json,
# analysis/impact_model.PRICES_FILE), priced by analysis/impact_model.escalated_cost (caching on between escalated
# steps within the model's cache lifetime, impact_model.CACHE_TTL_S; no caching as the upper bound).
CLOUD_PRICE_MODEL = "claude-sonnet-5-5"            # the table's escalation mid tier
CLOUD_PRICE_TIER = "standard"
CLOUD_PRICE_SENSITIVITY_MODEL = "claude-haiku-5-5"  # the table's cheap tier

# long-context recall probe
RECALL_NEEDLES = 8
RECALL_FILL = 0.85
RECALL_WARMUP_SEED = 0
RECALL_SEEDS = (1, 2, 3)
RECALL_NUM_PREDICT = 256
RECALL_FIRST_RATIO = 1.3                # tokens per chars/4 estimate assumed for the warm-up (conservative: under-fills)

PLATFORMS = {
    "evo-x2": {"host": "EVO-X2", "prefix": "x2_", "base_env": {}, "envelope_hw": "evo-x2", "queue": True,
               "models": BASE_MODELS, "b1_models": B1_MODELS,
               "rules_from": ("x2_r2_validation_v2.jsonl", "x2_r2_scale_validation.jsonl"),
               "server_log": r"C:\apu\ovn\ollama_serve_b5.log", "out": "x2_policy_compare.jsonl"},
    "evo-t2s": {"host": "EVO-T2S", "prefix": "t2s_", "base_env": {"OLLAMA_IGPU_ENABLE": "1"}, "envelope_hw": "evo-t2s",
                "queue": True, "models": ("llama3.1:8b",), "b1_models": (),
                "rules_from": ("t2s_r2_validation_igpu_v1.jsonl",),
                "server_log": r"C:\apu\ovn\ollama_serve_b5.log", "out": "t2s_policy_compare.jsonl"},
    # B5 on the evo-t2s Intel CPU (pre-registration "B5 on evo-t2s Intel CPU and Intel iGPU", 2026-10-22): stock
    # Ollama (no OLLAMA_IGPU_ENABLE in the job's server env), which on evo-t2s runs on the CPU and loads a 4096 default
    # window (t2s-r2-real-cpu). Same host and output prefix as "evo-t2s" (the iGPU platform).
    "evo-t2s-cpu": {"host": "EVO-T2S", "prefix": "t2s_", "base_env": {}, "envelope_hw": "evo-t2s", "queue": True,
                    "models": ("llama3.1:8b", "qwen3:14b"), "b1_models": (),
                    "rules_from": ("t2s_r2_validation_cpu_v1.jsonl",),
                    "server_log": r"C:\apu\ovn\ollama_serve_b5_cpu.log", "out": "t2s_policy_compare_cpu.jsonl"},
    "blade": {"host": "RITZLAPTOP", "prefix": "blade_", "base_env": {}, "envelope_hw": "blade", "queue": False,
              "models": ("qwen3:8b",), "b1_models": (),
              "rules_from": ("blade_r2_validation_qwen3_8b_v1.jsonl",),
              "server_log": None, "out": "blade_policy_compare_qwen3_8b_v1.jsonl"},
}
VERDICT_PLATFORMS = ("evo-x2", "evo-t2s", "blade")
JOB_ID = "x2_policy_compare"
# Queue position on evo-x2 (operator order of 2026-10-14 21:40 PT, docs/X2_QUEUE_PLAN.md): item 13, after B1, the
# family jobs and the B3 runs, before x2_window_override_methods (item 14, last). Inserted immediately before the first
# pending BEFORE_IDS entry, otherwise at the end; nothing else moves.
AFTER_IDS = ("x2_r2_family_mistral_small32_real", "x2_r2_scale_real", "x2_r2_scale_validation",
             "x2_scale_timing_probe")
BEFORE_IDS = ("x2_window_override_methods",)
PYTHON_EXE = sys.executable
SIDE_DIR = "b5"      # evo-x2: C:\apu\ovn\b5 holds policy_compare.py and this branch's x2_r2_agent.py


def arm_id(policy: str) -> str:
    return f"b5_{policy}_{POLICIES[policy]['name']}{SFX}"


def policy_of_arm(arm: str) -> str | None:
    m = re.match(r"^b5_([a-h])_", arm or "")
    return m.group(1) if m else None


def utc_iso():
    return ag.utc_iso()


def git_blob_sha1(path) -> str | None:
    try:
        b = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


# ── window probe (pure core; the runtime part is probe_windows) ─────────────────────────────────────

def ladder_for(native: int | None, ladder=LADDER) -> list[int]:
    """Descending num_ctx candidates: the ladder entries at or below the model's native context, plus the native
    context itself. No native context known: the whole ladder."""
    if not native:
        return sorted(set(ladder), reverse=True)
    return sorted({c for c in ladder if c <= native} | {int(native)}, reverse=True)


def point_starts(p: dict) -> bool:
    return bool(p.get("ok")) and p.get("loaded_context") == p.get("num_ctx")


def point_full_gpu(p: dict, share=FULL_GPU_SHARE) -> bool:
    size, vram = p.get("size"), p.get("size_vram")
    return point_starts(p) and bool(size) and vram is not None and vram >= share * size


def windows_from_points(points: list[dict]) -> dict:
    """L = the largest probed num_ctx that started (HTTP 200 and /api/ps context_length == num_ctx); G = the largest
    that started fully in GPU memory. None when no point qualifies."""
    starts = [p["num_ctx"] for p in points if point_starts(p)]
    full = [p["num_ctx"] for p in points if point_full_gpu(p)]
    return {"largest_starts": max(starts) if starts else None, "largest_full_gpu": max(full) if full else None}


def descend(probe_fn, candidates: list[int], need_full_gpu: bool = True) -> list[dict]:
    """probe_fn(num_ctx) -> point dict. Walks the candidates from the largest down and stops at the first one that is
    fully in GPU memory (need_full_gpu) or at the first that starts (need_full_gpu False). Returns every point."""
    points = []
    for c in candidates:
        p = probe_fn(c)
        p["num_ctx"] = c
        points.append(p)
        if (point_full_gpu(p) if need_full_gpu else point_starts(p)):
            break
    return points


# ── policy resolution (pure) ────────────────────────────────────────────────────────────────────────

def resolve_policies(win: dict) -> dict:
    """win: {"D": default window, "L": {env: largest that starts}, "G": largest full-GPU (base env)} with env keys
    "base", "q8_0", "q4_0". Returns {policy: config}. config: num_ctx_sent (None = not sent), effective_ctx, env
    ("base" / "q8_0" / "q4_0"), wrapper, proxy_ctx (router only), alias_of (None or an earlier policy), skip (None or
    the reason no session can run)."""
    D, L, G = win.get("D"), (win.get("L") or {}), win.get("G")
    lb = L.get("base")
    spec = {
        "a": (None, D, "base"), "b": (lb, lb, "base"), "c": (D, D, "base"), "d": (D, D, "base"),
        "e": (G, G, "base"), "f": (None, D, "base"), "g": (L.get("q8_0"), L.get("q8_0"), "q8_0"),
        "h": (L.get("q4_0"), L.get("q4_0"), "q4_0"),
    }
    why = {"a": "default window not measured (load without num_ctx failed)",
           "b": "no ladder window started (base env)", "c": "default window not measured",
           "d": "default window not measured", "e": "no ladder window fully in GPU memory",
           "f": "default window not measured", "g": "no ladder window started under q8_0",
           "h": "no ladder window started under q4_0"}
    out = {}
    for p in POLICIES:
        sent, eff, env = spec[p]
        cfg = {"policy": p, "name": POLICIES[p]["name"], "arm_id": arm_id(p), "num_ctx_sent": sent,
               "effective_ctx": eff, "env": env, "wrapper": POLICIES[p]["wrapper"], "alias_of": None, "skip": None}
        if p == "d":
            cfg["proxy_ctx"] = lb
            if lb is None and eff is not None:
                cfg["skip"] = "no full-history proxy window (no ladder window started)"
        if eff is None:
            cfg["skip"] = why[p]
        out[p] = cfg
    for i, p in enumerate(ALIAS_ORDER):
        if out[p]["skip"]:
            continue
        for q in ALIAS_ORDER[:i]:
            c = out[q]
            if not c["skip"] and c["alias_of"] is None and c["effective_ctx"] == out[p]["effective_ctx"] \
                    and c["env"] == out[p]["env"] and c["wrapper"] is None and out[p]["wrapper"] is None:
                out[p]["alias_of"] = q
                break
    return out


def distinct_policies(configs: dict) -> list[str]:
    return [p for p in POLICIES if not configs[p]["skip"] and configs[p]["alias_of"] is None]


def session_items(models, configs_by_model: dict, seeds=SEEDS, order=RUN_ORDER) -> list[tuple]:
    """[(seed, model, policy)] in run order: seed-major, then model, then RUN_ORDER; skipped and aliased policies out."""
    return [(s, m, p) for s in seeds for m in models for p in order
            if m in configs_by_model and p in distinct_policies(configs_by_model[m])]


def server_env_for(platform_env: dict, env_key: str) -> dict:
    """The full server env of a policy: the platform's base env plus the KV precision env (if any)."""
    env = dict(platform_env or {})
    if env_key in KV_ENV:
        env.update(KV_ENV[env_key])
    return env


# ── KV-type check on the server log (pure) ──────────────────────────────────────────────────────────

_KV_TYPE_RES = (re.compile(r"type_k\s*=\s*'?([a-z0-9_]+)", re.I), re.compile(r"\bK \(([a-z0-9_]+)\)"),
                re.compile(r"cache_type_k[\"=: ]+([a-z0-9_]+)", re.I))


def kv_check(log_text: str, requested: str | None) -> dict:
    """What the server log says about the KV cache: whether the server-config line carries the requested
    OLLAMA_KV_CACHE_TYPE (the env took effect), and every KV type llama.cpp reported. valid is False only when the env
    is missing from the config line or a reported type differs from the request; no reported type = unverified."""
    text = log_text or ""
    if requested is None:
        return {"requested": None, "valid": True, "observed": [], "config_has_env": None}
    cfg_lines = [l for l in text.splitlines() if "server config" in l]
    has_env = any(f"OLLAMA_KV_CACHE_TYPE:{requested}" in l for l in cfg_lines)
    fa = any(re.search(r"OLLAMA_FLASH_ATTENTION:(true|1)", l) for l in cfg_lines)
    observed = sorted({m.group(1).lower() for rx in _KV_TYPE_RES for m in rx.finditer(text)})
    bad = [t for t in observed if t != requested]
    valid = has_env and not bad
    return {"requested": requested, "config_has_env": has_env, "config_flash_attention": fa, "observed": observed,
            "valid": valid, "verified": bool(observed) and not bad,
            "reason": None if valid else ("env not in the server config line" if not has_env
                                           else f"llama.cpp reported KV type(s) {bad} != {requested}")}


# ── long-context recall probe (pure builders and scorer) ────────────────────────────────────────────

_WORDS = ("river", "copper", "lantern", "meadow", "signal", "harbor", "granite", "orchard", "velvet", "summit",
          "beacon", "tundra", "cobalt", "willow", "ember", "falcon", "prairie", "quartz", "saddle", "thicket",
          "canyon", "pebble", "ripple", "marble", "drizzle", "lattice", "harvest", "compass", "glacier", "timber")
_CODE_ALPHA = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def recall_needles(seed: int, n: int = RECALL_NEEDLES) -> dict[str, str]:
    rng = random.Random(7_000_003 * (seed + 1) + 11)
    return {f"V{i}": "".join(rng.choice(_CODE_ALPHA) for _ in range(8)) for i in range(1, n + 1)}


def build_recall_prompt(seed: int, target_tokens: int, ratio: float) -> tuple[str, dict]:
    """A filler document of about target_tokens model tokens (chars/4 estimate x ratio) with RECALL_NEEDLES needles at
    evenly spaced depths, then the question. Returns (prompt, needles)."""
    needles = recall_needles(seed)
    rng = random.Random(seed * 104_729 + 5)
    budget_chars = max(400, int(target_tokens / max(ratio, 0.1) * 4))
    sentences, chars = [], 0
    while chars < budget_chars:
        s = " ".join(rng.choice(_WORDS) for _ in range(12)).capitalize() + "."
        sentences.append(s)
        chars += len(s) + 1
    n = len(needles)
    for i, (k, v) in enumerate(needles.items(), start=1):
        pos = int(len(sentences) * i / (n + 1))
        sentences.insert(pos, f"Record: the access code for vault {k} is {v}.")
    q = ("\n\nQuestion: list the access code of every vault V1 to V%d named in the records above, as JSON "
         "{\"V1\": \"...\", ...}. Answer only with the JSON." % n)
    return " ".join(sentences) + q, needles


def score_recall(text: str | None, needles: dict) -> dict:
    t = text or ""
    found = [k for k, v in needles.items() if v in t]
    return {"found": len(found), "n": len(needles), "recall": len(found) / len(needles), "found_keys": found}


# ── wrappers around the agent runtime ───────────────────────────────────────────────────────────────

class _Passthrough:
    def __init__(self, inner):
        self.inner = inner

    def loaded_context(self, model):
        return self.inner.loaded_context(model)

    def unload(self, model):
        return self.inner.unload(model)

    def recover(self):
        return self.inner.recover()

    def begin_session(self, model, arm, seed):
        pass

    def end_session(self):
        pass


class PlainRuntime(_Passthrough):
    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        return (self.inner.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                else self.inner.chat(model, messages, num_ctx, tools, think))


class TrimRuntime(_Passthrough):
    """(c): x2_r2_client_trim.ClientTrimRuntime around the base runtime (no debug-log observation)."""

    def __init__(self, inner, counter, margin=CLIENT_TRIM_MARGIN, ct=None):
        super().__init__(inner)
        if ct is None:
            import x2_r2_client_trim as ct
        self.trt = ct.ClientTrimRuntime(inner, counter, margin=margin)

    def begin_session(self, model, arm, seed):
        self.trt.begin_session()

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        return (self.trt.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                else self.trt.chat(model, messages, num_ctx, tools, think))


class NumKeepPolicyRuntime(_Passthrough):
    """(f): measure S per session (measure(model, arm, seed) -> x2_num_keep.measure_system_tokens dict), then every
    request of the session carries options.num_keep = S (base.options_extra). emit gets one b5_num_keep row per
    session."""

    def __init__(self, inner, measure, emit):
        super().__init__(inner)
        self.measure, self.emit = measure, emit
        self.s = None

    def begin_session(self, model, arm, seed):
        self.inner.options_extra = None
        m = self.measure(model, arm, seed)
        self.s = int(m["system_prompt_tokens"])
        self.inner.options_extra = {"num_keep": self.s}
        self.emit({"record": "b5_num_keep", "model_id": model, "arm_id": arm, "seed": seed, "num_keep": self.s,
                   **m, "ts_utc": utc_iso()})

    def end_session(self):
        self.inner.options_extra = None
        self.s = None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        if self.s is None:
            raise RuntimeError("NumKeepPolicyRuntime.chat before begin_session")
        resp = (self.inner.chat(model, messages, num_ctx, tools, think, extra=extra) if extra
                else self.inner.chat(model, messages, num_ctx, tools, think))
        resp = dict(resp)
        resp["num_keep"] = {"sent": self.s, "system_prompt_tokens": self.s, "observed": None}
        return resp


def kept_by_turn_ids(messages: list[dict], kept_turn_ids: list[int]) -> list[dict]:
    """System messages plus the turns (1-based, by user message) the router kept, in order (src/dse/router.split_turns
    grouping)."""
    sys_idx, turns, keep = [], [], set(kept_turn_ids)
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            sys_idx.append(i)
        elif m.get("role") == "user" or not turns:
            turns.append([i])
        else:
            turns[-1].append(i)
    idx = sys_idx + [j for n, t in enumerate(turns, start=1) if n in keep for j in t]
    return [messages[j] for j in sorted(idx)]


class RouterPolicyRuntime(_Passthrough):
    """(d): before every call, router.decide(step) on the full transcript at window D. local -> sent untrimmed at D;
    local_trimmed -> the kept turns at D; cloud -> the full-history proxy: the same local model, full transcript, at
    proxy_ctx (ASSUMPTION A-B5-cloud); refuse -> no call (the call reads as an error, the turn is scored as usual).
    Every call emits one b5_route row (decision, tokens, sent window, measured duration, proxy reply tokens)."""

    def __init__(self, inner, make_router, window: int, proxy_ctx: int, emit, clock=time.time):
        super().__init__(inner)
        self.make_router, self.window, self.proxy_ctx, self.emit, self.clock = (make_router, window, proxy_ctx,
                                                                                 emit, clock)
        self.router = None
        self.session = None
        self._turn = 0
        self._call = 0

    def begin_session(self, model, arm, seed):
        self.router = self.make_router(model)
        self.session = {"model_id": model, "arm_id": arm, "seed": seed, "session_uid": uuid.uuid4().hex[:12]}
        self._turn = self._call = 0

    def end_session(self):
        self.router = None
        self.session = None

    def chat(self, model, messages, num_ctx, tools, think, extra=None):
        if self.router is None:
            raise RuntimeError("RouterPolicyRuntime.chat before begin_session")
        n_users = sum(1 for m in messages if m.get("role") == "user")
        if n_users != self._turn:
            self._turn, self._call = n_users, 0
        self._call += 1
        canary = n_users > 0 and n_users % ag.r2.CANARY_CHECK_EVERY == 0
        step = {"messages": messages, "num_predict": ag.MAX_TOKENS_PER_CALL, "tools": tools,
                "needs_turns": [1] if canary else [], "turn_idx": n_users, "call_idx": self._call}
        d = self.router.decide(step)
        t0 = self.clock()
        if d.target == "refuse":
            resp = {"outcome": "router_refused", "status": None, "error": d.reason[:300], "message": "",
                    "duration_s": 0.0}
            sent_ctx, n_sent = None, 0
        else:
            if d.target == "cloud":
                send, sent_ctx = messages, self.proxy_ctx
            elif d.target == "local_trimmed":
                send, sent_ctx = kept_by_turn_ids(messages, d.kept_turn_ids), self.window
            else:
                send, sent_ctx = messages, self.window
            n_sent = len(send)
            resp = (self.inner.chat(model, send, sent_ctx, tools, think, extra=extra) if extra
                    else self.inner.chat(model, send, sent_ctx, tools, think))
        dt = resp.get("duration_s")
        self.emit({"record": "b5_route", **self.session, "turn_idx": n_users, "call_idx": self._call,
                   "target": d.target, "reason": d.reason, "tokens": d.tokens, "kept_turn_ids": d.kept_turn_ids,
                   "dropped_turn_ids": d.dropped_turn_ids, "needs_turns": d.needs_turns,
                   "quality_floor_met": d.quality_floor_met, "router_est_cost_usd": d.est_cost_usd,
                   "router_budget_remaining_usd": d.budget_remaining_usd, "stub": d.stub,
                   "sent_num_ctx": sent_ctx, "n_messages_full": len(messages), "n_messages_sent": n_sent,
                   "duration_s": dt if dt is not None else self.clock() - t0,
                   "prompt_eval_count": resp.get("prompt_eval_count"), "eval_count": resp.get("eval_count"),
                   "outcome": resp.get("outcome"), "ts_utc": utc_iso()})
        return resp


def make_router_factory(platform: str, window: int, envelope=None, ledger_dir=None):
    """model -> a src/dse/router.Router at "ollama_ctx_<window>" on the platform's envelope machine (added empty when
    the envelope does not know it: the router then uses only the window, the needed turns and its budget)."""
    sys.path.insert(0, str(REPO))
    from src.dse.router import Router, Envelope  # noqa: PLC0415
    from src.cloud.client import CloudClient  # noqa: PLC0415
    env = Envelope.coerce(envelope)
    data = json.loads(json.dumps(env.data))
    hw = PLATFORMS[platform]["envelope_hw"]
    data["machines"].setdefault(hw, {"ollama_default_ctx": {}})
    ledger = Path(ledger_dir or ".") / "b5_router_stub_ledger_unused.jsonl"

    def make(model):
        client = CloudClient(mode="stub", ledger_path=ledger)
        ratio = Envelope(data).token_calib_ratio(model) or 1.0
        return Router(Envelope(data), ROUTER_BUDGET_USD, ROUTER_QUALITY_FLOOR, None, hw, f"ollama_ctx_{window}", model,
                      cloud_client=client, token_calib_ratio=ratio)
    return make


# ── metrics, Pareto frontier, verdict (pure, on rows) ───────────────────────────────────────────────

def _turns_by_attempt(rows):
    out = {}
    for r in rows:
        if r.get("record") == "r2a_turn" and r.get("mode") == MODE:
            out.setdefault(r.get("attempt_id"), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["turn_idx"])
    return out


def route_rows_by_attempt(rows) -> dict:
    """{attempt_id: [b5_route rows]}: route rows are joined to the session whose turn rows follow them in file order
    (a b5_route row precedes the r2a_turn row of its turn; the session uid ties the rows of one session together)."""
    out, pending = {}, {}
    for r in rows:
        if r.get("record") == "b5_route":
            pending.setdefault((r["model_id"], r["arm_id"], r["seed"]), []).append(r)
        elif r.get("record") == "r2a_turn" and r.get("mode") == MODE:
            key = (r["model_id"], r["arm_id"], r["seed"])
            uid_rows = pending.get(key) or []
            if uid_rows:
                uid = uid_rows[-1]["session_uid"]
                out.setdefault(r["attempt_id"], [])
                mine = [x for x in uid_rows if x["session_uid"] == uid and x["turn_idx"] == r["turn_idx"]]
                out[r["attempt_id"]].extend(mine)
    return out


def cloud_step_wall_s(eval_count) -> float:
    """ASSUMPTION A-B5-cloud-latency."""
    return CLOUD_TTFT_S_ASSUMED + (eval_count or 0) / CLOUD_DECODE_TOK_S_ASSUMED


def _ts(s):
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _impact_model():
    sys.path.insert(0, str(REPO / "analysis"))
    import impact_model
    return impact_model


def load_cloud_prices(repo=REPO):
    """The shared price table (impact_model.prices: results/cloud_prices_anthropic_20261014.json)."""
    return _impact_model().prices(repo)


def escalated_steps(route_rows: list[dict]) -> list[tuple]:
    """[(input_tokens, output_tokens, t_start_s)] of a session's cloud steps. Input = the router's calibrated estimate
    of system + tools + full history (tokens["system"] + tokens["full_history"]); output = the proxy reply's eval_count;
    t = the route row's timestamp (a step without one never reads from cache)."""
    out = []
    for i, r in enumerate(x for x in route_rows if x.get("target") == "cloud"):
        tok = r.get("tokens") or {}
        t = _ts(r.get("ts_utc"))
        out.append((int((tok.get("system") or 0) + (tok.get("full_history") or 0)), int(r.get("eval_count") or 0),
                    t if t is not None else 1e12 * (i + 1)))
    return out


def cloud_session_cost(route_rows: list[dict], prices, model=CLOUD_PRICE_MODEL, tier=CLOUD_PRICE_TIER,
                       caching=True) -> dict:
    """USD of one session's cloud steps at the shared price table, priced by impact_model.escalated_cost: caching on
    (main case) reads from cache the previous cloud step's input when it reached the model's minimum cacheable prefix
    and started at most impact_model.CACHE_TTL_S earlier, the rest of a cacheable input is a cache write; caching off
    prices every input token at the input rate."""
    esc = escalated_steps(route_rows)
    c = _impact_model().escalated_cost(esc, model, tier, prices)
    return {"usd": c["cached"] if caching else c["uncached"], "n_cloud_calls": c["n"], "cache_hits": c["hits"]}


def session_metrics(turns: list[dict], rules: list[str], route_rows=None, prices=None) -> dict:
    """One session: rule-in-use pass counts, history-tag recall on canary turns, wall-clock minutes (cloud steps
    re-timed per A-B5-cloud-latency; the proxy's measured time kept as a sensitivity), cloud USD (main and
    sensitivities)."""
    rule_pass = rule_n = 0
    for t in turns:
        for r in rules:
            v = t.get(r)
            if v is not None:
                rule_n += 1
                rule_pass += 1 if v else 0
    canary = [t for t in turns if t.get("canary_check")]
    hist_ok = sum(1 for t in canary if t.get("canary_hist_ok") is True)
    wall_s = sum((t.get("turn_wall_s") or 0.0) for t in turns)
    proxy_s = cloud_s = 0.0
    routes = route_rows or []
    for r in routes:
        if r.get("target") == "cloud":
            proxy_s += r.get("duration_s") or 0.0
            cloud_s += cloud_step_wall_s(r.get("eval_count"))
    out = {"n_turns": len(turns), "rule_pass": rule_pass, "rule_n": rule_n, "canary_turns": len(canary),
           "hist_ok": hist_ok, "wall_min": (wall_s - proxy_s + cloud_s) / 60.0,
           "wall_min_proxy_measured": wall_s / 60.0,
           "targets": {k: sum(1 for r in routes if r.get("target") == k) for k in ("local", "local_trimmed", "cloud",
                                                                                   "refuse")},
           "usd": 0.0, "usd_no_cache": 0.0, "usd_cheap_model": 0.0}
    if routes and prices is not None:
        out["usd"] = cloud_session_cost(routes, prices)["usd"]
        out["usd_no_cache"] = cloud_session_cost(routes, prices, caching=False)["usd"]
        out["usd_cheap_model"] = cloud_session_cost(routes, prices, model=CLOUD_PRICE_SENSITIVITY_MODEL)["usd"]
    return out


def resolution_of(rows, model) -> dict | None:
    r = next((r for r in reversed(rows) if r.get("record") == "b5_resolution" and r.get("model_id") == model), None)
    return r


def metrics_for_platform(rows: list[dict], prices=None, seeds=SEEDS, turns=TURNS) -> dict:
    """{model: {"configs", "balanced_seeds", "complete", "policies": {policy: metrics}, "problems"}}. Metrics per policy
    over the seeds completed (40 turns) for EVERY distinct policy of that model (balanced); an aliased policy carries
    its target's metrics. Rules in use per model come from the last run_start."""
    starts = [r for r in rows if r.get("record") == "run_start" and r.get("mode") == MODE]
    riu = {}
    for s in starts:
        riu.update(s.get("rules_in_use") or {})
    finished = any(r.get("record") == "run_end" for r in rows)
    tba = _turns_by_attempt(rows)
    routes = route_rows_by_attempt(rows)
    sessions = [s for s in ag.completed_sessions(rows) if s.get("mode") == MODE]
    invalid_env = {(r["model_id"], r["env"]) for r in rows if r.get("record") == "b5_kv_check"
                   and not (r.get("check") or {}).get("valid", True)}
    out = {}
    models = sorted({r["model_id"] for r in rows if r.get("record") == "b5_resolution"})
    for model in models:
        res = resolution_of(rows, model)
        configs = res["configs"]
        dist = distinct_policies(configs)
        problems = []
        for p in list(dist):
            if configs[p]["env"] != "base" and (model, configs[p]["env"]) in invalid_env:
                problems.append(f"{p}: KV type not applied ({configs[p]['env']}), sessions excluded")
                dist.remove(p)
        by_pol = {p: {} for p in dist}
        for s in sessions:
            p = policy_of_arm(s["arm_id"])
            if s["model_id"] == model and p in by_pol:
                t = tba.get(s["attempt_id"], [])
                if len(t) == turns:
                    by_pol[p][s["seed"]] = s["attempt_id"]
        balanced = [sd for sd in seeds if dist and all(sd in by_pol[p] for p in dist)]
        rules = list(riu.get(model) or [])
        if not rules:
            problems.append("no rules in use recorded")
        pol = {}
        for p in dist:
            ms = [session_metrics(tba[by_pol[p][sd]], rules, routes.get(by_pol[p][sd]), prices) for sd in balanced]
            if not ms:
                continue
            rn, cn = sum(m["rule_n"] for m in ms), sum(m["canary_turns"] for m in ms)
            pol[p] = {"rule_pass_rate": sum(m["rule_pass"] for m in ms) / rn if rn else None,
                      "history_recall": sum(m["hist_ok"] for m in ms) / cn if cn else None,
                      "wall_min": st.mean(m["wall_min"] for m in ms),
                      "wall_min_proxy_measured": st.mean(m["wall_min_proxy_measured"] for m in ms),
                      "cloud_usd": st.mean(m["usd"] for m in ms),
                      "cloud_usd_no_cache": st.mean(m["usd_no_cache"] for m in ms),
                      "cloud_usd_cheap_model": st.mean(m["usd_cheap_model"] for m in ms),
                      "n_sessions": len(ms),
                      "targets": {k: sum(m["targets"][k] for m in ms) for k in ms[0]["targets"]},
                      "window": configs[p]["effective_ctx"]}
        for p in POLICIES:
            a = configs[p].get("alias_of")
            if a and a in pol:
                pol[p] = dict(pol[a], alias_of=a)
        complete = (finished and bool(dist) and len(balanced) == len(seeds) and not any("excluded" in x for x in problems)
                    and ROUTER_POLICY in dist and bool(rules))
        if not finished:
            problems.append("no run_end")
        if len(balanced) != len(seeds):
            problems.append(f"balanced seeds {balanced}, expected {list(seeds)}")
        if ROUTER_POLICY not in dist:
            problems.append(f"router policy not run ({configs[ROUTER_POLICY].get('skip')})")
        out[model] = {"configs": configs, "balanced_seeds": balanced, "complete": complete, "policies": pol,
                      "problems": problems, "rules_in_use": rules}
    return out


METRIC_SENSE = (("rule_pass_rate", 1), ("history_recall", 1), ("wall_min", -1), ("cloud_usd", -1))


def dominates(x: dict, y: dict, sense=METRIC_SENSE) -> bool:
    """x dominates y: at least as good on every metric and strictly better on one (higher rates, lower minutes and
    USD). A missing metric (None) never counts as better."""
    better = False
    for k, s in sense:
        a, b = x.get(k), y.get(k)
        if a is None or b is None:
            return False
        if s * a < s * b:
            return False
        if s * a > s * b:
            better = True
    return better


def pareto_frontier(policies: dict) -> list[str]:
    return [p for p, m in policies.items() if not any(dominates(o, m) for q, o in policies.items() if q != p)]


def b5_verdict(platform_rows: dict, prices=None, required=VERDICT_PLATFORMS, seeds=SEEDS, turns=TURNS) -> dict:
    """The pre-registered B5 verdict. platform_rows: {platform: rows or None}. A cell = (platform, model); complete
    cells only. Supported: on every required platform (d) is on the frontier in every complete cell, and no fixed
    setting is on the frontier in every complete cell of every required platform. Refuted: (d) dominated in a complete
    cell of any platform, or (all required platforms complete) one fixed setting on the frontier everywhere."""
    per = {}
    for plat in required:
        rows = platform_rows.get(plat)
        if not rows:
            per[plat] = {"status": "not run", "cells": {}}
            continue
        cells = metrics_for_platform(rows, prices, seeds=seeds, turns=turns)
        done = {m: c for m, c in cells.items() if c["complete"]}
        out_cells = {}
        for m, c in cells.items():
            e = {"complete": c["complete"], "problems": c["problems"]}
            if c["complete"]:
                fr = pareto_frontier(c["policies"])
                e.update({"frontier": fr, "router_on_frontier": ROUTER_POLICY in fr,
                          "router_dominated_by": [q for q, o in c["policies"].items() if q != ROUTER_POLICY
                                                  and dominates(o, c["policies"][ROUTER_POLICY])],
                          "metrics": c["policies"]})
            out_cells[m] = e
        status = "complete" if done and all(c["complete"] for c in cells.values()) else (
            "partial" if done else "incomplete")
        router_ok = bool(done) and all(out_cells[m]["router_on_frontier"] for m in done)
        fixed_on = [p for p in FIXED_POLICIES if done and all(p in out_cells[m]["frontier"] for m in done)]
        per[plat] = {"status": status, "cells": out_cells, "router_on_frontier_all_cells": router_ok if done else None,
                     "fixed_on_frontier_all_cells": fixed_on if done else None}
    router_dominated = [f"{plat}/{m}" for plat, e in per.items() for m, c in e["cells"].items()
                        if c.get("complete") and not c["router_on_frontier"]]
    all_complete = all(per[p]["status"] == "complete" for p in required)
    fixed_everywhere = ([p for p in FIXED_POLICIES if all(p in (per[q]["fixed_on_frontier_all_cells"] or [])
                                                          for q in required)] if all_complete else None)
    if router_dominated:
        verdict = "refuted (router dominated in " + ", ".join(router_dominated) + ")"
    elif all_complete and fixed_everywhere:
        verdict = "refuted (fixed setting(s) on the frontier on every platform: " + ", ".join(fixed_everywhere) + ")"
    elif all_complete:
        verdict = "supported"
    else:
        missing = [p for p in required if per[p]["status"] != "complete"]
        verdict = "inconclusive (platforms not complete: " + ", ".join(missing) + ")"
    return {"verdict": verdict, "platforms": per, "router_dominated": router_dominated,
            "fixed_on_frontier_everywhere": fixed_everywhere,
            "assumptions": {"A-B5-cloud": "a cloud step = the same local model with the full history at window L",
                            "A-B5-cloud-latency": f"{CLOUD_TTFT_S_ASSUMED} s + output tokens / "
                                                  f"{CLOUD_DECODE_TOK_S_ASSUMED} tok/s",
                            "price": f"{CLOUD_PRICE_MODEL} {CLOUD_PRICE_TIER}, shared price table, caching on "
                                     f"between cloud steps (main)"}}


# ── B5 on evo-t2s Intel CPU and Intel iGPU (pre-registration 2026-10-22) ────────────────────────────────────────

T2S_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
T2S_MIN_SEEDS = SEEDS                  # a cell short of 5 seeds counts only if complete on the first 3 (B5's count)
T2S_PLATFORMS = ("evo-t2s-cpu", "evo-t2s")
T2S_CONSTRAINED = ("evo-t2s-cpu",)     # the platform whose default window fails (t2s-r2-real-cpu: gated 5/5 silent)


def handover_deadline_reached(platform, deps=None) -> bool:
    """evo-x2 / evo-t2s: True once t2s_queue.deadline_reached() (queue_deadline.json armed by harness/t2s_deadline.py);
    checked at every session boundary so the job stops cleanly before the watchdog's kill. Never on the Blade."""
    if platform == "blade":
        return False
    fn = (deps or {}).get("deadline_reached")
    if fn is None:
        try:
            import t2s_queue as tq
            fn = tq.deadline_reached
        except Exception:
            return False
    try:
        v = fn()
    except Exception:
        return False
    return v is not None and v is not False


def _t2s_cells(rows, prices=None) -> dict:
    """{model: cell} at 5 seeds; a model not complete at 5 is re-read at the first 3 (labelled seeds_used)."""
    five = metrics_for_platform(rows, prices, seeds=T2S_SEEDS)
    three = metrics_for_platform(rows, prices, seeds=T2S_MIN_SEEDS)
    out = {}
    for m, c in five.items():
        if c["complete"]:
            out[m] = dict(c, seeds_used=list(T2S_SEEDS))
        elif three.get(m, {}).get("complete"):
            out[m] = dict(three[m], seeds_used=list(T2S_MIN_SEEDS))
        else:
            out[m] = dict(c, seeds_used=None)
    return out


def router_beats_default(pol: dict) -> bool:
    """(d) at least as good as (a) on rule pass rate and history recall, strictly better on one."""
    d, a = pol.get(ROUTER_POLICY), pol.get("a")
    if not d or not a:
        return False
    keys = ("rule_pass_rate", "history_recall")
    if any(d.get(k) is None or a.get(k) is None for k in keys):
        return False
    return all(d[k] >= a[k] for k in keys) and any(d[k] > a[k] for k in keys)


def b5_t2s_verdict(platform_rows: dict, prices=None) -> dict:
    """Pre-registered verdict of B5 on evo-t2s (FINDINGS 2026-10-22). H-B5-T2S: where the default window fails, the
    router is worth having. Cells (platform, model) complete at 5 seeds, or at the first 3 (labelled).
    Supported: at least one complete constrained (CPU) cell, and in EVERY complete CPU cell (d) is on the Pareto
    frontier and beats (a) on quality (router_beats_default). Refuted: (d) dominated in any complete cell of either
    evo-t2s platform; or every complete CPU cell has (d) not beating (a) on quality. Inconclusive otherwise."""
    per = {}
    for plat in T2S_PLATFORMS:
        rows = platform_rows.get(plat)
        if not rows:
            per[plat] = {"status": "not run", "cells": {}}
            continue
        cells = _t2s_cells(rows, prices)
        out = {}
        for m, c in cells.items():
            done = c.get("seeds_used") is not None
            e = {"complete": done, "seeds_used": c.get("seeds_used"), "problems": c["problems"]}
            if done:
                fr = pareto_frontier(c["policies"])
                e.update({"frontier": fr, "router_on_frontier": ROUTER_POLICY in fr,
                          "router_beats_default": router_beats_default(c["policies"]),
                          "router_dominated_by": [q for q, o in c["policies"].items() if q != ROUTER_POLICY
                                                  and dominates(o, c["policies"][ROUTER_POLICY])],
                          "metrics": c["policies"]})
            out[m] = e
        per[plat] = {"status": "complete" if out and all(e["complete"] for e in out.values()) else
                     ("partial" if any(e["complete"] for e in out.values()) else "incomplete"), "cells": out}
    dominated = [f"{p}/{m}" for p, e in per.items() for m, c in e["cells"].items()
                 if c.get("complete") and not c["router_on_frontier"]]
    cpu = [(p, m, c) for p in T2S_CONSTRAINED for m, c in per.get(p, {}).get("cells", {}).items() if c.get("complete")]
    if dominated:
        verdict = "refuted (router dominated in " + ", ".join(dominated) + ")"
    elif cpu and not any(c["router_beats_default"] for _p, _m, c in cpu):
        verdict = "refuted (router does not beat the default on quality in any complete CPU cell)"
    elif cpu and all(c["router_on_frontier"] and c["router_beats_default"] for _p, _m, c in cpu):
        verdict = "supported (" + ", ".join(f"{p}/{m}" for p, m, _c in cpu) + ")"
    else:
        verdict = "inconclusive (" + ("no complete CPU cell" if not cpu else "mixed CPU cells") + ")"
    return {"verdict": verdict, "platforms": per, "router_dominated": dominated}


T2S_MEASURED_FILES = {"evo-t2s-cpu": {"default": "t2s_r2_real_cpu_v1.jsonl", "trim": "t2s_r2_mitigation_cpu_v1.jsonl"},
                      "evo-t2s": {"default": "t2s_r2_real_igpu_v1.jsonl", "trim": None}}
T2S_ASSUMED_L = {"llama3.1:8b": 131072, "qwen3:14b": 40960}   # native context; L is probed at run time
T2S_ESTIMATE_ASSUMPTIONS = (
    "minutes per session from MEASURED evo-t2s R2 sessions (t2s-r2-real-cpu, t2s-r2-real-igpu; CPU trim from "
    "t2s-r2-mitigation-cpu): (a), (d), (f) = the measured default-window session median (d's cloud-proxy steps not "
    "added); (c) = CPU: the measured trimming session median at 4096, iGPU: the default median x the evo-x2 trim "
    "factor; (b), (g), (h) at L = native context (ASSUMED; probed at run time) = the default median x the evo-x2 ratio "
    "of session minutes at L over the default window (x2_r2_real_v1/v1b); (e): skipped on the CPU (no GPU), alias of "
    "(b) on the iGPU (G = L assumed); recall and window probes as in estimate_platform")


def _t2s_measured(repo, platform) -> dict:
    """{model: {"D": loaded default window, "default_min": median, "trim_min": median or None}} from the files."""
    out = {}
    for kind, name in T2S_MEASURED_FILES[platform].items():
        if not name:
            continue
        rows = ag.read_rows(repo / "results" / name)
        acc, ctx = {}, {}
        for r in rows:
            if r.get("record") == "r2a_turn":
                a = acc.setdefault((r["model_id"], r["attempt_id"]), [0.0, 0])
                a[0] += r.get("turn_wall_s") or 0.0
                a[1] += 1
                if r.get("loaded_context"):
                    ctx.setdefault(r["model_id"], set()).add(r["loaded_context"])
        by = {}
        for (m, _a), (sec, n) in acc.items():
            if n == TURNS:
                by.setdefault(m, []).append(sec / 60.0)
        for m, v in by.items():
            e = out.setdefault(m, {"D": None, "default_min": None, "trim_min": None, "n": {}})
            e[f"{kind}_min"] = st.median(v)
            e["n"][kind] = len(v)
            if kind == "default":
                e["D"] = min(ctx.get(m) or {None})
    return out


def estimate_t2s(platform, models=None, seeds=T2S_SEEDS, repo=REPO) -> dict:
    """ESTIMATED hours of B5 on an evo-t2s platform from measured evo-t2s session minutes (T2S_ESTIMATE_ASSUMPTIONS)."""
    models = list(models or PLATFORMS["evo-t2s-cpu"]["models"])
    meas = _t2s_measured(repo, platform)
    x2 = _x2_session_minutes(repo)
    mf = _mitigation_factor(repo)
    rates = _x2_rates(repo)
    out = {"platform": platform, "label": "ESTIMATE (not measured)", "assumptions": T2S_ESTIMATE_ASSUMPTIONS,
           "seeds": list(seeds), "models": {}, "total_h": 0.0}
    for m in models:
        e = meas.get(m)
        if not e or not e["default_min"] or not e["D"]:
            out["models"][m] = {"total_h": None, "missing": "no measured evo-t2s default session"}
            continue
        D, L = e["D"], T2S_ASSUMED_L.get(m)
        dm = e["default_min"]
        ratio_L = (x2.get((m, L)) or 0) / x2[(m, D)] if x2.get((m, D)) and x2.get((m, L)) else None
        big = dm * ratio_L if ratio_L else None
        per = {"a": dm, "d": dm, "f": dm, "c": e["trim_min"] if e.get("trim_min") else dm * mf,
               "b": big, "g": big, "h": big}
        if platform == "evo-t2s":
            aliases = {"e": "b"}
        else:
            aliases = {}
        sess_h = sum(v or 0.0 for v in per.values()) * len(seeds) / 60.0
        factor = dm / x2[(m, D)] if x2.get((m, D)) else 1.0
        rec_h = 0.0
        rate = rates.get(m) or {}
        if rate.get("pe_rate") and L:
            per_call = (RECALL_FILL * L) / rate["pe_rate"] + RECALL_NUM_PREDICT / (rate.get("ev_rate") or 20.0)
            rec_h = len(RECALL_PRECISIONS) * (1 + len(RECALL_SEEDS)) * per_call / 3600.0 * factor
        probe_h = PROBE_H_PER_MODEL_ENV * (3 + 1)
        tot = sess_h + rec_h + probe_h
        out["models"][m] = {"D": D, "L_assumed": L, "min_per_session": per, "aliases": aliases,
                            "x2_ratio_L_over_D": ratio_L, "measured_n": e["n"], "sessions_h": sess_h,
                            "recall_h": rec_h, "probe_h": probe_h, "total_h": tot,
                            "per_seed_h": sum(v or 0.0 for v in per.values()) / 60.0}
        out["total_h"] += tot
    return out


def recall_summary(rows: list[dict]) -> dict:
    """{model: {precision: {"window", "recall_mean", "calls", "truncated"}}} from the measured b5_recall rows."""
    out = {}
    for r in rows:
        if r.get("record") != "b5_recall" or r.get("warmup"):
            continue
        e = out.setdefault(r["model_id"], {}).setdefault(r["precision"], {"window": r.get("num_ctx"), "recalls": [],
                                                                           "truncated": 0, "errors": 0})
        if r.get("outcome") != "ok":
            e["errors"] += 1
            continue
        e["recalls"].append(r["score"]["recall"])
        e["truncated"] += 1 if r.get("truncated") else 0
    for m in out.values():
        for e in m.values():
            e["recall_mean"] = st.mean(e["recalls"]) if e["recalls"] else None
            e["calls"] = len(e.pop("recalls"))
    return out


# ── hour estimates (labelled estimates; inputs from committed result files) ─────────────────────────

def _x2_session_minutes(repo=REPO) -> dict:
    """{(model, loaded_context): median minutes of a completed 40-turn real session} from x2_r2_real_v1/v1b."""
    rows = [r for f in ("x2_r2_real_v1.jsonl", "x2_r2_real_v1b.jsonl") for r in ag.read_rows(repo / "results" / f)]
    acc = {}
    for r in rows:
        if r.get("record") == "r2a_turn":
            k = (r["model_id"], r.get("loaded_context"), r["seed"], r["attempt_id"])
            a = acc.setdefault(k, [0.0, 0])
            a[0] += r.get("turn_wall_s") or 0.0
            a[1] += 1
    by = {}
    for (m, ctx, _s, _a), (sec, n) in acc.items():
        if n == TURNS:
            by.setdefault((m, ctx), []).append(sec / 60.0)
    return {k: st.median(v) for k, v in by.items()}


def _x2_rates(repo=REPO) -> dict:
    """{model: prompt-eval tok/s and eval tok/s} on the default-tier real sessions (turns 1-2, x2_r2_scale.rates)."""
    import x2_r2_scale as sc
    rows = [r for f in ("x2_r2_real_v1.jsonl", "x2_r2_real_v1b.jsonl") for r in ag.read_rows(repo / "results" / f)]
    out = {}
    for m in ("llama3.1:8b", "qwen3:14b", "qwen3:8b"):
        refs = sc.reference_sessions(rows, m, "ollama_default" + SFX, "real", turns=TURNS)
        if refs:
            out[m] = sc.pooled_rates(refs)
    return out


def _mitigation_factor(repo=REPO) -> float:
    """(c)'s render-and-count overhead: median minutes of x2_r2_mitigation_v1 sessions over x2_r2_real_v1 sessions at
    the same (model, num_ctx 4096 / 8192), floored at 1.0 (no speed-up is credited to trimming)."""
    mins = _x2_session_minutes(repo)
    rows = ag.read_rows(repo / "results" / "x2_r2_mitigation_v1.jsonl")
    acc = {}
    for r in rows:
        if r.get("record") == "r2a_turn":
            a = acc.setdefault((r["model_id"], r.get("loaded_context"), r["attempt_id"]), [0.0, 0])
            a[0] += r.get("turn_wall_s") or 0.0
            a[1] += 1
    ratios = []
    for (m, ctx, _a), (sec, n) in acc.items():
        if n == TURNS and (m, ctx) in mins:
            ratios.append((sec / 60.0) / mins[(m, ctx)])
    return max(1.0, st.median(ratios)) if ratios else 1.0


ESTIMATE_ASSUMPTIONS = {
    "evo-x2": "D = L = G = Lq8 = Lq4 = native context (evo-x2's Ollama default sizes to native; 128 GB unified memory), "
              "so (b) and (e) alias (a); session minutes = the measured x2_r2_real_v1/v1b median at the same loaded "
              "context",
    "evo-t2s": "OLLAMA_IGPU_ENABLE=1: D = 32768 (llama3.1:8b, measured), L = G = Lq8 = Lq4 = native 131072 (assumed: "
               "starts on the iGPU; probed at run time), (e) aliases (b); minutes = evo-x2 minutes at the same window "
               "x the iGPU/X2 latency ratio (scripts/t2s_week_queue.py input t2s_igpu_over_x2_ollama)",
    "blade": "qwen3:8b: D = 4096 and G = 8192 (blade_k1_v2.summary.json), L = Lq8 = Lq4 = native 40960 (assumed, partial "
             "offload); minutes = evo-x2 qwen3:8b minutes at the same window x analysis/blade_hours_estimate FACTORS "
             "(gpu_resident 1.0 up to 8192, edge 1.5 at 16384, offload 3.0 above)",
}
PROBE_H_PER_MODEL_ENV = 0.25            # nominal: a few one-token loads per server env


def _assumed_windows(platform: str, model: str, repo=REPO) -> dict:
    env = json.loads((repo / "src" / "dse" / "envelope_data.json").read_text(encoding="utf-8"))
    if platform == "evo-x2":
        key = model.replace(".", "").replace(":", "-")
        native = (env["machines"]["evo-x2"]["ollama_default_ctx"].get(key) or {}).get("ctx")
        return {"D": native, "L": {"base": native, "q8_0": native, "q4_0": native}, "G": native}
    if platform == "evo-t2s":
        return {"D": 32768, "L": {"base": 131072, "q8_0": 131072, "q4_0": 131072}, "G": 131072}
    k1 = json.loads((repo / "results" / "blade_k1_v2.summary.json").read_text(encoding="utf-8"))["models"][model]
    full = [int(c) for c, ok in (k1.get("fully_on_gpu_by_ctx") or {}).items() if ok]
    return {"D": k1.get("default_ctx"), "L": {"base": 40960, "q8_0": 40960, "q4_0": 40960},
            "G": max(full) if full else None}


def platform_factor(platform, window, repo=REPO) -> tuple[float, str]:
    """Time factor of a platform against evo-x2 at the same window (labelled estimate input)."""
    if platform == "evo-x2":
        return 1.0, "measured x2 median at this window"
    if platform == "evo-t2s":
        import importlib.util
        spec = importlib.util.spec_from_file_location("t2s_week_queue", repo / "scripts" / "t2s_week_queue.py")
        wq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wq)
        r = wq.ESTIMATE_INPUTS["t2s_igpu_over_x2_ollama"][0]
        return r, f"x2 median x iGPU/X2 ratio {r}"
    sys.path.insert(0, str(repo / "analysis"))
    import blade_hours_estimate as bhe
    f = bhe.FACTORS[bhe.CTX_FACTOR.get(window, "offload_32768")]
    return f, f"x2 median x Blade factor {f}"


def _minutes_at(platform, model, window, mins, repo=REPO):
    x2m = mins.get((model, window))
    f, basis = platform_factor(platform, window, repo)
    return (None if x2m is None else x2m * f), basis


def estimate_platform(platform: str, models=None, seeds=SEEDS, repo=REPO, deadline_h=None) -> dict:
    """Estimated hours of the B5 job on a platform (labelled estimate). Per model: the assumed windows, the resolved
    policies (with aliases), minutes per session per distinct policy, probe and recall allowances. evo-t2s-cpu: from
    measured evo-t2s session minutes (estimate_t2s)."""
    if platform == "evo-t2s-cpu":
        return estimate_t2s(platform, models, seeds if seeds != SEEDS else T2S_SEEDS, repo)
    models = list(models or PLATFORMS[platform]["models"])
    mins = _x2_session_minutes(repo)
    rates = _x2_rates(repo)
    mf = _mitigation_factor(repo)
    out = {"platform": platform, "label": "ESTIMATE (not measured)", "assumptions": ESTIMATE_ASSUMPTIONS[platform],
           "mitigation_factor": mf, "models": {}, "total_h": 0.0}
    order = []
    for model in models:
        win = _assumed_windows(platform, model, repo)
        cfg = resolve_policies(win)
        per = {}
        for p in distinct_policies(cfg):
            m, basis = _minutes_at(platform, model, cfg[p]["effective_ctx"], mins, repo)
            if m is not None and p == "c":
                m, basis = m * mf, basis + f" x trim overhead {mf:.2f}"
            per[p] = {"window": cfg[p]["effective_ctx"], "min_per_session": m, "basis": basis}
        sess_h = sum((v["min_per_session"] or 0.0) for v in per.values()) * len(seeds) / 60.0
        unknown = [p for p, v in per.items() if v["min_per_session"] is None]
        rate = rates.get(model) or {}
        rec_h = 0.0
        if rate.get("pe_rate"):
            for prec in RECALL_PRECISIONS:
                w = win["L"].get("base" if prec == "f16" else prec) or 0
                per_call = (RECALL_FILL * w) / rate["pe_rate"] + RECALL_NUM_PREDICT / (rate.get("ev_rate") or 20.0)
                rec_h += (1 + len(RECALL_SEEDS)) * per_call / 3600.0 * platform_factor(platform, w, repo)[0]
        probe_h = PROBE_H_PER_MODEL_ENV * (3 + 1)
        total = sess_h + rec_h + probe_h
        out["models"][model] = {"windows": win, "aliases": {p: c["alias_of"] for p, c in cfg.items() if c["alias_of"]},
                                "policies": per, "sessions_h": sess_h, "recall_h": rec_h, "probe_h": probe_h,
                                "total_h": total, "unknown_minutes": unknown}
        out["total_h"] += total
        for s in seeds:
            for p in RUN_ORDER:
                if p in per:
                    order.append((s, model, p, (per[p]["min_per_session"] or 0.0) / 60.0))
    if deadline_h is not None:
        cum, fit = 0.0, []
        for s, m, p, h in order:
            if cum + h > deadline_h:
                break
            cum += h
            fit.append((s, m, p))
        out["deadline_h"] = deadline_h
        out["sessions_within_deadline"] = fit
        out["sessions_total"] = len(order)
    return out


# ── platform servers ────────────────────────────────────────────────────────────────────────────────

class EvoServer:
    """evo-x2 / evo-t2s: host_config's WMI-launched `ollama serve` with this job's env and log path."""

    def __init__(self, hc, log_path):
        self.hc, self.log_path, self.env = hc, log_path, None

    def start(self, env: dict):
        self.hc.stop_ollama_server()
        ag._start_server(self.hc, dict(env) or None, self.log_path)
        if not self.hc.wait_for_ollama_ready(timeout_s=60):
            self.hc.stop_ollama_server()
            ag._start_server(self.hc, dict(env) or None, self.log_path)
            if not self.hc.wait_for_ollama_ready(timeout_s=90):
                raise RuntimeError("ollama did not become ready")
        self.env = dict(env)

    def stop(self):
        self.hc.stop_ollama_server()

    def log_text(self) -> str:
        try:
            return Path(self.log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def runtime(self):
        rt = ag.OllamaRuntime()
        rt.server_env, rt.server_log = (dict(self.env) or None), self.log_path
        return rt


class BladeServer:
    """The Blade: blade_common.LocalOllama (hidden local child, pinned 0.34.4 exe) with this job's env."""

    def __init__(self, exe, log_path):
        import blade_common as bc
        self.bc, self.exe, self.log_path, self.env, self.srv = bc, exe, log_path, None, None

    def start(self, env: dict):
        if self.srv is not None:
            self.srv.stop()
        self.srv = self.bc.LocalOllama(exe=self.exe, env=dict(env) or None, log_path=self.log_path)
        self.srv.start()
        if not self.srv.wait_ready():
            raise RuntimeError("ollama did not become ready")
        self.env = dict(env)

    def stop(self):
        if self.srv is not None:
            self.srv.stop()

    def log_text(self) -> str:
        try:
            return Path(self.log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def runtime(self):
        import blade_r2
        rt = blade_r2.BladeOllamaRuntime()
        rt.local_server = self.srv
        return rt


# ── runtime probes (need a server) ──────────────────────────────────────────────────────────────────

def native_context(runtime, model) -> int | None:
    status, data, _err, _dt = runtime._post("/api/show", {"model": model})
    info = (data or {}).get("model_info") or {}
    for k, v in info.items():
        if k.endswith(".context_length") and isinstance(v, (int, float)):
            return int(v)
    return None


def probe_load(runtime, model, num_ctx) -> dict:
    """One load: a one-token chat at num_ctx (None = not sent), then /api/ps; the model is unloaded afterwards."""
    runtime.unload(model)
    body = ag.native_chat_body(model, [{"role": "user", "content": "Reply with OK."}], num_ctx, None,
                               ag.MODELS.get(model))
    body["options"]["num_predict"] = 1
    t0 = time.monotonic()
    status, data, err, _dt = runtime._post("/api/chat", body)
    ps = {}
    for m in (runtime.client.get_ps().get("models") or []):
        if ag.same_tag(m.get("name") or m.get("model"), model):
            ps = m
    p = {"num_ctx_sent": num_ctx, "ok": status == 200 and data is not None, "http_status": status,
         "error": err, "loaded_context": ps.get("context_length"), "size": ps.get("size"),
         "size_vram": ps.get("size_vram"), "load_s": time.monotonic() - t0}
    runtime.unload(model)
    return p


def run_recall(runtime, model, num_ctx, precision, emit, log):
    """1 warm-up + len(RECALL_SEEDS) measured needle calls at num_ctx; the warm-up calibrates the token ratio."""
    ratio = RECALL_FIRST_RATIO
    target = int(RECALL_FILL * num_ctx)
    runtime.unload(model)
    for i, seed in enumerate((RECALL_WARMUP_SEED,) + RECALL_SEEDS):
        prompt, needles = build_recall_prompt(seed, target, ratio)
        est = ag.est_tokens(prompt)
        body = ag.native_chat_body(model, [{"role": "user", "content": prompt}], num_ctx, None, ag.MODELS.get(model))
        body["options"]["num_predict"] = RECALL_NUM_PREDICT
        resp = runtime._post_native(body)
        pec = resp.get("prompt_eval_count")
        loaded = runtime.loaded_context(model)
        trunc = bool(pec is not None and (pec < 0.95 * target * (0.7 if i == 0 else 1.0)
                                          or (loaded is not None and loaded < num_ctx)))
        emit({"record": "b5_recall", "model_id": model, "precision": precision, "num_ctx": num_ctx,
              "warmup": i == 0, "needle_seed": seed, "target_tokens": target, "est_tokens": est, "ratio_used": ratio,
              "prompt_eval_count": pec, "loaded_context": loaded, "truncated": trunc,
              "outcome": resp.get("outcome"), "http_status": resp.get("status"), "error": resp.get("error"),
              "duration_s": resp.get("duration_s"), "score": score_recall(resp.get("message"), needles),
              "answer": (resp.get("message") or "")[:600], "ts_utc": utc_iso()})
        if i == 0 and pec and est and pec < 0.95 * num_ctx:
            r = pec / est
            if ag.CALIB_RATIO_BOUNDS[0] <= r <= ag.CALIB_RATIO_BOUNDS[1]:
                ratio = r
        log(f"  recall {model} {precision} ctx={num_ctx} seed={seed} pec={pec} "
            f"found={score_recall(resp.get('message'), needles)['found']}")
    runtime.unload(model)


# ── job driver ──────────────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", choices=tuple(PLATFORMS), required=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-dir", default=None, help="where validation / price files are read (default: --out's dir)")
    ap.add_argument("--models", default=None, help="comma-separated (default: the platform's models)")
    ap.add_argument("--b1-models", default=None, help="evo-x2: comma-separated B1 models to consider (default: all)")
    ap.add_argument("--rules-from", default=None, help="comma-separated validation files (default: the platform's)")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--deadline-h", type=float, default=None, help="no new session starts after this many hours")
    ap.add_argument("--no-recall", action="store_true", help="skip the long-context recall probe")
    ap.add_argument("--ollama-exe", default=None, help="blade: default the pinned 0.34.4")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and estimated hours; no machine access")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--queue-insert", action="store_true")
    return ap


def b1_models_to_add(platform, candidates, rdir: Path) -> dict:
    """{model: {"add": bool, "reason"}}: a B1 model is added on evo-x2 only if it passes its own validation in
    x2_r2_scale_validation.jsonl (checked again by the preflight) and the H-scale probe projection gives minutes for
    its default tier with 6 distinct B5 policies x 3 seeds within B1_MODEL_CAP_H."""
    out = {}
    if platform != "evo-x2":
        return {m: {"add": False, "reason": "B1 models are evo-x2 only"} for m in candidates}
    prows = ag.read_rows(rdir / "x2_scale_timing_probe.jsonl")
    proj = next((r["projection"] for r in reversed(prows) if r.get("record") == "scale_projection"), None)
    for m in candidates:
        t = (((proj or {}).get("models") or {}).get(m) or {}).get("tiers", {}).get("ollama_default") or {}
        mps = t.get("minutes_per_session")
        if mps is None:
            out[m] = {"add": False, "reason": "no H-scale default-tier projection"}
            continue
        h = 6 * len(SEEDS) * mps / 60.0
        out[m] = {"add": h <= B1_MODEL_CAP_H, "projected_h": h,
                  "reason": f"projected {h:.1f} h {'<=' if h <= B1_MODEL_CAP_H else '>'} cap {B1_MODEL_CAP_H} h"}
    return out


def import_check() -> dict:
    """Imports everything a real run needs (router, cloud client, envelope, price table module, trim, num_keep) without
    a server; raises ImportError naming what is missing."""
    import host_config  # noqa: F401
    import x2_r2_client_trim  # noqa: F401
    import x2_num_keep  # noqa: F401
    import prompt_token_check  # noqa: F401
    make_router_factory("evo-x2", 4096)("llama3.1:8b")
    mods = {}
    for name in ("x2_r2_agent", "t2s_r2_session_growth", "t2s_k1_ollama", "host_config", "t2s_queue",
                 "x2_r2_client_trim", "x2_num_keep", "prompt_token_check", "proc_util", "src.dse.router",
                 "src.cloud.client", "envelope_model", "make_failure_map"):
        m = sys.modules.get(name)
        if m is not None and getattr(m, "__file__", None):
            mods[name] = {"file": str(Path(m.__file__).resolve()), "blob": git_blob_sha1(m.__file__)}
    return {"agent_blob": git_blob_sha1(ag.__file__),
            "agent_has_options_extra": hasattr(ag.OllamaRuntime, "options_extra"),
            "policies": {p: arm_id(p) for p in POLICIES}, "repo": str(REPO), "modules": mods,
            "queue_entry": queue_entry()["cmd"]}


def dry_run(args) -> dict:
    plats = [args.platform] if args.platform else list(PLATFORMS)
    out = {"label": "DRY RUN: plan and ESTIMATED hours (estimates, not measurements); nothing was started",
           "policies": {p: dict(POLICIES[p], arm=arm_id(p)) for p in POLICIES}, "seeds": list(SEEDS),
           "turns": TURNS, "platforms": {}}
    for plat in plats:
        models = [m for m in (args.models.split(",") if args.models else PLATFORMS[plat]["models"]) if m]
        e = estimate_platform(plat, models, deadline_h=args.deadline_h)
        e["queue_entry"] = queue_entry()["cmd"] if plat == "evo-x2" else None
        out["platforms"][plat] = e
    return out


def queue_entry(results_dir=r"C:\apu\ovn\results") -> dict:
    return {"id": JOB_ID, "cmd": [PYTHON_EXE, SIDE_DIR + "\\policy_compare.py", "--platform", "evo-x2", "--out",
                                  f"{results_dir}\\{PLATFORMS['evo-x2']['out']}"],
            "status": "pending", "projected_duration_h": None,
            "note": "B5 policy comparison (policies a-h incl. KV q8_0/q4_0), pre-registered in docs/FINDINGS.md "
                    "2026-10-14. Estimated hours: register row b5-plan-hours (estimate). Releases itself via "
                    "tq.advance() on exit; sets no pause."}


def insert_position(items: list[dict]) -> int:
    """Immediately before the first pending BEFORE_IDS entry, else at the end. Raises if the id is already queued or if
    a listed predecessor (AFTER_IDS) would come after the insert point."""
    if any(it["id"] == JOB_ID for it in items):
        raise ValueError(f"{JOB_ID} is already in the queue")
    pos = next((i for i, it in enumerate(items) if it["id"] in BEFORE_IDS and it.get("status") == "pending"),
               len(items))
    late = [it["id"] for it in items[pos:] if it["id"] in AFTER_IDS]
    if late:
        raise ValueError(f"predecessors {late} would run after {JOB_ID}")
    return pos


def queue_insert(log=print) -> list[dict]:
    import t2s_queue as tq
    tq.set_pause("policy_compare --queue-insert: one read-modify-write inserting x2_policy_compare before "
                 "x2_window_override_methods (else at the end); released by clear_pause() in this same script's "
                 "finally block within seconds")
    try:
        items = tq.read_queue()
        pos = insert_position(items)
        items.insert(pos, queue_entry())
        tq.write_queue(items)
        ids = [it["id"] for it in items]
        log(json.dumps({"inserted": JOB_ID, "position": ids.index(JOB_ID),
                        "after": [i for i in AFTER_IDS if i in ids], "n_items": len(ids)}))
        return items
    finally:
        tq.clear_pause()


def main(argv=None, advance=True, deps=None):
    """deps (tests): {"hc", "server", "runtime", "hostname", "make_router", "measure", "counter", "tq", "prices"}."""
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(args), indent=1, default=str))
        return "dry-run"
    if args.import_check:
        print(json.dumps(import_check(), indent=1, default=str))
        return "import-check"
    if args.queue_insert:
        queue_insert()
        return "queue-insert"
    if not args.platform or not args.out:
        ap.error("--platform and --out are required for a job")
    deps = deps or {}
    plat = PLATFORMS[args.platform]
    out_path = Path(args.out)
    if not out_path.name.startswith(plat["prefix"]):
        ap.error(f"{args.platform} output files are prefixed {plat['prefix']}: {out_path.name!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rdir = Path(args.results_dir) if args.results_dir else out_path.parent
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS
    t_start = time.monotonic()

    def log(msg):
        print(f"[{utc_iso()}] {msg}", flush=True)

    def emit(row):
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    note, rc = "completed", 0
    server = None
    try:
        import socket
        hostname = deps.get("hostname") or socket.gethostname()
        if hostname.upper() != plat["host"]:
            raise RuntimeError(f"policy_compare --platform {args.platform}: host {hostname!r} is not {plat['host']}")
        interactive = None
        if args.platform == "blade":
            neg_arm = None
            import blade_r2
            blade_r2.register_blade_arms()
            neg_arm = blade_r2.BLADE_NEG_ARM
        else:
            hc = deps.get("hc")
            if hc is None:
                import host_config as hc
            host_cfg = hc.require_host(hostname)
            try:
                interactive = hc.enforce_or_record_interactive_session(host_cfg)
            except SystemExit as e:
                note = f"stopped: {e}"[:400]
                log(note)
                return note
            neg_arm = ag.NEG_ARM
        models = [m for m in (args.models.split(",") if args.models else plat["models"]) if m]
        b1 = b1_models_to_add(args.platform, [m for m in (args.b1_models.split(",") if args.b1_models is not None
                                                          else plat["b1_models"]) if m], rdir)
        models += [m for m, d in b1.items() if d["add"]]
        rule_files = [p for p in (args.rules_from.split(",") if args.rules_from else plat["rules_from"]) if p]
        vfiles = [(Path(p).name, ag.read_rows(Path(p) if Path(p).is_absolute() or Path(p).exists() else rdir / p))
                  for p in rule_files]
        pf = ag.validation_preflight_per_model(vfiles, CALL2, models, neg_arm=neg_arm)
        refused, rules_in_use = pf["refused"], pf["rules_in_use"]
        models = list(pf["models_ok"])
        prices_mod = None
        # refuse before any model call when the router stack is missing (it is needed for policy d)
        make_router = deps.get("make_router")
        if make_router is None:
            sys.path.insert(0, str(REPO / "analysis"))
            make_router_factory(args.platform, 4096, ledger_dir=out_path.parent)  # import check
        emit({"record": "run_start", "mode": MODE, "platform": args.platform, "models": models,
              "models_requested": args.models, "b1_decision": b1, "rules_from": rule_files, "rules_in_use": rules_in_use,
              "validation_preflight": {k: pf[k] for k in ("ok", "models_ok", "refused", "source")},
              "seeds": list(seeds), "turns": TURNS, "call2_mode": CALL2, "policies": {p: arm_id(p) for p in POLICIES},
              "run_order": RUN_ORDER, "base_env": plat["base_env"], "agent_file": str(Path(ag.__file__).resolve()),
              "agent_blob": git_blob_sha1(ag.__file__), "self_blob": git_blob_sha1(__file__),
              "interactive_session": interactive, "deadline_h": args.deadline_h,
              "max_tokens_per_call": ag.MAX_TOKENS_PER_CALL, "keep_alive": ag.KEEP_ALIVE,
              "assumptions": {"A-B5-cloud": "cloud step = same local model, full history, window L",
                              "A-B5-cloud-latency": [CLOUD_TTFT_S_ASSUMED, CLOUD_DECODE_TOK_S_ASSUMED]},
              "ts_utc": utc_iso()})
        for m, reasons in refused.items():
            emit({"record": "refused_model", "model_id": m, "reasons": reasons, "ts_utc": utc_iso()})
            log(f"refused model {m}: {'; '.join(reasons)}")
        if not models:
            note = ("completed; refused to start, no model passed validation: "
                    + "; ".join(r for rs in refused.values() for r in rs))[:400]
            emit({"record": "refused", "reasons": refused, "ts_utc": utc_iso()})
            emit({"record": "run_end", "note": note, "ts_utc": utc_iso()})
            log(note)
            return note if args.platform != "blade" else 0
        if args.platform != "blade":
            import x2_num_keep as nk
            if deps.get("server") is None and nk.tray_app_running():
                raise RuntimeError("the Ollama tray app is running; B5 starts its own servers and does not stop the "
                                   "tray app")
        server = deps.get("server")
        if server is None:
            if args.platform == "blade":
                import blade_common as bc
                server = BladeServer(args.ollama_exe or bc.PINNED["ollama_exe"],
                                     bc.LOG_DIR / f"{out_path.stem}.ollama_serve.log")
            else:
                server = EvoServer(hc, plat["server_log"])
        base_env = dict(plat["base_env"])

        def ensure_env(env_key):
            want = server_env_for(base_env, env_key)
            if server.env != want:
                server.start(want)
                rt = deps.get("runtime") or server.runtime()
                return rt, True
            return None, False

        # ── 1. windows per model and env
        runtime, _ = ensure_env("base")
        avail = runtime.available_models()
        missing = [m for m in models if not any(ag.same_tag(m, a) for a in avail)]
        for m in missing:
            emit({"record": "missing_model", "model_id": m, "available": avail, "ts_utc": utc_iso()})
        models = [m for m in models if m not in missing]
        emit({"record": "runtime", "ollama_version": runtime.version(), "models_run": models, "ts_utc": utc_iso()})
        wins = {m: {"L": {}, "points": {}} for m in models}
        kv_ok = {}
        for env_key in ("base", "q8_0", "q4_0"):
            rt, _ = ensure_env(env_key)
            runtime = rt or runtime
            for m in models:
                native = native_context(runtime, m)
                wins[m]["native"] = native
                if env_key == "base":
                    dp = probe_load(runtime, m, None)
                    wins[m]["D"] = dp["loaded_context"] if dp["ok"] else None
                    wins[m]["points"]["default"] = dp
                pts = descend(lambda c: probe_load(runtime, m, c), ladder_for(native), need_full_gpu=(env_key == "base"))
                wins[m]["points"][env_key] = pts
                w = windows_from_points(pts)
                wins[m]["L"][env_key] = w["largest_starts"]
                if env_key == "base":
                    wins[m]["G"] = w["largest_full_gpu"]
                if env_key != "base":
                    chk = kv_check(server.log_text(), env_key)
                    kv_ok[(m, env_key)] = chk
                    emit({"record": "b5_kv_check", "model_id": m, "env": env_key, "check": chk, "ts_utc": utc_iso()})
                log(f"windows {m} {env_key}: native={native} L={wins[m]['L'][env_key]} "
                    f"D={wins[m].get('D')} G={wins[m].get('G')}")
        configs = {}
        for m in models:
            configs[m] = resolve_policies(wins[m])
            for env_key in ("q8_0", "q4_0"):
                chk = kv_ok.get((m, env_key)) or {}
                if not chk.get("valid", True):
                    for p in POLICIES:
                        if configs[m][p]["env"] == env_key and not configs[m][p]["skip"]:
                            configs[m][p]["skip"] = f"KV type not applied: {chk.get('reason')}"
            emit({"record": "b5_resolution", "model_id": m, "windows": wins[m], "configs": configs[m],
                  "distinct": distinct_policies(configs[m]), "ts_utc": utc_iso()})
        # ── 2. sessions
        prices = deps.get("prices")
        done = {(s["model_id"], s["arm_id"], s["seed"]) for s in ag.completed_sessions(ag.read_rows(out_path))
                if s.get("mode") == MODE}
        measure = deps.get("measure")
        counter = deps.get("counter")
        stopped_by_deadline = False
        for seed, model, p in session_items(models, configs, seeds):
            cfg = configs[model][p]
            if (model, cfg["arm_id"], seed) in done:
                log(f"skip (done): {model} {cfg['arm_id']} seed={seed}")
                continue
            if args.deadline_h is not None and (time.monotonic() - t_start) / 3600.0 >= args.deadline_h:
                stopped_by_deadline = True
                log(f"deadline {args.deadline_h} h reached; no new session")
                break
            if handover_deadline_reached(args.platform, deps):
                stopped_by_deadline = True
                log("handover deadline reached (t2s_queue.deadline_reached); no new session")
                break
            rt, _ = ensure_env(cfg["env"])
            runtime = rt or runtime
            ag.ARMS[cfg["arm_id"]] = {"num_ctx": cfg["num_ctx_sent"], "call2_tools": False, "call2_mode": CALL2}
            if cfg["wrapper"] == "client_trim":
                import x2_r2_client_trim as ct
                wrt = TrimRuntime(runtime, counter or ct.RenderTokenCounter(runtime))
            elif cfg["wrapper"] == "num_keep":
                wrt = NumKeepPolicyRuntime(runtime, measure or default_measure(runtime), emit)
            elif cfg["wrapper"] == "router":
                mk = make_router or make_router_factory(args.platform, cfg["effective_ctx"],
                                                        ledger_dir=out_path.parent)
                wrt = RouterPolicyRuntime(runtime, mk, cfg["effective_ctx"], cfg["proxy_ctx"], emit)
            else:
                wrt = PlainRuntime(runtime)
            emit({"record": "heartbeat", "mode": MODE, "model_id": model, "arm_id": cfg["arm_id"], "seed": seed,
                  "policy": p, "config": cfg, "server_env": server.env, "ts_utc": utc_iso()})
            log(f"session: {model} policy {p} ({cfg['name']}) ctx={cfg['effective_ctx']} seed={seed}")
            wrt.begin_session(model, cfg["arm_id"], seed)
            try:
                s = ag.run_session(wrt, model, cfg["arm_id"], seed, TURNS, emit, log, MODE)
            finally:
                wrt.end_session()
            log(f"session done: first_failure={s['first_failure_turn']} canary_misses="
                f"{s['canary_sys_misses']}+{s['canary_hist_misses']}")
        # ── 3. long-context recall
        if not args.no_recall and not stopped_by_deadline:
            for prec in RECALL_PRECISIONS:
                env_key = "base" if prec == "f16" else prec
                rt, _ = ensure_env(prec)
                runtime = rt or runtime
                for m in models:
                    w = wins[m]["L"].get(env_key)
                    if w is None:
                        emit({"record": "b5_recall_skipped", "model_id": m, "precision": prec,
                              "reason": "no window started", "ts_utc": utc_iso()})
                        continue
                    run_recall(runtime, m, w, prec, emit, log)
        # the report is a convenience (the register recomputes everything on the controller); it never fails the job
        try:
            if prices is None:
                prices = load_cloud_prices(REPO)
        except Exception:
            prices = None
        try:
            rows = ag.read_rows(out_path) + [{"record": "run_end"}]
            report = {"platform": args.platform, "metrics": metrics_for_platform(rows, prices),
                      "recall": recall_summary(rows), "refused": refused, "missing": missing,
                      "deadline_reached": stopped_by_deadline, "prices_loaded": prices is not None}
        except Exception as e:
            report = {"platform": args.platform, "report_error": repr(e)}
        Path(str(out_path) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        if stopped_by_deadline:
            note = (f"completed; deadline {args.deadline_h} h reached before every session ran"
                    if args.deadline_h is not None and (time.monotonic() - t_start) / 3600.0 >= args.deadline_h
                    else "completed; handover deadline reached before every session ran (resumable)")
        elif refused or missing:
            note = f"completed; refused: {sorted(refused)}; missing: {missing}"[:400]
        emit({"record": "run_end", "note": note, "ts_utc": utc_iso()})
    except Exception as e:
        import traceback
        note, rc = f"stopped: {e!r}"[:400], 2
        log(note)
        traceback.print_exc()
    finally:
        if server is not None:
            try:
                server.stop()
            except Exception as e:
                log(f"ollama stop failed: {e!r}")
        if advance and plat["queue"]:
            try:
                tq = deps.get("tq")
                if tq is None:
                    import t2s_queue as tq
                tq.advance(note)
            except Exception as e:
                log(f"queue advance failed: {e!r}")
    return note if plat["queue"] else rc


def default_measure(runtime):
    """x2_num_keep's measurement of S for one session (render and count with the model's tokenizer)."""
    import x2_num_keep as nk
    counters = {}

    def measure(model, arm, seed):
        ags = ag.build_agent_session(seed, TURNS)
        first_user = ag.user_message(ags.spec.turns[0], ags)
        if model not in counters:
            counters[model] = nk.tokenizer_count_fn(model)

        def render(m, msgs, num_ctx, tools, think):
            return runtime.render_only(m, msgs, num_ctx, tools, think)
        return nk.measure_system_tokens(render, counters[model], model, ags.system_prompt, first_user,
                                        ag.ARMS[arm]["num_ctx"], ag.MODELS.get(model), ag.r2.ollama_tools_payload())
    return measure


if __name__ == "__main__":
    r = main()
    if isinstance(r, int):
        sys.exit(r)
