"""do_call() builds its row dict as dict(<reserved keys>, **gate, **extra); any extra={} that also sets one of those
reserved keys (n_ctx, prompt_tokens, mmap, load_mode, co_runner, rep, mem_headroom_gb, load_s, item_id, kind, warmup,
server_pid, llama_build, or any thermal-gate field) raises a TypeError at call time -- found live in
t2s_night2.py's phase_c1 (mem_headroom_gb in both extra and the explicit kwarg), fixed in commit 300679f.

This both (a) regression-tests that fix and the other real call sites, and (b) is a static/grep sweep the way the
harness call sites were audited: no extra={...} literal or extra=dict(...) anywhere in the harness may set a
reserved key.
"""

import re
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import t2s_overnight as ov  # noqa: E402

RESERVED = {"n_ctx", "prompt_tokens", "mmap", "load_mode", "co_runner", "rep", "mem_headroom_gb", "load_s", "item_id",
            "kind", "warmup", "server_pid", "llama_build", "thermal_wait_s", "temp_at_gate_start", "temp_at_release",
            "idle_temp", "idle_pkg_w", "pkg_w_at_release", "gate_released_by"}


class _StubSrv:
    n_ctx, prompt_tokens, mmap, load_mode, backend = 8192, 100, False, "auto", "vulkan"
    pid = 1234
    start_info = {"load_s": 5.0, "build": "b10970"}

    def alive_and_ours(self):
        return True, []

    def chat(self, prompt, max_tokens, ignore_eos):
        return {"outcome": "ok", "ttft_s": 1.0, "decode_tok_s": 10.0, "e2e_s": 2.0, "output": "x",
                "completion_tokens": 5, "think_tag": False}


class _StubLab:
    idle_temp = idle_pkg = None

    def check(self):
        pass

    class tele:
        @staticmethod
        def thermal_gate(*a, **k):
            return {"thermal_wait_s": 0.0, "gate_released_by": "none"}

        @staticmethod
        def metrics(*a, **k):
            return {"igpu_mhz": None, "igpu_throttle_bits": None, "pkg_power_w": None, "rapl_pp0_w": None,
                    "rapl_pp1_w": None, "cpu_p_pct_perf": None, "cpu_e_pct_perf": None, "cpu_lpe_pct_perf": None,
                    "igpu_power_w": None, "igpu_temp_c_max": None, "shared_usage_mib": None, "total_committed_mib": None,
                    "pages_input_per_s": None, "hard_faults_per_s": None, "avail_mb_min": None, "avail_mb_max": None,
                    "temp_c_max": None}

    def row(self, section, mi, backend, **kw):
        return dict(kw)

    def emit(self, row):
        pass


def test_do_call_accepts_every_real_extra_dict_without_typeerror():
    """Representative extra={} payloads pulled from the real call sites, including the now-fixed C1 one."""
    payloads = [
        {},  # phase_c1 after the fix
        {"cpu_mask": "0xFFF0", "duty_cycle_pct": 100},  # night2 B1/B2
        {"cell": "nonp12", "cap": 100, "anchor": False},  # overnight Section B
        {"need_mib": 1000.0, "exceeds_budget": {"a": True}, "beyond_trained_ctx": False, "anchor": False,
         "fill_capped": False, "rope_flags": None},  # overnight Section A
        {"target_avail_mb": 4096, "need_mib": 1000.0, "anchor": False, "avail_before_server_mb": 5000},  # Section C
    ]
    for extra in payloads:
        with mock.patch.object(ov.time, "sleep", lambda *a: None):
            r = ov.do_call(_StubLab(), _StubSrv(), mi=mock.Mock(), section="X", item_id="i", prompt="p", n_tok=10,
                           warmup=False, rep=0, extra=extra)
        assert r is not None
        assert r["outcome"] == "ok"


def test_no_reserved_key_in_any_extra_literal_across_the_harness():
    """Static sweep: no extra={...} dict literal anywhere sets a reserved do_call() key. A hit here means a future
    TypeError like the C1 one is waiting to happen the first time that code path actually runs."""
    harness = Path(__file__).resolve().parents[1] / "harness"
    pat = re.compile(r'extra\s*=\s*\{([^}]*)\}')
    key_pat = re.compile(r'["\'](\w+)["\']\s*:')
    offenders = []
    for py in harness.glob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for m in pat.finditer(text):
            keys = set(key_pat.findall(m.group(1)))
            hit = keys & RESERVED
            if hit:
                offenders.append((py.name, hit, m.group(0)[:80]))
    assert not offenders, f"reserved keys found in extra={{}} literals: {offenders}"
