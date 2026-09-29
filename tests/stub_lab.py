"""Shared stub Lab/Server fixtures for dry-running a new or changed t2s_night2.py phase against every real call site
(L.Server, ov.start_row, ov.prompt_for, ov.measured_sequence/do_call, lab.row/emit/item_done/check) with realistic
arguments, but no real server process and no real co-runner. Exists because two crashes last week (the C1
duplicate-keyword TypeError, see tests/test_no_duplicate_row_kwargs.py) were the kind of thing a dry run against a
stub lab catches before a phase ever reaches a live machine -- every new or changed phase must be run through this
before it is queued.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))


class StubModelInfo:
    """Enough of t2s_lab.ModelInfo's public surface for a phase function to run without touching a real GGUF file."""
    def __init__(self, model_id, max_ctx_native=131072):
        self.model_id = model_id
        self.max_ctx_native = max_ctx_native
        self.path = f"C:\\apu\\models\\{model_id}.gguf"
        self.sha256 = "0" * 64
        self.quant = "Q4_K_M"


class StubServer:
    """Mimics t2s_lab.Server's public interface used by do_call/measured_sequence/start_row/prompt_for. tokenize()
    uses a fixed chars-per-token estimate so context.build_filler's iterative sizing loop (bounded to 4 iterations
    regardless) terminates without a real tokenizer."""
    def __init__(self, lab, mi, n_ctx, tag=None, mmap=None, backend="vulkan", extra=(), load_mode=None, ngl=99, fit=None):
        self.lab, self.mi, self.n_ctx, self.tag = lab, mi, n_ctx, tag
        self.mmap = mmap if mmap is not None else True
        self.load_mode = load_mode or "auto"
        self.backend = backend
        self.extra = list(extra)
        self.ngl, self.fit = ngl, fit
        self.pid = 4242
        self.start_info = {}
        self.log_path = "C:\\apu\\ovn\\results\\stub_dryrun_srv.txt"  # never actually written; read_logs()-style
        self._alive = True                                            # readers must tolerate a missing file

    def start(self, timeout=600):
        t0 = 1000000.0
        info = {"ok": True, "load_s": 3.0, "error": None, "exit_code": None, "pid": self.pid, "log": {},
                "build": "b10970-stub", "t_start": t0, "t_end": t0 + 3.0, "private_mib": 1000.0, "working_set_mib": 1200.0}
        self.start_info = info
        return info

    def _cmd(self):
        return ["llama-server.exe", "-m", getattr(self.mi, "path", "stub.gguf"), "-c", str(self.n_ctx)]

    def tokenize(self, prompt):
        return max(1, len(prompt) // 4)

    def alive_and_ours(self):
        return self._alive, [self.pid] if self._alive else []

    def chat(self, prompt, max_tokens, ignore_eos=True):
        return {"outcome": "ok", "ttft_s": 0.5, "decode_tok_s": 20.0, "e2e_s": 1.0, "output": "stub output",
                "completion_tokens": max_tokens, "think_tag": False, "error": None}

    def stop(self):
        self._alive = False


class _StubTele:
    @staticmethod
    def thermal_gate(*a, **k):
        return {"thermal_wait_s": 0.0, "gate_released_by": "none", "temp_at_gate_start": None,
                "temp_at_release": None, "idle_temp": None, "idle_pkg_w": None, "pkg_w_at_release": None}

    @staticmethod
    def metrics(*a, **k):
        return {"igpu_mhz": None, "igpu_throttle_bits": None, "pkg_power_w": None, "rapl_pp0_w": None,
                "rapl_pp1_w": None, "cpu_p_pct_perf": None, "cpu_e_pct_perf": None, "cpu_lpe_pct_perf": None,
                "igpu_power_w": None, "igpu_temp_c_max": None, "shared_usage_mib": None, "total_committed_mib": None,
                "pages_input_per_s": None, "hard_faults_per_s": None, "avail_mb_min": None, "avail_mb_max": None,
                "temp_c_max": None}

    @staticmethod
    def pkg_now():
        return None

    @staticmethod
    def temp_now():
        return None


class _StubScorers:
    @staticmethod
    def score(probe_dict, output):
        return 1.0, "stub_score"


class StubLab:
    """Enough of t2s_overnight.Lab's public surface for a phase function to run end to end. .rows collects every
    emitted row so a test can assert on axis tags, item_ids, section labels, etc."""
    def __init__(self, models=None):
        self.models = models or {}
        self.resources = {}
        self.done = set()
        self.rows = []
        self.records = []
        self.smoke = False
        self.idle_temp = None
        self.idle_pkg = None
        self.table = {}
        self.prefix = "C:\\apu\\ovn\\results\\stub_dryrun"
        self.tele = _StubTele()
        self.scorers = _StubScorers()

    def check(self):
        pass

    def row(self, section, mi, backend, **kw):
        r = dict(section=section, model_id=getattr(mi, "model_id", mi), backend=backend, **kw)
        return r

    def emit(self, row):
        self.rows.append(row)

    def item_done(self, item):
        self.done.add(item)
