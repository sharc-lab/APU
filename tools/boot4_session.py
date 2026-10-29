"""Shared boot-4 measurement session: gates, canary, per-point files, seal.

The runners call this module. It does not open a preregistration or an amendment.
Warm points are n_cached 12000, 24000, and 46000, delta text of 183 tokens, three
turn-2 repeats. Decode length comes from configs/delta_n.yaml acceptance.max_new_tokens.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import statistics
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

STUB_MARK = "measurement body is not started"
N_CACHED = (12000, 24000, 46000)
DELTA_TOKENS = 183
TURN2_REPEATS = 3
WARM_PLANNED_PROBES = 4
SMOKE_N_CACHED = 64
SMOKE_DELTA_TOKENS = 8
DECODE_N = (2000, 4000, 8000)
DECODE_REPEATS = 3
SMOKE_DECODE_N = 32
RESIDENT_LOW = 12000
RESIDENT_HIGH = 30000
RESIDENT_RESOLUTION = 500
# floor(4 / (C + 1)) is 1, so a canary follows every probe. The search
# itself visits at most low, high, and ceil(log2((high-low)/resolution)) mids.
RESIDENT_PLANNED_PROBES = 4
# DEMO-2 pilots P2/P3 (Task AS): an explicit SchedulerConfig for every child
# spec (it replaces GenAI's latency defaults) and a higher resident-ladder top.
# None = unchanged. Set only by tools/run_resident_limit.py / run_warm_kv.py flags.
SCHEDULER_OVERRIDE: dict[str, Any] | None = None
RESIDENT_HIGH_OVERRIDE: int | None = None


def set_pilot_overrides(
    *, scheduler: dict[str, Any] | None = None, resident_high: int | None = None
) -> None:
    global SCHEDULER_OVERRIDE, RESIDENT_HIGH_OVERRIDE
    SCHEDULER_OVERRIDE = dict(scheduler) if scheduler else None
    RESIDENT_HIGH_OVERRIDE = int(resident_high) if resident_high else None


def _resident_high() -> int:
    return RESIDENT_HIGH_OVERRIDE or RESIDENT_HIGH


# CC cells (Task BJ, docs/registrations/CC_CONTEXT_CEILING_2026-10-23.md): bisection grid,
# fixed points instead of bisection, turn-2 decode length and repeats, and child-side memory
# and compiled-KV records. Empty = unchanged. Set only by tools/run_resident_limit.py flags.
CC_OVERRIDE: dict[str, Any] = {}
CC_CHILD_FLAGS = (
    "gpu_mem_stats",
    "compiled_readback",
    "dxgi_trace",
    "reject_kv_mismatch",
    "trial_telemetry",
)
# Task BQ balloon-gate option (a): CC-BAL cells only; the balloon is tools/gpu_balloon.py, or
# (Task BR INV-1) the pinned balloon tools/pinned_balloon.py.
BALLOON_CELL_PREFIX = "CC-BAL-"
BALLOON_SCRIPT = "gpu_balloon.py"
BALLOON_SCRIPTS = (BALLOON_SCRIPT, "pinned_balloon.py")


def set_cc_overrides(
    *,
    cell: str | None = None,
    low: int | None = None,
    resolution: int | None = None,
    points: tuple[int, ...] | None = None,
    turn2_new_tokens: int | None = None,
    turn2_repeats: int | None = None,
    gpu_mem_stats: bool = False,
    compiled_readback: bool = False,
    probe_timeout_s: float | None = None,
    mem_trace_hz: float | None = None,
    grow: dict[str, int] | None = None,
    conc: dict[str, Any] | None = None,
    dxgi_trace: bool = False,
    reject_kv_mismatch: bool = False,
    canary_model_spec: str | None = None,
    balloon_floor_credit: bool = False,
    trial_telemetry: bool = False,
    bcd: dict[str, Any] | None = None,
) -> None:
    global CC_OVERRIDE
    out: dict[str, Any] = {}
    for key, value in (
        ("cell", cell),
        ("low", low),
        ("resolution", resolution),
        ("turn2_new_tokens", turn2_new_tokens),
        ("turn2_repeats", turn2_repeats),
        ("probe_timeout_s", probe_timeout_s),
        ("mem_trace_hz", mem_trace_hz),
        ("grow", grow),
        ("conc", conc),
        ("bcd", bcd),
    ):
        if value is not None:
            out[key] = value
    if points:
        out["points"] = tuple(int(n) for n in points)
    if gpu_mem_stats:
        out["gpu_mem_stats"] = True
    if compiled_readback:
        out["compiled_readback"] = True
    if dxgi_trace:
        out["dxgi_trace"] = True
    if reject_kv_mismatch:
        out["reject_kv_mismatch"] = True
    if canary_model_spec:
        out["canary_model_spec"] = str(canary_model_spec)
    if balloon_floor_credit:
        out["balloon_floor_credit"] = True
    if trial_telemetry:
        out["trial_telemetry"] = True
    CC_OVERRIDE = out


def _cc_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Harness config for one CC probe: a longer child timeout if --probe-timeout-s is set
    (Task BL A4). Without it the config object is returned unchanged."""
    t = CC_OVERRIDE.get("probe_timeout_s")
    if not t:
        return cfg
    out = copy.deepcopy(cfg)
    out["generation"]["timeout_s"] = float(t)
    return out


def model_window(model_dir: str | Path) -> dict[str, Any]:
    """Task BL A5: the checkpoint's position window, read from its config.json."""
    path = Path(model_dir) / "config.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300], "path": str(path)}
    return {
        "max_position_embeddings": doc.get("max_position_embeddings"),
        "rope_scaling": doc.get("rope_scaling"),
        "rope_theta": doc.get("rope_theta"),
        "path": str(path),
    }


def validity_fields(largest: int | None, window: dict[str, Any]) -> dict[str, Any]:
    """Task BL A5: L_mem, L_valid = min(L_mem, max_position_embeddings), and the flag."""
    mpe = window.get("max_position_embeddings")
    if largest is None or not isinstance(mpe, int):
        return {"L_mem": largest, "L_valid": None, "past_trained_positions": None}
    return {
        "L_mem": int(largest),
        "L_valid": min(int(largest), mpe),
        "past_trained_positions": int(largest) > mpe,
    }


def _cc_canary_specs(harness: dict[str, Any]) -> list[Path]:
    """The measured model first, then --canary-model-spec if set (Task BO: a second-model cell
    keeps the registered Qwen3-4B canary). Without the option the list is unchanged."""
    specs = [harness["model_spec"]]
    extra = CC_OVERRIDE.get("canary_model_spec")
    if extra:
        specs.append(Path(str(extra)))
    return specs


def _resident_low() -> int:
    return int(CC_OVERRIDE.get("low") or RESIDENT_LOW)


def _resident_resolution() -> int:
    return int(CC_OVERRIDE.get("resolution") or RESIDENT_RESOLUTION)


def runner_body_is_stub(path: Path) -> bool:
    return STUB_MARK in path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    """JSON written by this process or by PowerShell. Accepts a UTF-8 BOM."""
    from seam.json_io import load_json

    return load_json(path)


def advantage_percent(*, median_f16: float, median_other: float) -> float:
    """(median_other - median_f16) / median_other * 100. Lower TTFT is the advantage."""
    if median_other == 0:
        raise ValueError("median_other is 0")
    return (float(median_other) - float(median_f16)) / float(median_other) * 100.0


def grid_mid(lo: int, hi: int, resolution: int) -> int | None:
    """Next rung strictly between lo and hi, snapped down onto the resolution grid."""
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    if hi - lo <= resolution:
        return None
    snapped = ((lo + hi) // 2 // resolution) * resolution
    if snapped <= lo:
        snapped = lo + resolution
    if snapped >= hi:
        return None
    return snapped


def _passing_turn2(points: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for point in points:
        if not point.get("holds"):
            continue
        ttft = point.get("turn2_ttft_s")
        if ttft is None:
            continue
        out[str(point["n_cached"])] = float(ttft)
    return out


def bisect_holds(
    *,
    low: int,
    high: int,
    resolution: int,
    attempt: Any,
) -> dict[str, Any]:
    """Largest n on the grid that holds. Each rung is attempted once."""
    if (high - low) % resolution != 0:
        raise ValueError("high - low must be a multiple of resolution")
    seen: set[int] = set()
    points: list[dict[str, Any]] = []

    def once(n: int) -> dict[str, Any]:
        if n in seen:
            raise RuntimeError(f"rung {n} already attempted")
        if (n - low) % resolution != 0:
            raise RuntimeError(f"rung {n} is off the {resolution} grid")
        seen.add(n)
        point = attempt(n)
        points.append(point)
        return point

    low_point = once(low)
    if not low_point["holds"]:
        return {
            "status": "complete",
            "largest_n_cached": None,
            "failure_in_range": True,
            "points": points,
            "passing_turn2_ttft_s": {},
        }
    high_point = once(high)
    lo = low
    hi = high
    if high_point["holds"]:
        return {
            "status": "complete",
            "largest_n_cached": high,
            "failure_in_range": False,
            "note": "high held; the limit is above the search range",
            "points": points,
            "passing_turn2_ttft_s": _passing_turn2(points),
        }
    while True:
        mid = grid_mid(lo, hi, resolution)
        if mid is None:
            break
        point = once(mid)
        if point["holds"]:
            lo = mid
        else:
            hi = mid
    return {
        "status": "complete",
        "largest_n_cached": lo,
        "first_fail_n_cached": hi,
        "failure_in_range": True,
        "points": points,
        "passing_turn2_ttft_s": _passing_turn2(points),
    }


def bcd_next(level: int, held: bool, u: float, *, step: int, b: float, low: int, high: int) -> int:
    """Biased-coin up-and-down (Durham-Flournoy), STAT-1 (Task BR): after a failure step
    down; after a hold step up with probability b, else stay. Targets P(fail) = b / (1 + b),
    so b = (1 - q) / q targets the level that holds with probability q (q = 0.9 -> L90).
    The level stays on [low, high]."""
    if not held:
        return max(low, level - step)
    if u < b:
        return min(high, level + step)
    return level


def bcd_holds(
    *,
    start: int,
    step: int,
    trials: int,
    target_hold: float,
    seed: int,
    low: int,
    high: int,
    attempt: Any,
) -> dict[str, Any]:
    """STAT-1 block: `trials` fresh-child attempts on the biased-coin up-and-down rule.

    One uniform draw per trial from random.Random(seed), drawn whether or not it is used,
    so the coin sequence is fixed by the seed. Each point records its draw and the next level.
    largest_n_cached / first_fail_n_cached are descriptive; the estimand (L90, L50 by
    centered isotonic regression) is computed across blocks and sessions on the Mac."""
    import random

    if not 0.5 <= target_hold < 1.0:
        raise ValueError("target_hold must be in [0.5, 1)")
    if step <= 0 or trials < 1 or not low <= start <= high:
        raise ValueError("bad BCD parameters")
    b = (1.0 - target_hold) / target_hold
    rng = random.Random(int(seed))
    level = int(start)
    points: list[dict[str, Any]] = []
    seq: list[list[Any]] = []
    for i in range(int(trials)):
        point = attempt(level)
        u = rng.random()
        nxt = bcd_next(level, bool(point["holds"]), u, step=step, b=b, low=low, high=high)
        point["bcd"] = {"trial": i, "level": level, "u": u, "b": b, "next_level": nxt}
        points.append(point)
        seq.append([level, bool(point["holds"])])
        level = nxt
    held = [int(p["n_cached"]) for p in points if p.get("holds")]
    failed = [int(p["n_cached"]) for p in points if not p.get("holds")]
    out: dict[str, Any] = {
        "largest_n_cached": max(held) if held else None,
        "failure_in_range": bool(failed),
        "passing_turn2_ttft_s": _passing_turn2(points),
        "points": points,
        "note": "biased-coin up-and-down block (STAT-1), not a bisection",
        "bcd": {
            "start": int(start),
            "step": int(step),
            "trials": int(trials),
            "target_hold": float(target_hold),
            "b": b,
            "seed": int(seed),
            "low": int(low),
            "high": int(high),
            "sequence": seq,
            "next_level": level,
        },
    }
    if failed:
        out["first_fail_n_cached"] = min(failed)
    return out


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def balloon_private_bytes() -> dict[str, Any]:
    """The running balloon process (tools/gpu_balloon.py) and its measured private bytes.

    Candidates are python processes whose command line names the balloon script; the cmd.exe
    wrapper from spawn_detached.ps1 and any shell are ignored. A venv python.exe launcher and
    the interpreter it starts both match, so the one holding the memory is the candidate with
    no candidate child. Refuses unless exactly one such process is found and its private bytes
    can be read (psutil memory_info().private, Windows). The requested size is never used."""
    import psutil

    found: dict[int, Any] = {}
    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
        name = str(proc.info.get("name") or "").lower()
        cmd = " ".join(str(c) for c in (proc.info.get("cmdline") or []))
        if name.startswith("python") and any(b in cmd for b in BALLOON_SCRIPTS):
            found[int(proc.pid)] = proc
    parents = {int(p.info.get("ppid") or 0) for p in found.values()}
    leaves = [p for pid, p in found.items() if pid not in parents]
    if len(leaves) != 1:
        raise SystemExit(
            f"REFUSED -- balloon floor credit: {len(leaves)} balloon interpreter "
            f"processes (need 1; candidates {sorted(found)})"
        )
    proc = leaves[0]
    try:
        mem = proc.memory_info()
    except psutil.Error as exc:
        raise SystemExit(f"REFUSED -- balloon floor credit: bytes unreadable ({exc})") from exc
    private = getattr(mem, "private", None)
    if not isinstance(private, int) or private <= 0:
        raise SystemExit("REFUSED -- balloon floor credit: private bytes unreadable")
    return {
        "balloon_pid": int(proc.pid),
        "balloon_candidates": sorted(found),
        "balloon_private_bytes": private,
        "balloon_rss": mem.rss,
    }


def require_gates(*, skip_uptime: bool = False) -> dict[str, Any]:
    """All measurement gates. ``skip_uptime`` skips only the uptime gate; the gate
    record then carries the UNCOLD_UPTIME deviation (T2S, not rebooted).

    Task BQ option (a), CC-BAL cells with --balloon-floor-credit only: the Available floor
    (unchanged) is checked on Available + the balloon process's private bytes; the record
    keeps the observed Available, the balloon bytes, the credited sum, the floor and its
    citation. Every other gate is evaluated exactly as without the flag."""
    from seam.measurement_gates import evaluate_measurement_gates, probe_available_mb

    credit: dict[str, Any] | None = None
    kw: dict[str, Any] = {}
    if CC_OVERRIDE.get("balloon_floor_credit"):
        if not str(CC_OVERRIDE.get("cell") or "").startswith(BALLOON_CELL_PREFIX):
            raise SystemExit(f"REFUSED -- balloon floor credit needs a {BALLOON_CELL_PREFIX}* cell")
        observed = probe_available_mb()
        if observed is None:
            raise SystemExit("REFUSED -- balloon floor credit: Available MB unreadable")
        credit = {"observed_available_mb": float(observed), **balloon_private_bytes()}
        credit["balloon_private_mb"] = credit["balloon_private_bytes"] / 2**20
        credit["credited_available_mb"] = float(observed) + credit["balloon_private_mb"]
        kw["available_mb"] = credit["credited_available_mb"]
    report = evaluate_measurement_gates(ROOT, skip_uptime=skip_uptime, **kw)
    out = report.to_dict()
    if credit is not None:
        mem = next(g for g in out["gates"] if g["name"] == "available_mb")
        credit["floor_mb"] = mem["detail"].get("floor_mb")
        credit["floor_citation"] = mem["detail"].get("citation")
        credit["passed"] = mem["passed"]
        credit["rule"] = "Available + balloon private bytes >= floor (Task BQ option (a))"
        mem["detail"]["observed_is_credited"] = True
        out["balloon_floor_credit"] = credit
        print(json.dumps({"event": "balloon_floor_credit", **credit}, sort_keys=True), flush=True)
    if not report.all_passed:
        raise SystemExit("REFUSED -- gates: " + "; ".join(report.refusal_reasons))
    return out


def load_harness(model_spec: Path) -> dict[str, Any]:
    from seam.config import resolve_config
    from seam.model_provenance import load_local_spec
    from seam.tools.delta_n import _DELTA_N_PATH, _MEASUREMENT_PATH, _PLATFORM_PATH

    spec_path = model_spec if model_spec.is_absolute() else ROOT / model_spec
    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    cfg = resolved.data
    loaded = load_local_spec(spec_path)
    model_dir = str(loaded["ir_dir"])
    if not Path(model_dir).is_dir():
        raise SystemExit(f"REFUSED -- model IR missing: {model_dir}")
    if "arms" not in cfg:
        raise SystemExit("REFUSED -- resolved config missing arms")
    if "topology" not in cfg or "p_cpus" not in cfg["topology"]:
        raise SystemExit("REFUSED -- resolved config missing topology.p_cpus")
    acceptance = cfg.get("acceptance") or {}
    if acceptance.get("max_new_tokens") is None:
        raise SystemExit("REFUSED -- acceptance.max_new_tokens missing")
    return {
        "cfg": cfg,
        "model_spec": spec_path,
        "model_name": str(loaded.get("name") or spec_path.stem),
        "model_dir": model_dir,
        "arms": {a["id"]: a for a in cfg["arms"]},
        "p_cpus": [int(c) for c in cfg["topology"]["p_cpus"]],
        "filler_unit": str(cfg["ladder"]["filler_unit"]),
        "max_new_tokens": int(acceptance["max_new_tokens"]),
    }


def _child_spec(harness: dict[str, Any], arm_id: str, work: Path) -> dict[str, Any]:
    from seam.tools.ceiling_a import _child_spec_for_arm

    arm = harness["arms"].get(arm_id)
    if arm is None:
        raise SystemExit(f"REFUSED -- unknown arm {arm_id}")
    unused = work / "prompt_unused.txt"
    unused.write_text("unused\n", encoding="utf-8")
    spec = _child_spec_for_arm(
        cfg=harness["cfg"],
        arm=arm,
        model_dir=harness["model_dir"],
        prompt={"path": str(unused)},
        p_cpus=harness["p_cpus"],
    )
    spec["filler_unit"] = harness["filler_unit"]
    if SCHEDULER_OVERRIDE is not None:
        spec["scheduler_config"] = dict(SCHEDULER_OVERRIDE)
    for flag in CC_CHILD_FLAGS:
        if CC_OVERRIDE.get(flag):
            spec[flag] = True
    if CC_OVERRIDE.get("mem_trace_hz"):
        spec["mem_trace_hz"] = float(CC_OVERRIDE["mem_trace_hz"])
    return spec


def _canary_model(specs: list[Path]) -> Path:
    for spec in specs:
        if "Qwen3-4B-int4-ov" in spec.name:
            return spec if spec.is_absolute() else ROOT / spec
    raise SystemExit("REFUSED -- canary model spec Qwen3-4B-int4-ov is not in --model-spec")


def _open_canary(
    *,
    model_spec: Path,
    work: Path,
    plan_path: Path,
    planned: int,
    enforce_probe_budget: bool = True,
    up_front: bool = False,
) -> Any:
    from tools.ttft_slo_canary import CanaryBudgetRefuse, TtftSloCanaryGuard

    try:
        guard = TtftSloCanaryGuard(
            root=ROOT,
            model_spec=model_spec,
            work_dir=work / "canaries",
            plan_path=plan_path,
            planned_probe_count=int(planned),
            allow_unguarded=False,
            discard_warmup=True,
            enforce_probe_budget=enforce_probe_budget,
            arm_before_probes=up_front,
            up_front_budget=up_front,
        )
    except CanaryBudgetRefuse as exc:
        raise SystemExit(f"REFUSED -- {exc.detail}") from exc
    return guard


def _require_armed_up_front(
    *, guard: Any, out_dir: Path, summary: dict[str, Any], plan: dict[str, Any]
) -> None:
    """CC fix 2026-10-23e (as NPU amendment 2026-10-08 rule 1): after the up-front
    calibration the gate must be armed, or the run refuses before any probe."""
    if not guard.arm_before_probes or guard.gate.get("armed"):
        return
    summary.update(
        {
            "status": "REFUSED_CANARY",
            "abort_reason": "gate not armed after the up-front calibration; no probe ran",
            "points": [],
            "canaries": list(guard.canaries),
            "canary": guard.plan_fragment(),
            "ended_utc": _utc(),
        }
    )
    _write(out_dir / "summary.json", summary)
    plan["status"] = "REFUSED_CANARY"
    _write(out_dir / "plan.json", plan)
    raise SystemExit("REFUSED -- canary gate not armed after the up-front calibration")


def cc_up_front(points: tuple[int, ...]) -> bool:
    """One-point CC cells cannot arm with interleaved canaries: the opening fires one
    calibration canary and each probe at most one more, and C = 3. Those cells calibrate
    before the first probe (CC fix 2026-10-23e). Bisections and 2+ point cells are unchanged.
    """
    from tools.ttft_slo_canary import CALIBRATION_C

    return bool(points) and len(points) < CALIBRATION_C - 1


def trial_telemetry_fields(child: dict[str, Any]) -> dict[str, Any]:
    """Task BR (--trial-telemetry): the child's full exception, its memory trace at failure
    (summary and the raw 10 Hz samples with per-type GPU bytes) and time-to-fail (start of
    the failing generate to the exception). Empty values when the trial held."""
    rec = child.get("child") or {}
    at_fail = rec.get("memory_trace_at_failure") or {}
    return {
        "exception": rec.get("exception"),
        "memory_at_failure": rec.get("memory_at_failure"),
        "memory_trace_at_failure": rec.get("memory_trace_at_failure"),
        "memory_trace_raw_at_failure": rec.get("memory_trace_raw_at_failure"),
        "time_to_fail_s": at_fail.get("elapsed_to_exception_s"),
        "parent_wall_s": child.get("parent_wall_s"),
    }


def _point_failure(record: dict[str, Any]) -> dict[str, Any]:
    from tools.run_c1_ceiling import classify_c1_failure

    child = record.get("result") or record
    return classify_c1_failure(child)


def cell_status(points: list[dict[str, Any]]) -> str:
    """A recorded failure_kind is data. An unexplained non-pass is a harness failure."""
    saw_kind = False
    unexplained = False
    for point in points:
        if point.get("failure_kind"):
            saw_kind = True
        elif point.get("outcome") not in (None, "pass"):
            unexplained = True
        repeats = point.get("repeats")
        if isinstance(repeats, list):
            for repeat in repeats:
                if repeat.get("failure_kind"):
                    saw_kind = True
                elif repeat.get("outcome") not in (None, "pass"):
                    unexplained = True
            if point.get("median_decode_tok_s") is None and not any(
                item.get("failure_kind") for item in repeats
            ):
                unexplained = True
    if saw_kind:
        return "partial"
    if unexplained:
        return "failed"
    return "complete"


def cell_exit_code(status: str) -> int:
    """Point data continues the boot. Guard and harness statuses stop it."""
    if status in {"complete", "partial"}:
        return 0
    return 1


def _publish_cell_status(status: str, session_id: str | None = None) -> None:
    import os

    path = os.environ.get("SEAM_CELL_STATUS_PATH")
    if not path:
        return
    lines = [status]
    if session_id:
        lines.append(session_id)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of an empty sample")
    return float(statistics.median(values))


def seal_session(session_dir: Path) -> Path:
    """Copy plan, summary, and point files into a write-once sealed_<id> directory."""
    import os

    if os.environ.get("SEAM_REHEARSAL") == "1":
        raise SystemExit("REFUSED -- rehearsal does not seal")
    summary_path = session_dir / "summary.json"
    summary = read_json(summary_path)
    sid = str(summary["session_id"])
    dest = session_dir.parent / f"sealed_{sid}"
    if dest.exists():
        raise SystemExit(f"REFUSED -- seal destination exists: {dest}")
    dest.mkdir()
    rels: list[str] = []
    for name in ("plan.json", "summary.json"):
        shutil.copy2(session_dir / name, dest / name)
        rels.append(name)
    points = session_dir / "points"
    if points.is_dir():
        (dest / "points").mkdir()
        for path in sorted(points.glob("*.json")):
            shutil.copy2(path, dest / "points" / path.name)
            rels.append(f"points/{path.name}")
    manifest = {rel: hashlib.sha256((dest / rel).read_bytes()).hexdigest() for rel in rels}
    _write(dest / "manifest.sha256.json", manifest)
    (dest / ".sealed").write_text("sealed\n", encoding="utf-8")
    return dest


def _finish(
    *,
    guard: Any,
    out_dir: Path,
    summary: dict[str, Any],
    plan: dict[str, Any],
) -> int:
    from tools.ttft_slo_canary import CanaryUnarmedSealRefuse

    summary["canaries"] = list(guard.canaries)
    summary["canary"] = guard.plan_fragment()
    try:
        summary["canary_finalize"] = guard.finalize_or_refuse()
    except CanaryUnarmedSealRefuse as exc:
        summary["status"] = "REFUSED_CANARY"
        summary["ended_utc"] = _utc()
        _write(out_dir / "summary.json", summary)
        plan["status"] = "REFUSED_CANARY"
        _write(out_dir / "plan.json", plan)
        raise SystemExit(f"REFUSED -- {exc}") from exc
    summary["ended_utc"] = _utc()
    summary["sealed_dir"] = str(out_dir.parent / f"sealed_{summary['session_id']}")
    _write(out_dir / "summary.json", summary)
    plan["status"] = summary["status"]
    _write(out_dir / "plan.json", plan)
    sealed = seal_session(out_dir)
    print(json.dumps({"event": "sealed", "dir": str(sealed)}, sort_keys=True), flush=True)
    _publish_cell_status(str(summary["status"]), str(summary["session_id"]))
    return cell_exit_code(str(summary["status"]))


def run_warm_smoke(
    *,
    arm: str,
    model_spec: Path,
    turn2_path: str = "user_followup",
    record_prefix: bool = False,
) -> int:
    from seam.tools.delta_n import run_child

    harness = load_harness(model_spec)
    with tempfile.TemporaryDirectory(prefix="boot4-warm-smoke-") as tmp:
        work = Path(tmp)
        spec = _child_spec(harness, arm, work)
        spec["resident_two_turn"] = True
        spec["n_cached"] = SMOKE_N_CACHED
        spec["delta_tokens"] = SMOKE_DELTA_TOKENS
        spec["turn2_repeats"] = 1
        spec["turn1_max_new_tokens"] = 1
        spec["turn2_max_new_tokens"] = 1
        if turn2_path != "user_followup" or record_prefix:
            spec["turn2_path"] = turn2_path
            spec["record_prefix"] = record_prefix
        record = run_child(
            root=ROOT,
            work_dir=work,
            spec=spec,
            timeout_s=float(harness["cfg"]["generation"]["timeout_s"]),
            tag="smoke",
        )
    generation = (record.get("child") or {}).get("generation") or {}
    turn2 = generation.get("turn2") or []
    print(
        json.dumps(
            {
                "event": "warm_smoke",
                "arm": arm,
                "outcome": record.get("outcome"),
                "n_cached": SMOKE_N_CACHED,
                "delta_tokens": SMOKE_DELTA_TOKENS,
                "turn1_prefill_s": (generation.get("turn1") or {}).get("prefill_s"),
                "turn1_prompt_tokens": (generation.get("turn1") or {}).get(
                    "prompt_tokens_reported"
                ),
                "turn2_prefill_s": (turn2[0].get("prefill_s") if turn2 else None),
                "turn2_prompt_tokens": (turn2[0].get("prompt_tokens_reported") if turn2 else None),
                "turn2_path": generation.get("turn2_path"),
                "common_prefix_tokens": (turn2[0].get("common_prefix_tokens") if turn2 else None),
                "measured_delta_tokens": (turn2[0].get("measured_delta_tokens") if turn2 else None),
                "failure_mode": record.get("failure_mode"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if record.get("outcome") == "pass" else 1


def run_decode_smoke(*, arm: str, model_spec: Path) -> int:
    from seam.tools.delta_n import run_child

    harness = load_harness(model_spec)
    with tempfile.TemporaryDirectory(prefix="boot4-decode-smoke-") as tmp:
        work = Path(tmp)
        spec = _child_spec(harness, arm, work)
        spec["build_exact_n"] = SMOKE_DECODE_N
        spec["prompt_salt"] = "s"
        spec["max_new_tokens"] = harness["max_new_tokens"]
        spec["min_new_tokens"] = harness["max_new_tokens"]
        spec["decode_span_rate"] = True
        record = run_child(
            root=ROOT,
            work_dir=work,
            spec=spec,
            timeout_s=float(harness["cfg"]["generation"]["timeout_s"]),
            tag="smoke",
        )
    generation = (record.get("child") or {}).get("generation") or {}
    print(
        json.dumps(
            {
                "event": "decode_smoke",
                "arm": arm,
                "model": harness["model_name"],
                "outcome": record.get("outcome"),
                "n": SMOKE_DECODE_N,
                "max_new_tokens": harness["max_new_tokens"],
                "prefill_s": generation.get("prefill_s"),
                "decode_tok_s": generation.get("decode_tok_s"),
                "completion_tokens_reported": generation.get("completion_tokens_reported"),
                "decode_span": generation.get("decode_span"),
                "failure_mode": record.get("failure_mode"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if record.get("outcome") == "pass" else 1


def _warm_advantages(out_dir: Path, this_summary: dict[str, Any]) -> dict[str, Any] | None:
    by_arm: dict[str, dict[str, float]] = {}
    parent = out_dir.parent
    if parent.is_dir():
        for path in parent.glob("*/summary.json"):
            if path.parent.name.startswith("sealed_"):
                continue
            doc = read_json(path)
            if doc.get("kind") != "warm_kv":
                continue
            arm = doc.get("arm")
            medians = doc.get("turn2_median_prefill_s")
            if isinstance(arm, str) and isinstance(medians, dict) and medians:
                by_arm[arm] = {str(k): float(v) for k, v in medians.items()}
    arm = this_summary.get("arm")
    medians = this_summary.get("turn2_median_prefill_s")
    if isinstance(arm, str) and isinstance(medians, dict) and medians:
        by_arm[arm] = {str(k): float(v) for k, v in medians.items()}
    needed = ("gpu_only_f16", "gpu_only_u8", "gpu_only_u4")
    if any(name not in by_arm for name in needed):
        return None
    points = sorted(set(by_arm["gpu_only_f16"]))
    out: dict[str, Any] = {}
    for n in points:
        f16 = by_arm["gpu_only_f16"][n]
        out[n] = {
            "median_f16": f16,
            "median_u8": by_arm["gpu_only_u8"][n],
            "median_u4": by_arm["gpu_only_u4"][n],
            "f16_advantage_percent_vs_u8": advantage_percent(
                median_f16=f16, median_other=by_arm["gpu_only_u8"][n]
            ),
            "f16_advantage_percent_vs_u4": advantage_percent(
                median_f16=f16, median_other=by_arm["gpu_only_u4"][n]
            ),
        }
    return out


def run_warm(
    *,
    arm: str,
    model_spec: Path,
    session_id: str,
    out_dir: Path,
    n_cached: tuple[int, ...] | None = None,
    delta_tokens: int | None = None,
    turn2_path: str = "user_followup",
    record_prefix: bool = False,
) -> int:
    """WARM-KV cell. Defaults are the sealed WARM-KV configuration.

    Any override (DEMO-1 V5-CAL: n_cached, delta_tokens, turn2_path,
    record_prefix) writes kind warm_kv_v5cal and skips the cross-arm f16
    advantage table, which only applies to the WARM-KV points.
    """
    from seam.tools.delta_n import measured_repeat
    from tools.ttft_slo_canary import CanaryDriftAbort

    points_n = tuple(int(n) for n in (n_cached or N_CACHED))
    delta = int(delta_tokens if delta_tokens is not None else DELTA_TOKENS)
    custom = (
        points_n != tuple(N_CACHED)
        or delta != DELTA_TOKENS
        or turn2_path != "user_followup"
        or record_prefix
    )
    kind = "warm_kv_v5cal" if custom else "warm_kv"
    planned = max(WARM_PLANNED_PROBES, len(points_n))
    gates = require_gates()
    harness = load_harness(model_spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    _publish_cell_status("running", session_id)
    plan: dict[str, Any] = {
        "kind": kind,
        "session_id": session_id,
        "started_utc": _utc(),
        "arm": arm,
        "model_spec": str(harness["model_spec"]),
        "model_name": harness["model_name"],
        "n_cached": list(points_n),
        "delta_tokens": delta,
        "turn2_path": turn2_path,
        "record_prefix": record_prefix,
        "turn2_repeats": TURN2_REPEATS,
        "turn1_max_new_tokens": 1,
        "turn2_max_new_tokens": 1,
        "residency": "RESIDENT",
        "executed_points": len(points_n),
        "planned_probe_count": planned,
        "canary_budget_note": (
            "planned_probe_count is 4 so floor(planned/(C+1)) >= 1 with C=3. "
            "The cell executes the two n_cached points."
        ),
        "gates": gates,
        "status": "running",
    }
    plan_path = out_dir / "plan.json"
    _write(plan_path, plan)
    guard = _open_canary(
        model_spec=_canary_model([harness["model_spec"]]),
        work=out_dir / "work",
        plan_path=plan_path,
        planned=planned,
    )
    plan["canary"] = guard.plan_fragment()
    _write(plan_path, plan)
    points: list[dict[str, Any]] = []
    probes_log: list[dict[str, Any]] = []
    try:
        guard.opening()
        for n_cached in points_n:
            work = out_dir / "work"
            spec = _child_spec(harness, arm, work)
            spec["resident_two_turn"] = True
            spec["n_cached"] = int(n_cached)
            spec["delta_tokens"] = delta
            if custom:
                spec["turn2_path"] = turn2_path
                spec["record_prefix"] = record_prefix
            spec["turn2_repeats"] = TURN2_REPEATS
            spec["turn1_max_new_tokens"] = 1
            spec["turn2_max_new_tokens"] = 1
            measured = measured_repeat(
                root=ROOT,
                cfg=harness["cfg"],
                p_cpus=harness["p_cpus"],
                work_dir=work,
                child_spec=spec,
                label=f"warm-{arm}-n{n_cached}",
                tag=f"n{n_cached}",
            )
            child = measured["result"]
            classified = _point_failure(child)
            generation = (child.get("child") or {}).get("generation") or {}
            turn2 = list(generation.get("turn2") or [])
            prefills = [
                float(item["prefill_s"]) for item in turn2 if item.get("prefill_s") is not None
            ]
            point = {
                "arm": arm,
                "n_cached": int(n_cached),
                "rung": int(n_cached),
                "outcome": child.get("outcome"),
                "failure_kind": classified.get("failure_kind"),
                "failure_mode": child.get("failure_mode"),
                "memory": classified.get("memory") or {},
                "turn1_prefill_s": (generation.get("turn1") or {}).get("prefill_s"),
                "turn2_prefill_s": prefills,
                "turn2_median_prefill_s": _median(prefills)
                if len(prefills) == TURN2_REPEATS
                else None,
                "turn2_prompt_tokens_reported": [
                    item.get("prompt_tokens_reported") for item in turn2
                ],
            }
            if custom:
                point["delta_tokens"] = delta
                point["turn2_path"] = turn2_path
                for key in ("common_prefix_tokens", "measured_delta_tokens"):
                    point[key] = [item.get(key) for item in turn2]
            _write(out_dir / "points" / f"n{n_cached}.json", point)
            points.append(point)
            probes_log.append({"wall_s": child.get("parent_wall_s")})
            guard.after_probe(probes_log)
    except CanaryDriftAbort as exc:
        summary = {
            "kind": kind,
            "session_id": session_id,
            "arm": arm,
            "status": "FAIL_CANARY_DRIFT",
            "abort_reason": str(exc.detail),
            "points": points,
            "canaries": list(guard.canaries),
        }
        _write(out_dir / "summary.json", summary)
        raise SystemExit(f"REFUSED -- FAIL_CANARY_DRIFT: {exc.detail}") from exc

    medians = {
        str(point["n_cached"]): point["turn2_median_prefill_s"]
        for point in points
        if point.get("turn2_median_prefill_s") is not None
    }
    status = cell_status(points)
    summary = {
        "kind": kind,
        "session_id": session_id,
        "arm": arm,
        "model_name": harness["model_name"],
        "status": status,
        "points": points,
        "turn2_median_prefill_s": medians,
        "delta_tokens": delta,
        "n_cached": list(points_n),
    }
    if custom:
        summary["turn2_path"] = turn2_path
        summary["f16_advantage_percent"] = None
    else:
        summary["f16_advantage_percent"] = _warm_advantages(out_dir, summary)
    return _finish(guard=guard, out_dir=out_dir, summary=summary, plan=plan)


def run_decode(
    *,
    arm: str,
    model_specs: list[Path],
    session_id: str,
    out_dir: Path,
    n_list: tuple[int, ...] = DECODE_N,
    pilot: str | None = None,
    uncold_uptime: bool = False,
) -> int:
    """DECODE-MATCH cell. ``pilot`` (DEMO-2 P9/P2, tools/run_decode_match.py
    --pilot) allows one model spec and its registered n list; point files then
    carry the point index, since P9 repeats one n. Default output is unchanged."""
    from seam.tools.delta_n import measured_repeat
    from tools.ttft_slo_canary import CanaryDriftAbort

    if pilot is None and len(model_specs) < 2:
        raise SystemExit("REFUSED -- decode measurement requires both model specs")
    gates = require_gates(skip_uptime=uncold_uptime)
    loaded = [load_harness(spec) for spec in model_specs]
    out_dir.mkdir(parents=True, exist_ok=True)
    _publish_cell_status("running", session_id)
    planned = len(loaded) * len(n_list) * DECODE_REPEATS
    plan: dict[str, Any] = {
        "kind": "decode_match",
        "session_id": session_id,
        "started_utc": _utc(),
        "arm": arm,
        "models": [item["model_name"] for item in loaded],
        "n": list(n_list),
        "repeats": DECODE_REPEATS,
        "max_new_tokens": loaded[0]["max_new_tokens"],
        "min_new_tokens": loaded[0]["max_new_tokens"],
        "ignore_eos": True,
        "decode_tok_s": "63 / (t_last - t_first)",
        "planned_probe_count": planned,
        "gates": gates,
        "status": "running",
    }
    if pilot is not None:
        plan["pilot"] = pilot
    if uncold_uptime:
        # Task AY; absent otherwise so existing plans are unchanged.
        plan["deviation"] = {"kind": "UNCOLD_UPTIME", "reason": "T2S is not rebooted"}
    plan_path = out_dir / "plan.json"
    _write(plan_path, plan)
    guard = _open_canary(
        model_spec=_canary_model([item["model_spec"] for item in loaded]),
        work=out_dir / "work",
        plan_path=plan_path,
        planned=planned,
    )
    plan["canary"] = guard.plan_fragment()
    _write(plan_path, plan)
    points: list[dict[str, Any]] = []
    probes_log: list[dict[str, Any]] = []
    try:
        guard.opening()
        for harness in loaded:
            for point_index, n_tokens in enumerate(n_list):
                repeats: list[dict[str, Any]] = []
                for repeat in range(DECODE_REPEATS):
                    work = out_dir / "work"
                    spec = _child_spec(harness, arm, work)
                    spec["build_exact_n"] = int(n_tokens)
                    spec["prompt_salt"] = f"n{n_tokens}"
                    spec["max_new_tokens"] = harness["max_new_tokens"]
                    spec["min_new_tokens"] = harness["max_new_tokens"]
                    spec["decode_span_rate"] = True
                    measured = measured_repeat(
                        root=ROOT,
                        cfg=harness["cfg"],
                        p_cpus=harness["p_cpus"],
                        work_dir=work,
                        child_spec=spec,
                        label=f"decode-{harness['model_name']}-n{n_tokens}-r{repeat}",
                        tag=f"{harness['model_name']}_n{n_tokens}_r{repeat}",
                    )
                    child = measured["result"]
                    generation = (child.get("child") or {}).get("generation") or {}
                    repeats.append(
                        {
                            "repeat": repeat,
                            "outcome": child.get("outcome"),
                            "failure_kind": _point_failure(child).get("failure_kind"),
                            "prefill_s": generation.get("prefill_s"),
                            "decode_tok_s": generation.get("decode_tok_s"),
                            "completion_tokens_reported": generation.get(
                                "completion_tokens_reported"
                            ),
                            "decode_span": generation.get("decode_span"),
                        }
                    )
                    probes_log.append({"wall_s": child.get("parent_wall_s")})
                    guard.after_probe(probes_log)
                rates = [
                    float(item["decode_tok_s"])
                    for item in repeats
                    if item.get("decode_tok_s") is not None and item.get("outcome") == "pass"
                ]
                point = {
                    "model": harness["model_name"],
                    "arm": arm,
                    "n": int(n_tokens),
                    "repeats": repeats,
                    "median_decode_tok_s": _median(rates) if len(rates) == DECODE_REPEATS else None,
                }
                safe_name = harness["model_name"].replace(" ", "_")
                point_name = f"{safe_name}_n{n_tokens}.json"
                if pilot is not None:
                    point["point_index"] = point_index
                    point_name = f"{safe_name}_i{point_index:02d}_n{n_tokens}.json"
                _write(out_dir / "points" / point_name, point)
                points.append(point)
    except CanaryDriftAbort as exc:
        summary = {
            "kind": "decode_match",
            "session_id": session_id,
            "arm": arm,
            "status": "FAIL_CANARY_DRIFT",
            "abort_reason": str(exc.detail),
            "points": points,
            "canaries": list(guard.canaries),
        }
        _write(out_dir / "summary.json", summary)
        raise SystemExit(f"REFUSED -- FAIL_CANARY_DRIFT: {exc.detail}") from exc

    status = cell_status(points)
    summary = {
        "kind": "decode_match",
        "session_id": session_id,
        "arm": arm,
        "status": status,
        "models": [item["model_name"] for item in loaded],
        "n": list(n_list),
        "max_new_tokens": loaded[0]["max_new_tokens"],
        "points": points,
        "median_decode_tok_s": {
            f"{point['model']}:{point['n']}": point.get("median_decode_tok_s") for point in points
        },
    }
    if pilot is not None:
        summary["pilot"] = pilot
        summary["median_decode_tok_s"] = {
            f"{point['model']}:i{point['point_index']:02d}:n{point['n']}": point.get(
                "median_decode_tok_s"
            )
            for point in points
        }
    return _finish(guard=guard, out_dir=out_dir, summary=summary, plan=plan)


def run_resident_limit_smoke(*, arm: str, model_spec: Path) -> int:
    from seam.tools.delta_n import run_child

    harness = load_harness(model_spec)
    with tempfile.TemporaryDirectory(prefix="resident-limit-smoke-") as tmp:
        work = Path(tmp)
        spec = _child_spec(harness, arm, work)
        spec["resident_two_turn"] = True
        spec["n_cached"] = SMOKE_N_CACHED
        spec["delta_tokens"] = SMOKE_DELTA_TOKENS
        spec["turn2_repeats"] = 1
        spec["turn1_max_new_tokens"] = 1
        spec["turn2_max_new_tokens"] = 1
        record = run_child(
            root=ROOT,
            work_dir=work,
            spec=spec,
            timeout_s=float(harness["cfg"]["generation"]["timeout_s"]),
            tag="smoke",
        )
    generation = (record.get("child") or {}).get("generation") or {}
    turn2 = generation.get("turn2") or []
    print(
        json.dumps(
            {
                "event": "resident_limit_smoke",
                "arm": arm,
                "outcome": record.get("outcome"),
                "n_cached": SMOKE_N_CACHED,
                "delta_tokens": SMOKE_DELTA_TOKENS,
                "turn1_prefill_s": (generation.get("turn1") or {}).get("prefill_s"),
                "turn2_prefill_s": (turn2[0].get("prefill_s") if turn2 else None),
                "failure_mode": record.get("failure_mode"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if record.get("outcome") == "pass" else 1


def fixed_points(points: tuple[int, ...], attempt: Any) -> dict[str, Any]:
    """CC cells: attempt each listed n once, in the listed order (no bisection).

    largest_n_cached is the largest listed n that held; first_fail_n_cached the smallest that
    did not. Both are descriptive; a fixed-point cell does not search for a limit.
    """
    pts = [attempt(int(n)) for n in points]
    held = [int(p["n_cached"]) for p in pts if p.get("holds")]
    failed = [int(p["n_cached"]) for p in pts if not p.get("holds")]
    out: dict[str, Any] = {
        "largest_n_cached": max(held) if held else None,
        "failure_in_range": bool(failed),
        "passing_turn2_ttft_s": {
            str(p["n_cached"]): p.get("turn2_ttft_s") for p in pts if p.get("holds")
        },
        "points": pts,
        "note": "fixed points (CC cell), not a bisection",
    }
    if failed:
        out["first_fail_n_cached"] = min(failed)
    return out


def run_resident_limit(*, arm: str, model_spec: Path, session_id: str, out_dir: Path) -> int:
    """Bisect the largest n_cached a resident session holds. One attempt per rung."""
    from seam.tools.delta_n import measured_repeat
    from tools.ttft_slo_canary import CanaryDriftAbort

    gates = require_gates()
    harness = load_harness(model_spec)
    points = tuple(CC_OVERRIDE.get("points") or ())
    bcd: dict[str, Any] = dict(CC_OVERRIDE.get("bcd") or {})
    turn2_new_tokens = int(CC_OVERRIDE.get("turn2_new_tokens") or 1)
    turn2_repeats = int(CC_OVERRIDE.get("turn2_repeats") or 1)
    up_front = cc_up_front(points)
    planned = len(points) if up_front else max(RESIDENT_PLANNED_PROBES, len(points))
    if bcd:
        # STAT-1 (Task BR): a BCD block runs exactly `trials` children (>= 4, so the
        # interleaved canary schedule arms as for a bisection).
        planned = int(bcd["trials"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _publish_cell_status("running", session_id)
    plan: dict[str, Any] = {
        "kind": "resident_limit",
        "session_id": session_id,
        "started_utc": _utc(),
        "arm": arm,
        "model_spec": str(harness["model_spec"]),
        "model_name": harness["model_name"],
        "low": _resident_low(),
        "high": _resident_high(),
        "resolution": _resident_resolution(),
        "one_attempt_per_rung": True,
        "pass": "turn 1 completes and turn 2 runs",
        "delta_tokens": DELTA_TOKENS,
        "turn2_repeats": turn2_repeats,
        "residency": "RESIDENT",
        "planned_probe_count": planned,
        "gates": gates,
        "status": "running",
    }
    if SCHEDULER_OVERRIDE is not None:
        # DEMO-2 pilot P3 only; absent otherwise so existing plans are unchanged.
        plan["scheduler_config"] = dict(SCHEDULER_OVERRIDE)
    if CC_OVERRIDE:
        # CC cells only (Task BJ); absent otherwise so existing plans are unchanged.
        plan["cc"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in CC_OVERRIDE.items()}
        plan["turn2_max_new_tokens"] = turn2_new_tokens
        plan["model_window"] = model_window(harness["model_dir"])
    plan_path = out_dir / "plan.json"
    _write(plan_path, plan)
    guard = _open_canary(
        model_spec=_canary_model(_cc_canary_specs(harness)),
        work=out_dir / "work",
        plan_path=plan_path,
        planned=planned,
        up_front=up_front,
    )
    plan["canary"] = guard.plan_fragment()
    _write(plan_path, plan)
    probes_log: list[dict[str, Any]] = []
    recorded: list[dict[str, Any]] = []

    def attempt(n_cached: int) -> dict[str, Any]:
        # A repeated n (CC repeat cells only) gets a suffix so no record is overwritten.
        k = sum(1 for p in recorded if int(p["n_cached"]) == int(n_cached))
        sfx = f"_r{k}" if k else ""
        work = out_dir / "work"
        spec = _child_spec(harness, arm, work)
        spec["resident_two_turn"] = True
        spec["n_cached"] = int(n_cached)
        spec["delta_tokens"] = DELTA_TOKENS
        spec["turn2_repeats"] = turn2_repeats
        spec["turn1_max_new_tokens"] = 1
        spec["turn2_max_new_tokens"] = turn2_new_tokens
        measured = measured_repeat(
            root=ROOT,
            cfg=_cc_cfg(harness["cfg"]),
            p_cpus=harness["p_cpus"],
            work_dir=work,
            child_spec=spec,
            label=f"resident-{arm}-n{n_cached}{sfx}",
            tag=f"n{n_cached}{sfx}",
        )
        child = measured["result"]
        classified = _point_failure(child)
        generation = (child.get("child") or {}).get("generation") or {}
        turn1 = generation.get("turn1") or {}
        turn2 = list(generation.get("turn2") or [])
        prefills = [float(item["prefill_s"]) for item in turn2 if item.get("prefill_s") is not None]
        turn1_s = turn1.get("prefill_s")
        holds = turn1_s is not None and len(prefills) >= 1
        point = {
            "arm": arm,
            "n_cached": int(n_cached),
            "rung": int(n_cached),
            "holds": holds,
            "outcome": child.get("outcome"),
            "failure_kind": classified.get("failure_kind"),
            "failure_mode": child.get("failure_mode"),
            "memory": classified.get("memory") or {},
            "turn1_prefill_s": turn1_s,
            "turn2_prefill_s": prefills,
            "turn2_ttft_s": prefills[0] if prefills else None,
            "hung_after_result": bool(child.get("hung_after_result")),
            "hang_duration_s": child.get("hang_duration_s"),
            "hang_disposition": child.get("hang_disposition"),
        }
        if CC_OVERRIDE:
            point["turn2"] = turn2
            point["cc_mem"] = (child.get("child") or {}).get("cc_mem")
            mpe = (plan.get("model_window") or {}).get("max_position_embeddings")
            point["past_trained_positions"] = (
                int(n_cached) + DELTA_TOKENS > mpe if isinstance(mpe, int) else None
            )
            point["compiled_kv"] = [
                ld.get("compiled") for ld in ((child.get("child") or {}).get("loads") or [])
            ]
        if CC_OVERRIDE.get("trial_telemetry"):
            point.update(trial_telemetry_fields(child))
        _write(out_dir / "points" / f"n{n_cached}{sfx}.json", point)
        recorded.append(point)
        probes_log.append({"wall_s": child.get("parent_wall_s")})
        guard.after_probe(probes_log)
        return point

    try:
        guard.opening()
        _require_armed_up_front(
            guard=guard,
            out_dir=out_dir,
            summary={"kind": "resident_limit", "session_id": session_id, "arm": arm},
            plan=plan,
        )
        if points:
            found = fixed_points(points, attempt)
        elif bcd:
            found = bcd_holds(
                start=int(bcd["start"]),
                step=int(bcd["step"]),
                trials=int(bcd["trials"]),
                target_hold=float(bcd["target_hold"]),
                seed=int(bcd["seed"]),
                low=int(bcd["low"]),
                high=int(bcd["high"]),
                attempt=attempt,
            )
        else:
            found = bisect_holds(
                low=_resident_low(),
                high=_resident_high(),
                resolution=_resident_resolution(),
                attempt=attempt,
            )
        guard.closing(probes_log)
    except CanaryDriftAbort as exc:
        summary: dict[str, Any] = {
            "kind": "resident_limit",
            "session_id": session_id,
            "arm": arm,
            "status": "FAIL_CANARY_DRIFT",
            "abort_reason": str(exc.detail),
            "points": recorded,
            "canaries": list(guard.canaries),
        }
        _write(out_dir / "summary.json", summary)
        raise SystemExit(f"REFUSED -- FAIL_CANARY_DRIFT: {exc.detail}") from exc

    summary = {
        "kind": "resident_limit",
        "session_id": session_id,
        "arm": arm,
        "model_name": harness["model_name"],
        "status": "complete",
        "largest_n_cached": found["largest_n_cached"],
        "failure_in_range": found["failure_in_range"],
        "passing_turn2_ttft_s": found["passing_turn2_ttft_s"],
        "points": found["points"],
        "search": {
            "low": _resident_low(),
            "high": _resident_high(),
            "resolution": _resident_resolution(),
            "one_attempt": True,
        },
    }
    if found.get("note"):
        summary["note"] = found["note"]
    if found.get("bcd"):
        summary["bcd"] = found["bcd"]
    if "first_fail_n_cached" in found:
        summary["first_fail_n_cached"] = found["first_fail_n_cached"]
    if CC_OVERRIDE:
        summary["validity"] = validity_fields(
            found["largest_n_cached"], plan.get("model_window") or {}
        )
    return _finish(guard=guard, out_dir=out_dir, summary=summary, plan=plan)


def run_cc_series(*, arm: str, model_spec: Path, session_id: str, out_dir: Path) -> int:
    """CC-GROW (one growing session) or CC-CONC (N concurrent sessions, ascending, stop on
    the first failure). Task BL; set by tools/run_resident_limit.py --grow-step / --conc-sessions.
    """
    from seam.tools.delta_n import measured_repeat
    from tools.ttft_slo_canary import CanaryDriftAbort

    grow: dict[str, Any] = dict(CC_OVERRIDE.get("grow") or {})
    conc: dict[str, Any] = dict(CC_OVERRIDE.get("conc") or {})
    if bool(grow) == bool(conc):
        raise SystemExit("REFUSED -- run_cc_series needs exactly one of grow / conc")
    gates = require_gates()
    harness = load_harness(model_spec)
    kind = "cc_grow" if grow else "cc_conc"
    steps: list[int] = [0] if grow else [int(n) for n in conc["sessions"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    _publish_cell_status("running", session_id)
    plan: dict[str, Any] = {
        "kind": kind,
        "session_id": session_id,
        "started_utc": _utc(),
        "arm": arm,
        "model_spec": str(harness["model_spec"]),
        "model_name": harness["model_name"],
        "model_window": model_window(harness["model_dir"]),
        "cc": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CC_OVERRIDE.items()},
        "gates": gates,
        "status": "running",
    }
    if SCHEDULER_OVERRIDE is not None:
        plan["scheduler_config"] = dict(SCHEDULER_OVERRIDE)
    plan_path = out_dir / "plan.json"
    _write(plan_path, plan)
    # CC fix 2026-10-23e: calibrate before the first step (a grow series is one step and a
    # conc series stops at its first failure, so interleaved canaries cannot arm). Canaries
    # run in this process between children, never while a series child is resident.
    planned = len(steps)
    guard = _open_canary(
        model_spec=_canary_model(_cc_canary_specs(harness)),
        work=out_dir / "work",
        plan_path=plan_path,
        planned=planned,
        up_front=True,
    )
    plan["canary"] = guard.plan_fragment()
    _write(plan_path, plan)
    probes_log: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    try:
        guard.opening()
        _require_armed_up_front(
            guard=guard,
            out_dir=out_dir,
            summary={"kind": kind, "session_id": session_id, "arm": arm},
            plan=plan,
        )
        for n in steps:
            work = out_dir / "work"
            spec = _child_spec(harness, arm, work)
            tag = "grow" if grow else f"n{n}"
            if grow:
                spec["cc_grow"] = dict(grow)
            else:
                spec["cc_conc"] = {
                    "sessions": n,
                    "tokens": int(conc["tokens"]),
                    "new_tokens": int(conc.get("new_tokens", 16)),
                }
            measured = measured_repeat(
                root=ROOT,
                cfg=_cc_cfg(harness["cfg"]),
                p_cpus=harness["p_cpus"],
                work_dir=work,
                child_spec=spec,
                label=f"{kind}-{arm}-{tag}",
                tag=tag,
            )
            child = measured["result"]
            classified = _point_failure(child)
            point: dict[str, Any] = {
                "arm": arm,
                "step": n,
                "holds": child.get("outcome") == "pass",
                "outcome": child.get("outcome"),
                "failure_kind": classified.get("failure_kind"),
                "failure_mode": child.get("failure_mode"),
                "exception": (child.get("child") or {}).get("exception"),
                "generation": (child.get("child") or {}).get("generation"),
                "loads": (child.get("child") or {}).get("loads"),
                "hung_after_result": bool(child.get("hung_after_result")),
            }
            if CC_OVERRIDE.get("trial_telemetry"):
                point.update(trial_telemetry_fields(child))
            _write(out_dir / "points" / f"{tag}.json", point)
            points.append(point)
            probes_log.append({"wall_s": child.get("parent_wall_s")})
            guard.after_probe(probes_log)
            if not point["holds"]:
                break
        guard.closing(probes_log)
    except CanaryDriftAbort as exc:
        summary: dict[str, Any] = {
            "kind": kind,
            "session_id": session_id,
            "arm": arm,
            "status": "FAIL_CANARY_DRIFT",
            "abort_reason": str(exc.detail),
            "points": points,
            "canaries": list(guard.canaries),
        }
        _write(out_dir / "summary.json", summary)
        raise SystemExit(f"REFUSED -- FAIL_CANARY_DRIFT: {exc.detail}") from exc
    held = [p["step"] for p in points if p["holds"]]
    failed = [p["step"] for p in points if not p["holds"]]
    summary = {
        "kind": kind,
        "session_id": session_id,
        "arm": arm,
        "model_name": harness["model_name"],
        "status": "complete",
        "largest_holding_step": max(held) if held else None,
        "first_failing_step": min(failed) if failed else None,
        "points": points,
    }
    return _finish(guard=guard, out_dir=out_dir, summary=summary, plan=plan)


def new_session_id() -> str:
    return str(uuid.uuid4())
