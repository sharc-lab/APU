"""C-1 / C-2 ceiling sweep worker — KV-precision limits on int4 / gpu_only_*.

Detached payload for tools/launch_c1.ps1 and tools/launch_c2.ps1.
Binary-searches the largest n_cached per arm (gpu_only_f16, gpu_only_u8,
gpu_only_u4).

--criterion completion (default): pass = all repeats outcome==pass.
  Byte-identical control flow to the pre-criterion C-1 worker.
--criterion ttft_slo: pass = median(prefill_s) <= --slo-s (C-2 / AM-038).
  Drift canary (docs/CANARY_PROTOCOL.md) is mandatory on this path: fixed
  gpu_only_f16 / n=4000 / d=400 / RESIDENT cell; N from THIS run's probe
  wall times; C=3; threshold from early_max with 0.05 floor. A trip aborts
  with status FAIL_CANARY_DRIFT (CanaryDriftAbort — not a soft return).

CRITICAL (completion): every failed probe is classified as memory_wall vs
position_limit (or other), with the verbatim exception text retained.
Predictions are pre-registered in plan.json before any probe.
"""

from __future__ import annotations

import argparse
import atexit
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("gpu_only_f16", "gpu_only_u8", "gpu_only_u4")
POSITION_LIMIT = 40960
IR_BYTES = 2_290_768_181  # Qwen3-4B-int4-ov ir_bytes
# B/token residual+KV from sealed 41e419bd (README_characterizations / analysis)
K_PLUS_W = {
    "gpu_only_f16": 234_827,
    "gpu_only_u8": 171_532,
    "gpu_only_u4": 135_000,
}
DEFAULT_MODEL_SPEC = ROOT / "configs" / "models" / "Qwen3-4B-int4-ov.yaml"
LOW_DEFAULT = 12_000
HIGH_DEFAULT = 45_000
RESOLUTION = 250
REPEATS = 2
CRITERION_COMPLETION = "completion"
CRITERION_TTFT_SLO = "ttft_slo"
CRITERIA = (CRITERION_COMPLETION, CRITERION_TTFT_SLO)
SLO_S_DEFAULT = 10.0
DECODE_SLO_TOK_S = 6.0

# Sealed 41e419bd turn-1 medians (coarse grid) -- cited in AM-038 pre-reg only.
SEALED_41E_TURN1_PREFILL_MEDIAN = {
    "gpu_only_f16": {8000: 7.1832, 12000: 13.5377},
    "gpu_only_u8": {8000: 7.1326, 12000: 13.5273},
    "gpu_only_u4": {8000: 7.1324, 12000: 13.5001},
}

WATCHDOG_SCRIPT_BODY = r"""param([string]$LogPath, [int]$IntervalS)
$ErrorActionPreference = "Continue"
if ($IntervalS -lt 30) { $IntervalS = 30 }
while ($true) {
    Start-Sleep -Seconds $IntervalS
    $utc = (Get-Date).ToUniversalTime().ToString("o")
    $procs = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) {
        $rec = [ordered]@{ utc = $utc; event = "watchdog_poll"; n_found = 0; n_killed = 0 }
        Add-Content -LiteralPath $LogPath -Value (ConvertTo-Json -InputObject $rec -Compress)
        continue
    }
    $pids = @($procs | ForEach-Object { [int]$_.Id })
    $procs | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    $left = @(Get-Process -Name "WorkloadsSessionHost" -ErrorAction SilentlyContinue)
    $rec = [ordered]@{
        utc            = $utc
        event          = "watchdog_kill"
        n_found        = $pids.Count
        pids_found     = $pids
        n_killed       = $pids.Count - $left.Count
        pids_remaining = @($left | ForEach-Object { [int]$_.Id })
    }
    Add-Content -LiteralPath $LogPath -Value (ConvertTo-Json -InputObject $rec -Compress -Depth 4)
}
"""


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _start_wsh_watchdog(session_dir: Path, interval_s: int) -> dict[str, Any] | None:
    if interval_s <= 0:
        return None
    session_dir.mkdir(parents=True, exist_ok=True)
    kill_log = session_dir / "watchdog_kills.jsonl"
    watch_script = session_dir / "_wsh_watchdog.ps1"
    watch_script.write_text(WATCHDOG_SCRIPT_BODY, encoding="utf-8")
    init = {
        "utc": _utc(),
        "event": "watchdog_start_kill",
        "interval_s": interval_s,
        "pids_found": [],
        "n_killed": 0,
    }
    try:
        listed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                (
                    "$p=@(Get-Process -Name WorkloadsSessionHost -EA SilentlyContinue); "
                    "$ids=@($p | ForEach-Object { [int]$_.Id }); "
                    "if($p.Count -gt 0){$p|Stop-Process -Force -EA SilentlyContinue; "
                    "Start-Sleep -Seconds 1}; "
                    "$left=@(Get-Process -Name WorkloadsSessionHost -EA SilentlyContinue); "
                    "Write-Output (('ids=' + ($ids -join ',')) + '; left=' + $left.Count)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        init["start_kill_stdout"] = (listed.stdout or "").strip()
    except Exception as exc:
        init["start_kill_error"] = f"{type(exc).__name__}: {exc}"
    with kill_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(init, sort_keys=True) + "\n")
    creationflags = 0x00000200 if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-NoLogo",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(watch_script),
            "-LogPath",
            str(kill_log),
            "-IntervalS",
            str(interval_s),
        ],
        cwd=str(ROOT),
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    if proc.poll() is not None:
        raise SystemExit(f"REFUSED -- WSH watchdog exited immediately (pid={proc.pid})")

    def _stop() -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    atexit.register(_stop)
    _start_wsh_watchdog._alive_proc = proc  # type: ignore[attr-defined]
    return {
        "pid": proc.pid,
        "interval_s": interval_s,
        "kill_log": str(kill_log),
        "mechanism": "Popen_CREATE_NEW_PROCESS_GROUP",
    }


def classify_c1_failure(result: dict[str, Any]) -> dict[str, Any]:
    """Distinguish memory_wall vs position_limit; keep verbatim text."""
    from seam.tools.ceiling_a import classify_failure

    base = classify_failure(result)
    child = result.get("child") or {}
    exc = child.get("exception") or {}
    exc_type = str(exc.get("type") or "")
    exc_msg = str(exc.get("message") or "")
    failure_mode = str(result.get("failure_mode") or "")
    combined = f"{failure_mode} {exc_type} {exc_msg}"
    combined_l = combined.lower()
    verbatim = combined.strip() or None

    memory_markers = (
        "cl_out_of_resources",
        "out_of_resources",
        "out of memory",
        "oom",
        "std::bad_alloc",
        "bad_alloc",
        "cannot allocate",
        "not enough memory",
        "memoryerror",
        "failed to allocate",
        "allocation failure",
        "c0000005",
        "0xc0000005",
    )
    position_markers = (
        "max_position_embeddings",
        "position limit",
        "position_limit",
        "context length",
        "context_length",
        "exceeds max",
        "maximum context",
        "seq_len",
        "sequence length",
    )

    if result.get("outcome") == "pass":
        return {
            "class": "pass",
            "failure_kind": None,
            "verbatim": None,
            "base": base,
        }

    is_memory = any(m in combined_l for m in memory_markers)
    is_position = any(m in combined_l for m in position_markers)

    if is_memory and not is_position:
        kind = "memory_wall"
    elif is_position and not is_memory:
        kind = "position_limit"
    elif is_memory and is_position:
        kind = "ambiguous_memory_and_position"
    else:
        kind = "other"

    return {
        "class": base.get("class"),
        "failure_kind": kind,
        "verbatim": verbatim[:2000] if verbatim else None,
        "exception_type": exc_type or None,
        "exception_message": (exc_msg[:1200] if exc_msg else None),
        "failure_mode_raw": failure_mode or None,
        "base": base,
    }


def _predictions(m_available_mb: float) -> dict[str, Any]:
    m_bytes = float(m_available_mb) * 1024.0 * 1024.0
    headroom = m_bytes - float(IR_BYTES)
    per_arm: dict[str, Any] = {}
    for arm, kw in K_PLUS_W.items():
        n_mem = headroom / float(kw) if kw > 0 else None
        n_pos = float(POSITION_LIMIT)
        n_max = min(n_mem, n_pos) if n_mem is not None else n_pos
        bound = "MEMORY" if (n_mem is not None and n_mem < n_pos) else "POSITION"
        predicted_ceiling = int(round(n_mem)) if bound == "MEMORY" else POSITION_LIMIT
        per_arm[arm] = {
            "k_plus_w_B_per_token": kw,
            "n_max_memory": n_mem,
            "n_max_position": n_pos,
            "n_max": n_max,
            "binding": bound,
            "predicted_ceiling": predicted_ceiling,
        }
    return {
        "M_available_mb": m_available_mb,
        "W_ir_bytes": IR_BYTES,
        "formula": "n_max_memory=(M-W)/(k+w); n_max=min(n_max_memory, n_max_position)",
        "k_plus_w_source": "41e419bd measured B/token (f16 234827, u8 171532, u4 ~135000)",
        "arms": per_arm,
        "primary_prediction": {
            "claim": (
                "u8 and u4 give the SAME ceiling, 40960, because both are "
                "position-limited. u4 buys zero context over u8 on this machine "
                "while still costing prefill latency."
            ),
            "falsified_if": "u4 exceeds u8 beyond the bisection resolution (+/- 250 tokens)",
        },
        "secondary_prediction": {
            "claim": (
                "f16 walls by memory below 40960, near 35800 +/- 10% at M~10200 MB "
                "(recomputed from this run's M)."
            ),
            "falsified_if": "f16 reaches the position limit",
            "predicted_ceiling_at_this_M": per_arm["gpu_only_f16"]["predicted_ceiling"],
        },
        "declared_risk": {
            "extrapolation": (
                "Model fitted on n<=12000; predictions extrapolate to ~40000 "
                "(~3.4x beyond measured range)."
            ),
            "workspace_not_constant": (
                "Workspace residual ~87,371 B/token (f16) vs ~98,000 (u8/u4), "
                "a 12% spread. A miss on the secondary is as likely extrapolation "
                "error as model error — do not conflate them in the artifact."
            ),
        },
        "example_at_M_10200_mb": {
            "f16": 35788,
            "u8": 48994,
            "u4": 62257,
            "note": "operator-stated reference at M=10200 MB; this run uses measured M",
        },
    }


def _ttft_slo_predictions(*, slo_s: float, repeats: int) -> dict[str, Any]:
    """AM-038 pre-registration for --criterion ttft_slo (C-2). Both versions retained."""
    return {
        "source": ("docs/README_characterizations.md Finding 2 / sealed 41e419bd"),
        "amended_utc_note": (
            "PRE-DATA amendment before any C-2 probe: ordering claim withdrawn; "
            "replaced by turn-1 agreement within resolution. (AM-038)"
        ),
        "sealed_41e419bd_turn1_prefill_median_s": SEALED_41E_TURN1_PREFILL_MEDIAN,
        "coarse_bracket": {
            "n_ok_side": 8000,
            "n_fail_side": 12000,
            "note": (
                "All three arms under SLO at 8000 (~7.1-7.2 s) and over at "
                "12000 (~13.5 s) on sealed turn-1 medians."
            ),
        },
        "withdrawn_prediction": {
            "status": "WITHDRAWN",
            "withdrawn_before_any_probe": True,
            "claim": (
                "TTFT-bound limit orders f16 > u8 >= u4. Prefill is "
                "compute-bound; quantized arms pay dequantization, so f16 "
                "reaches a higher n before crossing the 10 s SLO."
            ),
            "order": "f16 > u8 >= u4",
            "falsified_if_when_active": [
                "the three TTFT limits agree within the 250-token resolution",
                "u4 exceeds f16",
            ],
            "withdrawal_reason": (
                "That ordering was inferred from the 15-27% f16 advantage in "
                "41e419bd Finding 2, which is a TURN-2 DELTA PREFILL "
                "measurement. C-2 bisects on TURN-1 bulk prefill. Finding 2's "
                "direct statement about turn-1 is that arms agree within 1%."
            ),
        },
        "primary_prediction": {
            "status": "ACTIVE",
            "claim": (
                "The three turn-1 TTFT limits AGREE within the 250-token "
                "resolution. KV precision does not move the cold-start "
                "context limit."
            ),
            "order": "f16 ~= u8 ~= u4 (within +/- 250 tokens)",
            "falsified_if": [
                "any pair of TTFT limits differs by more than 250 tokens",
            ],
            "consequence_if_holds": (
                "KV precision affects neither capability (memory does not "
                "bind, C-1) nor the cold-start SLO limit. Its only remaining "
                "value on this hardware is the headroom it frees, which "
                "nothing needs. That makes it the weakest of the six axes on "
                "this machine, and it should be reported that way rather "
                "than on its nominal purpose."
            ),
        },
        "separate_experiment_not_c2": {
            "name": "turn2_delta_prefill_ttft_limit",
            "status": "NOT_STARTED",
            "displacing": True,
            "note": (
                "The turn-2 delta-prefill limit binds under RESIDENT and is "
                "where the 15-27% f16 advantage lives. At n=12,000 turn-2 was "
                "roughly 0.16 of turn-1, so its 10 s crossing sits well above "
                "12,000 and needs a wider search. There the ordering "
                "f16 > u8 >= u4 is still the prediction, and it matters more, "
                "because residency is the configuration the workload actually "
                "runs in. Recorded here so it is not silently folded into C-2."
            ),
            "prediction_if_run": "f16 > u8 >= u4",
        },
        "slo": {
            "prefill_s_max": float(slo_s),
            "decode_tok_s_min": DECODE_SLO_TOK_S,
            "pass_uses_median_of_repeats": True,
            "repeats": int(repeats),
        },
        "record_note": ("Both withdrawn and replacement predictions stay in the record."),
    }


def _available_mb() -> float:
    from seam.tools.delta_n import memory_now

    return float(memory_now()["available_mb"])


def _probe_once(
    *,
    root: Path,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    model_dir: str,
    p_cpus: list[int],
    work_dir: Path,
    prompt: dict[str, Any],
    n_tokens: int,
    repeat_i: int,
    label_prefix: str = "c1",
) -> dict[str, Any]:
    from seam.tools.ceiling_a import _child_spec_for_arm
    from seam.tools.delta_n import measured_repeat

    child_spec = _child_spec_for_arm(
        cfg=cfg,
        arm=arm,
        model_dir=model_dir,
        prompt=prompt,
        p_cpus=p_cpus,
    )
    tag = f"{arm['id']}.n{n_tokens}.r{repeat_i}"
    record = measured_repeat(
        root=root,
        cfg=cfg,
        p_cpus=p_cpus,
        work_dir=work_dir,
        child_spec=child_spec,
        label=f"{label_prefix}/{arm['id']}/{n_tokens}/{repeat_i}",
        tag=tag,
    )
    result = record.get("result") or {}
    classification = classify_c1_failure(result)
    # run_child wraps the on-disk result.json as result["child"]; generation
    # metrics (prefill_s, decode_tok_s, wall_s) and completed live there.
    # Reading result["generation"] / result["completed"] yields nulls while
    # outcome (set on the wrapper) still reads "pass" -- that produced the
    # false REFUSED_LOW_OVER_SLO on e39aaa86.
    child = result.get("child") or {}
    gen = child.get("generation") or {}
    return {
        "arm_id": arm["id"],
        "n_tokens": n_tokens,
        "repeat_index": repeat_i,
        "outcome": result.get("outcome"),
        "completed": bool(child.get("completed")),
        "admissible": record.get("admissible"),
        "failure_classification": classification,
        "failure_kind": classification.get("failure_kind"),
        "verbatim_error": classification.get("verbatim"),
        "prefill_s": gen.get("prefill_s"),
        "decode_tok_s": gen.get("decode_tok_s"),
        "r_prefill_tok_s": gen.get("r_prefill_tok_s"),
        "wall_s": gen.get("wall_s"),
        "started_utc": record.get("started_utc"),
        "ended_utc": record.get("ended_utc"),
    }


def _eval_ttft_agreement(arm_results: list[dict[str, Any]], resolution: int) -> dict[str, Any]:
    limits: dict[str, int | None] = {}
    for r in arm_results:
        if r.get("status") != "complete":
            limits[r["arm_id"]] = None
            continue
        lim = r.get("ttft_limit_n")
        if lim is None:
            lim = r.get("ceiling_n_cached")
        limits[r["arm_id"]] = int(lim) if lim is not None else None
    f16, u8, u4 = (limits.get(a) for a in ARMS)
    if None in (f16, u8, u4):
        return {
            "evaluable": False,
            "limits": limits,
            "active_claim": "turn1_limits_agree_within_resolution",
            "reason": "one or more arms did not complete with a TTFT limit",
        }
    assert f16 is not None and u8 is not None and u4 is not None
    pairs = {
        "f16_u8": abs(f16 - u8),
        "f16_u4": abs(f16 - u4),
        "u8_u4": abs(u8 - u4),
    }
    span = max(f16, u8, u4) - min(f16, u8, u4)
    disagreeing = {k: v for k, v in pairs.items() if v > resolution}
    agree = len(disagreeing) == 0
    return {
        "evaluable": True,
        "active_claim": "turn1_limits_agree_within_resolution",
        "limits": limits,
        "pair_abs_diffs_tokens": pairs,
        "span_tokens": span,
        "agree_within_resolution": agree,
        "primary_prediction_held": agree,
        "falsified": not agree,
        "falsification_notes": (
            [
                f"pair {k} differs by {v} tokens (> {resolution})"
                for k, v in sorted(disagreeing.items())
            ]
            if disagreeing
            else []
        ),
        "withdrawn_ordering_forensic": {
            "order": "f16 > u8 >= u4",
            "held_on_this_data": (f16 > u8) and (u8 >= u4),
            "note": "Not the active claim; retained because both versions stay in the record.",
        },
    }


def bisect_arm(
    *,
    root: Path,
    cfg: dict[str, Any],
    arm: dict[str, Any],
    model_dir: str,
    p_cpus: list[int],
    work_dir: Path,
    prompt_cache: dict[int, dict[str, Any]],
    unit: str,
    low: int,
    high: int,
    resolution: int,
    repeats: int,
    probes_log: list[dict[str, Any]],
    criterion: str = CRITERION_COMPLETION,
    slo_s: float = SLO_S_DEFAULT,
    label_prefix: str = "c1",
    canary_guard: Any | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from seam.tools.delta_n import prompt_for

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    lo, hi = int(low), int(high)
    criterion = str(criterion)
    slo_s = float(slo_s)

    def probe_n(n: int) -> dict[str, Any]:
        prompt = prompt_for(
            root=root, tokenizer=tokenizer, n_tokens=n, unit=unit, cache=prompt_cache
        )
        reps = []
        for r in range(repeats):
            print(
                f"[{label_prefix}] probe arm={arm['id']} n={n} repeat={r}",
                flush=True,
            )
            rep = _probe_once(
                root=root,
                cfg=cfg,
                arm=arm,
                model_dir=model_dir,
                p_cpus=p_cpus,
                work_dir=work_dir,
                prompt=prompt,
                n_tokens=n,
                repeat_i=r,
                label_prefix=label_prefix,
            )
            if (
                criterion == CRITERION_TTFT_SLO
                and rep.get("outcome") == "pass"
                and rep.get("prefill_s") is None
            ):
                raise SystemExit(
                    "REFUSED -- ttft_slo: outcome=pass but prefill_s is null "
                    f"(arm={arm['id']} n={n} repeat={r}). Extraction must read "
                    "result['child']['generation']['prefill_s']; refusing to "
                    "treat a missing measurement as an SLO failure."
                )
            reps.append(rep)
            probes_log.append(rep)
            if canary_guard is not None:
                canary_guard.after_probe(probes_log)
        outcomes = [x.get("outcome") for x in reps]
        generate_ok = all(o == "pass" for o in outcomes) and len(outcomes) == repeats
        prefills = [x.get("prefill_s") for x in reps if x.get("prefill_s") is not None]
        decodes = [x.get("decode_tok_s") for x in reps if x.get("decode_tok_s") is not None]
        prefill_median = float(statistics.median(prefills)) if len(prefills) == repeats else None
        decode_median = float(statistics.median(decodes)) if decodes else None
        if criterion == CRITERION_COMPLETION:
            # Default path: identical pass rule to pre-criterion C-1.
            ok = generate_ok
        elif criterion == CRITERION_TTFT_SLO:
            ok = generate_ok and prefill_median is not None and prefill_median <= slo_s
        else:
            raise SystemExit(f"REFUSED -- unknown criterion {criterion!r}")
        fail_cls = next(
            (x["failure_classification"] for x in reps if x.get("outcome") != "pass"),
            None,
        )
        return {
            "n_tokens": n,
            "pass": ok,
            "generate_ok": generate_ok,
            "prefill_s_median": prefill_median,
            "decode_tok_s_median": decode_median,
            "decode_slo_ok": (decode_median is not None and decode_median >= DECODE_SLO_TOK_S),
            "repeats": reps,
            "failure_classification": fail_cls,
            "failure_kind": (fail_cls or {}).get("failure_kind"),
            "verbatim_error": (fail_cls or {}).get("verbatim"),
        }

    low_probe = probe_n(lo)
    if not low_probe["pass"]:
        if criterion == CRITERION_TTFT_SLO:
            return {
                "arm_id": arm["id"],
                "status": "REFUSED_LOW_OVER_SLO",
                "low": lo,
                "high": hi,
                "low_probe": low_probe,
                "note": f"low={lo} median prefill not under {slo_s}s SLO",
            }
        return {
            "arm_id": arm["id"],
            "status": "REFUSED_LOW_FAILED",
            "low": lo,
            "high": hi,
            "low_probe": low_probe,
            "note": "low known-good failed; host not admissible",
        }

    # Probe the declared high bound first. A pass here means the search range
    # contains no failure -- do NOT "converge" onto high as a fake ceiling.
    high_probe = probe_n(hi)
    if high_probe["pass"]:
        abort_reason = (
            "no_slo_crossing_in_range"
            if criterion == CRITERION_TTFT_SLO
            else "no_ceiling_found_in_range"
        )
        out: dict[str, Any] = {
            "arm_id": arm["id"],
            "status": "aborted",
            "abort_reason": abort_reason,
            "search": {"low_init": low, "high_init": high, "resolution": resolution},
            "highest_pass_n_cached": hi,
            "first_fail": None,
            "low_probe": low_probe,
            "high_probe": high_probe,
            "note": (
                "Every probe in [low, high] including high passed; "
                "no hard ceiling / SLO crossing observed in range."
            ),
        }
        if criterion == CRITERION_TTFT_SLO:
            out["ttft_limit_n"] = hi
            out["highest_slo_ok_n"] = hi
        return out

    best_pass = lo
    first_fail: dict[str, Any] | None = high_probe
    while (hi - lo) > resolution:
        mid = (lo + hi) // 2
        mid = max(lo + 1, mid)
        probe = probe_n(mid)
        if probe["pass"]:
            best_pass = mid
            lo = mid
        else:
            first_fail = probe
            hi = mid

    if first_fail is None:
        abort_reason = (
            "no_slo_crossing_in_range"
            if criterion == CRITERION_TTFT_SLO
            else "no_ceiling_found_in_range"
        )
        out = {
            "arm_id": arm["id"],
            "status": "aborted",
            "abort_reason": abort_reason,
            "search": {"low_init": low, "high_init": high, "resolution": resolution},
            "highest_pass_n_cached": best_pass,
            "first_fail": None,
            "note": (
                "Bisection ended without observing a failure; "
                "refusing to report high as ceiling."
            ),
        }
        if criterion == CRITERION_TTFT_SLO:
            out["ttft_limit_n"] = best_pass
            out["highest_slo_ok_n"] = best_pass
        return out

    out = {
        "arm_id": arm["id"],
        "status": "complete",
        "search": {"low_init": low, "high_init": high, "resolution": resolution},
        "ceiling_n_cached": best_pass,
        "first_fail": first_fail,
        "failure_kind_at_wall": (first_fail or {}).get("failure_kind"),
        "verbatim_at_wall": (first_fail or {}).get("verbatim_error"),
    }
    if criterion == CRITERION_TTFT_SLO:
        out["ttft_limit_n"] = best_pass
        out["highest_slo_ok_n"] = best_pass
        out["first_slo_fail_n"] = (first_fail or {}).get("n_tokens")
        out["first_slo_fail"] = first_fail
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--low", type=int, default=LOW_DEFAULT)
    parser.add_argument("--high", type=int, default=HIGH_DEFAULT)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--watchdog-interval-s", type=int, default=60)
    parser.add_argument(
        "--criterion",
        default=CRITERION_COMPLETION,
        choices=list(CRITERIA),
        help=(
            "completion (default): all repeats outcome==pass. "
            "ttft_slo: median(prefill_s)<=--slo-s."
        ),
    )
    parser.add_argument(
        "--slo-s",
        type=float,
        default=SLO_S_DEFAULT,
        help="prefill_s median threshold for --criterion ttft_slo (default 10)",
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated arm ids (default gpu_only_f16,gpu_only_u8,gpu_only_u4)",
    )
    parser.add_argument(
        "--planned-probe-count",
        type=int,
        default=None,
        help=(
            "ttft_slo canary budget (INF-1b). Default: estimate from arms/search. "
            "Refuse start if floor(planned/(C+1)) < 1."
        ),
    )
    parser.add_argument(
        "--allow-unguarded",
        action="store_true",
        help=(
            "Allow finalize/seal when canary armed==false; writes UNGUARDED into "
            "summary (and seal). Default: refuse unarmed complete."
        ),
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model_spec = args.model_spec if args.model_spec.is_absolute() else ROOT / args.model_spec
    criterion = str(args.criterion)
    slo_s = float(args.slo_s)
    label_prefix = "c2" if criterion == CRITERION_TTFT_SLO else "c1"

    from seam.config import resolve_config
    from seam.model_provenance import load_local_spec
    from seam.tools.delta_n import _DELTA_N_PATH, _MEASUREMENT_PATH, _PLATFORM_PATH

    resolved = resolve_config(
        [ROOT / _PLATFORM_PATH, ROOT / _MEASUREMENT_PATH, ROOT / _DELTA_N_PATH],
        repo_root=ROOT,
    )
    cfg = resolved.data
    # Pin model via FetchedModelSpec (int4).
    spec = load_local_spec(model_spec)
    model_dir = str(spec["ir_dir"])
    if not Path(model_dir).is_dir():
        raise SystemExit(f"REFUSED -- model IR missing: {model_dir}")
    # delta_n arms live under cfg; topology may be merged from platform.
    if "arms" not in cfg:
        raise SystemExit("REFUSED -- resolved config missing arms (delta_n.yaml)")
    if "topology" not in cfg or "p_cpus" not in cfg["topology"]:
        raise SystemExit("REFUSED -- resolved config missing topology.p_cpus")
    arms_by_id = {a["id"]: a for a in cfg["arms"]}
    arm_ids = [a.strip() for a in str(args.arms).split(",") if a.strip()]
    for aid in arm_ids:
        if aid not in arms_by_id:
            raise SystemExit(f"REFUSED -- unknown arm {aid}")
        # Confirm f16 pin requests KV_CACHE_PRECISION=f16
        if aid == "gpu_only_f16":
            props = arms_by_id[aid].get("properties") or {}
            if str(props.get("KV_CACHE_PRECISION") or "").lower() != "f16":
                raise SystemExit(
                    "REFUSED -- gpu_only_f16 must request KV_CACHE_PRECISION=f16 " f"(got {props})"
                )

    if criterion == CRITERION_TTFT_SLO:
        predictions = _ttft_slo_predictions(slo_s=slo_s, repeats=int(args.repeats))
        kind = "c2_ttft_bound_limit"
    else:
        m_mb = _available_mb()
        predictions = _predictions(m_mb)
        kind = "c1_kv_precision_ceiling"

    watch = _start_wsh_watchdog(out_dir, int(args.watchdog_interval_s))

    plan = {
        "kind": kind,
        "session_id": args.session_id,
        "started_utc": _utc(),
        "model_spec": str(model_spec),
        "ir_sha256": spec.get("ir_sha256"),
        "ir_bytes": IR_BYTES,
        "arms": arm_ids,
        "criterion": criterion,
        "slo_s": slo_s if criterion == CRITERION_TTFT_SLO else None,
        "search": {
            "low": int(args.low),
            "high": int(args.high),
            "resolution_tokens": int(args.resolution),
            "repeats_per_probe": int(args.repeats),
            "pass_requires_all_repeats": True,
            "criterion": criterion,
            "slo_s": slo_s if criterion == CRITERION_TTFT_SLO else None,
        },
        "pre_registered_predictions": predictions,
        "gpu_only_f16_pin_note": (
            "gpu_only_f16 requests KV_CACHE_PRECISION=f16 and has matched "
            "readback=f16 on prior cells (not the old bare gpu_only that "
            "read back dynamic)."
        ),
        "watchdog": watch,
        "status": "running",
    }
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": f"{label_prefix}.plan_written",
                "session_id": args.session_id,
                "criterion": criterion,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    p_cpus = [int(c) for c in cfg["topology"]["p_cpus"]]
    unit = str((cfg.get("ladder") or {}).get("filler_unit") or "")
    if not unit:
        raise SystemExit("REFUSED -- ladder.filler_unit missing")
    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True)
    prompt_cache: dict[int, dict[str, Any]] = {}
    probes_log: list[dict[str, Any]] = []
    arm_results: list[dict[str, Any]] = []

    canary_guard = None
    if criterion == CRITERION_TTFT_SLO:
        from tools.ttft_slo_canary import (
            CanaryBudgetRefuse,
            CanaryDriftAbort,
            CanaryUnarmedSealRefuse,
            TtftSloCanaryGuard,
            UNARMED_REFUSE,
            estimate_bisect_planned_probes,
        )

        planned = args.planned_probe_count
        if planned is None:
            planned = estimate_bisect_planned_probes(
                n_arms=len(arm_ids),
                low=int(args.low),
                high=int(args.high),
                resolution=int(args.resolution),
                repeats=int(args.repeats),
            )
        plan["planned_probe_count"] = int(planned)
        plan["allow_unguarded"] = bool(args.allow_unguarded)
        try:
            canary_guard = TtftSloCanaryGuard(
                root=ROOT,
                model_spec=model_spec,
                work_dir=work_dir / "canaries",
                plan_path=out_dir / "plan.json",
                planned_probe_count=int(planned),
                allow_unguarded=bool(args.allow_unguarded),
            )
        except CanaryBudgetRefuse as exc:
            print(f"REFUSED -- {exc.detail}", flush=True)
            raise SystemExit(f"REFUSED -- {exc.detail}") from exc
        plan["canary"] = canary_guard.plan_fragment()
        (out_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            canary_guard.opening()
        except CanaryDriftAbort as exc:
            session_status = "FAIL_CANARY_DRIFT"
            summary = {
                "kind": kind,
                "session_id": args.session_id,
                "ended_utc": _utc(),
                "status": session_status,
                "abort_reason": "FAIL_CANARY_DRIFT",
                "canary_trip_detail": exc.detail,
                "canary": canary_guard.plan_fragment(),
                "canaries": canary_guard.canaries,
                "criterion": criterion,
                "pre_registered_predictions": predictions,
                "arm_results": [],
                "primary_claim_eval": None,
                "n_probes": 0,
            }
            (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            plan["status"] = session_status
            plan["abort_reason"] = "FAIL_CANARY_DRIFT"
            plan["canary"] = canary_guard.plan_fragment()
            plan["canaries"] = canary_guard.canaries
            plan["ended_utc"] = summary["ended_utc"]
            (out_dir / "plan.json").write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "ok": False,
                        "status": session_status,
                        "abort_reason": "FAIL_CANARY_DRIFT",
                        "detail": exc.detail,
                        "session_id": args.session_id,
                        "out": str(out_dir),
                    },
                    indent=2,
                )
            )
            return 2
        plan["canary"] = canary_guard.plan_fragment()
        (out_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    try:
        for aid in arm_ids:
            print(f"[{label_prefix}] BEGIN arm={aid}", flush=True)
            res = bisect_arm(
                root=ROOT,
                cfg=cfg,
                arm=arms_by_id[aid],
                model_dir=model_dir,
                p_cpus=p_cpus,
                work_dir=work_dir,
                prompt_cache=prompt_cache,
                unit=unit,
                low=int(args.low),
                high=int(args.high),
                resolution=int(args.resolution),
                repeats=int(args.repeats),
                probes_log=probes_log,
                criterion=criterion,
                slo_s=slo_s,
                label_prefix=label_prefix,
                canary_guard=canary_guard,
            )
            arm_results.append(res)
            (out_dir / "probes.ndjson").write_text(
                "".join(json.dumps(p, sort_keys=True) + "\n" for p in probes_log),
                encoding="utf-8",
            )
            (out_dir / "arm_results.json").write_text(
                json.dumps(arm_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if canary_guard is not None:
                plan["canary"] = canary_guard.plan_fragment()
                plan["canaries"] = canary_guard.canaries
                (out_dir / "plan.json").write_text(
                    json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
    except Exception as exc:
        # Import locally so completion path stays free of the canary module.
        from tools.ttft_slo_canary import CanaryDriftAbort

        if not isinstance(exc, CanaryDriftAbort):
            raise
        session_status = "FAIL_CANARY_DRIFT"
        summary = {
            "kind": kind,
            "session_id": args.session_id,
            "ended_utc": _utc(),
            "status": session_status,
            "abort_reason": "FAIL_CANARY_DRIFT",
            "canary_trip_detail": exc.detail,
            "canary": canary_guard.plan_fragment() if canary_guard else None,
            "canaries": canary_guard.canaries if canary_guard else [],
            "criterion": criterion,
            "pre_registered_predictions": predictions,
            "arm_results": arm_results,
            "primary_claim_eval": None,
            "n_probes": len(probes_log),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out_dir / "probes.ndjson").write_text(
            "".join(json.dumps(p, sort_keys=True) + "\n" for p in probes_log),
            encoding="utf-8",
        )
        plan["status"] = session_status
        plan["abort_reason"] = "FAIL_CANARY_DRIFT"
        plan["canary"] = canary_guard.plan_fragment() if canary_guard else None
        plan["canaries"] = canary_guard.canaries if canary_guard else []
        plan["arm_results"] = arm_results
        plan["ended_utc"] = summary["ended_utc"]
        (out_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": session_status,
                    "abort_reason": "FAIL_CANARY_DRIFT",
                    "detail": exc.detail,
                    "session_id": args.session_id,
                    "out": str(out_dir),
                },
                indent=2,
            )
        )
        return 2

    by_id = {r["arm_id"]: r for r in arm_results}
    if criterion == CRITERION_TTFT_SLO:
        primary = _eval_ttft_agreement(arm_results, int(args.resolution))
    else:
        primary = None
        if (
            by_id.get("gpu_only_u8", {}).get("status") == "complete"
            and by_id.get("gpu_only_u4", {}).get("status") == "complete"
        ):
            c8 = int(by_id["gpu_only_u8"]["ceiling_n_cached"])
            c4 = int(by_id["gpu_only_u4"]["ceiling_n_cached"])
            primary = {
                "u8_ceiling": c8,
                "u4_ceiling": c4,
                "same_within_resolution": abs(c8 - c4) <= int(args.resolution),
                "both_at_position_limit": c8 >= POSITION_LIMIT and c4 >= POSITION_LIMIT,
                "primary_prediction_held": abs(c8 - c4) <= int(args.resolution),
            }

    aborted = [
        r
        for r in arm_results
        if r.get("status") == "aborted"
        or r.get("abort_reason") in ("no_ceiling_found_in_range", "no_slo_crossing_in_range")
    ]
    refused = [r for r in arm_results if str(r.get("status", "")).startswith("REFUSED")]
    if aborted and not any(r.get("status") == "complete" for r in arm_results):
        session_status = "aborted"
        abort_reason = aborted[0].get("abort_reason")
    elif refused and not any(r.get("status") == "complete" for r in arm_results):
        session_status = "refused"
        abort_reason = None
    elif aborted:
        session_status = "partial_aborted"
        abort_reason = aborted[0].get("abort_reason")
    else:
        session_status = "complete"
        abort_reason = None

    summary: dict[str, Any] = {
        "kind": kind,
        "session_id": args.session_id,
        "ended_utc": _utc(),
        "status": session_status,
        "abort_reason": abort_reason,
        "criterion": criterion,
        "pre_registered_predictions": predictions,
        "arm_results": arm_results,
        "primary_claim_eval": primary,
        "n_probes": len(probes_log),
    }
    if canary_guard is not None:
        summary["canary"] = canary_guard.plan_fragment()
        summary["canaries"] = canary_guard.canaries
        plan["canary"] = summary["canary"]
        plan["canaries"] = canary_guard.canaries
        if session_status == "complete":
            try:
                fin = canary_guard.finalize_or_refuse()
            except CanaryUnarmedSealRefuse as exc:
                session_status = UNARMED_REFUSE
                abort_reason = UNARMED_REFUSE
                summary["status"] = session_status
                summary["abort_reason"] = abort_reason
                summary["canary_unarmed_detail"] = exc.detail
                print(f"REFUSED -- {exc.detail}", flush=True)
            else:
                if fin.get("UNGUARDED"):
                    summary["UNGUARDED"] = True
                    summary["unguarded_reason"] = fin.get("note")
                    plan["UNGUARDED"] = True
    if criterion == CRITERION_TTFT_SLO:
        summary["ttft_limits"] = {r["arm_id"]: r.get("ttft_limit_n") for r in arm_results}
        summary["per_probe_medians"] = {
            r["arm_id"]: [
                {
                    "n": p.get("n_tokens"),
                    "prefill_s_median": p.get("prefill_s_median"),
                    "decode_tok_s_median": p.get("decode_tok_s_median"),
                    "slo_ok": p.get("pass"),
                }
                for p in (r.get("probes") or [])
            ]
            for r in arm_results
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan["status"] = session_status
    plan["abort_reason"] = abort_reason
    plan["ended_utc"] = summary["ended_utc"]
    plan["arm_results"] = arm_results
    plan["primary_claim_eval"] = primary
    (out_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan["status"] = session_status
    plan["abort_reason"] = abort_reason
    print(
        json.dumps(
            {
                "ok": session_status == "complete",
                "status": session_status,
                "abort_reason": abort_reason,
                "criterion": criterion,
                "session_id": args.session_id,
                "out": str(out_dir),
                "UNGUARDED": bool(summary.get("UNGUARDED")),
            },
            indent=2,
        )
    )
    return 0 if session_status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
