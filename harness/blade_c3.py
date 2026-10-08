"""Blade C3 (night 2, job 1): does NVIDIA's "CUDA - Sysmem Fallback Policy" select the C1 failure regime?

C1 (FINDINGS "C1, Blade") found a silent spill: from ctx 38912 the KV cache that no longer fits in 8 GB lands in
system memory and TTFT jumps 3.7x to 5.1x, with the policy value UNKNOWN (never changed, not readable). C3 reruns C1's
spill points with the policy set explicitly, then restored:

  half A  policy "Prefer No Sysmem Fallback": ctx 36864 (last clean point), 38912, 40960, 43008; 1 warm-up + 5
          measured calls per ctx (C1's design), order shuffled with ORDER_SEED. Prediction (not assumed): the spilled
          points stop spilling and fail loudly at load or at the first call instead.
  half B  policy restored to "Driver Default": ctx 40960 and 43008, 1 warm-up + 3 measured calls (does the C1 spill
          come back with the restored setting).

Everything per call is C1's harness (harness/blade_spill_sweep.py run_throughput_ctx: qwen3-4b-instruct, f16 KV,
-fa on -ngl 99 -np 1 -t 4, 90% fill, thermal gate, stale-server guard, nvidia-smi + dmon + per-PID Shared Usage).
Unlike C1 there is no skip-after-failure rule: every ctx is attempted, a failure is the measurement.

The operator flips the setting at the keyboard (docs/BLADE_PLAN.md "C3 operator steps"). Before each half the job
writes C:\\apu\\blade\\c3_WAITING_<half>.txt and waits, with NO timeout, for C:\\apu\\blade\\c3_confirm_<half>.flag
created after the wait began whose first line is exactly the expected setting text ("Prefer No Sysmem Fallback" or
"Driver Default"). A flag left from an earlier run (older than the wait) is renamed .stale and ignored. nvidia-smi
does not expose this policy, so the confirmation text is the record of the setting; the half-A rows themselves
(Shared Usage excess, server start or failure) are the behavioural check.

Usage:
  py -3.12 harness/blade_c3.py --out results/blade_c3_sysmem_fallback_v1.jsonl
  py -3.12 harness/blade_c3.py --out results/blade_dryrun/blade_dryrun_c3_gate.jsonl --dry-run-gate
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import blade_common as bc  # noqa: E402

HALVES = (
    {"half": "A", "policy": "Prefer No Sysmem Fallback", "ctxs": (36864, 38912, 40960, 43008), "n_measured": 5},
    {"half": "B", "policy": "Driver Default", "ctxs": (40960, 43008), "n_measured": 3},
)
ORDER_SEED = 20261008
POLL_S = 10
HEARTBEAT_S = 600


def ordered_ctxs(half: dict) -> list[int]:
    ctxs = list(half["ctxs"])
    if half["half"] == "A":
        random.Random(ORDER_SEED).shuffle(ctxs)
    return ctxs


def waiting_text(half: dict, flag: Path) -> str:
    return (f"Blade C3 is waiting for half {half['half']}.\n"
            f"1. Open NVIDIA Control Panel > 3D Settings > Manage 3D settings > Global Settings.\n"
            f"2. Set 'CUDA - Sysmem Fallback Policy' to '{half['policy']}', press Apply.\n"
            f"3. Re-open the drop-down and confirm it reads '{half['policy']}'.\n"
            f"4. Create {flag} with exactly this first line:\n{half['policy']}\n"
            f"The job does not start this half on its own; there is no timeout.\n")


def wait_for_confirmation(half: dict, flag_dir: Path, emit, log, sleep=time.sleep, now=time.time,
                          poll_s=POLL_S, heartbeat_s=HEARTBEAT_S, max_polls=None) -> dict:
    """Blocks until the half's confirmation flag exists, is newer than the wait, and names the expected policy.
    max_polls is for tests only (None = wait forever). Returns the c3_gate_confirmed row."""
    flag_dir = Path(flag_dir)
    flag_dir.mkdir(parents=True, exist_ok=True)
    flag = flag_dir / f"c3_confirm_{half['half']}.flag"
    waiting = flag_dir / f"c3_WAITING_{half['half']}.txt"
    opened = now()
    if flag.exists():
        stale = flag.with_suffix(".flag.stale")
        if stale.exists():
            stale.unlink()
        flag.rename(stale)
        emit({"record": "c3_gate_stale_flag_moved", "half": half["half"], "to": str(stale)})
    waiting.write_text(waiting_text(half, flag), encoding="utf-8")
    emit({"record": "c3_gate_open", "half": half["half"], "expected_policy": half["policy"], "flag": str(flag),
          "waiting_file": str(waiting), "opened_epoch": opened})
    log(f"C3 half {half['half']}: waiting for {flag} ('{half['policy']}'), no timeout")
    polls, last_beat, rejected_mtime = 0, opened, None
    while True:
        if flag.exists():
            mtime = flag.stat().st_mtime
            text = flag.read_text(encoding="utf-8", errors="replace")
            first = (text.splitlines() or [""])[0].strip()
            if mtime >= opened and first == half["policy"]:
                waiting.unlink(missing_ok=True)
                row = emit({"record": "c3_gate_confirmed", "half": half["half"], "policy": half["policy"],
                            "flag_text": text[:500], "flag_mtime_epoch": mtime, "waited_s": round(now() - opened, 1),
                            "readback": "operator confirmation (nvidia-smi does not expose this policy)"})
                log(f"C3 half {half['half']}: confirmed '{first}'")
                return row
            if mtime != rejected_mtime:
                rejected_mtime = mtime
                emit({"record": "c3_gate_rejected", "half": half["half"], "flag_first_line": first[:200],
                      "expected": half["policy"], "older_than_wait": mtime < opened})
                log(f"C3 half {half['half']}: flag rejected (first line {first!r}, expected {half['policy']!r})")
        polls += 1
        if max_polls is not None and polls >= max_polls:
            raise TimeoutError("test-only max_polls reached")
        if now() - last_beat >= heartbeat_s:
            last_beat = now()
            emit({"record": "c3_gate_waiting", "half": half["half"], "waited_s": round(last_beat - opened, 1)})
        sleep(poll_s)


def halves_done(rows: list[dict]) -> set:
    return {r["half"] for r in rows if r.get("record") == "c3_half_done"}


def run_half(half: dict, out: Path, emit, log, flag_dir: Path):
    import blade_spill_sweep as bss
    import blade_telemetry as bt
    gate = wait_for_confirmation(half, flag_dir, emit, log)
    bss.verify_model_sha256()
    env = bt.environment_manifest()
    env["cuda_sysmem_fallback_policy"] = f"{half['policy']} (operator confirmation, flag {gate['flag_mtime_epoch']})"
    run = bss.Run(f"blade_c3_half{half['half']}", out.parent, scratch=False)
    run.jsonl = out          # all rows in one file; telemetry and server logs keep the per-half stem
    run.manifest = {"experiment": f"C3 half {half['half']}", "policy": half["policy"], "environment": env,
                    "ctxs": ordered_ctxs(half), "n_warmup": bss.N_WARMUP, "n_measured": half["n_measured"],
                    "order_seed": ORDER_SEED, "server_bin": bss.SERVER_BIN, "model_sha256": bss.MODEL_SHA256}
    run.save_manifest()
    bss.N_MEASURED = half["n_measured"]
    run.tele.start_gpu()
    try:
        for ctx in ordered_ctxs(half):
            tag = f"half{half['half']}_ctx{ctx}"
            log(f"=== C3 {tag} ===")
            try:
                res = bss.run_throughput_ctx(run, ctx, tag, f"c3_{half['half']}", 0)
            except bss.StopExperiment:
                raise
            except Exception as e:  # a crash at this ctx is data; record it and go on
                res = {"failed": True, "reason": repr(e)[:300]}
            emit({"record": "c3_ctx_result", "half": half["half"], "policy": half["policy"], "ctx": ctx, **res})
            time.sleep(5)
    finally:
        run.tele.stop()
        time.sleep(2)
        run.manifest["completed_utc"] = bc.utc_iso()
        run.save_manifest()
    emit({"record": "c3_half_done", "half": half["half"], "policy": half["policy"], "telemetry_prefix": run.tele.prefix})


def dry_run_gate(out: Path, emit, log) -> None:
    """The gate logic only, with a fake operator: a temp flag dir, a wrong flag first, then the right one. No server,
    no measurement, no NVIDIA setting touched."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="blade_c3_gate_dryrun_"))
    for half in HALVES:
        def operator(h=half):
            time.sleep(1.0)
            (d / f"c3_confirm_{h['half']}.flag").write_text("wrong value\n", encoding="utf-8")
            time.sleep(1.5)
            (d / f"c3_confirm_{h['half']}.flag").write_text(h["policy"] + "\nfake operator (dry run)\n",
                                                            encoding="utf-8")
        t = threading.Thread(target=operator, daemon=True)
        t.start()
        wait_for_confirmation(half, d, emit, log, poll_s=0.2)
        t.join()
        emit({"record": "c3_dry_run_half_skipped", "half": half["half"], "ctxs": ordered_ctxs(half),
              "why": "dry run: measurement part not run"})


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flag-dir", default=str(bc.BLADE_DIR))
    ap.add_argument("--dry-run-gate", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = bc.out_path_ok(args.out, args.dry_run_gate)
    emit = bc.jsonl_emitter(out)

    def log(msg):
        print(f"[{bc.utc_iso()}] {msg}", flush=True)

    bc.require_blade()
    versions = bc.versions_record()
    emit({"record": "run_start", "job": "blade_c3", "argv": sys.argv, "dry_run": args.dry_run_gate,
          "versions": versions, "version_problems": bc.version_problems(versions), "halves": HALVES})
    if args.dry_run_gate:
        dry_run_gate(out, emit, log)
        emit({"record": "run_stopped", "note": "dry run: gate logic only"})
        return 0
    # C3 uses llama-server only: the Ollama pin does not apply, the driver and llama.cpp pins do
    problems = [p for p in bc.version_problems(versions) if not p.startswith("ollama")]
    if problems:
        emit({"record": "refused", "reasons": problems})
        log(f"refused (versions): {problems}")
        return 2
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = halves_done(rows)
    try:
        for half in HALVES:
            if half["half"] in done:
                log(f"C3 half {half['half']} already done, skipped")
                continue
            run_half(half, out, emit, log, Path(args.flag_dir))
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit({"record": "run_stopped", "note": f"stopped: {e!r}"[:400]})
        return 2
    emit({"record": "run_end", "note": "completed"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
