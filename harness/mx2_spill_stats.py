"""MX2 spill cost with statistics (item 3(b), evo-x2, 2026-10-06 replan). Reuses mx2_validation.py's own
run_regime_point unchanged -- same LlamaServerConfig/LlamaServerSession path, same fixed 2048-token prompt,
n_predict=128, same GPU dedicated/shared-usage measurement -- at 5 n_ctx points for llama-3.3-70b, 5 reps
each, and reports decode tok/s and TTFT with 95% CIs per point plus per-process GPU dedicated/shared usage.

Claim under test: crossing the device-local heap costs nothing on evo-x2 (i.e. decode tok/s and TTFT at
n_ctx points above the device-local heap size are not measurably worse than points below it).

Points: 20480, 65536, 100000, 115200, 131072 (n_ctx=131072 is this model's own effective ceiling -- requests
above it clamp, confirmed in docs/FINDINGS.md's MX2 correction; included anyway since it is one of the 5
tiers specified).

Usage: py -3.12 harness/mx2_spill_stats.py --out results/mx2_spill_stats.jsonl [--reps 5]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if (_here / "analysis").is_dir():
    REPO = _here
else:
    REPO = _here.parents[0]
sys.path.insert(0, str(_here))
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))

from mx2_validation import run_regime_point, utc_iso, emit  # noqa: E402

MODEL_ID = "llama-3.3-70b"
N_CTX_POINTS = [20480, 65536, 100000, 115200, 131072]
DEFAULT_REPS = 5

# Student's t critical value, two-sided 95%, for small n (n-1 degrees of freedom); n=5 reps -> df=4.
T_CRIT_BY_DF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def mean_ci95(values):
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None, None, 0
    mean = sum(vals) / n
    if n < 2:
        return mean, None, n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = T_CRIT_BY_DF.get(n - 1, 1.96)
    return mean, t * se, n


def already_done(out_path):
    done = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("record") == "mx2v_regime_point" and r.get("model_id") == MODEL_ID:
            done.add((r["point_type"], r["requested_n_ctx"], r["rep"]))
    return done


def run(out_path, reps, log=print):
    done = already_done(out_path)
    for n_ctx in N_CTX_POINTS:
        point_type = f"spill_stats_{n_ctx}"
        for rep in range(reps):
            key = (point_type, n_ctx, rep)
            if key in done:
                continue
            emit(out_path, {"record": "heartbeat", "n_ctx": n_ctx, "rep": rep, "ts_utc": utc_iso()})
            run_regime_point(MODEL_ID, point_type, n_ctx, rep, out_path, log)

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()
           if l.strip() and json.loads(l).get("record") == "mx2v_regime_point"
           and json.loads(l).get("model_id") == MODEL_ID]
    log("\n--- summary: decode tok/s and TTFT, mean +/- 95% CI, per n_ctx point ---")
    for n_ctx in N_CTX_POINTS:
        point_rows = [r for r in rows if r.get("requested_n_ctx") == n_ctx and r.get("started")]
        decode_vals = [r.get("decode_tok_s") for r in point_rows]
        ttft_vals = [r.get("ttft_ms") for r in point_rows]
        shared_vals = [r.get("gpu_shared_mib") for r in point_rows]
        dedicated_vals = [r.get("gpu_dedicated_mib") for r in point_rows]
        d_mean, d_ci, d_n = mean_ci95(decode_vals)
        t_mean, t_ci, t_n = mean_ci95(ttft_vals)
        actual = point_rows[0].get("actual_n_ctx") if point_rows else None
        clamped = point_rows[0].get("clamped") if point_rows else None
        log(f"n_ctx={n_ctx} (actual={actual}, clamped={clamped}): "
            f"decode_tok_s={d_mean} +/- {d_ci} (n={d_n}), ttft_ms={t_mean} +/- {t_ci} (n={t_n}), "
            f"gpu_shared_mib(mean)={sum(v for v in shared_vals if v is not None) / max(1, len([v for v in shared_vals if v is not None])) if shared_vals else None}, "
            f"gpu_dedicated_mib(mean)={sum(v for v in dedicated_vals if v is not None) / max(1, len([v for v in dedicated_vals if v is not None])) if dedicated_vals else None}, "
            f"n_rows={len(point_rows)}/{reps}")
    emit(out_path, {"record": "run_end", "ts_utc": utc_iso()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    args = ap.parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {out_path}")

    note = "completed"
    try:
        run(out_path, args.reps)
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        print(note)
        traceback.print_exc()
    finally:
        try:
            import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}")


if __name__ == "__main__":
    main()
