# SEAM measurement protocol

## Five gates (launchers)

Before spawn, launchers refuse unless: uptime/cold-boot gate; AC online;
Best Performance power plan; Available MB ≥ floor (7000 for C-2/H-1);
tier-1 processes absent (Cursor/chrome/msedge/claude/vmmem).

## Canary (INF-1 / INF-1b)

Fixed cell: `gpu_only_f16`, `n_cached=4000`, `delta=400`, `RESIDENT`.
`C=3`, `rel_drift_floor=0.05`, `onset_s=657`.

```
N = min(floor(onset_s / mean_probe_wall_s),
        floor(planned_probe_count / (C + 1)))
```

Refuse to start if budget cannot fit C + 1 canaries. Refuse to seal with
`armed == false` unless `-AllowUnguarded` (writes `UNGUARDED`). Trip →
`FAIL_CANARY_DRIFT`.

## Sealed runs

Write-once trees with `tree_sha256` / `.sealed`. Every published number
traces to a `run_id` in `zjohnson2005/gnn-hls-accel`.

## Interleaved arms

Matrices interleave arms; canaries use one fixed cell.

## Pre-registered predictions

Written into `plan.json` before first probe (e.g. C-2 AM-038). Amendments
require authorization in `AMENDMENTS.md` (canonical repo).

## KV readback refusal

Normalized precision path:
`loads[i].kv_cache_precision.readback.normalized`.
`None` or missing → REFUSE. Mismatch → `KV_PRECISION_MISMATCH`.
