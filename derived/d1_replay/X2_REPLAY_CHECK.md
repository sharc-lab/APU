# X-2 replay check

Generated: 2026-09-08T23:04:30.433396+00:00

Replay the four X-2 cells from component curves (int4-4B, kv=u8).
No pass/fail threshold — report the error.

| session | placement | residency | predicted s | sealed s | error s | error % |
|---|---|---|---:|---:|---:|---:|
| cb781dbf | cpu-p | NON_RESIDENT | 7350.3 | 8148.5 | -798.2 | -9.8 |
| 9fdedb46 | cpu-p | RESIDENT | 1251.5 | 1576.4 | -324.9 | -20.6 |
| afd1aa21 | gpu_only | NON_RESIDENT | 817.1 | 750.5 | 66.7 | 8.9 |
| 0963168f | gpu_only | RESIDENT | 242.5 | 489.5 | -247.0 | -50.5 |

## cpu-p decode source / cancellation

### cb781dbf (NON_RESIDENT)

LOST - Aug-7 14.9 tok/s@2048 has no sealed run_id; W-2 has zero cpu-p cells; decode tagged ASSUMED(from=gpu fit). X-2 cpu-p ~14.9 tok/s is holdout, not a fit source.

Cancellation: decode error +13.1 tok/s against a 63.7 s/turn gap, net -9.8%

### 9fdedb46 (RESIDENT)

LOST - Aug-7 14.9 tok/s@2048 has no sealed run_id; W-2 has zero cpu-p cells; decode tagged ASSUMED(from=gpu fit). X-2 cpu-p ~14.9 tok/s is holdout, not a fit source.

Cancellation: decode error +13.2 tok/s against a 7.3 s/turn gap, net -20.6%

