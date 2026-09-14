# H-1 predictions (D-1c cloud predictor)

Generated: 2026-09-08T23:04:30.432383+00:00

Cloud $ from linear predictor on turns remaining at escalation.
Exact sealed-20 token lookup kept as a test only (not the reported number).

## Input run_ids

- C-2 TTFT / gpu prefill (PROVISIONAL): `62395fdb-1899-415f-b708-6adc81a24dda`
- cpu-p prefill curve: arm A `b5ce21e5-9f29-46f4-8319-f74adcdeb628` seal `ad7b9288-42e6-42e8-be3e-ad1a1b1abc4c`
- RESIDENT delta-prefill: `41e419bd-f3e9-43b1-8364-0ebd89fa086b` (SD-001)
- Decode BW+c fit: `a784f5ec-5615-4fea-a680-a07874426ae4` (gpu_only; cpu-p decode ASSUMED from gpu fit)
- Commit intercept: `83127e1b-9d6e-4103-bee6-2a63c00f479f`
- Quality int4: `6225d6e1-4e0a-41c9-90bb-695ecc5fbe0a`
- Quality int8: `1d8db970-4c18-4bcf-824d-d9c141b6eb22`
- Cloud cost fit: `cloud_multi_turn_report.json` (20 entries)
- Cloud completion 13/20: PROVISIONAL n=20

## Cloud cost fit

usd = 0.010377 + 0.071267×turns_remaining; R²=0.7132; residual_se=0.0544; n_points=70; ASSUMED(fit=20 entries cloud_multi_turn_report.json)

C-3 confirm (per-entry): {'c3_verdict': 'closer to linear in remaining cloud turns (fitted exponent 1.027); quadratic claim not supported', 'c3_fitted_exponent': 1.026974487407871, 'c3_linear_r2_aggregated': 0.9798644526439462, 'per_entry_all_from_turn_r2': 0.7132240894033286, 'per_entry_full_entry_only_r2': 0.4363912622899151, 'per_entry_full_entry_only_slope': 0.06418357894736841, 'per_entry_full_entry_only_intercept': 0.045709673684210556, 'n_points_all_from_turn': 70, 'n_points_full_entry': 20, 'linear_confirmed': True}

Exact-lookup test (20 sealed): pred $5.407044 vs meas $5.407044; max |err| $1.11022e-16

## Emission granularity / R2b policy

ENTRY: failure_bucket==multi_turn:empty_turn_model_response on sealed W-3 ledgers. Future W-3 runs also record per_turn.emitted_parseable_tool_call (boolean); sealed runs untouched. emission_escalate still uses entry granularity until a turn-level policy is specified.

R2b policy: on an empty-turn response, the cloud redoes that turn and the
entry stays on cloud. Completion/emission reported as floor/indep range.

## Predictions

| id | policy | trace_n | config | cloud $ [lo, hi] | completion | emission | SLO frac (local) | session time sum (local s) |
|---|---|---:|---|---|---:|---:|---:|---:|
| R0 | cloud_only | 200 | gpu_only RESIDENT u8 int4-4B | 54.0704 [51.1228, 57.0180] | 0.6500 | 1.0000 | n/a | 0.0 |
| R1 | agnostic_default | 200 | cpu-p NON_RESIDENT u8 int4-4B | 0.0000 [0.0000, 0.0000] | 0.1000 | 0.7750 | 0.0000 | 77197.8 |
| R2a | slo_escalate | 200 | gpu_only RESIDENT u8 int4-4B | 0.0000 [0.0000, 0.0000] | 0.1000 | 0.7750 | 1.0000 | 2641.5 |
| R2b | emission_escalate | 200 | gpu_only RESIDENT u8 int4-4B | 13.0812 [11.8752, 14.2871] | floor=0.1000/indep=0.2463 | floor=0.7750/indep=0.9213 | 1.0000 | 1983.7 |

## Decode fit residuals (W-2)

| weight | n | measured | predicted | residual |
|---|---:|---:|---:|---:|
| int4 | 2000 | 28.4572 | 29.8917 | -1.4344 |
| int4 | 5000 | 27.9493 | 25.7599 | 2.1893 |
| int4 | 12000 | 19.9773 | 19.9216 | 0.0557 |
| int8 | 2000 | 19.1844 | 19.7545 | -0.5701 |
| int8 | 5000 | 18.3040 | 18.1682 | 0.1358 |
| int8 | 12000 | 15.1290 | 15.5054 | -0.3763 |
