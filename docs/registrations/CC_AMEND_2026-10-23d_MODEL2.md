# CC amendment 2026-10-23d: second model (Qwen3-8B INT4), plus wording reconciliations

Registered Fri Oct 23 2026, about 1:30 PM ET (Task BO), before any Qwen3-8B CC cell ran. Mac only, $0.
Approval: Zach, Fri Oct 23 about 1:05 PM ET, "Add Qwen3-8B" (TASK_BO_WEEKEND_PARALLEL.md). This is the
one addition under the PLAN LOCK (docs/PLAN_LOCK_2026-10-23.md, addendum line dated 2026-10-23).

It edits no existing prediction or gate. New predictions: derived/cc/CC_M2_PREDICTIONS.json (DERIVED;
tools/cc_m2_predict.py; tests/test_cc_bo.py). Runners do not read this file.

## 1. Model and pin
- Qwen3-8B INT4, configs/models/Qwen3-8B-int4-ov.v2.yaml: revision 5c47abf4..., ir_sha256
  cd97874b..., openvino_model.bin f2e6960f...; INT4_ASYM, group 128, scale estimation on.
  - The IR is already on the XPS. No download.
  - Its quantization differs from the 4B IR (INT4_SYM + AWQ). Weight bytes enter only through W.
- Config read 2026-10-23 (huggingface.co/Qwen/Qwen3-8B config.json):
  - 36 layers, hidden 4,096, intermediate 12,288, 32 query heads, 8 KV heads, head_dim 128;
  - max_position_embeddings 40,960, rope_scaling null, untied embeddings.
- KV shapes equal Qwen3-4B's, so k is f16 147,456 / i8 78,336 / u4 41,472 B per token.

## 2. Predictions (DERIVED, before any run; bands +-15%, the held-out rule)
- **a from config.** a = 4 x (2 x intermediate + 2 x hidden) = 131,072 B per token.
  - The same rule gives 98,304 for Qwen3-4B, 1.6% from the fitted 96,768.
- **B.** B_8B = B_4B(fit, 3.412 GiB) - (W_8B - W_4B) = 0.998 GiB.
  - W is the IR bytes: 4,882,865,352 vs 2,290,768,181.
  - The OS tier and the fixed overhead cancel.
  - A19 substitution rule (no refit): if the A19 record at the M2 run shows a GPU-shared tier S
    different from the CC v3 queue-start record, use B_8B + (S_m2 - S_v3).
  - Caveat: B_8B is small, so a 0.1 GiB error in the weights' resident size moves the limits about
    10%.

| Quantity | Point | Band (+-15%) |
|---|---|---|
| allocator slope a, f16 and u4 | 131,072 B/token | 111,411 - 150,733 (f16 vs u4 within 10%) |
| uncapped f16 limit | 3,846 | 3,269 - 4,423 |
| uncapped i8 limit | 5,116 | 4,348 - 5,883 |
| uncapped u4 limit | 6,209 | 5,277 - 7,140 |
| u4 cap-256 limit | 25,022 | 21,269 - 28,775 |
| f16 cap-256 limit | 7,037 | info only |
| KV-compression bound (k_f16 + a) / a | 2.125x | - |

## 3. Cells (XPS, after the CC v3 queue; about 4-6 h)
- **Allocator slope (A20 style):** f16 and u4 uncapped, points 1,024 / 2,048 / 3,072, each 3 times.
  - The A20 points (4K-12K) are above the 8B f16 limit, so smaller points are used. They are all
    below the f16 band.
- **Uncapped bisections, 250-token grid:** f16 1,000-8,000; i8 2,000-10,000; u4 2,000-12,000. Then
  repeats at L* and L* + 250 (r = 3).
- **u4 cap-256 bisection:** 4,000-40,000 at 1,000, with repeats (r = 3).
- **Logging as A6 / A19 / A31 / A32:** --gpu-mem-stats, --compiled-readback, --reject-kv-mismatch,
  --mem-trace-hz 10, --dxgi-trace if the A19 probe works, child timeout 3,600 s.
- **Canary.** The canary stays the registered Qwen3-4B cell (new default-off option
  `--canary-model-spec configs\models\Qwen3-4B-int4-ov.yaml`).
- **Queue:** ~/seam_closeout_1013/xps/run_queue_xps_m2.sh (new). It refuses while the XPS lock is held
  (a running v3 queue). It runs only after Zach pushes the commit that registers this amendment.

## 4. Wording reconciliation (allowed under the lock; no threshold changes)
- **Nov 6 gate, one wording everywhere:**
  - target: held-out median absolute error <= 10%;
  - red flag (R8): median > 15%;
  - at least 30 held-out cells;
  - the binding ceiling called correctly in >= 90% of cells.
  - This replaces the Award-report wording (<= 5% median / <= 10% worst case), which was never a
    registered gate.
- **Three pre-existing tensions,** recorded as "to be read against A19 max-allocation, A21, CC-GROW,
  A32", not as verdicts:
  - (1) 15367c3a (XPS, DEMO-2 pilot arm A, i8 true default, LONGCTX 15-30) ran a generate at
    67,947 tokens (entry 138 step 5; analysis/demo2_pilots/BB/CONTEXT_OVER_LIMIT.md; the entry is
    scored a local fail). That is far above the B fitted from the single-prefill limits, on a
    resident, prefix-cached session. Read it against A19 (max allocation), CC-GROW and A32
    (failure class).
  - (2) The sealed 8-bit points 20,000 (cf3555d5, u8 request compiled i8) and 18,000 (a5d19dcc,
    true default) lie below the A21 band 20,921-21,486. 20,000 is inside the R1 range of
    19,900-22,600; 18,000 is not, but it is a different compile (no KV property). Read it against
    A21.
  - (3) V5-CAL cached-prefix turns of 17-35 s at delta 8,511 (a178e9c5, e1ed06b4). The A23 law
    gives about 3.8 s at L 20,000 / delta 2,048, and about 10 s at the V5-CAL shape. Read it
    against CC-GROW (A23).

## 5. Retrospective check (labelled; not held-out; analysis/cc/retro_8b.md)
- Sealed 8B points through the same law: f16 3,156 (322b2f86, first fail 3,312) vs 3,846 predicted;
  u8 request 4,062 (69234ed5) vs 5,116 (i8).
- The law overpredicts by 22-26%, outside +-15%. Those runs used an older driver and stack and a cold
  single-generate tool (c2_ttft), so they are not a test of this amendment.
- They are an early warning that the second-model cells may miss their bands. This is reported to
  Zach now (lock rule: notify; nothing changes).
