# Retrospective: sealed Qwen3-8B INT4 points through the Context Roofline law

Task BO part F, Fri Oct 23 2026. **Retrospective, older driver/stack. Not held-out and not scored.**
The registered predictions for the second model are in derived/cc/CC_M2_PREDICTIONS.json.

## Inputs (all from the repo)
- Law: L = B_8B / (k + a_8B), with a_8B = 131,072 B per token (config rule) and
  B_8B = 3.412 GiB - (4,882,865,352 - 2,290,768,181) B = 0.998 GiB (tools/cc_m2_predict.py).
- k: f16 147,456; 8-bit 78,336 B per token.

## Points

| Run | Arm (requested -> compiled) | Measured | Law | Law / measured |
|---|---|---|---|---|
| 322b2f86 (sealed) | f16 | 3,156 (first fail 3,312, memory_wall) | 3,846 | +21.9% |
| 69234ed5 (sealed) | u8 -> i8 (CONFIG_PROVENANCE rule) | 4,062 (memory_wall) | 5,116 | +25.9% |

## Reading
- The law overpredicts both points by 22-26%. That is outside the +-15% band the held-out cells use.
- **Why these points are not a test:**
  - Both are c2_ttft cold single-generate runs (one child per repeat), not the resident two-turn
    probe.
  - They are from September 2026 (CAP-3), on an older driver (8622-8724 era, CONFIG_PROVENANCE).
  - The 4B budget B was fitted on the resident probe.
- **Why they still matter:** the 8B budget is only about 1 GiB, so any extra resident weight overhead
  (untied embeddings, INT4_ASYM zero points, compiled buffers) is a large fraction of it. An overhead
  of about 0.2 GiB would explain the gap.
  - This is an early warning that the registered 8B cells may miss their bands. Under the plan lock
    it is reported to Zach; nothing changes.
