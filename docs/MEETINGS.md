# Meetings

## 2026-10-13, meeting with Prof. Hao

**Asked of us:**
1. Quantify every problem in money and time, and rank the problems by cost.
2. Show that the problem cannot be fixed by one flag, then show our fix.
3. Keep vendor-specific findings as evidence for general claims (the claims are general; each vendor is a data point).

**Priorities:** cost metrics and scale-up, in parallel.

**Action items:**
- Problem ranking by cost (Rithwik: docs/IMPACT.md, analysis/impact_model.py; plan v2 Part A).
- Scaling to larger models (Rithwik: R2 at 30B to 70B on evo-x2; plan v2 Part B).
- Persistent window override on all vendors (Zach).
- KV eviction that protects guardrails (Zach).

**Zach's update, 2026-10-13 (as reported by Zach; not yet in the numbers register, registration pending under plan
item A6, so these are not cited as our results):**
- XPS: longest resident context 15,000 tokens with f16 KV, 20,000 with u8, 26,500 with u4; int8 weights decode 30 to
  34% slower than int4 (commit 180dfbb9).
- evo-t2s: u8 KV reaches the same limits as f16 for the 8B (16,437) and nearly for the 4B (18,500 vs 18,687).
- NPU: 0/12 with int4 g128 vs GPU 12/12 (commit 78fc3478); matches the GPU word for word with int8 channel-wise
  (commit b172b287).
- Hybrid demo prediction: the XPS keeps about 44% of long-task turns local at about USD 0.70 per task; evo-t2s 91 to
  100% local under USD 0.15; about USD 80 predicted and a USD 200 ceiling.
