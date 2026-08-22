# Result File Provenance and Corrections

This file records corrections to metadata embedded in historical result files.
Result files are not rewritten after the fact; corrections are documented here
and must be applied when interpreting or pooling result rows.

---

## Correction 1: host string "blade_rtx4070" → "blade14_rtx4070"

**Date of correction:** 2026-08-22  
**Affected field:** `hardware` (per-row string field in JSON result files)

**Reason:** The measurement host was identified as a Razer Blade 15 in all
documents and harness scripts prior to 2026-08-22. The correct model is
**Razer Blade 14 (RZ09-0508)**. The config key was renamed from
`blade_rtx4070` to `blade14_rtx4070` in the same commit, and all docs were
updated. Result rows written before this commit carry the string
`"hardware": "blade_rtx4070"`, which should be read as `"blade14_rtx4070"`.
The hardware is otherwise unchanged: RTX 4070 Laptop GPU, 8188 MiB GDDR6
VRAM, discrete memory architecture. No measurement values are affected.

**Files carrying the uncorrected string:**

| File | Rows affected |
|------|---------------|
| `results/art_truncation.json` | 480 (all rows) |
| `results/art_headroom.json` | all rows |
| `results/partial_truncation.json` | all rows |
| `results/filler_composition.json` | all rows |
| `results/model2_truncation.json` | all rows |
| `results/selfreport_arms.json` | all rows |
| `results/stage_c_20260818T040408Z.jsonl` | all rows |
| `results/position_pressure_analysis.json` | all rows |
| `results/ablation_cha04_20260817.jsonl` | all rows |

**Reading rule:** any row with `"hardware": "blade_rtx4070"` was collected on
the Razer Blade 14 RZ09-0508, RTX 4070 dGPU, discrete VRAM, and is
equivalent to rows labelled `"blade14_rtx4070"` from this commit forward.

---

## Note: AMD Radeon 780M iGPU added to host inventory (2026-08-22)

The Razer Blade 14 RZ09-0508 also carries an **AMD Radeon 780M integrated
GPU** (unified memory, LPDDR5, 15.6 GB shared GPU memory pool, 31.3 GB system
RAM total). This iGPU was not previously documented. It is now registered as
`blade14_780m` in `configs/hardware/`. No experiments have been run on the
780M as of this date. When they are, their rows will carry
`"hardware": "blade14_780m"` and must not be pooled with `blade14_rtx4070`
rows or with future AMD Strix Halo data.

---

## Note: stage_a_scale.json — missing instrumentation fields (pre-fix rows)

**Affected file:** `results/stage_a_scale.json`  
**Affected rows:** The 51 rows where `classification_method == "unavailable"` (all
rows not replaced by the Stage A rerun).

**Missing fields:** `eval_count`, `prompt_eval_count`, `done_reason`. The
`thinking` field is present only as a 120-character snippet under the key
`thinking_snippet`; the full thinking trace was not stored.

**Cause:** These rows were written by `harness/stage_a_scale.py` at
`MIN_PREDICT=512` before the harness instrumentation fix that added the above
fields to all harnesses. The fix raised `MIN_PREDICT` to 1024 and stores the
full thinking trace under `"thinking"`.

**Reading rule:** For the 51 pre-fix rows, `done_reason` is unknown. These rows
cannot be classified as `budget_exhausted` or `null_response` from the data
alone. The field `classification_method: "unavailable"` marks them
explicitly. The 21 rows that produced empty output (`output == ""`) were
rerun with `harness/stage_a_rerun.py` at `MIN_PREDICT=1024`; those
replacement rows carry `classification_method: "done_reason"` and `rerun:
true`, and their `done_reason` field is authoritative.
