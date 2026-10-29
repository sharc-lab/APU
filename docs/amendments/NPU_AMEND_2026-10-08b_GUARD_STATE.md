# NPU-1 amendment 2026-10-08b: calibrate in measurement state (pre-run)

Registered and committed before any runner change and before any further
NPU-1 run. It adds to docs/amendments/NPU_AMEND_2026-10-08_GUARD.md and does
not edit it. Where they differ, this file governs NPU-1 from here on. The
runner does not open this file.

## Basis

The 1024 rerun 9d6d7f66-0c81-4792-b3a6-a8d20586ba6b (EVO-T2S, 2026-10-08
16:27-16:29 UTC, git f86a137) ended FAIL_CANARY_DRIFT, guarded (armed).

Canary values (turn-1 prefill_s, turn-1 end_ws_bytes) from
work/canaries/canary.N.json:

| run | canary | after_probe_count | turn1 s | ws GB | note |
|---|---|---|---|---|---|
| 9d6d7f66 | 0 | -1 | 1.0317 | 7.58 | calibration |
| 9d6d7f66 | 1 | -1 | 1.0137 | 7.68 | calibration |
| 9d6d7f66 | 2 | -1 | 0.9829 | 7.69 | calibration; ref 1.0137, threshold_t1 0.0771 (healthy floor) |
| 9d6d7f66 | 3 | 3 | 0.9113 | 10.17 | rel drift 0.101 > 0.0771, trip |
| 8ccfb38c | 0 | -1 | 0.9977 | 7.53 | opening |
| 8ccfb38c | 1 | 3 | 0.9115 | 10.11 | |

In both sessions, between the last canary before the probes and the first
canary after them, turn 1 got about 9-10% faster and the working set grew by
about 2.5 GB (committed_mb 13155 -> 15706 in 9d6d7f66, 13077 -> 15618 in
8ccfb38c). The paired GPU LLMPipeline is created between those canaries
(tools/run_npu_profile.py, f86a137: arm at :866, GPU pipeline at :872).
So the calibration ran in a different host state from the probes it
guards, and its reference does not represent that state. We do not know the
mechanism. We read it as a step in host state, not as drift during
measurement.

This choice depends only on canary values and host state (working set,
commit). It does not depend on any NPU probe result. 9d6d7f66's 3 probes
are not cited.

## Rule

For the NPU-1 bisection session (npu1-setting, not --load-only):

1. After the load and the capacity check (2026-10-07 item 2), the runner
   creates the paired GPU LLMPipeline, the same pipeline the probes use.
2. On it, the runner runs one untimed warm-up generate. Prompt: the
   rendered prompt the 64-token rung uses
   (seam.tools.boot4_text.rendered_exact_prompt, target = --low, default
   64, filler unit from configs/delta_n.yaml, salt "npu"). Config: the
   probes' generation config (2026-10-07 item 3, 2026-10-08 item 2). It is
   not a probe. It does not enter probes_log, the cadence count, or any
   rung. It is recorded in plan.json as gpu_warmup. If it does not return,
   the session is REFUSED_GPU_WARMUP, and no canary or probe runs.
3. Then the C = 3 calibration canaries run back to back, and then the
   first probe (2026-10-08 item 1).

Unchanged from 2026-10-08: the drift rule, threshold = max(2 x early_max,
healthy_floor), the healthy floors, C = 3, the N formula and cadence, the
closing canary, REFUSED_UNARMED_CANARY when calibration does not arm, and
no unguarded seals. The NPU pipeline and the capacity probes are unchanged.

## Observation on 04d189da (no relabel)

04d189da (sealed, guarded) also ran its opening canary before the GPU
pipeline existed. Its canaries were:

- c0 1.0095 s at 7.28 GB, before the pipeline;
- c1 0.9109 s at 10.17 GB, after probe 4;
- c2 0.9911 s at 10.15 GB, after probe 8, the one that armed;
- then 1.0245 s and 1.0135 s at ~10.2 GB.

Its calibration (c0-c2: ref 0.9911, early_max_t1 0.081) therefore likely
included this step, and its threshold_t1 0.161907 came out about 2x the
healthy floor. It stays sealed and is not relabeled.

That run also shows that the faster turn 1 was seen only on the first
canary after the GPU pipeline was created. Later canaries at the same
~10.2 GB working set were back near 1.0 s. So the step may be a transient
that follows pipeline creation, not a lasting level. The warm-up generate
in rule item 2 is meant to move calibration past that point. If the
transient lasts longer than one warm-up generate, calibration can still
land inside it. A later canary would then read slower than the reference
and could trip. Such a trip is a guard result under this rule, not a reason
to change the rule after the run.

## Existing runs

- 9d6d7f66: kept as a record. FAIL_CANARY_DRIFT, guarded (armed), not
  cited, not void. Its 3 probes are not cited.
- 8ccfb38c, 04d189da, 680b031a, 28cf811d, 041d9f13: none.

Next NPU-1 run: the guarded 1024 rerun under this amendment.
