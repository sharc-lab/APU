# NPU-1 amendment 2026-10-08: guard schedule and decode (pre-run)

Registered and committed before any runner change and before any further
NPU-1 run. It adds to docs/NPU_PROTOCOL.md, derived/npu/NPU1_PREREG.json and
docs/amendments/NPU_AMEND_2026-10-07.md. It does not edit them. Where this
file and those differ, this file governs NPU-1 from here on. The runner does
not open this file.

Basis: the 1024 rerun 8ccfb38c-b1d0-43af-b9d5-18627c2ab00f (EVO-T2S,
2026-10-07 23:54-23:57 UTC, git 75436f8). It ended REFUSED_UNARMED_CANARY,
UNGUARDED. Its result is visible to us (64 and 1024 rungs timed, binding
field TOOLCHAIN_CAP). Nothing from it is cited and prediction (b) is not
scored from it. The choices below do not depend on that result: item 1
follows from the requirement alone, and item 2 follows from the runner's own
decode rule. Neither changes which rungs are visited, the SLO, or any
threshold.

## 1. Guard schedule: calibrate before the first probe

Requirement: the GPU drift canary is armed before any probe whose result
can be cited, whatever the cell's actual probe count.

Why 8ccfb38c did not arm. Under the 2026-10-05 amendment the cadence is
N = min(floor(657 / mean_probe_wall_s), floor(planned / (C + 1))), with
planned the bisection upper bound (18 at 1024) and C = 3. The gate arms on
the C-th OK canary: the opening canary plus one every N probes. With N = 4
the gate arms after probe 8. The single template (2026-10-07 item 3) makes
the 1024 rung feasible. It passed the SLO, so the bisection stopped after
the 64 and 1024 rungs: 6 probes, 2 canaries, never armed. 04d189da armed
only because its 1024 rung was refused and the bisection ran all 18 probes.

Options considered:

- (i) Derive N from the minimum guaranteed probe count instead of the
  planned count. Rejected. Arming needs C canaries, and in this design they
  are spread between probes, so at least C - 1 intervals of probes always
  run before the gate arms. That breaks the requirement for every probe
  count. With a minimum of 6 probes, N = 1 and probes 1-2 still precede
  arming.
- (ii) Run the arming canaries before the measured probes. Chosen. It is
  the only option where arming comes before probe 1 whatever the probe count
  turns out to be.

Rule, for the NPU-1 bisection session (npu1-setting, not --load-only):

1. After the load and the 2026-10-07 capacity check, and before the paired
   GPU pipeline and any probe, the runner runs C = 3 canaries back to back
   (the same fixed cell, gpu_only_f16 RESIDENT n_cached 4000 delta 400).
   The gate must be armed after them. If it is not armed, the session is
   REFUSED_UNARMED_CANARY, and no probe runs.
2. During the bisection, canaries fire every N probes as before. N's
   formula, onset_s 657, C = 3, and the planned count are unchanged.
3. After the last probe, if any probe ran since the last canary, one closing
   canary runs. Each probe is then followed by an armed check before the
   seal.
4. Any armed canary that trips ends the session with FAIL_CANARY_DRIFT, as
   before. That includes the interval and closing canaries.
5. The seal still refuses when armed is false, unless AllowUnguarded writes
   UNGUARDED. That path is unchanged.

Unchanged: the drift rule, threshold = max(2 x early_max, healthy_floor),
the healthy floors, the pool, C, onset_s, the N formula and its budget
preflight, the warm-up setting (no warm-up discard in the NPU runner, as in
04d189da), and the non-gating NPU n=400 series. Calibrating on back-to-back
canaries can only give an early_max as small as or smaller than spread-out
canaries. The threshold then sits at or above the healthy floor, as the
formula requires. That is not a loosening.

Out of scope: the load-only and LOAD_FAIL paths. They time no probe and
keep the opening canary as before.

## 2. Decode at the limit

Why decode_tok_s_at_limit was null in 8ccfb38c. The runner computes decode
tok/s only from streamed tokens, as (tokens - 1) / (last - first). A
generate that streams one token has no decode rate. In 8ccfb38c every rung
row has a prefill (streamer first-token time, so at least one token
streamed) and a null decode (so at most one). P1's text was empty. We infer
that the NPU's first token under the single template was end-of-sequence.
The completion count was not recorded, which is why it is an inference. In
04d189da the double template made the model emit text, so several tokens
streamed and decode was recorded (16.53 tok/s).

Rule: every NPU and GPU generate sets min_new_tokens = max_new_tokens = 8,
so each generate streams exactly 8 tokens. Decode tok/s is measured over 7
intervals. max_new_tokens, greedy decoding, apply_chat_template False, the
list call form and MIN_RESPONSE_LEN = 8 are unchanged. TTFT, the SLO and the
binding are unchanged, because TTFT is the time to the first token. Per row
the runner also records the pipeline completion count and the streamer token
count. When decode cannot be measured it records why. When
decode_tok_s_at_limit is null and a limit exists, the summary says why.
Prediction (c) is not scored from a null value.

The summary also carries the capacity check record. Before this change it
was in plan.json only.

## 3. Existing runs

- 04d189da (guarded, 1024): none. It is kept as sealed and is not relabeled.
  It was guarded under the schedule then in force. That schedule armed at
  canary 2, after probe 8. The cited limit rung (960) ran after the gate
  armed.
- 8ccfb38c: kept as a record. Status REFUSED_UNARMED_CANARY, UNGUARDED. Not
  cited, not void. Prediction (b) is not scored from it.
- 680b031a, 28cf811d, 041d9f13: none.

Next NPU-1 run: the guarded 1024 rerun under this amendment. Then 2048,
4096, 8192 and 16384 load-only, per 2026-10-07 item 4.
