# NPU-1 amendment 2026-10-08c: validity-set limit, ladder change (pre-run)

Registered and committed before any further NPU-1 run. It adds to
docs/amendments/NPU_AMEND_2026-10-07.md, NPU_AMEND_2026-10-08_GUARD.md and
NPU_AMEND_2026-10-08b_GUARD_STATE.md. It does not edit them or
derived/npu/NPU1_PREREG.json. Where they differ, this file governs NPU-1
from here on. No code changes with this amendment. The runner does not open
this file.

## Basis: 34c2240f

Run 34c2240f-245e-4388-9a84-27ca7f577162: EVO-T2S, 2026-10-08 18:48-18:53
UTC, git 0b06dbb, MAX_PROMPT_LEN 1024. Not yet sealed; it is sealed at
closeout.

- Guard: armed after the GPU warm-up and 3 calibration canaries (turn 1
  0.998 / 1.013 / 1.012 s, ref 1.0122, threshold_t1 0.0771). 8 canaries;
  the largest drift was +5.0%. Status complete.
- Capacity check passed.
  - P1: 960 tokens accepted, pipeline_input_tokens 960, 8 streamed
    tokens. Text "</think>\n\n" four times, which the rule marked valid.
  - P2: 1088 tokens, refused by the runtime length check ("up to 1024
    tokens").
- Rungs, all with exact_match_vs_gpu false on every row. The rung output
  texts were not stored.
  - 64, 512 and 768: valid. Prefill 1.28 s at every length. Decode
    17.0-17.3 tok/s over 8 streamed tokens.
  - 832, 896 and 1024: refused by repeat_4gram (fraction 0.8).
- Summary: binding TOOLCHAIN_CAP, ttft_limit_n 768, decode_tok_s_at_limit
  17.205.

## 1. The 768 limit is set by the validity rule

34c2240f's limit of 768 comes from the validity rule. That rule is
repeat_4gram on 8 forced tokens (2026-10-08 item 2), counted as whitespace
4-grams. Latency did not set it, and neither did the cap: prefill stayed at
1.28 s and the cap is 1024.

The NPU's outputs look degenerate:
- P1 is "</think>" repeated, which is 4 whitespace tokens, so it has a
  single 4-gram and cannot repeat.
- No row matches the GPU.

So the validity rule cannot separate real text from degenerate text here,
in either direction.

Therefore:

- The 768 limit is not cited.
- Prediction (b) is not scored until NPU-NATSTOP and a validity amendment
  settle the question.
- The prefill and decode timings stay recorded (guarded, run_id 34c2240f).
  They carry the same caveat: they were measured on output that may be
  degenerate.
- The binding label TOOLCHAIN_CAP is the literal output of the rule:
  no timed rung missed the SLO. The protocol has no validity-bound
  category. A validity-refused rung is neither LATENCY nor the cap. This
  amendment adds no category; the validity amendment will decide.

## 2. Ladder: changed, not dropped

Prediction (a) needs only the load and the realized capacity, not the
bisection. So 2048, 4096 and 8192 now run as load-only sessions with the
2026-10-07 capacity check, one detached session per setting, to score
prediction (a). 16384 runs load-only as planned. It is predicted not to
hold.

Scoring (a), per setting S:

- (a) holds at S when the load succeeds, and P1 (S - 64 tokens) is
  accepted with pipeline_input_tokens == S - 64, and P2 (S + 64 tokens) is
  refused by the runtime length check (the message contains
  "input_ids.get_size() <= m_max_prompt_len").
- P1's validity is recorded and is not used for (a). The runner's
  capacity_check.passed still requires P1 to be valid, and a failure there
  gives session status REFUSED_CAPACITY. So (a) is scored from the recorded
  P1 and P2 fields, whatever the session status (loaded_bisect_deferred or
  REFUSED_CAPACITY).
- At 16384, "does not load at the requested value" holds when the load
  fails (LOAD_FAIL), or when the per-setting criterion above fails.
- At 1024, the same criterion is applied to 34c2240f's capacity-check
  record once that run is sealed.

The load-only sessions time nothing that is cited. The runner's load-only
path runs one opening canary (arm(4)), which cannot arm, so these sessions
are UNGUARDED by construction. (a) is a load and length-check outcome, not a
timing, so it does not depend on the drift guard. load_s is recorded and
is not cited from these sessions.

The full bisection at 2048, 4096 and 8192 stays PLANNED. It is gated on
NPU-NATSTOP and the validity amendment.

## Unchanged

The SLO, low and resolution, repeats, the capacity check, the guard rules
(2026-10-08, 2026-10-08b), the validity rule as written, the binding
definitions, and the predictions' text.

## Existing runs

- 34c2240f: kept; it is sealed at closeout. Guarded. Its timings are
  recorded. Its limit is not cited, and (b) is not scored from it.
- 9d6d7f66, 8ccfb38c, 04d189da, 680b031a, 28cf811d, 041d9f13: none.
