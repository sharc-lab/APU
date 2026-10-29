# NPU amendment 2026-10-13b: NPU-INT8 envelope (pre-run)

Registered and committed before any code that runs it and before any
session it describes. It adds to the NPU amendments 2026-10-07, 2026-10-08,
2026-10-08b, 2026-10-08c, 2026-10-12 and 2026-10-13 and edits none of them,
derived/npu/NPU1_PREREG.json or derived/npu/NPU_NATSTOP_RULE.md. It governs
NPU-1 ladder sessions on the int8 IR only. The int4 ladder, its PARKED
65536 rung and its CHANGED bisections stay as recorded. tools/run_npu_profile.py
does not open this file; the constants below are written in both places and
a test checks they agree.

## Basis

- NPU-FORMAT-F1 natstop_f1_b172b287-2ff9-4f1f-87dc-dd5e20a705a8: verdict FA
  (sealed tree 7b1d24d38d9d6e898b1ae0450a6b110b55c83ff81e3d7c06e4a87eea8ef3a2cb,
  MATCH; re-derived by npu_natstop.py --classify). On the int8 IR the NPU
  SINGLE cell passes validity and content on both chat prompts, and all four
  NPU cells are token-identical to their GPU pairs over the recorded ids.
- The IR the run read (constants census; rt_info and openvino_config state
  no quantization): INT8_ASYM, per channel (group -1), 253 u8 weight
  constants with u8 zero points, openvino_model.xml sha256
  3f6af24761fd17d8282160d486bb3614b8c7277867d46b7f596e2a1fa3e9a553 (matches
  the pin in configs/models/Qwen3-4B-int8-ov.yaml).
- Scope of F1: Qwen3-4B INT8_ASYM channel-wise IR on openvino-genai
  2026.2.1, 2 chat prompts, 64 tokens. Bits, symmetry, calibration and
  grouping all differ from the int4 g128 IR, so FA does not say which one
  matters (2026-10-13, "What F1 cannot separate").

## Question

On the int8 IR, where does the NPU stop: toolchain cap, latency, or output
validity, per MAX_PROMPT_LEN setting, and does the output at the limit rung
still answer a question?

## Design (per setting S, one detached session)

Model: configs/models/Qwen3-4B-int8-ov.yaml. A session refuses unless
openvino_model.xml exists, its sha256 equals the spec pin, and the mode read
from the IR is INT8_*.

Unchanged from the NPU-1 protocol and amendments: MAX_PROMPT_LEN as the
load-time axis (1024, 2048, 4096, 8192, then doubling); load properties
MAX_PROMPT_LEN S, MIN_RESPONSE_LEN 8, NPUW_LLM_PREFILL_CHUNK_SIZE 1024 (the
F1 properties at S = 2048); single template (prompt rendered with thinking
disabled, apply_chat_template False, list call form); filler prompts from
rendered_exact_prompt (salt "npu"); the 10 s TTFT SLO; low 64, resolution
64, 3 repeats per rung; the guard of 2026-10-08 and 2026-10-08b (GPU
pipeline plus one untimed warm-up, C = 3 calibration canaries before the
first probe, interval canaries, closing canary). The GPU canary and the
paired GPU pipeline use the same int8 spec; canary thresholds stay
max(2 x early drift, pooled healthy floor).

Changed for int8 sessions:
1. Capacity check: P1 (S - 64 accepted, pipeline count exact) and P2
   (S + 64 refused by the runtime length check) as 2026-10-07. P1 keeps its
   8-token generate; its validity is recorded and does not gate the session.
2. Bisection high is S - 64. MIN_RESPONSE_LEN 8 leaves S + 8 - n tokens of
   room, so a rung above S - 56 cannot produce the 64 tokens the validity
   rule needs.
3. Per rung, repeat 1 is the validity generate: NPU, then paired GPU, same
   prompt, greedy, max_new_tokens 64, min_new_tokens not set. Repeats 2 and 3
   are timing generates on the NPU only, greedy, max_new_tokens 1. All three
   give a TTFT sample.
4. A rung is valid when the repeat-1 NPU generation passes the 2026-10-12
   rule: generate returned, pipeline input count equals the realized count,
   at least K = 62 real tokens (floor: the paired GPU's real-token count when
   the GPU stopped by EOS before 64; never below 1), id 4-gram repeat at most
   T = 27/61. Real tokens as 2026-10-12 (not in the stop set, not whitespace
   after removing think tags).
5. A rung passes when it is valid, all 3 repeats report a TTFT, and the
   median TTFT is at most 10 s. The limit is the highest passing visited rung.
6. Binding (first match): LOAD_FAIL (no load); TOOLCHAIN_CAP (the high rung
   S - 64 passes); LATENCY (the lowest visited rung above the limit, or the
   low rung when nothing passes, is valid and misses the SLO); VALIDITY (that
   rung is not valid).
7. Content check at the limit rung L, after the closing canary, untimed. The
   prompt is the filler prompt of exactly L tokens whose user content ends
   with a blank line and the F1 chat1 question "What is the capital of
   France? Answer in one sentence." NPU, then GPU, greedy, max_new_tokens 64,
   min_new_tokens not set. Outcome:
   - CONTENT_OK: the GPU passes (valid with itself as reference, real text
     contains "paris", case-insensitive) and the NPU passes (valid against
     the GPU, real text contains "paris").
   - CONTENT_FAIL: the GPU passes and the NPU does not.
   - CONTENT_REFERENCE_UNUSABLE: the GPU does not pass. No content claim.
   - NO_LIMIT: no rung passed; no content probe.
8. A limit is cited (with its run_id) only from a guarded (armed), complete
   session whose content check is CONTENT_OK. Anything else is recorded, not
   cited.

Load-only sessions (16384, 32768): load, capacity check, one opening canary
(UNGUARDED by construction, as 2026-10-08c), no bisection, no content check.

## Predictions

Sources: sealed or recorded runs only. Figures marked R are recorded, not
cited, and used only to form the estimate.
- int8 NPU load at 2048: 139.42 s, mean of the four F1 NPU cells (R,
  natstop_f1_b172b287). int4 loads per setting (R): 81.87 (1024, 34c2240f),
  95.07 (2048, mean of f704d565 and ca3467ad), 117.09 (4096, 47630f52), 195.05
  (8192, 413d4a2e), 508.77 (16384, d626d0bc), 1963.26 (32768, 3a672cd5).
  int8 load estimate = int4 load x 139.42 / 95.07 = x 1.4665.
- int8 NPU TTFT at 21-32 prompt tokens: 4.150 s, mean of the four F1 NPU
  cells (R). int4 NPU prefill at 64: 1.2873 s, mean of the 34c2240f rung (R).
  Ratio 3.2238. int4 prefill points (R): 1.3241 at 960 (34c2240f P1), 2.77 at
  1984, 6.54 at 4032, 15.94 at 8128, 58.61 at 16320, 245.76 at 32704 (the
  load-only P1 prefills). int8 TTFT estimate at n = 3.2238 x the int4 value,
  linear between points.
- int8 NPU decode: 2.9036 s per token = 0.344 tok/s, from F1 child walls
  ((329.032 - 139.797) - (166.406 - 139.772)) / (64 - 8) (R; a wall
  difference, not a streamed decode measurement).

| S | high | int8 TTFT at high (est.) | binding (predicted) | limit (predicted) |
|---|---|---|---|---|
| 1024 | 960 | 4.27 s | TOOLCHAIN_CAP | 960 |
| 2048 | 1984 | 8.93 s | TOOLCHAIN_CAP | 1984 |
| 4096 | 4032 | 21.08 s | LATENCY | in [1984, 3072), point 2112 |
| 8192 | 8128 | 51.39 s | LATENCY | in [1984, 3072), point 2112 |

- P-INT8-a (capacity): P1 accepted with an exact count and P2 refused by the
  length check at every setting run, 1024 through 32768. Basis: the int4
  ladder held at every setting through 32768 (2026-10-09, 2026-10-12); the
  cap is the runtime length check, not the weights. Falsified if any run
  setting fails P1 or P2.
- P-INT8-b (binding): the table. Falsified per setting if the binding
  differs, or the limit at 4096 or 8192 falls outside [1984, 3072). 2048 is
  the low-margin call (8.93 s against 10 s).
- P-INT8-v (validity): every visited rung up to the high rung is valid.
  Basis: F1 token identity with the GPU. Falsified if any visited rung is
  not valid. This is the first test of int8 validity on filler prompts.
- P-INT8-c (content): CONTENT_OK at every setting with a limit. Falsified by
  CONTENT_FAIL. CONTENT_REFERENCE_UNUSABLE scores nothing.
- P-INT8-d (decode, directional): NPU decode tok/s from the repeat-1
  streamer at the limit rung is below 1.0 (point 0.34). Falsified if at
  least 1.0.

## Wall-time estimates (upper bounds; formula in run_npu_profile.py)

Per session: int8 load + P1 (TTFT + 7 decode tokens) + 10 s GPU setup + rung
points x (validity repeat: TTFT + 63 x 2.9036 s + 4 s GPU; 2 timing repeats:
TTFT + 0.5 s), every rung priced at the high rung's TTFT + canaries x 12 s
(3 calibration, one per N = planned/4 probes, closing; 12 s per canary from
the 34c2240f canary spacing, R) + content probe (as a validity repeat) + 30 s.
Rung points = 2 + ceil(log2((high - 64) / 64)).

| S | session | estimate |
|---|---|---|
| 1024 | envelope | 1689 s |
| 2048 | envelope | 2016 s |
| 4096 | envelope | 2567 s |
| 8192 | envelope | 3823 s |
| 16384 | load-only (the envelope bound 9027 s exceeds 7200 s) | 1034 s |
| 32768 | load-only | 3770 s |
| 65536 | not scheduled: int8 load alone estimated near 11100 s (int4 32768 load x 3.86, the 16384 to 32768 growth, x 1.4665) | - |

## Schedule

- T2S, by Wed Oct 14 (end of window 1), in this order, one detached session
  each: 1024, 2048, 4096, 8192 envelope (10095 s estimated together), then
  16384 load-only (1034 s) if time remains. Each needs Zach's OK.
- Window 2 (Oct 19-25): whatever did not run, 32768 load-only, and NPU-1-CW.
- A rehearsal (-Rehearsal, stub worker) precedes the first real session.

## Code (after this file)

- tools/launch_t2s_npu1.ps1 and launch_boot1.ps1 profile npu-1 gain
  -ModelSpec (default configs\models\Qwen3-4B-int4-ov.yaml, unchanged) and
  -Envelope. -Envelope requires a non-default spec; a non-default spec
  without -Envelope is refused. The launcher uses the envelope estimate.
- tools/run_npu_profile.py --envelope: items 1-8 above, the IR read
  (seam.ir_quantization) in plan.json, weight_precision from the IR. Without
  --envelope the int4 path is unchanged.
