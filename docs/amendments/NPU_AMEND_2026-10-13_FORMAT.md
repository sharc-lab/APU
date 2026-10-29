# NPU amendment 2026-10-13: export-format test F1 (pre-run)

Registered and committed before any code that runs F1 and before any F1
cell. It adds to the NPU amendments 2026-10-07, 2026-10-08, 2026-10-08b,
2026-10-08c and 2026-10-12 and edits none of them. tools/npu_natstop.py does
not open this file; the content-check constants and the outcome rule below
are written in both places, and a test checks they agree.

## Basis

- NPU-NATSTOP natstop_6b39a76e: verdict N3 (sealed tree 347f64b7..., MATCH).
- NPU-NATSTOP-B natstop_b_78fc3478-67bf-44b5-bb50-7d5f476dc43b: verdict B3
  (sealed tree ab5aee39013227309cb26ca0547cc7de74214547dd47703b0477863da8d3010c,
  MATCH). The one form-valid NPU cell (SINGLE chat1) is a counting sequence
  unrelated to the question. Across both runs no NPU cell gave an on-topic
  answer; every GPU cell wrote coherent on-topic text.
- Both runs used configs/models/Qwen3-4B-int4-ov.yaml: OpenVINO/Qwen3-4B-int4-ov
  rev b467368d16b75df14055562fe927ae8e1f15f7ef, INT4_SYM, group size 128,
  ratio 1.0 (publisher README), on openvino-genai 2026.2.1.
- Every NPU claim so far is scoped to "Qwen3-4B INT4_SYM group size 128 IR
  on openvino-genai 2026.2.1".

## Hypothesis

The NPU failure is specific to the INT4 group-size-128 export. NPU-ready
published exports use INT4 symmetric channel-wise weights (group size -1,
ratio 1.0).

## Test F1

IR: the existing 4B int8 IR, the NPU-2 model. No download, no conversion.
- Spec configs/models/Qwen3-4B-int8-ov.yaml: OpenVINO/Qwen3-4B-int8-ov rev
  fd9c3cc8adf942151e48904b7341a57eae58d8d5, ir_sha256
  f7d323d0eb94a6bc9a1bf807416877cbb988cfbd95e7f9ef12403951b90886ae,
  openvino_model.xml sha256
  3f6af24761fd17d8282160d486bb3614b8c7277867d46b7f596e2a1fa3e9a553.
- Quantization per the publisher README (spec, source: readme): INT8_ASYM,
  ratio null, group size null (int8 weight compression is per channel),
  no AWQ, no scale estimation, no dataset.
- The run reads the quantization metadata from the IR itself (rt_info,
  openvino_config.json, and a census of the compressed weight constants:
  element types, group sizes, zero points) and records it in plan.json.
  Where the IR disagrees with the README, the IR read is the record.
- A real run refuses unless openvino_model.xml exists and its sha256 equals
  the spec's ir_files pin.

Cells: the NATSTOP-B chat prompt set, unchanged.
- chat1: "What is the capital of France? Answer in one sentence."
- chat2: "Give three tips for staying focused while studying."
- NPU SINGLE, NPU DOUBLE, GPU SINGLE, GPU DOUBLE on the same IR, per prompt
  in that order; 8 cells.
- NPU load properties as NATSTOP-B: MAX_PROMPT_LEN 2048, MIN_RESPONSE_LEN 8,
  NPUW_LLM_PREFILL_CHUNK_SIZE 1024. Greedy, max_new_tokens 64,
  min_new_tokens not set.
- Each cell runs in its own child process, one after another, child timeout
  600 s. Two NPU pipelines are never loaded at once.

### Pass rule per cell (fixed now)

A cell passes when it is valid AND its content check holds.
- Valid: the 2026-10-12 validity rule, unchanged (at least K = 62 real
  tokens, floor the paired GPU cell's real-token count when that cell
  stopped by EOS, never below 1; id 4-gram repeat at most T = 27/61). The
  reference is the GPU cell with the same prompt and template.
- Content check, on the cell's real text (full text with every "<think>"
  and "</think>" removed, stripped; the text the validity rule uses):
  - chat1: the text contains "paris", case-insensitive.
  - chat2: at least 2 distinct tip lines. A tip line is a line that, after
    stripping, matches
    `^(?:#{1,6}\s*)?(?:\*\*)?(?:\d{1,2}[.)]|[-*+]|\u2022)(?:\*\*)?\s+(.+)$`
    (an optional markdown heading or bold marker, then a numeral of 1-2
    digits with "." or ")", or a bullet "-", "*", "+" or U+2022, then
    whitespace) and whose remainder has at least 5 words. A word is a
    whitespace-separated piece with at least one ASCII letter; each such
    word is at least one real token, so 5 words imply at least 5 real
    tokens. Two tip lines are distinct when their remainders differ after
    lowercasing and collapsing whitespace.
- Words, not token ids, so the check is re-derivable from the stored text
  without a tokenizer.

On natstop_b_78fc3478 (int4 g128) this rule gives: GPU SINGLE chat1 and
chat2 pass; GPU DOUBLE chat1 passes; GPU DOUBLE chat2 fails content (a think
trace cut at 64 tokens lists no tips yet); every NPU cell fails (NPU SINGLE
chat1 is valid but has no "paris"). At 64 tokens a cell that thinks first
cannot list tips, so F1's FA in practice rests on a template where the NPU
answers without thinking.

### Outcome rule (first match wins)

- FA: there is a template under which the NPU cell passes on both chat1 and
  chat2. The format hypothesis is supported; NPU-1-CW is prioritized.
- FB: all 4 NPU cells are ok (child exit 0, complete, loaded, generate
  returned) and none passes, and GPU SINGLE passes on both prompts. The
  failure is not specific to int4 g128; NPU-1-CW is still needed before any
  general claim.
- FC: anything else. Report to Zach. This includes an NPU cell that does
  not run past load, crashes or times out (the NPU-2 failure mode, which
  says nothing about output format), a GPU SINGLE reference that fails, and
  mixed results.

GPU SINGLE is the reference for FB, as in the NATSTOP-B B4 check. GPU DOUBLE
cells are recorded and are each NPU DOUBLE cell's validity reference.

### What F1 cannot separate

The int8 IR differs from the int4 IR in bit width (8 vs 4), grouping (per
channel vs group 128), symmetry (ASYM vs SYM) and calibration (none vs AWQ
plus scale estimation on wikitext2). FA says the failure belongs to the int4
g128 export as a whole, not to its group size alone; that attribution needs
NPU-1-CW (INT4_SYM, group size -1, ratio 1.0), which changes grouping and
keeps 4-bit symmetric weights.

## NPU-2 isolated-child plan: CHANGED (superseded by F1)

NPU-2's next attempt was an isolated-child load plus a 64-token generate of
the int8 IR on the NPU (openvino #35641), blocked on a protocol amendment.
Each F1 NPU cell is that: one child process, int8 IR, NPU load, one
64-token greedy generate, plus the paired GPU cell. So the NPU-2 plan is
CHANGED, not dropped: superseded by F1, reason above. F1 is a diagnostic;
it does not score NPU-2 prediction (d), which stays UNSCORED. Whether int8
runs past load in an isolated child is recorded per NPU cell (load_error,
child exit code, generate_returned).

## Code and output

- tools/npu_natstop.py --prompt-set f1 --model-spec <int8 spec>. The f1 set
  refuses the int4 default spec. Run dir derived/npu/natstop_f1_<run_id>/.
  Last stdout line: NATSTOP_F1_VERDICT <FA|FB|FC> <dir>. The NATSTOP-B
  scope verdict (B1-B4) is also recorded in the summary, not used.
- tools/launch_t2s_natstop.ps1 -PromptSet f1 -ModelSpec
  configs\models\Qwen3-4B-int8-ov.yaml; log
  derived/c2_ttft/_launches/npu-natstop-f1_<UTC>.log, ending
  NATSTOP_COMPLETE.
- Diagnostic, not a measurement. Nothing timed is cited. About 12 min
  (4 NPU children at about 130 s for the int8 load, 4 GPU children at
  about 10 s). Needs Zach's OK. Not run as of this amendment.
