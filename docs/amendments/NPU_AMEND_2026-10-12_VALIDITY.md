# NPU amendment 2026-10-12: output validity after NPU-NATSTOP N3 (pre-run)

Registered and committed before any further NPU generation. It adds to the
NPU-1 amendments 2026-10-07, 2026-10-08, 2026-10-08b and 2026-10-08c and
does not edit them, derived/npu/NPU1_PREREG.json or
derived/npu/NPU_NATSTOP_RULE.md. Where they differ, this file governs NPU
generation from here on. tools/npu_natstop.py does not open this file; the
constants and prompt texts below are written in both places, and a test
checks they are equal.

## Basis: natstop_6b39a76e (verdict N3)

Run natstop_6b39a76e-0df4-4a90-84c1-4c1dd15bff35: EVO-T2S, 2026-10-12
18:24-18:38 UTC, git 177350d. Sealed tree
derived/npu/sealed_natstop_6b39a76e-0df4-4a90-84c1-4c1dd15bff35,
tree_sha256 347f64b7f666f5b92573721034d702cd3cfa5bca7b3b1ec794b988e1942a1b59,
MATCH (RECONSTRUCTED from host listings NPU_SHA256_20261012b.txt and
NPU_SOURCE_MANIFEST_20261012b.txt, 50 files). The verdict is N3, recorded
by the script and re-derived by `npu_natstop.py --classify` under the rule
committed in 61efb6c. All 16 children exited 0 and every cell is ok.

- GPU, all 8 cells: writes text. Stop max, 64 ids. SINGLE starts "It
  seems like you've ..."; DOUBLE starts "<think>\nOkay, ...". Id 4-gram
  repeat is 0.0 in 7 cells and 27/61 (0.4426) in GPU DOUBLE 64, which
  quotes the repeated filler back.
- NPU SINGLE at 64, 768, 960 and 1984: the first token is EOS (id 151645).
  1 id, empty text.
- NPU DOUBLE:
  - 64: "<think>" then "</think>" repeats, 23 ids, id 4-gram repeat 0.75,
    no real text.
  - 768 and 960: "<think>" and blank lines, then EOS (3 ids).
  - 1984: EOS as the first token (1 id, kv room 64).

## (a) Validity rule

A generation is valid only if both of these hold:
1. It has at least K real tokens. When the paired GPU cell (same prompt,
   same template) stopped by EOS before K, the floor is that cell's real
   token count instead. It is never below 1.
2. Its id 4-gram repeat fraction is at most T.

A real token is a generated id that is not in the pipeline's eos/stop set
and whose decoded text, with "<think>" and "</think>" removed, is not
whitespace-only. The child records the count as real_token_count. The id
4-gram repeat fraction is the share of token-id 4-grams that repeat an
earlier one (seam.npu_validity.fourgram_repeat_fraction over ids).

The constants come from the 8 GPU cells of natstop_6b39a76e, the only
healthy reference generated on these prompts at max_new_tokens 64. They were
fixed from those cells alone, before any NPU generation is judged by them.
- K = 62: the fewest real tokens in any GPU cell. In the DOUBLE cells, 64
  ids less "<think>" and the newline after it. All SINGLE cells have 64.
- T = 27/61 (0.4426): the highest GPU id 4-gram repeat (GPU DOUBLE 64).
  T is the largest value a working device produced, not a tuned cut-off.

The rule replaces the whitespace-word repeat_4gram check of amendment
2026-10-08 for any NPU validity judgement. Under it, every NPU cell of
natstop_6b39a76e is invalid.

## (b) Consequence of N3 for NPU-1

- NPU-1 limits are recorded only as the toolchain cap and as load and
  prefill measurements. No NPU decode figure, quality figure or
  validity-bound limit is cited. The 768 of 34c2240f stays uncited
  (2026-10-08c).
- Full bisections at 2048, 4096 and 8192: CHANGED, not dropped. Reason:
  N3. A rung passes only with a valid generation, and the NPU produces none
  on this toolchain, so a bisection would only measure the validity rule.
  They revive if a later toolchain gives a valid NPU generation on the
  NATSTOP filler prompts.
- Prediction (b) ("1024 is TOOLCHAIN_CAP, 8192 is LATENCY below 8192"):
  unscorable as registered. Its LATENCY outcome needs a valid generation at
  the limit rung, which N3 rules out. TOOLCHAIN_CAP at 1024 stays the
  literal rule output (2026-10-08c), not a score.
- Prediction (c) ("NPU decode tok/s below 34.3"): unscorable. Decode of a
  degenerate or EOS stream is not task decode. 34c2240f's ~17.2 tok/s (8
  forced tokens) stays recorded, not cited.
- Prediction (a) is unchanged (scored MISS 2026-10-09; 32768 holds).

## (c) Scope check NPU-NATSTOP-B (pre-registered here)

Question: is N3 specific to the long filler prompts, or is there no usable
NPU output on this toolchain at all?

Design:
- Same IR, load properties and children as NATSTOP:
  - NPU: MAX_PROMPT_LEN 2048 + MIN_RESPONSE_LEN 8, NPUW_LLM_PREFILL_CHUNK_SIZE 1024.
  - Paired GPU pipeline.
  - SINGLE and DOUBLE templates.
  - Greedy, max_new_tokens 64, min_new_tokens not set.
  - One child per cell, run in sequence, child timeout 600 s.
- 2 prompts x 2 devices x 2 templates = 8 cells. The prompt is the user
  turn, rendered with the chat template, thinking disabled
  (seam.tools.boot4_text.render_user). The fixed texts:
  - chat1: "What is the capital of France? Answer in one sentence."
  - chat2: "Give three tips for staying focused while studying."
- Order: for each prompt, NPU SINGLE, NPU DOUBLE, GPU SINGLE, GPU DOUBLE.

Rule (code: tools/npu_natstop.py classify_scope; first match wins; valid
is rule (a); an NPU cell's reference is the GPU cell with the same prompt
and template):
- B4: some GPU SINGLE cell is not valid. The reference is unusable; report
  to Zach, no claim.
- B1: no NPU cell is valid, on either prompt or template. The paper says
  "no usable NPU output on this toolchain".
- B2: NPU SINGLE is valid on both prompts, or NPU DOUBLE is valid on both.
  The N3 claim is limited to long filler prompts.
- B3: anything else (mixed by prompt). Report to Zach. N3 stands as
  measured, with no wider claim.

Output: derived/npu/natstop_b_<run_id>/ (plan.json, cell_*.json,
summary.json). Last stdout line: NATSTOP_B_VERDICT <B1|B2|B3|B4> <dir>.
Launcher: tools/launch_t2s_natstop.ps1 -PromptSet chat; log
derived/c2_ttft/_launches/npu-natstop-b_<UTC>.log. Diagnostic, not a
measurement. Nothing timed is cited. Not run as of this amendment.

## (d) Ladder

65536 load-only stays on the ladder, PLANNED for the next T2S window.
Load and the cap are not affected by N3: prediction (a) is scored from
the load and the P1/P2 capacity fields, not from output validity.
