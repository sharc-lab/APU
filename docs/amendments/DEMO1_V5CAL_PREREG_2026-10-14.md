# DEMO-1 V5-CAL preregistration (design; committed before any V5-CAL run)

Adds to docs/amendments/DEMO1_AMEND_2026-10-13c_CONFIG.md (ed8ffb2) and edits
none of its predictions. Runners must not read this file. Status: DESIGN
ONLY. Code, estimate table, tests and launch commands are not built yet;
no V5-CAL run before they are committed.

## Why
V5a/V5b are indeterminate (all 15 entries): each jump step adds D 8,109-8,208
tokens on cached P 2,159-7,356; sealed warm data stops at D 400 (u8) / 183 (u4).

## Points (XPS GPU 4B int4, RESIDENT, turn-2 TTFT; local-only, $0)
- Same prompt construction and prefix-cache path as sealed WARM-KV
  (seam/tools/_delta_n_child.py _resident_two_turn: turn-1 fill of n_cached
  rendered tokens, then 3 turn-2 repeats with distinct raw delta texts),
  same repeats (3) and canary (TtftSloCanaryGuard, discard warm-up, C = 3).
- delta_tokens 8,500, so D (turn-2 prompt minus common prefix) is about 8,517,
  at or above the block-high jump D (8,208 + 256 = 8,464). No jump step
  needs extrapolation in D.
- n_cached 2,200 / 3,320 / 4,080 / 6,290 / 7,380: each at or above the largest
  jump P of one cluster (2,159-2,180; 3,058-3,298; 4,023-4,057; 6,229-6,267;
  7,189-7,356) plus the turn-1 generation-prompt break. No jump step needs
  extrapolation in P.
- At u8 and at u4: 10 points.
- One V5-path point (u8, n_cached 4,080, delta 8,500): turn-2 history is
  user, assistant (the turn-1 reply), user, so the common prefix is the
  turn-1 prompt minus the 4-token empty think block, exactly the H1 path.
- Each point records turn-1 and turn-2 prompt tokens and the measured common
  prefix (rendered token ids), so P and D are measured, not assumed.

## Correction to amendment 2026-10-13c
The sealed WARM-KV turn 2 appends a second user message, so it too diverges
from turn 1 at the generation prompt (a few tokens, expected 6; to be
measured with the pinned tokenizer). "Exact prefix" in the amendment should
read "prefix minus the turn-1 generation prompt".

## Budget
One boot, two cells (u8 with 6 points, u4 with 5), estimate method as boot 4
(sealed cold medians: turn 1 bounded by cold(n_cached); each turn 2 bounded
by cold(n_cached + 8,500); plus warm-up bound and 752.1359013 s canary
overhead per cell). Must be under 6,000 s; table to be committed with the code.

## Rule for using the points
A V5 addendum recomputes V5a/V5b with these points by the amendment's method
(monotone bound, block bounds 0 / +256) and is committed before any V5 run.
V5 is not worth running if, with these points, all 15 entries are predicted
to escalate at both precisions (or none flips relative to the V4 step model);
then the addendum records the predicted null and V5 is dropped or deferred.
