# DEMO-2 pre-freeze rules (Task BF0 part A), registered 2026-10-22

Registered Thu Oct 22 2026 (ET) before the freeze, before any held-out run, and before the author
(Claude Code, Task BF0) opened any calibration ledger, log or cells output. Written from the
pre-freeze audit (~/seam_closeout_1013/research_20261021/DEMO2_pre_freeze_audit.md, IDs X/H/M
below) and the registered docs only. Approvals: docs/amendments/APPROVALS_2026-10-22.md (Zach,
about 1:15 PM ET, "yes on all").

Adds to, and edits none of:
- DEMO2_ISO_QUALITY_2026-10-21.md ("ISO");
- DEMO2_ARMS_REVISED_2026-10-21.md;
- DEMO2_PER_DEVICE_2026-10-21.md;
- DEMO2_AMEND_2026-10-16_TRUE_DEFAULT.md;
- APPROVALS_2026-10-21.md.

Where this note and an earlier note differ, this note governs from now on. Runners do not read it.

## 1. Cloud price formula (X2)
- Per cloud request, from its `cloud_requests` usage record:
  `cost = (input*p_in + cache_write*1.25*p_in + cache_read*0.10*p_in + output*p_out) / 1e6`.
  - input = `input_tokens`;
  - cache_write = `cache_creation_input_tokens`;
  - cache_read = `cache_read_input_tokens`;
  - output = `output_tokens`.
- Rates: seam/budget.py `PricingTable.tier_for` (seam/budget.py:143-163) on
  configs/pricing/anthropic.yaml, for the record's `model` at the call's UTC date. The formula is
  `usd_cost` (seam/budget.py:196-208), with the 5-minute write rate, since `ephemeral` sets no TTL.
  - For claude-sonnet-5 from 2026-09-01: p_in $3.00, p_out $15.00, read $0.30 (0.10x), write
    $3.75 (1.25x).
- The old formula (`_cloud_usd`, tools/bfcl_feasibility_probe.py:3382-3385; the runner's
  `cloud_usd`) prices input + output only. It is used for no DEMO-2 number from now on.
- A request with `ok` true and any of the four fields missing (None) is REFUSED. It is not
  priced as zero. A request with `ok` false has no billed usage and costs 0.
- The same formula is used:
  - in CostGuard (the runner's charge per cloud turn), before the calibration cloud log;
  - in the replay, which reprices every turn from `cloud_requests` and never sums the logged
    `cloud_usd`.

## 2. Multiplicity (X3; approval 4)
- Confirmatory family:
  - The ONLY confirmatory test is dC_iso of P vs A, one per host.
  - Fixed sequence: T2S first at alpha 0.05. XPS is tested at alpha 0.05 only if T2S rejects.
  - On the XPS, "P" means the arm the section 6 rule selects (P or P').
- Claim rule: "P is cheaper than A at A's quality" on a host only if that host's test is reached
  in the sequence, its two-sided bootstrap p < 0.05, and its 95% CI excludes 0.
- Everything else is SECONDARY:
  - every other arm and contrast, dQ_iso and McNemar, $ per passed task, area, sensitivities,
    failure/latency/context tables;
  - it is reported with unadjusted 95% CIs, labelled "secondary";
  - a secondary is interpreted only if the primary on that host rejects. Otherwise it is
    descriptive.
- This replaces ISO section 4's Holm family, the per-host `holm()` in the replay, and TASK_BG's
  "Holm across hosts x arms x endpoints".

## 3. Infinite bootstrap draws (X4)
- A draw where X does not reach Q* has C_X(Q*) = +inf, so dC_iso = -inf.
  - It is KEPT and counted as <= 0 in the two-sided p and in the percentile CI. A CI end may be
    -inf.
  - The count of such draws is reported.
- If the point estimate itself is -inf, it is reported as "X does not reach Q*", and the test
  does not reject.

## 4. Secondary contrasts and predictions (X6 approval 2; H5)
- Secondary contrasts, run with `--reference`, per host:
  - P vs R-full, P vs I-4K, C vs R-full, P vs C;
  - plus every arm vs A;
  - plus C vs I-4K (the consumer half of deliverable A).
- Each contrast reports dC_iso (and $ per passed task at the base points).
- DERIVED predictions for every contrast:
  - file derived/h1_hybrid/demo2/DEMO2_BF0_PREDICTIONS.json, sha256
    c250131daff849a8a9a560531fb24194c20b56405d4c9c4c3e6bfdc4a143820f, generator
    tools/demo2_bf0_predict.py;
  - the model is the BE iso model, with per-arm local-pass multipliers stated in the generator;
  - scored in % of the reference's C_R(Q*): HIT if the measured point dC_iso_rel is inside the
    band.
- Predicted dC_iso_rel, point [band]:

| contrast | XPS | T2S |
|---|---|---|
| P vs A | 6.2 [0.0, 16.6] | 19.6 [0.0, 52.3] |
| P' vs A | 1.5 [0.0, 3.0] | 4.8 [0.0, 9.6] |
| C, B, R-full vs A | 0.0 [-3.6, 3.8] | 0.0 [-11.5, 12.1] |
| I-4K vs A | -7.1 [-13.6, 0.0] | -22.4 [-42.8, 0.0] |
| P vs R-full, P vs C | 6.2 [0.0, 16.6] | 19.6 [0.0, 52.3] |
| P vs I-4K | 2.3 [1.3, 4.2] | 2.3 [1.3, 4.2] |
| C vs R-full | 0.0 [-3.6, 3.8] | 0.0 [-11.5, 12.1] |
| C vs I-4K | 1.3 [0.6, 1.9] | 1.3 [0.6, 1.9] |

- Plainly: the model prices only the local pass rate. So arms that run the same 4B are
  predicted equal at iso-quality. P' and I-4K are the stated exceptions.
- The relative figures do not depend on the cloud price, so caching does not change them. This
  answers H5's "uncached" point for the relative predictions.
- The BC $ predictions (DEMO2_BC_PREDICTIONS.json) remain registered for any live hybrid cell.
  They are live and uncached.

## 5. Escalation cost model (X7; approval 3)
- Primary: "splice (cache-optimistic)".
  - Escalation at turn t merges local turns [0, t) with the cloud log's turns [t, end).
  - Cost is the repriced cost of the cloud log's turns t..end.
  - It is optimistic: those turns carry cache reads of a cloud prefix that a real handoff would
    not have.
- Registered sensitivities, reported for every contrast:
  - **re-run:** an entry escalated at any turn costs the whole cloud entry (turns 0..end), and
    its pass is the cloud-only pass;
  - **cold handoff:** as splice, but in the first cloud request of turn t, cache_read tokens are
    priced at the write rate (1.25 p_in). The handed-off prefix is written once. Later requests
    are unchanged.
- Reported per arm: the share of escalations at turn t > 0 (the share where splice and re-run
  differ).
- Live hybrid validation (stretch) uses the same splice. It is reported as such.

## 6. P' rule (H1), redefined before XPS cal_p / cal_pp are opened
- Inputs (both runs complete or resumed to 30/30 under section 7):
  - A = XPS cal_a1 (window_20261022_1012_be; its deviation is in section 12);
  - P = XPS cal_p (Qwen3-8B INT4).
- P' (Qwen3-4B INT8, cal_pp) replaces P on the XPS only if ANY of these holds:
  - **(a)** P's local pass count < A's. Local pass count = the number of DEMO2_CALIB_30 entries
    with `local_pass` true in `policies/local_only/entry_quality.json`. A `skipped_hang` row
    counts as a fail.
  - **(b)** P decode p5 < 12 tok/s.
    - The sample is P's turn-ledger `decode_tok_s` over every local turn where it is not null.
      This is the value the router's 6 tok/s trigger reads.
    - p5 is nearest-rank: sort ascending, take element ceil(0.05 n), 1-based.
  - **(c)** P has more entries than A with a `ttft>10.0` trigger, counted on the same 30
    entries. An entry counts once if any local turn's `passive_triggers.mechanical` contains
    the substring `ttft>10.0`.
- Otherwise P stays.
- Applied once, at the freeze. All three P-vs-A values are recorded.
- It replaces ISO section 3's P' rule, which counted (c) only and did not name the A run.
- If cal_p or cal_a1 is incomplete, the rule cannot be computed, and P' is NOT selected by
  default. The freeze records NOT YET MEASURED, and Zach decides. That decision is a recorded
  deviation.

## 7. Hangs in replay (H3)
- A hang, stall or turn-budget fire at turn t_h is a local error at t_h.
  - t_h comes from the `skipped_hang` row's `hang_ref.turn` in the continuation run's
    entry_quality.json.
- The replay escalates at t_h, using the local turns [0, t_h) of that entry when an arm log holds
  them.
- A fired worker writes no partial ledger row. So when t_h > 0 and those turns are absent, the
  entry is replayed as a turn-0 escalation.
  - This bound charges the hanging arm the full cloud entry; it is conservative against the arm
    that hangs.
  - The entry is counted in `hang_prefix_missing`.
- A hang listed in an arm's hang_events.json without a `skipped_hang` row in any of that arm's
  continuation runs, and without a ledger row, is REFUSED.
- A `skipped_hang` row without `hang_ref.turn` is also REFUSED.

## 8. Material win, pivot and P7 under the new primary (H4)
- Noise terms, per host, from calibration:
  - d_usd: |dC_iso(A2 vs A1)| / C_A1(Q*_A1), by the replay of cal_a2 against cal_a1 with
    cal_cloud;
  - d_lat: |p50_1 - p50_2| / mean(p50_1, p50_2) of local-turn `t_generate`, cal_a1 vs cal_a2.
- They replace P7's paid A/A, which has not run and is not needed for these rules.
- X materially beats A on a host, on point estimates (secondary, descriptive), if:
  - dQ_iso at A's cost is > -0.15, AND
  - at least one of:
    - dC_iso >= max(25%, 2 x d_usd) of C_A(Q*);
    - p50 local-turn latency is lower by >= max(20%, 2 x d_lat);
    - context held is higher by >= 20%.
- Pivot: if model-fixed C does not materially beat A on a host, that host records "vendor
  defaults capture most of the single-device gain; remaining levers: <measured>", as registered.
  - The SEARCH phase stays registered.
  - Its start is a decision for Zach after Oct 26, not automatic in this sprint.
- P is reported separately and never satisfies the pivot for C.

## 9. Held-out cloud cap (H11)
- cap = max($20, 2 x cal_cloud_cost x 1.1), with cal_cloud_cost the repriced total of the
  calibration cloud log (30 entries; held-out has 60).
- If the cap is over $20, Zach approves it before ho_cloud launches.
- Truncation: if the cap stops ho_cloud early, entries without a cloud record are excluded from
  BOTH sides of every contrast, and the count is reported.

## 10. Memory and context split (H12)
- A local turn is flagged LOW_MEM when its local_turn_telemetry.jsonl `memory.min_available_mb`
  < 1,000.
- Flags are counted per arm and host. A flagged turn is not removed.
- Every endpoint is reported for the whole pin and separately for its base half
  (multi_turn_base) and its long-context half (multi_turn_long_context).
- Calibration reaches about 7k prompt tokens; held-out reaches about 68k (audit H12).

## 11. Frontier estimator, matched point and A/A (H13, H14)
- The frontier is the analytic expectation over random orderings, not one ordering.
  - Down (thinning the n_T triggered entries) and up (escalating the n_U untriggered entries at
    turn 0) are each the straight segment from the base point to the all-moved point.
  - C_X(Q) and Q_X(c) are read off these two segments; fractional moves are randomized
    policies.
  - The bootstrap recomputes the segments per draw.
  - This replaces ISO section 4's "random order, seed 20261023" frontier.
- Matched point pi for McNemar (secondary): the point on X's expected frontier at cost C* (A's
  base cost).
  - The realized entry set moves k = floor(lambda* n_seg) entries, in ascending
    sha256("20261023:" + entry_id) order (fixed now, outcome-blind).
  - dQ_iso is the expected-frontier value at C*.
- A/A (H14): every CI is conditional on the one shared cloud log, so cloud-side run-to-run
  variance is not in it; this is stated wherever a CI appears. A1 vs A2 (cal_a1, cal_a2) is run
  through the same dC pipeline. A second cloud log is not approved.

## 12. Deviation and disclosure table (H6, X5)
| when (ET) | what | effect |
|---|---|---|
| Oct 21 21:00 | BC (17a102e) registered revised arms; DERIVED: revised C not a material $ win vs A on either host (T2S +73% cost) | - |
| Oct 21 21:40-21:45 | research report "Dollar win against the default" written; Zach approves BE | - |
| Oct 21 22:11 | BE (145047f) moved the primary to iso-quality replay and added arm P | made after BC's predicted C loss was known; the 2026-10-16 pivot rule would have sent C-vs-A to the SEARCH phase, and BE did not cite that reversal |
| Oct 22 ~11:07-11:21 | audit notes parsed calibration ledgers: T2S cal_a1/a2/c/p, XPS cal_a1 (e.g. max TTFT 0.91 s, T2S A/A 0/30) | the H1 redefinition (section 6) was written knowing that criterion (c) reads 0 vs 0 on A; no "nothing was read" claim is made |
| Oct 22 11:10:39 | XPS cal_a1 stopped by the banned-app check (msedge) | deviation: cal_a1 used as completed in its window; T2S cal_a1 is byte-identical (audit) |
| Oct 22 (X5) | C runs as specified (K 1,024 h70t30 from L1, age-mask on, budget per device); the L6 gate (ARMS_REVISED sections 6-7) was never evaluated (ladder ran UNTIL=4); no L6 cells run | approval 1 |
| Oct 22 | this note, part A, committed (see below) | before the author opened any calibration ledger |

- Part A commit time: the commit that adds this file (git log is authoritative). The author's
  first calibration-ledger read is after it.

## 13. Secondaries and section 5 text (M3, M4, M6)
- Registered secondaries, per arm and host, base/long split:
  - failure behaviour: silent vs loud local failures, generates past 32,768 and 40,960, loops
    (max_steps), stalls / hangs / turn-budget fires, generation errors, LOW_MEM turns;
  - local-turn latency: TTFT and `t_generate` p50 / p90;
  - context: the largest n_ctx a local generate completed at, and raw tool-output tokens
    absorbed before the first escalation.
- ISO section 5's refusal list is replaced by what the replay actually refuses:
  - a missing turn ledger;
  - local turns without passive_triggers (`--derive-triggers` is refused on DEMO2_HELDOUT_60);
  - a `local_pass` of None;
  - duplicate entry ids across an arm's dirs or in the cloud log;
  - cloud entries with fewer turns than the entry's user turns;
  - cloud requests missing cache fields;
  - pin ids missing in an arm or the cloud log;
  - the section 7 hang cases.
- M4: the label "device-provisioned (model tier/precision), not model-fixed" stays. No T2S 8B
  INT4 cell is run.
- M6: one governing freeze time, Fri Oct 23 2026 12:00 PM ET, as a hard ceiling (earlier is
  allowed). It supersedes "Sat noon" in DEMO2_PER_DEVICE section 6.

## 14. Sunday rule on the T2S (H10)
- The T2S queue launches nothing, including a resume continuation, that cannot finish by
  8:44 PM ET Sun Oct 25. `fits()` is also checked in every resume.
- If OUR worker is still running at 9:15 PM Sun, the queue:
  - stops it (logged PAST_FINISH). Only our python processes are stopped: python.exe whose
    executable path is our repo's `.venv-seam\Scripts\python.exe`;
  - parks the tree;
  - runs the hand-back check by 9:44 PM.
- Nothing that is not ours is killed.
