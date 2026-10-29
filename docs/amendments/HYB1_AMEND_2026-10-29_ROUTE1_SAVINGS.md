# HYB-1 amendment: ROUTE-1, ceiling-planned sessions: cost vs the default hybrid

Status: REGISTERED - NO DATA - [private]
Filed: 2026-10-29 (Task BT; approval docs/amendments/APPROVALS_2026-10-29.md). Filing time ET: 2026-10-29 17:15 EDT
Amends: docs/registrations/BR_HYB1_2026-10-26.md (sha256 c1454ebbdbe38e6d472a240f6531ce53518f7c82719ee0ced2021498cacb5642,
re-checked at filing: MATCH). BR_HYB1_2026-10-26.md and derived/br/BR_HYB1_PREDICTIONS.json keep their bytes.
Predictions: derived/route1/ROUTE1_PREDICTIONS.json sha256 9188d7e5b353306ee2598c2da58dad4ab0f03f934eb4013f01601ac50e17fc95
This file: sha256 cbf56609547c9f260fce196f5c1f306eceb1cdbe63a7f527462f5933113228aa (computed with the value on this line replaced by 64 "0" characters;
the true sha256 of the file as committed is in docs/PLAN_LOCK_2026-10-23.md, Addendum ROUTE-1)
COST-1 (inputs X16-W, X-Q, PRICE): docs/registrations/COST1_RAMEQ_2026-10-29.md, derived/cost1/COST1_PREDICTIONS.json.
[private]
Source of the design: "Ceiling informed routing cost result.md", Appendix A (research_20261021).

Labels: MEASURED (run_id), DERIVED (formula or script), ASSUMED (a modeling choice), CITED (external source).

## 1. Purpose and relation to HYB-1
Adds experiment ROUTE-1. Changes no HYB-1 hypothesis, metric, cell, threshold, title or thesis. Lifts HYB-1's
"Not claimed: savings" ONLY for the contrast ROUTE-1 vs min(D1, D2). All-cloud is a normalizer only.
Anything computed from data seen before this filing is exploratory.

## 2. Headline wording (X, Y filled only from frozen data under the DRs)
"On a 16 GB Intel AI PC running Intel's default runtime, the measured context-ceiling model turns
route-at-start into a configure-and-plan decision that cuts cloud $ per 1,000 agent sessions by X% versus the
default hybrid router (D1/D2) at equal task success, and closes Y% of the gap to the oracle, with zero failed
local attempts."
Always reported beside it: "-N failed local attempts and -M seconds of wasted wait per session, at no extra
cloud cost."

## 3. Inputs as of filing
- i8 default ceiling 19,750 (b038e362), 18,000 (fd074be1). MEASURED.
- True-default KV i8 (597f40a9). MEASURED.
- 10 s TTFT edge at 10,000 tokens on the i8 default (c2246b1f). MEASURED.
- TTFT 22.7 s u4 at 16K (d42040c1); 48.1 s u4-c256 at 16K (8997aa4e); 53 s per turn near 60K (d7e8b43e).
  MEASURED.
- u4-c256 capacity 70,000 (f8df3847, 0/4 repeats). MEASURED. i8-c256 38,000 (1e9671a0). MEASURED.
- Time-to-fail 0.442 s (00c7fa60, turn-1 OOM). MEASURED. Pipeline compile 3.77 s (96612911). MEASURED.
- BR constants: k_f16 147,456, k_i8 84,096, k_u4 ~47,232, a 87,520, B_BR 3,267,086,560 B, W 40,960.
- Session law L_total(T) = (B - a*min(T,c))/k; error band -2.5% / +5.9% (team-reported; run_ids to be
  attached from analysis/cc_20261026/SCORES.md ADDENDUM 5 before scoring). DERIVED.
- DEMO-2 arm A TTFT p50 0.30 s at n_ctx p50 3,936 (d482c621). MEASURED.
- M1 prior (code and doc text only; no result JSON opened): RESIDENT (by continuous-batching prefix-cache
  reuse, not start_chat). Evidence: tools/run_h1_hybrid.py:1483-1484 ("RESIDENT: omit SchedulerConfig (CB
  default prefix caching ON)"), tools/run_h1_hybrid.py:1658 (each turn calls pipe.generate([prompt]) with
  the full prompt, so reuse comes from prefix blocks), docs/amendments/DEMO2_AMEND_2026-10-16_TRUE_DEFAULT.md
  lines 138-148 ("Arm A runs prefix caching on"; TTFT p50 0.30 s at n_ctx 3,936). Unverified until the
  M1 two-call probe; branch S vs R is decided by M1 and Gate B, not by this prior.

## 4. Controls and policies (definitions exact)
- D1 default hybrid = OpenVINO GenAI LLMPipeline(model,"GPU") true default (i8 read-back, unchunked, 57%,
  W 40,960); residency = AS MEASURED BY M1; local-first; on exception or no first token within 10 s ->
  cancel, re-issue to cloud with full transcript and cache_control on the stable prefix; sticky; 0 local
  retries. Sensitivities: D1-nb (non-sticky per call), 3 retries.
- D2 = D1 + pre-call check: Qwen3 tokens + max_new > declared window -> cloud, sticky. Declared window =
  runtime-reported (OpenVINO 40,960; Ollama num_ctx via /api/show). Margin 0 primary (LiteLLM
  enable_pre_call_checks semantics), 0.05 sensitivity.
- ROUTE-1 = at session start choose a config from {i8 default, i8+resident, i8-c256+resident,
  u4-c256+resident} maximizing predicted local turns s.t. memory k*n_ctx + a*min(n_new,c) <= B_eff*(1-m);
  n_ctx + max_new <= W - 512; predicted TTFT(n_new|n_ctx) <= 10 s; config passed COST-1 X-Q. Tie-break
  toward the default. Planned migration turn = first t with predicted C_t > min(L_session(config),
  W-512)*(1-m); delta-hat re-estimated each turn from observed growth; per-call guard backstop; one cloud
  tier pinned per session; transcript handoff at migration. m = 0.059 (session-law upper error) unless the
  STAT-1 L90 margin is larger.
  Implementation note (filed with this amendment, before any data; it fixes how the rule above is computed
  and reproduces the registered hand-computed fixtures, turn 22 for i8+resident and turn 26 for
  u4-c256+resident): the margin (1-m) multiplies the memory ceiling of resident configs (the session law,
  whose error band m comes from). The window cap W-512 is exact and is not multiplied by (1-m). For the
  non-resident i8 default, the memory ceiling is the single-shot law B/(k+a).
- Oracle = perfect per-turn feasibility, TTFT and cache state; minimizes $ with no failure and no SLO miss.
  Y = (D - ROUTE-1)/(D - Oracle), with D = the control giving the smaller saving.
- Ablations: A0 = HYB-1 P2-U (guard at default config); A1 = D1 on ROUTE-1's config, no predictor;
  A2 = ROUTE-1 with a = 0; A3 = B-4K, B-8K, B-rand (seed 20261031) at ROUTE-1's cloud share.

## 5. Workloads and pins
- tau2 via Exgentic agent-llm-traces (v1 70036b93, v2 4b8ad4ab; full hashes to be copied from the dataset
  page without opening data; re-tokenize with the Qwen3 tokenizer) = primary cost replay.
- BFCL v4 multi-turn base + long_context = secondary and live success guard.
- TraceLab v0.0.2 (HYB-1 pin) = dilution check.
- Exgentic SWE-bench = cost-only sensitivity.
- OSWorld a11y text mode = stretch, unregistered outcomes.
- Nothing is opened before the Fri Nov 6 fleet gate.

## 6. Device measurements M1-M8
ROUTE-1 cells run at the lowest priority, after locked gates, SESS-1, STAT-1, COST-1 P1 and the HYB-1 live
replay. Zero T2S hours.

| # | What | Why | Existing evidence | Producing queue | New cell | Priority |
|---|---|---|---|---|---|---|
| M1 | default-stack residency read-back | branch S vs R | arm A TTFT p50 0.30 s at 3,936 (d482c621); code prior RESIDENT (section 3) | none | two-call TTFT probe, shared 16K prefix (~5 min) | 1 |
| M2 | cold TTFT(n) per config (i8, u4, i8-c256, u4-c256), 8 lengths, r >= 3 | D1 timeout; ROUTE-1 SLO | d42040c1, 8997aa4e, d7e8b43e, c2246b1f | HYB-1 live replay (R-ttft) | i8 curve if missing | 3 |
| M2b | resident TTFT(n_new given n_ctx), n_new 0.5-2K, n_ctx 10-40K | SLO for resident turns | none (2-5 s ASSUMED) | none | one cell ~1.5 XPS-h | 2 |
| M3 | time-to-fail, turn >= 2 | D1 wasted time | 0.442 s at turn 1 (00c7fa60) | INV/SESS with --trial-telemetry | capture on turn-2 path | 4 |
| M4 | time-to-ready and KV survival after a failure | recovery cost | not built (reaper conflict) | none | new child, in-process recovery | 5 |
| M5 | session law for resident i8 (~37.3K at delta 1,500) and u4-c256 | biggest swing | SESS evidence; band -2.5/+5.9% | SESS-1 -> Gate B Nov 8 | i8 resident cell only if SESS-1 lacks it | 2 |
| M6 | window overrun at 42K / 45K u4-c256: loud or silent | A1 silent failure; margin | 43 silent generates > 40,960 (BR_HYB1) | SESS-1 (--points 42000,45000) | none | 2 |
| M7 | hold rate near W at u4-c256 | capacity input | 70,000 0/4 (f8df3847) | STAT-1; COST-1 X16-W | none | 2 |
| M8 | Qwen3-4B success at 20-40K under u4 | iso-success | none | COST-1 X-Q (config side) | live paired test (spend) | 6 |

## 7. Cost model
- Anthropic cache rules: 5-min TTL from request start, refreshed on hit; exact prefix; 20-block lookback;
  tools -> system -> messages invalidation; 512-token minimum; model or tier switch = cold (ASSUMED, verify
  live).
- Price sheets: HYB-1 2026-10-26 (primary) plus the dated line "Gemini 3.1 Flash-Lite $0.25 in / $0.025
  cached / $1.50 out, cache write = input (ASSUMED), checked 2026-10-29,
  https://ai.google.dev/gemini-api/docs/pricing" (carried from the report; not re-fetched in this task).
- Tiers: Haiku 5.5 (x5 above 100K), Flash-Lite 3.1, Sonnet 5.5 (headline), Opus 5.5. Forced hit grid
  {55, 75, 90, 96}%.
- Local energy at 18.2 c/kWh as a separate line. Wait seconds as a separate axis, never in the headline $.
  Rework-adjusted $ always reported.

## 8. Statistics
Session = unit; all policies paired on the same sessions; cluster bootstrap 10,000 (Exgentic: tasks then
traces; TraceLab: users then sessions); Holm across H-R0..H-R8; success non-inferiority delta = 5 pp,
one-sided 95% lower bound, crossing sessions. Power: ~53% at 120 pairs, 10% discordance; 247 pairs for 80%.

## 9. Predictions and decision rules

| ID | Contrast | Prediction | Basis |
|---|---|---|---|
| H-R0 | A0 vs D1 | Cloud $ differ by **<= 2%**. Failed local attempts **>= 95% fewer** | DERIVED $235.60 = $235.60 |
| H-R1 | ROUTE-1 vs min(D1, D2), Sonnet, sessions crossing 19,750 | **Branch S** (M1 shows the default re-prefills, or Gate B fails at i8): **>= 25% lower**. **Branch R** (default resident and Gate B passes at i8): **3-12% lower** | DERIVED base: S 35.8-42.4%; R 5.5-8.4% |
| H-R1u | Same, unconditional per workload | tau2: S **>= 15%**, R **<= 5%**. TraceLab **<= 5%** in both branches | DERIVED shapes; COST-1 P-$1 |
| H-R2 | Oracle gap closed (crossing sessions) | **Y >= 70%** | DERIVED: 100% (u4-c256), 68-84% (i8 + resident) |
| H-R3 | A1 vs ROUTE-1 | $ within **+/-5%**. A1 has **>= 1** failure or silent overrun per session crossing W | The config carries the $; the predictor carries safety |
| H-R4 | A2 vs ROUTE-1, unchunked configs | **>= 10** failed attempts per base session; within 1% under chunk-256 | DERIVED (13; the 2.04x phantom) |
| H-R5 | Price tiers | The % saving on Haiku and Flash-Lite lands within -5/+15 pp of Sonnet's. Absolute Haiku $ is **<= 1/10** of Sonnet's | DERIVED 42.9 / 47.7 vs 35.8%; $6.33 vs $84.35 |
| H-R6 | Migration accuracy | +/-1 turn in **>= 90%** of crossing sessions | Session-law error band |
| H-R7 | Success, crossing sessions | Lower bound of ROUTE-1 - D1 **> -5 pp** | Lowest confidence of all the predictions |
| H-R8 | Safety and energy | ROUTE-1 **0** hard failures (R-guard). Extra local energy **<= 2%** of the Sonnet saving | DERIVED <1% |

| Rule | Condition | What may be claimed |
|---|---|---|
| DR-R1 | H-R1 lower CI bound clears its branch floor, and H-R7 holds | The headline sentence with X and Y |
| DR-R2 | H-R1 holds, H-R7 fails | Headline only as **rework-adjusted** $. If that is under 10%, no dollar headline: lead with failures and time |
| DR-R3 | Branch R | [private] |
| DR-R4 | H-R3 holds | Say plainly that the dollars come from the configuration the model selects. The model's share is choosing it safely per device: on the 32 GB tier the same rule says "do not reconfigure" |
| DR-R5 | Live billed $ misses +/-10% | Live numbers carry the headline; the simulator becomes exploratory |
| DR-R6 | COST-1 X-Q <= 26/30 (its DR-7) | Drop u4 from the config set and use i8-c256 + resident (38,000, MEASURED 1e9671a0). The base edge becomes about 30% (DERIVED $165.10) |
| DR-R7 | [private] | [private] |

## 10. Simulator validation criteria
Billed $ within +/-10% via cache_creation_input_tokens / cache_read_input_tokens; migration turn +/-1 in
>= 90% of crossing sessions; HYB-1 R-fit, R-ttft, R-guard.

## 11. Spend
$0 now. Tranches A (cache validation, $3-5), B (BFCL paired success, $7-15), C (tau2 subset, $2-4); cap $30,
hard stop $40; each tranche needs Zach's written approval logged in PLAN_LOCK BEFORE any call; stop at +50%
over estimate.

## 12. Schedule (ET)
Register by Sat Oct 31; scope replies by Tue Nov 3; Gate A Wed Nov 4; code-freeze tag Thu Nov 5; scope
deadline Thu Nov 5 11:59 PM (DR-R7); fleet gate Fri Nov 6 (traces open only after it); Gate B Sun Nov 8
(session-law branch); sims Nov 9-14; framing Sun Nov 15; live validation Nov 16-21 if approved; freeze Sun
Nov 22 (params hashed).

## 13. Not claimed
Savings at a fixed configuration; quality beyond the non-inferiority margin; learned or difficulty-aware
routing; anything on Rithwik's hardware or traces; wait time in $.

## Code (synthetic sessions only)
seam/route1/ (session, prices, cache_sim, device, policies, planner, guard, sim) and tests/test_route1_*.py.
TTFT parameters in seam/route1/device.py are PLACEHOLDER until the Nov 22 freeze.
