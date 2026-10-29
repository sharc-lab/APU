# ROUTE-1 addendum: DUET (split by SLO class, measured resident-TTFT model, D3 and the ablation ladder)

Status: REGISTERED - NO DATA - [private]
Filed: 2026-10-29 (Task BT; approval docs/amendments/APPROVALS_2026-10-29.md). Filing time ET: 2026-10-29 17:20 EDT
Adds to: docs/amendments/HYB1_AMEND_2026-10-29_ROUTE1_SAVINGS.md (sha256
ba269f15e5c846518936491cab2a7ea9f895738c9cc8c45c3dfd02a215dbaf4e) and derived/route1/ROUTE1_PREDICTIONS.json
(sha256 9188d7e5b353306ee2598c2da58dad4ab0f03f934eb4013f01601ac50e17fc95); both re-checked at filing:
MATCH. Their bytes are not edited.
Predictions: derived/route1/DUET_ADDENDUM_PREDICTIONS.json sha256 c54cf7958aeccfef07a32e3e4ce4da8503281888408788d24bdbb0739a2c77dd
This file: sha256 d24ba9f0f765b092da18c3cc0a6b3faf54149ae16aa3e87d1542e09d943f6280 (computed with the value on this line replaced by 64 "0" characters;
the true sha256 of the file as committed is in docs/PLAN_LOCK_2026-10-23.md, Addendum ROUTE-1/DUET)
[private]
Source of the design: "Hardware aware cost routing policy.md", Appendix A (research_20261021).

Labels: MEASURED (run_id), DERIVED (formula or simulator), ASSUMED (a modeling choice), HYPOTHESIS (a
mechanism that needs a measurement before it can carry a claim).

## 1. Purpose
A dated addendum to ROUTE-1, filed before any routing data. The original H-R1 and H-R1u stay registered and
are still reported. This addendum adds split predictions, controls, a ladder and errata. Anything computed
from data seen before filing is exploratory.

## 2. Errata (input labels only; no registered prediction is recomputed)
- kappa unit: resident-turn kappa is in microseconds per token^2 (s = kappa * n_new * n_ctx), not ns.
  Per-cell values (u4 unchunked) 0.462 / 0.428 / 0.407 us (0c7ae0a3, d7e8b43e, 10cb79e4); f16 0.450 /
  0.335 / 0.281 us (e99bb2ad, 71b7d9d8, 9cb921fd); u4 c256 0.578 us (08e52da9); f16 c256 0.808 us
  (2bfae783). DERIVED from the cells' per-turn prefill_s.
- M2b "resident TTFT 2-5 s ASSUMED" is replaced by the DERIVED model
  TTFT_ext(n,P) = P*(alpha + beta*n)*(P/16500)^0.15, alpha = 0.079e-3 s/token,
  beta = 0.339e-6 s/token^2, conservative +7..+22% vs anchors: 183@16,384 measured 1.89-2.20 s (INV-1);
  1K@16,561 5.97 s; 2K@16,473 11.2 s; 4K@16,429 20.4 s; 1K@33,121 13.0 s; 2K@61,771 53.1 s.
  Cold: 0.28 + 0.276e-3*P + 66.7e-9*P^2 s. No i8 resident cell exists: i8 uses the u4 fit (ASSUMED) until
  measurement D-M2.
  Note: at the 4K@16,429 anchor (n/P 0.24) the model favours REPREFILL narrowly (cold 22.8 s vs extend
  23.6 s) while the measured extend (20.4 s) won; the true crossover is likely 0.3-0.4. D-M2 decides.
- Consequence (DERIVED): under ROUTE-1 constraint (c) the base shape migrates at turn ~10 in interactive
  mode, not 26. ROUTE-1's 30-36% applies to background sessions only.

## 3. SLO classes
INTERACTIVE (TTFT SLO 10 s, chance-constrained) and BACKGROUND (no TTFT SLO; memory and window bind). Each
session is labelled by the agent framework before routing.

## 4. Controls (definitions exact)
- D3 = D1 resident + client tool-output filter (keep ratio 0.25) + compaction at handoff (keep 0.25, Haiku
  5.5 executor, priced) + Haiku 100K cliff compaction + ski-rental keep-alive + cache hygiene (append-only
  transcript, pinned tools/system/params, shared breakpoint).
- D3-bg = D1 with timeout off (i8 session law) + the same tricks.
- D1-nb = D1 non-sticky (sensitivity).

## 5. Policy DUET
Fixed, non-learned, feasibility-and-configuration only; tier fixed per session.
Rungs: H1 predictive admission by the resident-TTFT law; H2 configuration by SLO class; H3 delta routing
(REPREFILL when cold < extend and memory allows; EXCURSION-STUB: cloud serves a big-delta turn on its warm
append-only copy, device appends a 60-token stub); H4 SSD park/restore, live DXGI admission, share-aware
runtime; H5 observability guard; H7 analysis-only counterfactual (extend at cold per-pair efficiency).
Every arm, including D1 and D3, gets the same cloud-side tricks when the ladder says so.

## 6. Ladder
L0, L0s, C1..C4 (= D3), H1..H5, H7, Oracle. Negative ablations with direction predictions: kappa-blind
(assume 3.5 s resident TTFT), n_new-blind (kappa*P only), KV-only (a = 0), drift-blind (L50), share-blind,
observability-blind, declared-window trigger.

## 7. Predictions (hashed; floors about half of each DERIVED value, not tuned)
- H-R1-Bg   background: unchanged ROUTE-1 branch values (S >= 25%, R 3-12%).
- H-R1-Iu   interactive uniform: DUET vs D1 resident |delta| <= 2%; vs D1 stateless 5-10% lower (DERIVED
            0% and 6.7%).
- H-R1-Ih   interactive, sessions with >= 15% big-delta turns: DUET-HW (H1-H3, no filter, no compaction)
            >= 25% lower $ than D1 (DERIVED 46.6%). EXPLORATORY until promoted.
- H-D3-I    D3 -> DUET: bursty interactive >= 20% lower; uniform interactive |delta| <= 2%.
- H-D3-B    D3-bg -> DUET-bg >= 8% lower (DERIVED 16.2%).
- H-D3-T    tool-heavy: D3 -> DUET <= 12% lower (DERIVED 6.6-8.1%).
- H-ABL-1   kappa-blind: >= 10 SLO-violating admitted turns per uniform session (DERIVED 14-16).
- H-ABL-2   n_new-blind: leaves at the first big tool output, same turn as D1 (DERIVED turn 4).

## 8. EXPLORATORY list
Filter ratio, compaction ratio and executor, stub payload, BOOMERANG, keep-alive targeting, anchor
breakpoint, 1h-prefix/5m-tail TTL mixing, H4-H6.
Promotion rule (decided only at the Sun Nov 15 framing gate): D-M2 confirms the resident-TTFT model within
+/-20% at i8 and u4, AND the quality pairs pass one-sided non-inferiority at delta = 5 pp. Otherwise they
stay exploratory.

## 9. Matched-quality reporting on every rung
$ at matched pass rate; pass rate at matched $; rework-adjusted $ = dp_fail * task$ * 1.76 / 0.848; a
matched-cost random two-model mix. Wait seconds reported separately, never in $.

## 10. New measurements
Lowest XPS priority, after locked gates, SESS-1/STAT-1, COST-1 P1, HYB-1 replay.
- D-M2 small-delta resident sweep n_new {128, 512, 1500, 4000, 8000} x P {8K, 16K, 24K, 35K} at i8 and u4,
  r = 3, plus REPREFILL and per-kernel timing (~2-3 XPS-h).
- D-M4 llama.cpp SYCL resident + slot save/restore (~1.5 h).
- D-M5 co-runner balloon (~1 h).
- D-M6 turn-2 time-to-fail/ready/KV survival (~0.5 h).
- Cloud tranche A+ adds ~$2 to tranche A.

## 11. Spend
$0 now. ROUTE-1 cap $30, hard stop $40, unchanged. Every tranche needs Zach's written approval logged in
PLAN_LOCK before any call.

## 12. Schedule (ET)
File by Sat Oct 31 (hard limit before Fri Nov 6); scope replies by Tue Nov 3; code freeze and scope deadline
Thu Nov 5 11:59 PM (DR-R7 applies to DUET); fleet gate Fri Nov 6; Gate B Sun Nov 8 (component 3 in or out);
XPS cells Nov 7-14; framing and promotion Sun Nov 15; tranches Nov 16-21 if approved; freeze Sun Nov 22
(params hashed).

## 13. Scope
[private]
## 14. Not claimed
Learned or difficulty-aware routing; quality beyond the margin; savings from cloud-side tricks as hardware
savings; anything on Rithwik's hardware or traces; wait in $.

## Code (synthetic sessions only)
seam/duet/ (params, ttft, device, shapes, ledger [imports seam/route1/cache_sim], policies, planner,
keepalive, ladder); tests/test_duet_fixtures.py, tests/test_duet_rules.py; synthetic ladder table
derived/duet/LADDER_SYNTHETIC_2026-10-29.md. All registered fixtures reproduce with the constants as
written (no tuning).
