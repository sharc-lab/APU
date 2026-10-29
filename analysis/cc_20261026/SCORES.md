# CC registered predictions scored (Task BK, Mon Oct 26 2026)

Each item is scored by its own file's rule (CC_PREDICTIONS / CC_BL / CC_BM; DEMO-2 F2 in
analysis/demo2_heldout/SCORES.md). Nothing is re-derived after seeing data. Values come from the
sealed records via analysis/cc_20261026/{collect,bytes,peak_slopes,provenance,measure}.py.
- Registered L* is the bisection's largest held point, as registered.
- Robust ceilings (ADDENDUM 1 rule) are in measured.json "repeats".
- MISS rows use the same type size as HIT rows.

## CC_PREDICTIONS (BJ) - XPS limits, ratios, gate
| Item | Registered | Measured (run_id) | Verdict |
|---|---|---|---|
| f16 c1024 | 17,200 [10,192, 27,229] | 21,000 (b48e4cfe) | HIT |
| f16 c256 | 22,359 [10,192, 27,901] | 22,000 (d782cf80) | HIT |
| f16 unl explicit | 15,000 [14,000, 16,000] | 14,000 (aa914c4c) | HIT (edge) |
| 8-bit c1024 (i8) | 25,463 [19,186, 51,255] | 36,000 (3e45f554) | HIT |
| 8-bit c256 (i8) | 38,676 [19,186, 52,520] | 38,000 (1e9671a0) | HIT |
| 8-bit unl explicit | 20,000 [19,000, 21,000] | 18,000 (fd074be1) | MISS |
| u4 c1024 | 34,234 [28,090, 96,814] | 67,000 (f7542f68) | HIT |
| u4 c256 | 63,319 [36,239, 99,204] | 70,000 (f8df3847) | HIT |
| u4 unl explicit | 26,500 [25,500, 27,500] | 25,000 (5943c992) | MISS |
| u4/f16 c1024 | 1.99 [1.99, 3.56] | 3.19 | HIT |
| u4/f16 c256 | 2.83 [2.83, 3.56] | 3.18 | HIT |
| Mechanism gate: u4 c256 >= 40,000 | HIT | registered L* 70,000; repeats 0/4 at 70,000 and 0/4 at 71,000 (0580176a); bisection held 61,000 and 67,000 once | HIT on the registered L*; robust ceiling NOT established above 16,384 (STAT-1 binomial at 63,000 decides) |
| Hypothesis rule (section 3.3 order) | - | c1024 within 10% of c256 on all three precisions | H-act selected; BL confirm rule also met (f16 c256 >= 21,000, u4 c256 >= 60,000) |

## CC_PREDICTIONS (BJ) - bytes (section 3.2; same rule on both hosts)
| Item | Registered | Measured | Verdict |
|---|---|---|---|
| Settled KV slope, f16 | 147,456 +-25% | 147,456 B/token, every f16 cell, both hosts | HIT |
| Settled KV slope, 8-bit | 78,336 +-25% | 84,096 (i8 compiled; all i8 cells) | HIT |
| Settled KV slope, u4 | 41,472 +-25% | 47,232 (all u4 cells) | HIT |
| Peak-minus-settled, uncapped | [71,956, 129,024] | 0 B/token as registered (10 Hz peak = after-turn-1 state) | MISS |
| Peak-minus-settled, c1024 | <= 65,536 | 0 | HIT |
| Peak-minus-settled, c256 | <= 16,384 | 0 | HIT |
- **Reading of the uncapped MISS (DERIVED, not a rescore):** the plugin keeps the prefill buffers
  after turn 1. "Settled after turn 1" therefore contains them, and the 10 Hz peak never exceeds it.
  - Against the KV-only state after turn 2, peak minus settled is 87,384-87,517 B/token uncapped and
    about 0 capped.
  - usm_device after turn 1 grows 87,056 B/token uncapped and 0 capped, in every cell on both hosts
    (e.g. 50d5a3a4, e442a753, e4e797e7, 5943c992).
- The 1 Hz PDH sidecar cannot resolve the transient (its peak-minus-settled is noise, from -188 kB to
  +35 kB per token). The child-side records are the byte source for the figures.

## CC_PREDICTIONS (BJ) - T2S and cliff
| Item | Registered | Measured | Verdict |
|---|---|---|---|
| T2S holds at the window in every arm | holds | 40,448 held 3/3 at f16 / i8 / u4 (7e249f7e, 6dfd1a12, 2d329db2), plus every slope cell | HIT |
| CC-CEIL decode tok/s vs bandwidth | line | not computed in BK | UNRESOLVED (TARGET) |
| Cliff below L* | holds; TTFT exponent 1.8-2.4 | 12,000 held (83a3c422). One passing point, so no exponent. TTFT cells give 1.58 (u4, 4K-16K, d42040c1) | holds HIT; exponent UNRESOLVED |
| Cliff above L* | loud failure in turn 1, no silent >10x step | 13,500 / 14,250 / 15,000 / 15,750 / 18,750 / 22,500 all failed loud in turn 1 | HIT |
| cliff_u4 | - | 36e0ec24 refused (3 quiescence blocks) | NOT RUN |

## CC_BL (A1-A18)
| Item | Registered | Measured | Verdict |
|---|---|---|---|
| A8 cap-256 TTFT penalty at 16,384 | <= 15% | +112% (48.1 s vs 22.7 s; 8997aa4e vs d42040c1); c1024 +86% (a8018646) | MISS |
| A9 grow, 2K turns | no band; H-act f16 24,676 / u4 87,736; MIXED f16 13,153 / u4 21,231 / u4 c256 63,319 | last held (ADDENDUM 5) f16 18,532 (71b7d9d8), u4 61,771 (d7e8b43e), u4 c256 65,889 (08e52da9), f16 c256 16,473 (2bfae783) | no rule registered; between MIXED and H-act; see the session law below |
| A11 CC-WINDOW T2S | 40,448 holds r = 3 | 3/3 in every arm | HIT |
| A12 uncapped peak-minus-settled | [72,576, 120,960] | 0 as registered; 87.4 kB after turn 2 (DERIVED) | MISS (as registered) |
| A12 capped | <= 24,192 | about 0 | HIT |
| A13 (i) unbounded | fails at N = 6 | held 3, failed at 4 (device_alloc, a854b0da) | MISS |
| A13 (ii) cap 256 dynamic | fails at N = 10 | held 9; N = 10 timed out at 3,600 s (fb3ad7d0) | UNRESOLVED (censored) |
| A13 (iii) fixed 40 GB | preemption from N = 9 | construction refused at N = 8: budget_in_bytes check (186addfa) | MISS |
| A13 (iv) fixed 50 GB | construction fails | budget_in_bytes check at construction (c38c2259, e659627b) | HIT |
| A7 CC-RT, A10 CC-VALID, A15, A16, A18 | - | not run (conditional / spec pin) | NOT RUN |

## CC_BM (A19-A36)
| Item | Registered | Measured | Verdict |
|---|---|---|---|
| A19 T2S | shared = total - 16,384 MiB | 48,634 = 65,018 - 16,384 | HIT |
| A19 XPS | slider x ullTotalPhys (+-256 MiB) | 50% 7,988.1, 57% 9,106.4, 74% 11,822.4 MiB = slider x 15,976.2 | HIT |
| A20 allocator a | 93-98 kB; falsified < 70 or > 130 or f16/u4 > 10% apart | f16 87.52 (e4e797e7), u4 93.28 (ae16fcd5); T2S identical (50d5a3a4, 460ec188) | f16 MISS, u4 HIT; not falsified (R3 PASS) |
| A21 8-bit rebisect | 20,921-21,486; law fails outside 19,900-22,600 | 19,750 (de5f641f + b038e362; 4/4 at 19,500 / 19,750 in the rep cell) | MISS; R1 RED FLAG ("Keep going") |
| A22 GPU balloon f16 | -4,397 per GiB | 0 GiB 13,250 (ba7a1e89), 1 GiB 14,250 (0014465c); 2 GiB not run (98c9a927, quiescence) | MISS |
| A22 GPU balloon u4 | -7,767 per GiB | 25,000 / 25,000 / 24,750 at 0 / 1 / 2 GiB (d6616dfd, 598e882d, f9953743) | MISS |
| A22 ratio | 1.767 [1.59, 1.95] | u4 slope about 0 | MISS (R2) |
| A22 CPU balloon | abs dL < 5% | +5.7% at 2 GiB (0f3950a3), +7.5% at 4 GiB (62d59ab8), vs 13,250 | MISS |
| A23 grow TTFT | delta x (7.2e9 + 589,824 L) / 1.015e13 | 3.6-6.2x slower: u4 2K turns at 59.7K history 53.0 s vs 8.6 s; f16 at 16.5K 12.2 s vs 3.4 s | MISS |
| A23 failure mode | device_alloc at L_mem, not TTFT | every grow failure is a device error in the turn (CL_OUT_OF_RESOURCES; c256 cells: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) | HIT |
| A24 TTFT law | L_TTFT(20 s) = 16,729 +-15% | 15,232 (-8.9%; fit on d42040c1 turn-1 rows); 5 s 6,541, 30 s 19,193 | HIT |
| A25 slider | +681 (f16) / +1,204 (u4) tokens per point, R^2 >= 0.98, ratio 1.77 +-10% | u4 -208 per point (R^2 0.9999 on a NEGATIVE line), f16 -107 (50-57%); 74% f16 failed its 8,000 low rung | MISS (sign); R2 RED FLAG |
| A29 T2S balloon conc (ii) | fails at 10 / 9 / 8 for 0 / 4.5 / 9 GiB | 9 held in all three; 10 timed out in all three (fb3ad7d0, cf1237a2, 093364f9) | 0 GiB UNRESOLVED; 4.5 GiB MISS; 9 GiB MISS |
| A30 Ollama default | XPS 4,096 / T2S 32,768 | Ollama absent on both hosts | NOT RUN |
| A31 KV provenance | compiled = requested, PA | 378 / 378 loads match, backend PA | HIT (R5 PASS) |
| A32 activation law | uncapped failures are device_alloc in prefill near L_mem | 120 device_alloc + 14 oneDNN "could not execute a primitive" in prefill; 4 device errors in turn 2, all in the 74% boot (9e31d4cb, 39d38c85) and 1 at 37,000 i8 c1024 | HIT, with the 74% turn-2 exception noted |
| A32 bookkeeping | construction budget_check below the budget | none in any headline cell (only the fixed-cache conc cells) | MISS (bookkeeping not supported) |
| A33 cache_size | equal +-256 (source); derived 25,891 | 13,000 with 2 GiB (aaadf214) vs 24,250 with 0 (3cd03df7) | MISS (both) |
| A34 cap 256 | bookkeeping: no change; activation: f16 24,676, u4 >= 29,000 | f16 22,000, u4 70,000 | bookkeeping MISS; activation HIT for u4 (f16 -10.8%, no band) |
| A26 energy | fit only | power CSV kept (69f56f0c); fit not done | TARGET |
| M2 (Qwen3-8B) | CC_M2 / BR_M2B | not run | NOT RUN |

## Session law (ADDENDUM 2 + 5; POST HOC / DERIVED, not a held-out test)
- Corrected law: L_hist = (B_BR - a T)/k - T, with compiled k, a 87,520 and B_BR 3.267 GB (BR).
- Residuals against the last held turn:
  - f16 +5.9 / -1.9 / +5.1% at 1K / 2K / 4K (e99bb2ad, 71b7d9d8, 9cb921fd);
  - u4 -1.6 / -2.5 / 0.0% (0c7ae0a3, d7e8b43e, 10cb79e4);
  - u4 c256 2K -1.1% (08e52da9);
  - f16 c256 2K -17.5% (2bfae783; a CL_EXEC_STATUS event error, not CL_OUT_OF_RESOURCES).
- Turn-size slope from 1K to 4K (measured f16 -5,307, u4 -7,707):
  - the old law explains 34% (f16) and 74% (u4);
  - the corrected law explains 92% and 114%.
- A23 is scored as registered above (MISS).
- 15367c3a (i8, a growing session at 67,947) remains unexplained: about 1.8x the i8 session law at
  small turns (DERIVED). SESS-1 has no i8 arm.
