# First XPS CC v3 data vs the red flags (Task BP, Fri Oct 23 2026 ~4:40 PM ET)

Read-only, from the parked copies on the Mac: ~/seam_closeout_1013/xps/window_20261023_140349_cc_v3
(head 241e4d3) and window_20261023_132334_cc_v3 (A19 record). Every value carries its run_id. Derived
values are labelled DERIVED. Post-hoc checks are labelled POST HOC: they are not predictions and are
not scored. Not scored here: BK scores the cells at closeout.

## 1. R6 (A19 OS tier), both hosts: PASS
- XPS, window_20261023_132334_cc_v3/a19_dxgi_start.json:
  - ullTotalPhys 16,752,234,496 B (15,976.2 MiB);
  - Intel(R) Graphics SharedSystemMemory 9,548,773,662 B (9,106.42 MiB).
- Zach's manual record (Fri Oct 23 about 4:25 PM ET, Intel Graphics Software): Shared GPU Memory
  Override ON at 57%.
  - 0.57 x 16,752,234,496 = 9,548,773,663 B, within 1 byte of the measured value.
  - That is the registered A19 "slider x ullTotalPhys" clause, so R6 PASSES.
  - Without that clause (0.50 x M = 7,988.1 MiB) the XPS would be +1,118 MiB off.
- The 57% figure is the driver's documented default. Intel's support article 000101789
  ("Shared GPU Memory Override", read 2026-10-23) states that the default value is 57%. The setting
  arrived with Intel Graphics Software 25.26.1602.2 / driver 32.0.101.6974.
  - Not verified here: whether the override toggle ships ON on this Dell image, or whether a user
    turned it on. Zach recorded it as the driver default.
- T2S, window_20261023_130443_cc_v3: SharedSystemMemory 48,634 MiB = ullTotalPhys 65,018 - 16,384
  MiB, i.e. the Windows formula. The override clause does not apply there: 57% would give 37,060 MiB.
- Consequence for H2 (wording only): the GPU-visible tier is set by the Windows formula, OR by the
  vendor driver where its override is active. On the XPS that is the Intel driver at its documented
  57% default. The default XPS tier is therefore vendor-set, not OS-set.

## 2. R1 (A21 8-bit, i8 compiled and read back): RED FLAG by the registered range
- a21_8bit de5f641f: holds 19,500; fails 19,750, 20,250, 21,000, 24,000 (all turn-1 RuntimeError,
  memory_wall); L* 19,500. Compiled kv_cache_precision i8 in every point.
- a21_8bit_rep b038e362 (points 19,500, 19,750 x 2): all four held. 19,750 therefore held 2 of 3
  attempts. Best estimate L* = 19,750; 20,000 was never tried; 20,250 failed.
- R1 range 19,900-22,600: 19,750 is 150 tokens (0.75%) under its lower edge. The A21 band
  20,921-21,486 is missed by 5.6%.
- The grid caveat: L* lies in [19,750, 20,250). The 250-token grid cannot place it on either side
  of 19,900 with certainty; the point estimate is below.

## 3. Is the miss a window-level B shift or a k/a misfit? (DERIVED)
Uncapped limits in this window against the registered law (a = 96,768, B = 3.412 GiB):

| cell | run_id | L* | law | miss |
|---|---|---|---|---|
| u4 unlx | 5943c992 | 25,000 (fails 26,000) | 26,502 | -5.7% (grid 1,000: up to -1.9%) |
| i8 (A21) | de5f641f + b038e362 | 19,750 | 20,921 | -5.6% |
| f16 (bal_g0, 0 GiB balloon process resident) | ba7a1e89 | 13,250 | 15,000 | -11.7% |

- Implied B = L* x (k + a):
  - with the fitted a: u4 3.219 GiB, i8 3.221 GiB, f16 3.014 GiB;
  - with the A20 a (section 4): u4 3.137, i8 3.157 (u4's a), f16 2.900 GiB.
- Allocator reading at the last held point (GPU_MEMORY_STATISTICS after turn 1, minus the weights'
  usm_host at load):
  - i8 19,750: 3.157 GiB;
  - u4 25,000: 3.138 GiB;
  - f16 c256 22,000: 3.042 GiB;
  - u4 c256 70,000: 3.100 GiB;
  - f16 13,250: 2.901 GiB.
- Reading:
  - u4 and i8 imply the same B (3.22 GiB with the fitted a; 3.14-3.16 GiB from the allocator),
    about 0.19 GiB (5.6%) below the fitted 3.412 GiB. One shared budget shift explains both. A
    k or a misfit would move i8 and u4 by different fractions. So the R1 miss is a window-level
    B shift, not a k/a misfit.
  - The f16 uncapped proxy is lower still. It ran with the balloon process resident at 0 GiB.
    Read it against mech_f16_unlx (v3 row 50) before any conclusion.

## 4. A20 allocator a (R3): PASS; f16 misses its band
- Method: after-turn-1 GPU_MEMORY_STATISTICS (cl_mem + usm_device + usm_host) minus the weights
  (usm_host at load) minus n x k, regressed on n_cached (9 points each).
- Results:
  - alloc_f16 e4e797e7: a = 87,520 B/token, R^2 1.000;
  - alloc_u4 ae16fcd5: a = 93,280 B/token, R^2 1.000.
- They differ by 6.6% (< 10%), and both lie in 70-130 kB, so R3 PASSES.
- A20 band 93-98 kB: u4 HIT, f16 MISS (prediction miss, not a flag).
- Why u4 minus f16 = 5,760 B/token (measured split):
  - usm_device per token is identical in both arms (87,056 B/token). That is the activation term.
  - The whole difference is in cl_mem, the KV pool:
    - f16 147,648 B/token vs k 147,456 (+192, block rounding);
    - u4 47,294 vs k 41,472 (+5,822);
    - i8 (de5f641f) 84,113 vs k 78,336 (+5,777).
  - Quantized KV blocks therefore store about 5.8 kB/token beyond our k. That is likely the
    scale/zero-point storage at a finer granularity or precision than k assumes (an inference; not
    checked in source). The A20 decomposition subtracts only k, so this lands in a for u4.

## 5. R4 / R7 (cap 256): PASS
- Cells and the law:
  - mech_f16_c256 d782cf80: L* 22,000 (fails 23,000); H-act 24,676 (-10.8%).
  - mech_u4_c256 f8df3847: L* 70,000 (fails 71,000); H-act 87,736 (-20.2%).
  - Gate >= 40,000: PASS. H-act is refuted only at <= 29,000: not refuted, so R7 PASSES.
- Cap 256 raises both ceilings. Every failure is a turn-1 (prefill) RuntimeError classed
  memory_wall, with no cache_size set, so no construction-time budget assertion. R4 PASSES.
- Gains (DERIVED):
  - f16 capped / uncapped: 22,000 / 13,250 = 1.66x (1.47x against the sealed 15,000, 553b3a5c);
  - u4: 70,000 / 25,000 = 2.80x.
- f16 -> u4 ratio:
  - uncapped 25,000 / 13,250 = 1.89x (sealed 26,500 / 15,000 = 1.77x);
  - capped 70,000 / 22,000 = 3.18x (registered DERIVED 3.56x).
- POST HOC (not a prediction): the capped ceilings fit (B_alloc - 256 x 87,100) / k_eff. Here
  B_alloc = 3.15 GiB from the u4/i8 holds, and k_eff is the measured KV-pool bytes per token
  (f16 147,563; u4 47,243). That gives f16 22,770 (measured 22,000-22,999) and u4 71,120 (measured
  70,000-70,999). The u4 capped miss against H-act is mostly the 5.8 kB/token KV overhead plus the
  window B shift.

## 6. R5: PASS so far (XPS)
Compiled KV read back as requested (f16 / u4 / i8) in every point of the cells above, with A31 on.

## 7. Balloon cells and the available-memory floor (Part G)
- bal_g0_f16 ba7a1e89 ran: L* 13,250.
- bal_g1_f16 5b2ca33c and bal_g2_f16 75fd18fc: the balloon was verified (1,058 / 2,092 MB GPU
  shared usage), then the child refused at start with "REFUSED -- gates:
  available_mb:available_mb_below_floor".
- The floor is 7,000 MB: configs/platforms/aipc-c1.yaml measurement_gates.pre_run_available_mb_min,
  citing "2026-08-09 clean session / 7477 MB Available; arm A lowest free_physical_mb_start 6926".
  It is evaluated in seam/measurement_gates.py:374-449 by tools/boot4_session.py require_gates().
- Available MB from the cell sidecar CSVs:

| cell | before balloon | with balloon |
|---|---|---|
| g0 | ~9,300 | (0 GiB) |
| g1 | 8,967 | 6,424-6,811 |
| g2 | 8,840 | 4,436-4,678 |

- The balloon process costs about 1.2 GB of Available beyond its own bytes (its OpenVINO GPU
  context). So every balloon of 1 GiB or more, and both CPU balloons, falls below 7,000.
- The registry already noted that the 4 GiB CPU cell might be refused ("not loosened").
- So A22 cannot run on the XPS at 241e4d3, or at the canary-fix head.
- Options (none implemented; Zach decides; balloon sizes, points and predictions unchanged):
  - (a) Recommended. A default-off runner flag for CC-BAL cells only. The floor is checked on
    Available plus the balloon process's private bytes, measured at gate time. Both raw and
    credited values are written to the gate record. The 7,000 floor is unchanged; the check
    excludes the deliberate load the cell itself adds.
  - (b) Evaluate the gates before balloon_start and pass that gate record to the runner. Same
    rigor; more queue plumbing.
  - (c) Lower the floor by the verified balloon bytes. This is insufficient: 2 GiB gives 4,560 +
    2,048 = 6,608 < 7,000.
- Even with a fix, two rows sit below their registered low rung by the A22 per-GiB predictions:
  - bal_g2_f16 at about 4,500 vs low 8,000;
  - bal_g2_u4 at about 9,500 vs low 12,000.
  - Both bisections would stop after one failing probe: no L*, and no canary arming (that
    one-probe case is the residual noted in CC_FIX_2026-10-23e).
- T2S A29 (floor 24,000 MB, configs/platforms/evo-t2.yaml):
  - Available at a T2S cell start was 60,700 MB (alloc_f16 50d5a3a4 sidecar).
  - 9 GiB + ~1.2 GB leaves about 50,000, and 4.5 GiB about 54,800. Both clear the floor at 241e4d3.
