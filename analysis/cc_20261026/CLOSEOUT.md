# Weekend closeout, Part A (Task BK, Mon Oct 26 2026; Mac only, $0, no ssh)

## A1. Windows, rc != 0, NOT_RUN, ALERT (from cells.txt / steps.txt / ALERTS.txt)
| Window | Cells | rc != 0 / NOT_RUN / ALERT |
|---|---|---|
| xps 20261023_140349_cc_v3 (Fri-Sat) | 39 | rc=1: bal_g1_f16 5b2ca33c, bal_g2_f16 75fd18fc, bal_g2_u4 7cfbb796, bal_c4_f16 210c6ffd (Available floor, 241e4d3); cliff_u4 36e0ec24 (3 quiescence refusals after the XPS woke and charged: NOT RUN, not a result). NOT_RUN: CC-RT (conditional), CC-VALID (VALID_SPEC unset). rep_f16_c256 run_id=none in cells.txt: e7f0c307 (network hang during the copy; see A3) |
| xps 20261023_174609_cc_v3b | 1 | STOP: reh_grow rc=1 (pre-fix head) |
| xps 20261024_175932_cc_v3b | 18 | rc=1: bal_g2_f16 98c9a927 (3 quiescence refusals with the 2 GiB GPU balloon). NOT_RUN_COVERED: bal_g1_u4 (598e882d), bal_c2_f16 (0f3950a3); CC-RT conditional |
| xps 20261025_174345 / 185526 / 204059_cc_v3 (slider 50 / 74 / 57) | 2 each | 74%: slider_f16 39d38c85 rc=1, REFUSED_CANARY (low rung 8,000 failed in turn 2; unsealed). CC-VALID / CC-RT NOT_RUN |
| t2s 20261023_131037_be_v2 (held-out) | 4 | STOP 15:03:08 Oct 23 on a foreign python before ho_b |
| t2s 20261023_160550_cc_v3 | 6 (+1) | conc_iv c38c2259 rc=1 (budget_check at construction); ALERT 22:28 Oct 23 python still running after 2,400 s (conc_iv, about 14 h); STOP 14:25 Oct 24: park slope_f16_unl failed 3 times; its sealed run b4a96d44 reached the Mac (MATCH) without a cells.txt line |
| t2s 20261024_143009_cc_v3 (relaunch FROM=41, cedb9d3) | 13 | conc_iii 186addfa rc=1 (budget_check at construction, N = 8) |
| t2s 20261024_175945_be_v2 (resume FROM=34, cedb9d3) | 2 | ho_i4k d9873c58 rc=1 (BFCL checker IndexError; no summary) |
| t2s 20261024_182749_cc_v3 | 1 | conc_iv e659627b rc=1 (budget_check at construction) |
- No PAST_FINISH lines in these windows.

## A2. T2S hand-back
- Final T2S window (Oct 24 18:35 ET) and every CC window: tree clean, PYTHON_COUNT=0, SIDECARS=1.
- The SIDECARS count is the number of processes whose command line matches 'cc_mem_sidecar'. The
  PowerShell that runs the count carries that string in its own command line, so 1 is very likely the
  query itself. It reads 1 in every window, including those whose sidecar had already exited.
- The sidecar also stops itself after -MaxHours 8.
- Not verified (no ssh by rule). Zach tells Rithwik; the be_v2 windows do not count sidecars.
- 20261023_160550 hand-back also listed untracked derived/delta_prefill/b4a96d44 (the unparked
  slope_f16_unl). The next window's hand-back was clean.

## A3. Records
- derived/cc/windows_20261023_25/: byte copies of the 9 CC windows (sidecar CSVs kept in
  ~/seam_closeout_1013, sha256 in CSV_MANIFEST.sha256; 80 CSVs).
- derived/demo2_heldout/: the two T2S be_v2 windows.
- Seals verified on the Mac (tools/seal_verify.py):
  - CC copies: 70 MATCH_MANIFEST (includes duplicate nested copies) + 1 MANIFEST_COVERAGE_GAP;
  - held-out copies: 10 MATCH.
- One exception: the outer copy of e7f0c307 (rep_f16_c256) has a 0-byte summary.json (truncated
  transfer during the network hang), so it reads MANIFEST_COVERAGE_GAP. The nested re-copy
  (rep_f16_c256/20261023_140349_rep_f16_c256/...) verifies MATCH_MANIFEST and is the one used.
  Nothing in either tree was edited.
- Run tables: analysis/cc_20261026/RUNS.md (95 rows), analysis/demo2_heldout/RUNS.md.

## A4. CC-REH answers (they decide the byte source)
1. GPU_MEMORY_STATISTICS grows with n in every cell, per type:
   - cl_mem (KV pool) at exactly compiled k;
   - usm_device after turn 1 at 87,056 B/token uncapped and 0 capped;
   - usm_host constant (weights, about 2.27 GB).
2. Compiled readback: 378 / 378 loads compiled the requested KV type with backend PA
   (provenance.json). There was no SDPA fallback.
3. Sidecar CSV: present for every resident cell. Its 1 Hz PDH peak cannot resolve the prefill
   transient, so its peak-minus-settled is noise.
- Byte source for the figures: the child-side per-type records (fig1, fig9). The sidecar is used for
  Available and page faults only.

## Weekend operations (ADDENDUM 1.3, as stated by Zach; consistent with the logs above)
- The Mac slept on battery Fri-Sat overnight and both links were lost:
  - the XPS v3 hung about 17 h in a scp with no timeout;
  - T2S conc_iv ran about 14 h;
  - the T2S queue stopped 14:25 Sat on park and was relaunched FROM=41 at cedb9d3.
- cliff_u4 36e0ec24 was refused after the XPS woke and charged: NOT RUN.

## Available vs L* (ADDENDUM 1.2; DESCRIPTIVE)
- Available after load (median over the probes), unballooned 57% single-shot cells:
  - u4 25,000 at 8,569 MB (5943c992), 25,000 at 5,617 (d6616dfd), 24,250 at 7,018 (3cd03df7);
  - f16 13,250 at 5,119 (ba7a1e89), 14,000 at 5,862 (aa914c4c);
  - i8 19,500 at 8,505 (de5f641f), 18,000 at 6,351 (fd074be1).
- u4 L* does not track Available (R^2 0.001 over 5.6-8.6 GB). The balloons say the same: a verified
  1 GiB GPU balloon (598e882d) and 2 / 4 GiB CPU balloons (0f3950a3, 62d59ab8) did not lower the
  ceiling.
- The open B question therefore points away from system Available and toward a per-process or
  runtime budget (INV-1).
- One exception is recorded, not explained: the 74% boot's first two children started at
  3,492 / 3,884 MB and failed in turn 2 (39d38c85, 9e31d4cb).
