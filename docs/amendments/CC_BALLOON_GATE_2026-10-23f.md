# CC note 2026-10-23f: balloon-gate option (a), --balloon-floor-credit (CC-BAL cells only)

Recorded Fri Oct 23 2026, about 5:25 PM ET (Task BQ). Mac only, $0.
- Approved by Zach about 5:10 PM ET: "Approve option (a)" (docs/amendments/APPROVALS_2026-10-23.md).
- Under the PLAN LOCK this is a procedural gate fix that lets registered cells run. No cell, arm,
  balloon size, point, band, prediction or threshold changed.
- The 7,000 MB floor (configs/platforms/aipc-c1.yaml) is unchanged. Runners do not read this file.

## 1. Rule
- The flag is tools/run_resident_limit.py --balloon-floor-credit. It is off by default and refused
  unless --cc-cell starts with "CC-BAL-".
- With the flag, require_gates() (tools/boot4_session.py) passes the Available gate when Available
  + the private bytes of the running balloon process is at least the floor.
  - The private bytes are read from the process itself (psutil memory_info().private).
  - The requested balloon size is never used.
- The balloon process is the python interpreter whose command line names tools/gpu_balloon.py:
  - The cmd.exe wrapper from spawn_detached.ps1 is ignored, and so is any shell.
  - A venv launcher and the interpreter it starts both match, so the process with no matching
    child is the one used.
- Refusals:
  - zero or several such processes;
  - unreadable private bytes;
  - unreadable Available.
- Record (plan.json gates.balloon_floor_credit, and an "event": "balloon_floor_credit" log line,
  so it survives a refusal):
  - observed Available;
  - balloon pid, candidate pids, private bytes, rss;
  - the credited sum;
  - the floor and its citation;
  - pass or fail.
  - The available_mb gate detail carries observed_is_credited: true.
- Every other gate, and every cell without the flag, is evaluated exactly as before (tests:
  tests/test_cc_bq_balloon_gate.py).

## 2. Correction to the BP reading (analysis/cc/bp_xps_first_data_20261023.md section 7)
- tools/gpu_balloon.py keeps the numpy host buffer for every chunk (held.append(host)). A GPU
  balloon of X GiB therefore also holds about X GiB of host memory.
- That, not a 1.2 GB runtime context, explains the Available drops:
  - about 2.2-2.5 GB for 1 GiB;
  - about 4.2 GB for 2 GiB.
- The balloon tool is not changed (sizes are locked). The credited private bytes include the host
  copies.
- Whether the GPU-side allocation is also charged to the process's private bytes is not yet known.
  If it is not, a 2 GiB cell may still refuse on the credited sum, which would be a correct refusal.

## 3. Expected outcomes (no change to bands, points or sizes)
- Rows 32 (bal_g2_f16) and 35 (bal_g2_u4) are expected to fail their low rung: about 4,500 vs
  8,000 and about 9,500 vs 12,000, from the A22 per-GiB predictions.
  - They report "below low rung", not an L*.
  - A bisection stopped at its low rung cannot arm the canary (the residual in CC_FIX_2026-10-23e),
    so these cells come back unsealed.
- Already measured at 241e4d3 without the flag, under the registered gate:

| cell | run_id | balloon | gate Available | result |
|---|---|---|---|---|
| bal_g0_f16 | ba7a1e89 | 0 GiB | ~9,300 MB | L* 13,250 |
| bal_g0_u4 | d6616dfd | 0 GiB | 9,131 MB | L* 25,000, first fail 25,250 |
| bal_g1_u4 | 598e882d | 1 GiB, 1,058 MB GPU shared verified | 7,582 MB, above the floor unaided | L* 25,000, first fail 25,250 |
| bal_c2_f16 | 0f3950a3 | 2 GiB CPU, ready | 7,094 MB, above the floor unaided | L* 14,000, first fail 14,250 |

  - v3b does not re-run row 34 (bal_g1_u4) or row 36 (bal_c2_f16): 598e882d and 0f3950a3 have the
    same args, a ready balloon and the registered gate. v3b re-runs rows 31, 32, 35 and 37 with the
    flag (bal_c4_f16 210c6ffd and bal_g2_u4 7cfbb796 were also refused by the floor at 241e4d3).
  - The CPU 2 GiB cell is 5.7% above the f16 0 GiB cell (14,000 vs 13,250). A22 registered
    |dL| < 5% for CPU balloons, so this is a small miss in the upward direction (for BK to score).
- How BK reads them: one A22 table per arm. Each row carries its gate mode: "registered" for the
  three above, "credited (option (a))" for the v3b rows. The measurement protocol is otherwise
  identical. The slope per GiB is fitted across 0/1/2 GiB from both kinds of row.

## 4. Observation for the closeout (no decision here)
- The u4 ceiling did not move with a verified 1 GiB GPU-shared balloon: d6616dfd and 598e882d
  both have L* 25,000, the same point sequence, and the same peak GPU_MEMORY_STATISTICS of 5.25 GiB
  at 25,000. A22 predicted -7,767 tokens per GiB.
- The per-process GPU_DEVICE_TOTAL_MEM_SIZE on the XPS is 8,503,984,128 B (7.92 GiB), from the
  load record.
- One reading (a hypothesis, not tested) is that the binding limit is the GPU plugin's
  per-process accounting, which a side-process balloon does not touch.
- This is reported to Zach under the lock (R2 / R8 context). The 2 GiB and f16 balloon rows, and
  A25 (slider), test it.
