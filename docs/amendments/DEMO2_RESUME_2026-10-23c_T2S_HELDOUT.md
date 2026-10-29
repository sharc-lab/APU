# DEMO-2 note 2026-10-23c: resuming the T2S held-out at ho_b (no arm or rule change)

Recorded Fri Oct 23 2026, about 4:45 PM ET (Task BP part E). Mac only, $0, read-only check. It
changes no arm, rule, pin, threshold or prediction. Runners do not read this file.

## 1. What happened
- Window window_20261023_131037_be_v2 (run_queue_t2s_be_v2.sh, sha256 2d69f1fc, the frozen hash in
  DEMO2_FREEZE_2026-10-23.md; EXPECT_HEAD 72e499f):
  - ho_a 1f13b75e, ho_p 9eb71a3a, ho_c 89676c43 and ho_rfull 188a34b8 all ended rc=0 with no
    watchdog fire.
  - At 15:03:08 ET the queue stopped before ho_b: a banned/foreign "python" process was seen.
    banned_detail.txt is empty (the process had exited by the detail call). The hand-back check was
    clean.
  - ho_b (row 34) and ho_i4k (row 35) did not run.
- The script pulls with --ff-only and requires head == EXPECT_HEAD. Origin main is no longer 72e499f,
  so a resume needs a new EXPECT_HEAD.

## 2. Code-path check (72e499f -> the head to be pushed)
- Path: tools/launch_h1.ps1 -T2S -> tools/run_h1_hybrid.py + tools/run_h1_canary_opening.py ->
  tools/canary_child.py -> tools/smoke_delta_prefill.py, plus t2s_queue_watchdog.py and
  cloud_ping.py. 67 .py files, including imports inside functions.
- 72e499f -> 241e4d3: the only changed .py file in the path is seam/ov_kv_precision.py. It adds
  i8 names, a byte entry and aliases.
  - For arm gpu_only (ho_b and ho_i4k), requested_kv_cache_precision is None, so the match is
    always True.
  - The canary's f16 maps exactly as before.
  - The true-default i8 check uses seam/genai_runtime.py's own normalizer, which is unchanged.
- configs/delta_n.yaml, which the path reads, gains one arm (gpu_only_i8). DEMO-2 selects gpu_only
  and the canary selects gpu_only_f16. Neither arm's text changed, and neither did arm_order.
- 241e4d3 -> fb5f901 (Task BP part A): tools/ttft_slo_canary.py changed, and the DEMO-2 opening
  canary imports it.
  - The change is default off. With its defaults, assert_canary_budget_fits and
    derive_canary_every_n returned identical output for planned 0-59 and four wall-time series
    (240 cases, 0 mismatches; Mac check).
  - plan_fragment adds keys only when up_front_budget is set.
- Other files changed after 241e4d3 (tools/boot4_session.py, run_resident_limit.py, redflags.py,
  fleet / figs / queue_status tools, docs) are not imported by the DEMO-2 path.
- Verdict: the DEMO-2 code path behaves identically. Any commit pushed after this one must be
  re-checked the same way before the resume.

## 3. Deviation recorded
- plan.json git.git_head will differ: 72e499f for ho_a/p/c/rfull, the pushed head for ho_b/ho_i4k.
  The plan hashes no config file.
- Same pin DEMO2_HELDOUT_60.json (CC74A37E), same arms and flags, same budget 16,384, same script
  sha256.

## 4. Resume (Zach runs it Sat/Sun only after the T2S CC v3 queue ends; it reboots the T2S)
REBOOT_OK=1 BUDGET=16384 HELDOUT_OK=1 EXPECT_HEAD=<pushed head> FROM=34 bash ~/seam_closeout_1013/xps/run_queue_t2s_be_v2.sh

The script stops at 9:44 PM ET Sun Oct 25 (hand-back).

## 5. What needs ho_b / ho_i4k
- Primary (P vs A dC_iso, DEMO2_PREFREEZE_2026-10-22.md:41): needs ho_a + ho_p (done) and the shared
  cloud-only log ho_cloud (XPS row 30, deferred until after Oct 27). Neither ho_b nor ho_i4k is
  needed.
- F2 family (DEMO2_AMEND_2026-10-23_OBJECTIVES.md:94, m = 6): A vs P and A vs C only. Not needed.
- Secondaries (PREFREEZE.md:64-67, interpreted only if the host's primary rejects, :50):
  - ho_i4k is needed for P vs I-4K, C vs I-4K and I-4K vs A.
  - ho_b is needed for B vs A ("every arm vs A"). B exists only on the T2S.
