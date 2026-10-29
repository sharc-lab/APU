# DEMO-2 registration freeze: values computed by the registered rules (Task BF)

Written Thu Oct 22 2026, about 8 PM ET, by Claude Code (Task BF), Mac only, no ssh, $0. This is ahead of
the Fri Oct 23 12:00 PM ET hard ceiling (PF 13). The freeze head is the commit that adds this file.

This note fills values only. It changes no rule, threshold, arm, pin or prediction. Rules:
- DEMO2_PREFREEZE_2026-10-22.md ("PF", sha256 e24f4ce92395a6b2fed9b065cee933eb808a704539aacc7d24d29998b2e6ae8b),
  which governs where it differs from:
- DEMO2_ISO_QUALITY_2026-10-21.md, DEMO2_ARMS_REVISED_2026-10-21.md, DEMO2_PER_DEVICE_2026-10-21.md.

Records: derived/demo2_pilots/windows_20261022/ (byte copies; 19/19 h1 seals re-verified on the Mac).
Values: analysis/demo2_pilots/BF/freeze_values.json (tools/demo2_freeze_values.py).

## 0. Code fix made before any value was taken (24f9ed9)
- The BF0 replay passed the same BFCL checker name for every arm and mode. The checker caches
  tool-class instances by that name, so a later splice score ran on state an earlier call left.
- Found here: A2 vs A on byte-identical local logs gave splice dC_iso -0.0042 (base passes 6 vs 5).
- Fix: one name per call. A2 vs A is then exactly 0 on both hosts.
- No rule changed. The pre-fix outputs are kept (calib_replay_*_SCORER_REUSE_BUG.*) and used for nothing.

## 1. Closeout of what ran (all $0 except cal_cloud)
| item | run_id | result |
|---|---|---|
| T2S download 8B INT4 | - | ir_sha256 cd97874b... = pin configs/models/Qwen3-8B-int4-ov(.v2).yaml |
| T2S download 8B INT8 | - | eac56228... = pin Qwen3-8B-int8-ov.yaml |
| XPS download 8B INT4 | - | cd97874b... = pin |
| XPS download 4B INT8 | - | f7d323d0... = pin Qwen3-4B-int8-ov.yaml |
| T2S R0 readbacks | 5fe2d65d, 13cca5ff, d2a8c095, b13a7060 | verdict valid (a, u4, pl, u4pl) |
| T2S R1 i8 | 5a6583aa-9127-41f4-b233-0e61690ae443 | refined_largest_pass 16,384 (16,384: TTFT max 8.77 s, decode min 26.3 tok/s; 17,408: 9.36 s) |
| T2S R1 u4 | 7f36c923-2ad2-4862-8d3f-7018170c4897 | 16,384 (observation only) |
| XPS R0 readbacks | 9bcfdc51, 8efcb6dd | valid |
| XPS R1 i8 | dd2ecf51-abf1-42a9-94d6-ed9c863ef42e | 8,192 (8,192: 8.13 s, decode min 22.0; 9,216: 9.59 s): confirms 8,192 |
| XPS R1 u4 | ab95aeb2-3262-4423-a08f-051fd5aad43a | 8,192 (observation only) |
| T2S cal_a1 / a2 / c / p | 75fa4d1c / 000010fc / dd62e520 / c91bc055 | 30/30 each, head 145047f |
| XPS cal_a1 | 7cf9b18d | 30/30; window stopped after it (msedge), PF 12 |
| XPS cal_a2 / c / p / pp | 7d4e5117 / a200bd1a / f511af02 / 4407accc | 30/30 each, head 145047f |
| XPS cal_cloud | a403befe-deac-418e-82c9-db9260fa809b | 30/30, head 5212e66, plan.json cloud_price_formula present (PF 1 CostGuard); $2.4522 repriced = logged; 22/30 cloud pass |

- T2S R2-R6: not run. R3/R4 NOT_BUILT (registered).
- Every calibration cell: no watchdog fire, no hang, no resume, no generation error, no Traceback.

## 2. Values
### a. Budgets (PER_DEVICE section 1)
- **T2S B = 16,384**, from R1 i8 5a6583aa refined_largest_pass 16,384, at the 16,384 cap.
- **XPS B = 8,192** (registered), confirmed by R1 i8 dd2ecf51.
- u4 (7f36c923, ab95aeb2): 16,384 and 8,192. Observation only.

### b. P' rule (PF 6). Inputs: A = XPS cal_a1 7cf9b18d, P = XPS cal_p f511af02 (8B INT4), both 30/30
| criterion | P | A | holds |
|---|---|---|---|
| (a) local pass count | 3 | 4 | **yes** |
| (b) decode p5, nearest rank, 103 local turns | 16.09 tok/s (>= 12) | - | no |
| (c) entries with `ttft>10.0` | 0 | 0 | no |
- **P' is selected on the XPS (criterion a). PSPEC = `configs\models\Qwen3-4B-int8-ov.yaml`.**
- For reference only (not a rule input): cal_pp 4407accc, 4 local passes, decode p5 17.20 tok/s.

### c. A/A per host, cal_a1 vs cal_a2 (PF 8)
| host | runs | discordant entries | step-level decoded-call mismatch | t_generate p50 a1 / a2 (s) | d_lat | d_usd |
|---|---|---|---|---|---|---|
| T2S | 75fa4d1c vs 000010fc | 0 / 30 | 0 / 104 (0.0) | 3.175 / 3.161 | 0.0045 | 0.0 |
| XPS | 7cf9b18d vs 7d4e5117 | 0 / 30 | 0 / 104 (0.0) | 4.387 / 4.186 | 0.0471 | 0.0 |
- Step mismatch: steps aligned by (entry, turn, step index); a step in only one run counts as a
  mismatch; denominator = the union of positions.
- d_usd = |dC_iso(A2 vs A1)| / C_A1(Q*), from the calibration replay with cal_cloud, splice:
  dC_iso 0.0 on C_A1(Q*) $0.014442 per task.
- So PF 8's material-win thresholds stay at their floors on both hosts: $ 25%, latency 20%
  (2 x d_lat = 0.9% T2S, 9.4% XPS).
- Fact: the 4B INT4 decoded steps of cal_a1 are identical on the two hosts (30/30 entries).

### d. Calibration replay (pipeline check only; no threshold tuned on it)
Command per host (pin DEMO2_CALIB_30 sha256 28f00f54...; cloud a403befe; BFCL unpacked dir
apu_characterization/out/cap01/live_sources/bfcl-wheel/unpacked), head 24f9ed9:
`tools/demo2_iso_quality_replay.py --pin derived/h1_hybrid/demo2/DEMO2_CALIB_30.json --cloud <cal_cloud> --arm A=<cal_a1> --arm A2=<cal_a2> --arm C=<cal_c> --arm P=<cal_p> [--arm Pp=<cal_pp>] --contrast A2:A --host <H> --bfcl-unpacked <dir> --out analysis/demo2_pilots/BF/calib_replay_<H>.json`
- Refusals: none on either host. Every turn had passive_triggers. 0 derived triggers, 0 hangs,
  0 hang_prefix_missing, 0 LOW_MEM.
- Splice base points (passes / 30, $ per task):
  - T2S: A 6, $0.01444; A2 6, $0.01444; C 6, $0.01494; P 8, $0.01211.
  - XPS: A 6, $0.01444; A2 6; C 6, $0.01494; P 3, $0.00257; Pp 4, $0.00195.
- dC_iso (splice):
  - A2 vs A: 0.0 on both hosts.
  - C vs A: -0.00049 on both hosts.
  - T2S P vs A: +0.01444, CI [-0.00675, 0.02369].
  - XPS P vs A: -0.00166.
  - XPS Pp vs A: +0.00349.
- The replay's default primary on the XPS is the arm named P. For held-out, the XPS P cell runs
  PSPEC (P'), so P names the selected arm, as PF 2 says.

### e. Held-out cloud cap (PF 9)
- cal_cloud repriced total = $2.4522414 (a403befe).
- cap = max($20, 2 x 2.4522414 x 1.1 = $5.395) = **$20.00**.
- Not over $20, so no approval is needed. HO_CLOUD_CAP=20, without HO_CLOUD_CAP_APPROVED.

### f. Live-run safety (PF 10) in cal_p / cal_pp and the other calibration cells
- Watchdog: no fire in any cell.
- Resume: none needed.
- PIPELINE_REBUILT: 0. Calibration ran without -RebuildOnError, which is new in held-out.
- LOW_MEM turns: 0 of 103 in every cell. Minimum available memory: XPS cal_p 3,706 MB, cal_pp
  4,729 MB, cal_a1 2,263 MB; T2S cal_p 50,255 MB.
- Largest local n_ctx: XPS cal_p 6,609, cal_pp 6,603; T2S cal_p 8,475.

## 3. Input sha256 (files under derived/demo2_pilots/windows_20261022/ unless a path is given)
Each h1 .sealed records the tree_sha256 of its run.
| file | sha256 |
|---|---|
| t2s_1012_be/cal_a1/.../75fa4d1c.../.sealed | c11f1bce9320a3ef632b4eedabe474e7a7a461d10aef4fa2fe7e1d4ab3109820 |
| t2s_1012_be/cal_a2/.../000010fc.../.sealed | 983b2ed17efbcb284f3a4221a0294b70db34e8c8a97dbe5aaa3408a34f9eb07f |
| t2s_1012_be/cal_c/.../dd62e520.../.sealed | 9d3efec216106d3178a03b5772314c5a503b806a0068be5e8000b2750e7ed814 |
| t2s_1012_be/cal_p/.../c91bc055.../.sealed | a10e3fcc88f1c0ed1dfa6f0ece62b65ddc83bf1242032da7c4aa929e34c7c406 |
| xps_1012_be/cal_a1/.../7cf9b18d.../.sealed | e5c35dcfc467a5a9c4dd56dd90feb7d231e581c24c6decb10e51eaa8fc863e82 |
| xps_1149_be/cal_a2/.../7d4e5117.../.sealed | 8fcdc08990fd2604a4f3e7c6b36906c0f362306571c0990236ecc458a0c21c49 |
| xps_1149_be/cal_c/.../a200bd1a.../.sealed | 276e46807824168326f621d23253194eb6ba0190cede96b8b9a08b9a3244b9d5 |
| xps_1149_be/cal_p/.../f511af02.../.sealed | 33c0d2dd1a37eb6412b03c7dc5ce45093690e66c6b6a870f2d840147e2b3f099 |
| xps_1149_be/cal_pp/.../4407accc.../.sealed | 8c8467df8192ce13b14287f841bcbbb2cd603a3edf299330539591b904e39160 |
| xps_1901_be_v2_cal_cloud/cal_cloud/.../a403befe.../.sealed | b33e253e905f8fbe6e2864526a1424206009cae813cf11cfcda89ddc474234ce |
| t2s_0654_ladder/r1_i8/.../5a6583aa.../run.json | f9165d4121072accd1c4b1c3d9816db747028da3c2799e5ad7d8c48e4c05cdfd |
| t2s_0654_ladder/r1_u4/.../7f36c923.../run.json | 04729e331b63f71e522fde8407f8f6feefe0918fb3d3548e6433ee54cc4967f5 |
| xps_0656_ladder/r1x_i8/.../dd2ecf51.../run.json | 85e70e57cfa365f6cd4ce609c15c1bc98b62a85b14d6802b4019a1dee2a5df0d |
| xps_0656_ladder/r1x_u4/.../ab95aeb2.../run.json | 82db125fbb2f8c5e5a0058df36f8da544e7b59e1c7a512159da221d9574129e9 |
| t2s_downloads_20261021_2215/Qwen3-8B-int4-ov.host_spec.yaml | 648b20f44e3efe6232bc52620ad7cdd92b78dacd8d16ac18f5c4e63a42c1c47e |
| t2s_downloads_20261021_2215/Qwen3-8B-int8-ov.host_spec.yaml | 25c5970881e7868465d01e134321038b2af41502208aeeba5ee2b38a3b75caf5 |
| xps_downloads_0653/Qwen3-4B-int8-ov.host_spec.yaml | a2b3a659d06f924070ff45e93c56a148327790b30d0938a0c23477026afe2d8a |
| xps_downloads_0653/Qwen3-8B-int4-ov.host_spec.yaml | b085f0e5bda17309fc215ad64e8c39fc072816a57aacddce4dea0071c65b73ee |
| analysis/demo2_pilots/BF/freeze_values.json | 32085e1c75dbb9409d448680ff628a4e277a1cc8b5beea195162488e1c5ccdb1 |
| analysis/demo2_pilots/BF/calib_replay_T2S.json | b4cec3bc39b35f33156d9dbb023b9dd1d725151064122cbf479681f097653584 |
| analysis/demo2_pilots/BF/calib_replay_XPS.json | b70623eb2506221a21b31eb250f8c389b4f50eb51943a2e35095e249a4c20b39 |
| derived/h1_hybrid/demo2/DEMO2_CALIB_30.json | 28f00f54080eee8a909dcfe562057e71411c9795afc84b4d9d5a8b38fbc5d895 |

## 4. Frozen queue scripts (outside the repo, ~/seam_closeout_1013/xps/)
| file | sha256 |
|---|---|
| run_queue_t2s_be_v2.sh | 2d69f1fc1c8dfb41ed3a5bf1801ca1aaff37b5f153ede46190c39828182beb66 |
| run_queue_xps_be_v2.sh | 13a88cdf45272338166c75edab55dad83c32b6d65f5dfa062269f161cbd97a5a |
| park_untracked_v2.ps1 | e2bf2fc643bff81cea2db594d5bdcdd68e020fdfe91a9386eb0e54464f4d63c7 |
- These equal the BF0 hashes and the hashes the cal_cloud window logged (steps.txt).
- A different hash at launch (each window logs its own in steps.txt) is a deviation.

## 5. Held-out commands (values filled; <freeze head> = the commit that adds this file)
- `! git -C ~/gnn-hls-accel push origin main`
- T2S:
  `REBOOT_OK=1 BUDGET=16384 HELDOUT_OK=1 EXPECT_HEAD=<freeze head> FROM=30 bash ~/seam_closeout_1013/xps/run_queue_t2s_be_v2.sh`
- XPS:
  `HELDOUT_OK=1 EXPECT_HEAD=<freeze head> PSPEC='configs\models\Qwen3-4B-int8-ov.yaml' HO_CLOUD_CAP=20 FROM=30 bash ~/seam_closeout_1013/xps/run_queue_xps_be_v2.sh`
- Both scripts caffeinate themselves. Keep the Mac lid open and on AC.
- Do not run the T2S ladder R2 alongside held-out.
- The freeze head must be the tip of origin/main when the hosts pull: both scripts stop if the
  host head after `git pull --ff-only` is not EXPECT_HEAD.
