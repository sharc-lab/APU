# evo-t2s week plan (machine back about 2026-10-15)

Status: staged, nothing run. evo-t2s is off limits (no SSH, no scp, no queue changes) until the operator says so.
Every command below that touches evo-t2s is for that moment, not before. No evo-x2 time is used by this plan.

Files:
- `queues/t2s_week_queue.json`: the queue, in `harness/t2s_queue.py` item format, built by
  `scripts/t2s_week_queue.py build` (a test fails if the file and the builder ever differ).
- `scripts/t2s_week_queue.py`: `build`, `dry-run` (fakes only), `inputs` (recomputes every estimate input from
  `results/`), `load` (run on evo-t2s).
- `harness/t2s_week_preflight.py`: queue job 1, the read-only start check every later job is gated on.
- `harness/t2s_r2_agent.py`: host-neutral evo-t2s wrapper around `harness/x2_r2_agent.py` (no edits to that file).
- `harness/t2s_outcome_table.py`: extended with `--models`, `--subset-n`, `--subset-seed`, `--canary-gate` and a
  queue exit; the plain invocation behaves as before.
- `harness/t2s_night2.py`: new opt-in phase `px2i` (Intel bandwidth hog).
- `harness/argparse_probe.py`: parses a harness's command line without running it (dry run, capability check).

## Plan

Order is priority order: a cut at the end loses the least valuable work. Hours: "measured" means the same job shape
was timed on evo-t2s; "scaled" means an evo-x2 timing times a measured T2S/X2 ratio; "nominal" means no timing exists.
Every input is a number recomputed from a committed result file (`py -3.12 scripts/t2s_week_queue.py inputs`; the test
`test_estimate_inputs_match_the_result_files` checks them within 1%).

| # | step | queue id | hours | basis | cumulative |
|---|---|---|---|---|---|
| 0 | access restored, handover file confirmed, re-enable checks (operator, by hand) | none | (operator) | | |
| 1 | start check: host, handover file, versions, models, files | t2s_wk_preflight | 0.1 | nominal | 0.1 |
| 2a | R2 validation + both controls, Ollama default (CPU, 4096) | t2s_wk_r2_val_cpu | 5.1 | scaled | 5.2 |
| 2b | R2 validation + both controls, OLLAMA_IGPU_ENABLE=1 (iGPU, 32768) | t2s_wk_r2_val_igpu | 1.8 | scaled | 6.9 |
| 2c | R2 native, CPU 4096 default, 2 models x 5 seeds x 40 turns | t2s_wk_r2_real_cpu | 15.1 | scaled | 22.0 |
| 2d | R2 native, iGPU 32768 default, 2 models x 5 seeds x 40 turns | t2s_wk_r2_real_igpu | 35.4 | scaled | 57.4 |
| 3 | R2 mitigation (client-side trim) at the native 4096 | t2s_wk_r2_mitigation_cpu | 12.0 | scaled | 69.4 |
| 4 | Intel bandwidth hog PX2I, N0/B4/e8/N1, Level Zero | t2s_wk_px2i | 1.1 | scaled | 70.5 |
| 5 | outcome-table subset, 100 items, 2 models, 3 configs | t2s_wk_outcome_subset | 34.3 | measured | 104.8 |
| 6 | R2 mechanism logging at the 4096 default | t2s_wk_r2_mechanism_cpu | 1.4 | scaled | 106.2 |

**Total 106.2 h (4.4 days) against the 72 h (3 day) budget line.** Steps 1 to 4 fit (70.5 h); steps 5 and 6 are
past the line (`beyond_budget: true` in the queue file). See "Budget and cut options".

How each estimate is formed (inputs and sources in `scripts/t2s_week_queue.py` ESTIMATE_INPUTS):
- CPU factor 9.45 = evo-t2s CPU seconds per call (49.27 s, median latency of HTTP-200 `ollama_default` rows in
  `results/t2s_outcome_table_full.jsonl`, 4096 window, 2050 tokens processed) / evo-x2 seconds per call in a 4096
  session (6.95 min median llama3.1:8b `ollama_ctx_4096` session in `results/x2_r2_real_v1.jsonl`, 80 calls). Low
  confidence: the T2S calls were single long prompts, R2 calls are a growing transcript.
- iGPU ratio 3.31 = median per-item latency T2S `ollama_igpu_enable` / X2 `ollama_default`, llama3.1:8b, same items,
  `t2s_outcome_table_full.jsonl` vs `x2_outcome_table_v3.jsonl`.
- Vulkan ratio 1.99 = the same for T2S `llama_server_vulkan` / X2 `llama_server` (the cleanest like-for-like pair).
- 2a/2b: `x2_r2_validation_v2.jsonl` file span 32.1 min (both models, all validation arms) x CPU factor / iGPU ratio.
- 2c: 5 seeds x (llama3.1:8b 6.95 + qwen3:14b 12.24 min, X2 4096 medians) x CPU factor.
- 2d: 5 seeds x (22.73 + 105.57 min, X2 `ollama_ctx_32768` medians; the T2S iGPU default is 32768) x iGPU ratio.
- 3: 3 seeds x the 2c per-session CPU times x 1.32 (X2 mechanism 4096 session 9.17 min over the real 4096 session 6.95
  min: the render, tokenize and debug-log overhead the mitigation job shares with the mechanism job).
- 4: X2 PX2 time per condition (qwen3-8b 32.77 min and qwen3-14b 35.99 min for 9 conditions,
  `t2s_night2_20260930T135145Z.jsonl`) x 4 conditions x Vulkan ratio, plus 0.1 h smoke and calibration.
- 5: 9.98 min per item for llama3.1:8b over all three configs including server starts (`t2s_outcome_table_full.jsonl`,
  file span / 40 items) x 100 items x 2 models, plus 1 h for the canaries. qwen3-8b is assumed equal to llama3.1:8b
  (not measured under these three configs on evo-t2s).
- 6: one llama3.1:8b CPU 4096 session (66 min) x 1.32.

## Step 0: access restored (operator, by hand, only after the operator says SSH is allowed)

Do these in order. Nothing is queued or enabled until all pass.

1. **Handover file.** `scripts/t2s_handover.py` (the 2026-10-06 handover to Zach) writes
   `C:\apu\ovn\HANDOVER_DONE.json` as its last step (stopped pids, registry check, watchdog state, process list).
   Confirm it exists and read it before anything else:
   `C:\Windows\System32\OpenSSH\ssh.exe sharc@100.72.40.24 "hostname; Get-Content -Raw C:\apu\ovn\HANDOVER_DONE.json"`
   Expected: `EVO-T2S`, a JSON record with the watchdog "Disabled". If the file is missing the handover never finished
   and the machine state is unknown: stop and ask the operator. The preflight job re-checks this file.
2. **Nobody logged in.** `query user` must say "No User exists". `host_config.HOSTS["EVO-T2S"]["interactive_guard"]`
   is True (asserted in `tests/test_host_config_interactive_guard.py` and `tests/test_t2s_week_queue.py`), so every
   queued job also refuses on its own if Zach is logged in.
3. **Queue state and maintenance lock.** `Get-Content C:\apu\ovn\queue_state.json`: no entry may be "running".
   `Test-Path C:\apu\ovn\queue_pause.flag` must be False; if True, read its reason and ask the operator, do not clear
   it blindly. `queue_empty.flag` may exist (it is cleared by the next launch).
4. **Stray processes.** `Get-Process ollama,'ollama app',llama-server -ErrorAction SilentlyContinue | Select Id,Path`
   must be empty. If not, they are ours only if the path is under C:\apu or Ollama's own install dir
   (`stale_server_cleanup.is_ours_shared_machine`); the queue's pre-launch cleanup (`t2s_queue.launch_next`) sweeps them
   before every launch, and `stop_ollama_server` also kills Ollama's model runners since cd5c5bf.
5. **Windows Update.** Read only, never reboot without the operator:
   `Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending'`,
   `Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired'`,
   `Get-HotFix | Sort-Object InstalledOn -Descending | Select-Object -First 3`. A pending reboot is reported to the
   operator (a reboot mid-week kills the running job; the watchdog would mark it crashed and move on). The preflight
   job records the same two keys as a warning.
6. **Versions** (next section). Ollama must be 0.34.4 and llama.cpp b10970 before anything is queued.
7. **SYSTEM resolution of the Ollama exe and models dir** (0b1ab71, 992f6c8). The watchdog runs as SYSTEM, whose
   LOCALAPPDATA is the system profile, so `host_config._this_host_entry()` resolves the per-user paths from the
   HOSTS entry (user `sharc`): `C:\Users\sharc\AppData\Local\Programs\Ollama\ollama.exe` and
   `C:\Users\sharc\.ollama\models`. Check both exist:
   `Test-Path C:\Users\sharc\AppData\Local\Programs\Ollama\ollama.exe; Test-Path C:\Users\sharc\.ollama\models\manifests`.
   With the side-by-side 0.34.4 pinned (below), `OLLAMA_BIN` takes precedence over the HOSTS path in
   `_resolve_ollama_exe_for_serve`, and the models dir stays the HOSTS one.
8. **Deploy** the committed files (byte-exact, `scripts/deploy_evo.py`, refuses uncommitted files and a wrong host):
   ```
   py -3.12 scripts/deploy_evo.py C:\apu\ovn harness/t2s_queue.py harness/queue_watchdog.py harness/stale_server_cleanup.py harness/host_config.py harness/proc_util.py harness/argparse_probe.py harness/t2s_week_preflight.py harness/t2s_r2_agent.py harness/x2_r2_agent.py harness/t2s_r2_session_growth.py harness/x2_r2_mechanism.py harness/prompt_token_check.py harness/t2s_k1_ollama.py harness/t2s_outcome_table.py harness/chat_template_source.py harness/t2s_night2.py harness/t2s_lab.py harness/t2s_overnight.py harness/t2s_amech.py harness/level_zero_sysman.py harness/win_cpu_topology.py harness/bw_hog.py harness/spin_hog_affinity.py harness/t2s_m3_power_coupling.py harness/run_provenance.py harness/server_guard.py scripts/t2s_week_queue.py queues/t2s_week_queue.json --host evo-t2s
   ```
   `C:\apu\ovn\analysis\trace_weighted_pack.py` and `C:\apu\ovn\results\workload_pack\` (grade.py, items/, the weights
   file) are the flat-layout copies the earlier T2S outcome run used; the preflight checks they are present. If any is
   missing, copy it from the repo into the same relative path.
9. **Sync.** `scripts/sync_results.py` includes evo-t2s again only when the operator says so: until then run it as
   `--host evo-x2`. Note (found while writing this plan, read-only): the controller's `APU-SyncResults` scheduled task
   currently runs `sync_results.py --host both`, so it already tries evo-t2s every 2 h. That is for the operator to
   decide; this plan changes no scheduled task.
10. **Load the queue, then the watchdog.** The load (one command, from the controller):
   ```
   C:\Windows\System32\OpenSSH\ssh.exe sharc@100.72.40.24 "C:\Users\sharc\AppData\Local\Programs\Python\Python312\python.exe C:\apu\ovn\t2s_week_queue.py load C:\apu\ovn\t2s_week_queue.json"
   ```
   It appends the 9 entries as pending after whatever is in `queue_state.json`, refuses (writes nothing) if any
   `t2s_wk_` id is already there, and never edits an existing entry. Then re-enable (not re-register) the watchdog
   that the handover disabled: `Enable-ScheduledTask -TaskName APU-QueueWatchdog`. Its next tick (at most 10 min)
   launches `t2s_wk_preflight`; every job's own exit advances to the next. No pause flag is set at any point, so
   nothing waits on a person.

## Versions: pin Ollama 0.34.4 and llama.cpp b10970

Target, from committed files: Ollama **0.34.4** on evo-x2 (`results/x2_r2_mechanism.jsonl` and
`results/x2_r2_validation_v2b.jsonl`, the `runtime` record's `ollama_version`; `docs/X2_CHANGELOG.md` winget install).
llama.cpp **b10970-bfdc32183**: the `llama_build` of every evo-t2s and evo-x2 row that records one, binaries in
`C:\apu\bin\llama-b10970\` on both machines.

evo-t2s's only recorded Ollama version is **0.33.2** (`docs/T2S_CHANGELOG.md`, 2026-09-29 winget install); no evo-t2s
result row records an Ollama version at all. So expect a mismatch.

Verification (read only):
```
& "C:\Users\sharc\AppData\Local\Programs\Ollama\ollama.exe" --version
& "C:\apu\bin\llama-b10970\llama-server.exe" --version
& "C:\apu\bin\llama-b10970\llama-tokenize.exe" --version
Get-FileHash C:\apu\bin\llama-b10970\llama-server.exe -Algorithm SHA256
```
Expected: `0.34.4`; `version: 10970 (bfdc3218)` for both llama.cpp tools.

If evo-t2s has a different Ollama version: install 0.34.4 **side by side, never replacing the existing install**
(the existing install is Zach's machine's state, and replacing it silently would change every earlier evo-t2s row's
runtime without a record). Use the release's standalone Windows zip (`ollama-windows-amd64.zip` on the v0.34.4 GitHub
release page; confirm the asset name there), not the installer, which upgrades the per-user install in place:
```
New-Item -ItemType Directory -Force C:\apu\bin\ollama-0.34.4
Expand-Archive <downloaded zip> -DestinationPath C:\apu\bin\ollama-0.34.4
& C:\apu\bin\ollama-0.34.4\ollama.exe --version
Get-FileHash <downloaded zip>, C:\apu\bin\ollama-0.34.4\ollama.exe -Algorithm SHA256
Set-Content -Encoding ascii C:\apu\ovn\ollama_pin.json '{"ollama_bin": "C:\\apu\\bin\\ollama-0.34.4\\ollama.exe", "version": "0.34.4"}'
```
`t2s_week_preflight.apply_ollama_pin()` reads the pin file and sets `OLLAMA_BIN`, which
`host_config._resolve_ollama_exe_for_serve()` checks first; `t2s_r2_agent.py` and `t2s_outcome_table.py` (subset path)
apply it before any server start, and the preflight job verifies the pinned exe reports 0.34.4. The models store is
shared (`C:\Users\sharc\.ollama\models`), so nothing is re-pulled. Its runner (`lib\ollama\llama-server.exe` under the
pin dir) still matches `stop_ollama_server`'s `*\Ollama\*` path filter. If llama.cpp is not b10970: stop and ask the
operator (every T2S llama-server row so far is b10970; a different build is a different runtime).

Record the install in `docs/T2S_CHANGELOG.md`, in its own format:
```
## 2026-10-1x -- Ollama 0.34.4 installed side by side (C:\apu\bin\ollama-0.34.4)

**What:** the v0.34.4 Windows zip extracted to C:\apu\bin\ollama-0.34.4 (zip SHA-256 <...>, ollama.exe SHA-256 <...>),
and C:\apu\ovn\ollama_pin.json pointing the queued jobs at it. The per-user install
(C:\Users\sharc\AppData\Local\Programs\Ollama, 0.33.2) is untouched.
**Why:** every evo-x2 R2 and outcome-table row was measured with Ollama 0.34.4; the T2S week (docs/T2S_WEEK_PLAN.md)
must run the same version.
**Revert:** Remove-Item -Recurse -Force C:\apu\bin\ollama-0.34.4, C:\apu\ovn\ollama_pin.json
```

## Changes needed in harness/x2_r2_agent.py

`harness/x2_r2_agent.py` (as of 62c0f8b) cannot run on evo-t2s. Not changed here; these are the exact changes for its
owner. Until they are on main and deployed, every R2 entry of the queue refuses with a `stopped:` note that names what
is missing (`t2s_r2_agent.missing_capabilities`), the queue marks it error and moves on, and the entries gated on it
stay pending. The dry run lists them as `blocked_until_x2_r2_agent_changes`.

1. **Host guard** (main, the `if socket.gethostname().upper() != "EVO-X2": raise RuntimeError(... "runs on EVO-X2
   only" ...)` line): replace with `host_cfg = hc.require_host(socket.gethostname())` and
   `hc.enforce_or_record_interactive_session(host_cfg)` (refuses on evo-t2s when someone is logged in, records console
   state on evo-x2), and add `"hw_id": host_cfg["hw_id"]` to the run_start record. The wrapper detects this change by
   the absence of the literal "runs on EVO-X2 only" in `main`'s source.
2. **`--server-env KEY=VALUE`** (repeatable, `action="append"`): merged into `server_env` (with `OLLAMA_DEBUG` for the
   mechanism mode), passed to `_start_server(hc, server_env, server_log)` and set on `runtime.server_env`, so
   `OllamaRuntime.recover()` restarts the server with the same environment (without this, a recovery restart would
   silently fall back from the iGPU to the CPU). Call `hc.stop_ollama_server()` before the first start whenever
   server_env is non-empty (start_ollama_server is a no-op while any Ollama runs), and record server_env in the
   `runtime` record (it already is for the mechanism job).
3. **`--tiers a,b`** (comma-separated base arm ids from `_BASE_ARMS`): passed as `tiers=` to `strong_plan` and
   `mechanism_plan` (and to the mitigation plan once merged). Validate every name against `_BASE_ARMS`.
4. **`--seeds s1,s2`** (comma-separated ints): passed as `seeds=` to `strong_plan` (and the mitigation plan). Not used
   by the queue as built; it is what the cut options below need.
5. **x2_r2_mitigation_v1** (another agent's job, not on main as of this plan): the T2S mitigation entry uses its
   documented flags from the pre-registration (`--mode mitigation --client-trim margin=0.05`, docs/FINDINGS.md
   "PRE-REGISTRATION: R2 mitigation x2_r2_mitigation_v1") plus `--tiers ollama_ctx_4096`. **The entry depends on the
   x2_r2_mitigation_v1 merge**; if its final flag names differ, update `job_specs()` in `scripts/t2s_week_queue.py`,
   rebuild, and the dry run checks the new argv against the merged parser.

`harness/t2s_r2_agent.py` already forwards `--server-env OLLAMA_IGPU_ENABLE=1` (igpu_enable), `--tiers` and `--seeds`,
calls `x2_r2_agent.main(argv, advance=False)` and advances the queue once itself.

## Step 2: R2 native on Intel

Harness `harness/x2_r2_agent.py` through `harness/t2s_r2_agent.py`. Models llama3.1:8b and qwen3:14b. v1 protocol
(`--call2-tools off`, the operator decision of 2026-10-08 for every Ollama run). Two runtime configurations:
- **cpu_default**: stock Ollama. On evo-t2s Ollama drops the Arc iGPU by policy, runs on the CPU, and sizes the
  default context at 4096 (docs/FINDINGS.md, the `vram-based default context ... default_num_ctx=4096` log line).
- **igpu_enable**: `OLLAMA_IGPU_ENABLE=1` in this job's own server environment only (never persisted); measured
  default context 32768 for llama3.1:8b (FINDINGS, the OLLAMA_IGPU_ENABLE table). qwen3:14b's iGPU default has not
  been measured; the wrapper records it.

Per configuration, validation first: `--mode validation` runs, per model, the negative control and baseline (arm b,
num_ctx 131072, 3 seeds x 10 turns), the positive control (num_ctx 8192, 1 seed x 15 turns) and the diagnostic arm
(`validation_plan("off")`). Then the real run, gated on that validation entry: `--mode real --plan strong --tiers
ollama_default` (the native default, no num_ctx sent), `SEEDS_STRONG` (5 seeds), 40 turns, `--order seed_major` (an
interrupted run leaves every model with as many seeds as possible), `--rules-from` the same configuration's validation
file with `--require-validation-gates --per-model-refusal` (a model whose own controls failed is refused, the other
still runs). After the run the wrapper appends a `t2s_runtime_check` row: loaded context per model on the
ollama_default turns against 4096 (cpu) or 32768 (iGPU); a mismatch goes into the exit note, rows are never altered.

Outputs: `results/t2s_r2_validation_cpu_v1.jsonl`, `results/t2s_r2_validation_igpu_v1.jsonl`,
`results/t2s_r2_real_cpu_v1.jsonl`, `results/t2s_r2_real_igpu_v1.jsonl`.

## Step 3: mitigation at the native 4096

Same design as x2_r2_mitigation_v1 (pre-registration in docs/FINDINGS.md): client-side trim to
floor(num_ctx x 0.95) - 384 prompt tokens, render-and-count with the model's own tokenizer (`llama-tokenize`, b10970,
`harness/prompt_token_check.py`), OLLAMA_DEBUG=1 per-call log parsing, llama3.1:8b and qwen3:14b, 3 seeds, 40 turns,
v1 protocol, rules in use from this machine's own CPU validation file. evo-t2s's native default is 4096 on the CPU, so
the tier is `ollama_ctx_4096` with `cpu_default` (an explicit num_ctx equal to the native default, which the trim
budget needs). 8192 is not run on evo-t2s (it is not a native default here). Output
`results/t2s_r2_mitigation_cpu_v1.jsonl`. Depends on the x2_r2_mitigation_v1 merge.

## Step 4: Intel bandwidth hog (PX2I)

PX2 itself refuses evo-t2s (`px2_topology` raises on shared-L2 E-core clusters, by design). New opt-in phase `px2i` in
`harness/t2s_night2.py` (`--phases px2i`), using the core map read live on evo-t2s (2026-09-28):
- N0: no co-runner; B4: STREAM-triad bandwidth hog (`harness/bw_hog.py`) on E-cluster A (CPUs 4-7); e8: spin hog on
  all 8 E-cores (4-11, the Intel B4 phase's e8 mask 0x0FF0); N1: no co-runner (N1 vs N0 is the drift check).
- qwen3-8b and qwen3-14b; B4/e8 in a per-model shuffled order (seed recorded); server pinned to the P-cores 0-3 in
  every condition; ctx 8192, fill 2048, 128 output tokens, 1 warm-up + 5 measured calls per condition (PX2's
  constants); one 45 s solo bandwidth calibration on the B4 cores.
- Level Zero Sysman (`harness/level_zero_sysman.py` via `t2s_lab.Telemetry`) per call (igpu_mhz,
  igpu_throttle_bits, pkg_power_w), plus a `px2i_level_zero` record per condition with medians over that condition's
  own call rows and whether Sysman was available.
- Smoke gated (`SMOKE_GATED_PHASES`), like every new night2 phase. Rows are an evo-t2s arm, never pooled with PX2.
Open design point for the operator: the listed conditions contrast bandwidth on 4 cores (B4) with spin on 8 cores
(e8). A matched-core spin condition (spin on CPUs 4-7) would isolate bandwidth from power at equal core count, as S4
vs B4 does in PX2; it adds about 0.25 h (one condition, two models) if wanted.

## Step 5: outcome-table subset

`harness/t2s_outcome_table.py --models llama3.1:8b,qwen3-8b --subset-n 100 --subset-seed 20261007 --canary-gate
--deadline-h 36`, output `results/t2s_outcome_subset_v1.jsonl`.
- Subset: `trace_weighted_subset`, a copy of `x2_outcome_table.qwen32b_subset` (Efraimidis-Spirakis weighted sampling
  without replacement over the trace weights, fixed seed). Seed 20261007 is `QWEN32B_SUBSET_SEED`, so it is the same
  100 items as evo-x2's qwen3-32b subset; `tests/test_t2s_outcome_subset.py` reproduces the recorded
  `qwen32b_subset` record of `results/x2_outcome_table_v3.jsonl` exactly. Recorded once as a `t2s_subset` record.
- Three configurations per model: ollama_default (CPU, 4096), ollama_igpu_enable, llama_server_vulkan (b10970,
  `-c` sized to the item). Item-major: each item runs every model under every config.
- Canary gate per (model, config): 5 shortest gsm8k + 5 shortest function_calling items first; error rate above 10% or
  mean score below 0.5 halts that (model, config) for the run with an ALERT record (x2_outcome_table's rule, parity
  tested).
- GSM8K: the #### fix and the lenient column (5fff514) apply to every row (`scored_fields`, SCORER_VERSION 2,
  score_strict / format_ok / score_lenient); `test_queued_outcome_job_rows_use_the_fixed_gsm8k_scorer_and_lenient_column`
  runs the real Ollama row path for both queued models with a fake server and checks a `#### 42` answer scores 1.0
  with the lenient column present. Note: the 100-item subset itself has no gsm8k item (67 longdoc, 33 trace_length_mix,
  from the recorded subset), so gsm8k scoring reaches only the canary rows.
- Thinking off: Ollama `"think": false`; llama-server for the subset path starts with `--reasoning-budget 0` and sends
  `chat_template_kwargs.enable_thinking=false` (x2_outcome_table's mechanism); rows record thinking_setting,
  thinking_leak and reasoning_chars.
- Chat template source and SHA-256 on every row (`harness/chat_template_source.py`).
- Known risk from the last T2S run (`t2s_outcome_table_full.jsonl`): 18 of 40 ollama_default rows and 19 of 40
  ollama_igpu_enable rows were HTTP 500 allocation failures, and only 20 of 40 items got a Vulkan row. The canary gate
  will stop a configuration that fails the same way instead of filling the file with errors.

## Step 6: mechanism logging (if time remains)

`--mode mechanism --tiers ollama_default` with `cpu_default`: one llama3.1:8b 40-turn session at the native 4096
default under OLLAMA_DEBUG=1, per-call log slices and render checks (`harness/x2_r2_mechanism.py`). Output
`results/t2s_r2_mechanism_cpu_v1.jsonl`.

## Queue file

`queues/t2s_week_queue.json`: 9 entries, ids `t2s_wk_*`, status pending, `gate.requires_done` (t2s_queue's gate)
as in the plan table: every job waits for `t2s_wk_preflight`; the real and mitigation runs wait for their own
configuration's validation entry. Each entry's `note` carries its estimate, basis and cumulative hours (overwritten by
the job's exit note when it finishes); `est_hours`, `est_basis`, `cum_hours`, `beyond_budget`, `expected_out` and
`depends_on_changes` stay. Interpreter `C:\Users\sharc\AppData\Local\Programs\Python\Python312\python.exe`, an absolute
path, because the watchdog launches as SYSTEM where `%USERPROFILE%` is the system profile.

Every job calls `t2s_queue.advance()` exactly once on exit, on every path, verified by test:
- `t2s_week_preflight.py`: `test_preflight_advances_once`, `test_preflight_fails_on_a_different_ollama_version_and_still_advances_once`
- `t2s_r2_agent.py`: `test_main_advances_exactly_once_*` (success, runtime mismatch, missing capabilities, wrong host,
  logged-in user; the inner x2_r2_agent.main is always called with advance=False)
- `t2s_outcome_table.py`: `test_main_advances_once_when_queued` (success and exception; no advance when not queued)
- `t2s_night2.py`: `test_night2_main_advances_once` (one `tq.advance(note)`, in the finally block)

Dry run (`py -3.12 scripts/t2s_week_queue.py dry-run`, no SSH, no machine): for every entry it checks the id prefix,
that the gate is an earlier entry, the evo-t2s interpreter, a flat script name that exists in `harness/` and in the
preflight's deployed-file list, no evo-x2 user/host in the command, `t2s_`-prefixed outputs that collide with no
`results/x2_*` name and do not already exist, and parses the argv with the harness's own parser
(`harness/argparse_probe.py`; for R2 entries the forwarded argv against the real x2_r2_agent parser, with the flags it
does not have yet reported as pending), night2 phase names, outcome-table model keys; then the total hours.
Result at this commit: all 9 entries pass, total 106.2 h, 70.5 h within the 72 h line; the 6 R2 entries are
reported blocked until the x2_r2_agent changes above.

## Budget and cut options

The full design is 106.2 h; the 72 h line falls inside step 5. Options, hours saved (same estimate inputs):
- iGPU real run, qwen3:14b at 3 seeds instead of 5 (needs `--seeds`, change 4): saves 11.6 h.
- Outcome subset with llama3.1:8b only: saves 16.6 h (or 50 items for both models: the same 16.6 h).
- CPU real run, qwen3:14b at 3 seeds: saves 3.9 h.
- Move step 6 (1.4 h) ahead of step 5 so it lands inside the line.
The CPU estimates (2a, 2c, 3, 6) use one measured CPU per-call time and are the least certain; the first CPU
validation entry's own wall clock should replace them before deciding a cut.

## Open questions for the operator

1. Which cut (above), if any, for a 3-day window.
2. Ollama on evo-t2s: confirm the side-by-side 0.34.4 install (and the zip asset) before anything is queued.
3. The controller's APU-SyncResults task runs `--host both` today, so it already tries evo-t2s every 2 h; keep, or
   change it to `--host evo-x2` until the machine is back?
4. PX2I: add a matched-core spin condition (CPUs 4-7) next to B4? About 0.25 h.
5. The outcome subset has no gsm8k items; is that acceptable, or should the subset be stratified to include some?
6. qwen3:14b at num_ctx 131072 (negative control) on the CPU and on the iGPU: about 20 GB of KV cache at f16 on a
   63.5 GB machine; the last T2S outcome run saw allocation failures at far smaller sizes. If the validation entry
   fails on memory, is a smaller negative-control context acceptable for evo-t2s?
