# evo-t2s week plan (machine back about 2026-10-15)

Status (2026-10-15): RUNNING in the window 2026-10-15 00:00 ET to 2026-10-18 23:59 ET; see "Window 2026-10-15" at
the end for what was done at the start and the handover deadline. No evo-x2 time is used by this plan.
`<profile>` below is the evo-t2s user's profile directory (C: drive, Users, then `HOSTS["EVO-T2S"]["user"]` from
`harness/host_config.py`); `scripts/t2s_week_queue.host_python()` builds the interpreter path from it.

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
| 5 | R2 mechanism logging at the 4096 default | t2s_wk_r2_mechanism_cpu | 1.4 | scaled | 72.0 (71.96 exact) |
| deferred | outcome-table subset, 100 items, 2 models, 3 configs | (not queued) | 34.3 | measured | |

**Operator cut (2026-10-08): steps 1 to 4 plus the mechanism run, 72.0 h (71.96 exact, from the queue builder) of the 72 h line.** The outcome-table subset
is DEFERRED (reason: the evo-x2 outcome table covers quality; the T2S subset would add a second platform's quality
numbers, not a new finding). It is not in the queue file; its job and harness path stay in the repo for later.

**Trim rule if the measured hours exceed 72 once on the machine** (decided by the operator, applied in this order,
re-estimating from the first jobs' own wall clocks after step 2a): trim the lowest-value R2 tier first, never the
mechanism run. The lowest-value tier is the iGPU 32768 real run (2d): it repeats a pattern already measured on evo-x2
(history lost silently, rules kept), while the CPU 4096 tier (2c) is the one that tests rule loss on Intel. So:
(1) 2d qwen3:14b to 3 seeds (`--seeds`), (2) 2d llama3.1:8b to 3 seeds, (3) drop 2d qwen3:14b, and only then
(4) 2c qwen3:14b to 3 seeds. Steps 2a, 2b, 3, 4 and 5 (mechanism) are not trimmed.

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
   HOSTS entry (user `sharc`): `<profile>\AppData\Local\Programs\Ollama\ollama.exe` and
   `<profile>\.ollama\models`. Check both exist:
   `Test-Path <profile>\AppData\Local\Programs\Ollama\ollama.exe; Test-Path <profile>\.ollama\models\manifests`.
   With the side-by-side 0.34.4 pinned (below), `OLLAMA_BIN` takes precedence over the HOSTS path in
   `_resolve_ollama_exe_for_serve`, and the models dir stays the HOSTS one.
8. **Deploy** the committed files (byte-exact, `scripts/deploy_evo.py`, refuses uncommitted files and a wrong host):
   ```
   py -3.12 scripts/deploy_evo.py C:\apu\ovn harness/t2s_queue.py harness/queue_watchdog.py harness/stale_server_cleanup.py harness/host_config.py harness/proc_util.py harness/argparse_probe.py harness/t2s_week_preflight.py harness/t2s_r2_agent.py harness/x2_r2_agent.py harness/t2s_r2_session_growth.py harness/x2_r2_mechanism.py harness/x2_r2_client_trim.py harness/prompt_token_check.py harness/t2s_k1_ollama.py harness/t2s_outcome_table.py harness/chat_template_source.py harness/t2s_night2.py harness/t2s_lab.py harness/t2s_overnight.py harness/t2s_amech.py harness/level_zero_sysman.py harness/win_cpu_topology.py harness/bw_hog.py harness/spin_hog_affinity.py harness/t2s_m3_power_coupling.py harness/run_provenance.py harness/server_guard.py scripts/t2s_week_queue.py queues/t2s_week_queue.json --host evo-t2s
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
   C:\Windows\System32\OpenSSH\ssh.exe sharc@100.72.40.24 "<profile>\AppData\Local\Programs\Python\Python312\python.exe C:\apu\ovn\t2s_week_queue.py load C:\apu\ovn\t2s_week_queue.json"
   ```
   It appends the 8 entries as pending after whatever is in `queue_state.json`, refuses (writes nothing) if any
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
& "<profile>\AppData\Local\Programs\Ollama\ollama.exe" --version
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
shared (`<profile>\.ollama\models`), so nothing is re-pulled. Its runner (`lib\ollama\llama-server.exe` under the
pin dir) still matches `stop_ollama_server`'s `*\Ollama\*` path filter. If llama.cpp is not b10970: stop and ask the
operator (every T2S llama-server row so far is b10970; a different build is a different runtime).

Record the install in `docs/T2S_CHANGELOG.md`, in its own format:
```
## 2026-10-1x -- Ollama 0.34.4 installed side by side (C:\apu\bin\ollama-0.34.4)

**What:** the v0.34.4 Windows zip extracted to C:\apu\bin\ollama-0.34.4 (zip SHA-256 <...>, ollama.exe SHA-256 <...>),
and C:\apu\ovn\ollama_pin.json pointing the queued jobs at it. The per-user install
(<profile>\AppData\Local\Programs\Ollama, 0.33.2) is untouched.
**Why:** every evo-x2 R2 and outcome-table row was measured with Ollama 0.34.4; the T2S week (docs/T2S_WEEK_PLAN.md)
must run the same version.
**Revert:** Remove-Item -Recurse -Force C:\apu\bin\ollama-0.34.4, C:\apu\ovn\ollama_pin.json
```

## Changes needed in harness/x2_r2_agent.py (DONE 2026-10-08, commit 7ce6df1; run_start also records interactive_session)

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
3. **`--tiers a,b`** (comma-separated base arm ids from `_BASE_ARMS`): passed as `tiers=` to `strong_plan`,
   `mechanism_plan` and `mitigation_plan` (all three already take a `tiers` parameter). Validate every name against
   `_BASE_ARMS`.
4. **`--seeds s1,s2`** (comma-separated ints): passed as `seeds=` to `strong_plan` and `mitigation_plan`. Not used
   by the queue as built; it is what the cut options below need.
5. **x2_r2_mitigation_v1** is on main (4b46765: `--mode mitigation --client-trim margin=0.05`, `MITIGATION_TIERS`
   4096 and 8192, `SEEDS`, both default models, `harness/x2_r2_client_trim.py`). The T2S mitigation entry reuses those
   flags as they are and adds only `--tiers ollama_ctx_4096` (change 3). Its tokenizer and models dir resolve through
   `prompt_token_check.TOKENIZE_EXE` (`C:\apu\bin\llama-b10970\llama-tokenize.exe`) and
   `prompt_token_check.default_models_dir()` (host_config's per-host store), both already correct on evo-t2s.
   The dry run parses the entry against the merged parser (`--mode mitigation` and `--client-trim` accepted); note
   the parser's post-parse `ap.error` checks (for example `--client-trim` only with mitigation) are not reached by the
   dry run, which stops at `parse_args`.

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
`results/t2s_r2_mitigation_cpu_v1.jsonl`. Uses the merged x2_r2_mitigation_v1 flags (4b46765); still needs changes 1 and 3.

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

`queues/t2s_week_queue.json`: 8 entries (outcome subset deferred, not queued), ids `t2s_wk_*`, status pending, `gate.requires_done` (t2s_queue's gate)
as in the plan table: every job waits for `t2s_wk_preflight`; the real and mitigation runs wait for their own
configuration's validation entry. Each entry's `note` carries its estimate, basis and cumulative hours (overwritten by
the job's exit note when it finishes); `est_hours`, `est_basis`, `cum_hours`, `beyond_budget`, `expected_out` and
`depends_on_changes` stay. Interpreter `<profile>\AppData\Local\Programs\Python\Python312\python.exe`, an absolute
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
Result after the operator cut and the x2_r2_agent changes (2026-10-08, commit 7ce6df1): all 8 entries pass, none
blocked, total 71.96 h (printed 72.0) within the 72 h line; `t2s_wk_outcome_subset` listed as deferred.

## Budget and cut options

**Superseded by the operator cut above (2026-10-08).** Kept for the record: the full design is 106.2 h; the 72 h line falls inside step 5. Options, hours saved (same estimate inputs):
- iGPU real run, qwen3:14b at 3 seeds instead of 5 (needs `--seeds`, change 4): saves 11.6 h.
- Outcome subset with llama3.1:8b only: saves 16.6 h (or 50 items for both models: the same 16.6 h).
- CPU real run, qwen3:14b at 3 seeds: saves 3.9 h.
- Move step 6 (1.4 h) ahead of step 5 so it lands inside the line.
The CPU estimates (2a, 2c, 3, 6) use one measured CPU per-call time and are the least certain; the first CPU
validation entry's own wall clock should replace them before deciding a cut.

## Optional entry: context-window override methods (staged 2026-10-13, operator plan v2 Part A5)

Not in `queues/t2s_week_queue.json` (the week's queue, estimates and dry run are unchanged). `py -3.12
scripts/t2s_week_queue.py build-optional` writes `queues/t2s_week_optional.json` with one entry,
`t2s_wk_window_override_methods`: `window_override_methods.py --out results/t2s_window_override_methods_v1.jsonl
--queue`, gated on `t2s_wk_preflight`, nominal 0.3 h. The same job as evo-x2's `x2_window_override_methods`
(docs/FINDINGS.md "Context-window override methods on Ollama 0.34.4"): llama3.1:8b on the pinned 0.34.4 (via
`ollama_pin.json`), its own server on port 11500, request num_ctx / Modelfile num_ctx on a derived tag (created and
deleted by the job) / OLLAMA_CONTEXT_LENGTH, persistence across restarts, and a read-only GPU memory record. The
interactive-session guard applies (refuses while anyone is logged in). To use it, only if time remains: deploy
`window_override_methods.py` flat to C:\apu\ovn and `python t2s_week_queue.py load t2s_week_optional.json` on evo-t2s.

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

## Window 2026-10-15 (what was done at the start)

Facts and changes: `results/t2s_window_20261015_preflight.json`, `docs/T2S_CHANGELOG.md` (2026-10-15 entries).
- Step 0 findings: the handover record is the 2026-10-02 one (not 10-06); the 2026-10-02 handover pause flag was still
  set and was replaced and cleared by the load below; Ollama per-user 0.33.2, so 0.34.4 was installed side by side and
  pinned as in "Versions" above; llama.cpp b10970 confirmed; **qwen3:14b was missing from the Ollama store**.
- Added job `t2s_wk_model_pull` (`harness/t2s_model_pull.py`, `queues/t2s_week_pulls.json`, nominal 0.5 h) ahead of
  the preflight. Load command used, one read-modify-write under the queue pause cleared in its finally block (refuses
  if an old entry is pending or running):
  `python t2s_week_queue.py load-window t2s_week_pulls.json t2s_week_queue.json`.
- Handover deadline (operator instruction 4): `C:\apu\ovn\queue_deadline.json`, armed with
  `python t2s_deadline.py arm --no-launch-after 2026-10-19T03:00:00Z --hard-deadline 2026-10-19T03:30:00Z`. After
  03:00Z (23:00 ET) nothing launches and R2 jobs stop at their next session boundary; a job still running is killed at
  the first watchdog tick after 03:20Z (so before 03:30Z); then the watchdog sets the pause "handover to Zach
  2026-10-18; resume only in the next confirmed window", stops our Ollama/llama-server/python and writes
  HANDOVER_DONE.json (`harness/t2s_deadline.py`, tests `tests/test_t2s_deadline.py`). Released only by the next
  window's operator (`t2s_queue.clear_pause()`).
- Trim rule: re-estimate from the first jobs' wall clocks after 2a; projected end at start was about 72.5 h after the
  first launch (2026-10-15T04:48Z), about 21 h before the deadline.

## Staged for 2026-10-26 to 2026-11-08 (pending Zach's confirmation; NOT loaded)

**Update 2026-10-22 (operator decision: per-cell policy sets, qwen3:14b staged without (b)).** The llama3.1:8b cells
keep a, b, c, d, e, f; the qwen3:14b cell on the Intel CPU at 4096 runs a, c, d, e, f at 3 seeds. Each B5 entry passes
its own `--policies` (`scripts/t2s_week_queue.oct26_cell_policies`). Pasted from [t2s-b5-plan-hours] (ESTIMATES; B5
from measured evo-t2s session minutes, [t2s-r2-session-minutes]): "launch budget 311.0 h (24.0 h slack before
no-launch)"; "t2s_o26_b5_cpu_llama31_8b [policies a,b,c,d,e,f] 87.3 h (yes; ends 87.6 h, 247.4 h before no-launch ...)";
"t2s_o26_b5_igpu_llama31_8b [policies a,b,c,d,e,f] 85.9 h (yes; ends 173.5 h, 161.5 h before no-launch ...)";
"t2s_o26_b5_cpu_qwen3_14b [policies a,c,d,e,f] 37.3 h (yes; ends 210.8 h, 124.2 h before no-launch; 12.0 h/seed, 3 of
3 seeds fit, cell counts yes (first 3 seeds))"; "t2s_o26_b6_model_pull 1.0 h (yes; ends 211.8 h, 123.2 h before
no-launch)"; "t2s_o26_b6_val_cpu 34.6 h (yes; ends 246.4 h, 88.6 h before no-launch)"; "t2s_o26_b6_val_igpu 17.3 h
(yes; ends 263.7 h, 71.3 h before no-launch)"; B6 not staged: "t2s_o26_b6_real_cpu_4096 105.5 h (does not fit: 105.5 h
from 263.7 h ends at 369.2 h, budget 311.0 h)" and "t2s_o26_b6_real_igpu_default 299.0 h (does not fit ...)". So the
queue is preflight, A5, the three B5 cells, the B6 pull and the two B6 validations; the B6 real runs are future work
for this window. Dated amendment: docs/FINDINGS.md "AMENDMENT 2026-10-22 to the B5-on-evo-t2s pre-registration". The
updates below are superseded where they differ.

**Update 2026-10-21 (later): (b) kept, qwen3:14b at 3 seeds if it fits, 24 h slack, B6 fills the rest.** The evo-t2s
B5 policy set is a, b, c, d, e, f (each entry passes `--policies a,b,c,d,e,f`). Every estimated entry must end within
the launch budget, which is the available hours minus 24 h before the no-launch time. Pasted from [t2s-b5-plan-hours]
(ESTIMATES; B5 from measured evo-t2s session minutes, [t2s-r2-session-minutes]): "available 335.0 h"; "launch budget
311.0 h (24.0 h slack before no-launch)"; "t2s_o26_b5_cpu_llama31_8b 87.3 h (yes; ends 87.6 h, 247.4 h before
no-launch; 17.1 h/seed, 5 of 5 seeds fit, cell counts yes (5 seeds))"; "t2s_o26_b5_igpu_llama31_8b 85.9 h (yes; ends
173.5 h, 161.5 h before no-launch; 16.9 h/seed, 5 of 5 seeds fit, cell counts yes (5 seeds))"; "137.5 h are left of
the 311.0 h launch budget; evo-t2s-cpu qwen3:14b at 3 seeds needs 190.2 h (63.0 h/seed): does not fit, not staged".
Most of the qwen3:14b time is (b) at its native window, estimated by the evo-x2 ratio (assumption stated in the row's
basis). B6 then fills in order: "t2s_o26_b6_model_pull 1.0 h (yes; ends 174.5 h, 160.5 h before no-launch)";
"t2s_o26_b6_val_cpu 34.6 h (yes; ends 209.1 h, 125.9 h before no-launch)"; "t2s_o26_b6_val_igpu 17.3 h (yes; ends
226.4 h, 108.6 h before no-launch)"; not staged: "t2s_o26_b6_real_cpu_4096 105.5 h (does not fit: 105.5 h from 226.4 h
ends at 331.9 h, budget 311.0 h)" and "t2s_o26_b6_real_igpu_default 299.0 h (does not fit ...)". B6 hours are evo-x2
H-scale measured minutes times the largest measured evo-t2s / evo-x2 R2 session ratio at the same window ("factor
15.12" on the CPU at 4096, "factor 7.57" on the iGPU at 32768); the iGPU window of these models is ASSUMED 32768 and
the pull hour is nominal. So the queue is preflight, A5, the two llama3.1:8b cells, the B6 pull and the two B6
validations. qwen3:14b on Intel and the B6 real runs are future work for this window. Dated amendment: docs/FINDINGS.md
"AMENDMENT 2026-10-21 (later) to the B5-on-evo-t2s pre-registration and to the B6 staging note". The updates below
are superseded where they differ.

**Update 2026-10-21 22:30 (operator item 6: reduced policy set on evo-t2s; qwen3:14b back in at 2 seeds).** On evo-t2s
B5 runs policies (a) default, (c) trim, (d) router, (e) auto-sized window, (f) num_keep only; (b), (g), (h) are not run
there (evo-x2 keeps all). Each B5 entry passes `--policies a,c,d,e,f` (`harness/policy_compare.T2S_POLICIES`, also the
platform default for `evo-t2s` and `evo-t2s-cpu`); window probes and recall calls under the q8_0 / q4_0 server envs
are not run, since only (g) / (h) need them. On the CPU (e) has no window (no GPU) and is not run, by the 2026-10-14
rule. Order: llama3.1:8b Intel CPU 5 seeds, llama3.1:8b Intel iGPU 5 seeds, then qwen3:14b on the Intel CPU at its
4096 default window at 2 seeds, staged only if it fits. Plan, pasted from [t2s-b5-plan-hours] (ESTIMATES from measured
evo-t2s session minutes, [t2s-r2-session-minutes]): "available 335.0 h"; "B5 staged cells 145.2 h";
"t2s_o26_b5_cpu_llama31_8b 34.0 h (yes; 6.4 h/seed, 5 of 5 seeds fit, cell counts yes (5 seeds))";
"t2s_o26_b5_igpu_llama31_8b 85.9 h (yes; 16.9 h/seed, 5 of 5 seeds fit, cell counts yes (5 seeds))";
"t2s_o26_b5_cpu_qwen3_14b 25.3 h (yes; 12.0 h/seed, 2 of 2 seeds fit, cell counts no (reported only, below 3 seeds))";
"evo-t2s-cpu qwen3:14b at 2 seeds needs 25.3 h (12.0 h/seed): fits, staged after the llama3.1:8b cells". So qwen3:14b
on the Intel CPU goes back into `queues/t2s_oct26_queue.json` as `t2s_o26_b5_cpu_qwen3_14b` (dry-run passes, fresh
build matches the file). At 2 seeds it is below the pre-registered 3-seed minimum, so it is reported, not a complete
cell. The qwen3:14b iGPU cell stays future work. The B6 entries follow B5 and now have hours left to start in (they
carry no estimate: no timing of those models on evo-t2s). Dated amendment: docs/FINDINGS.md "AMENDMENT 2026-10-21
(22:30) to the B5-on-evo-t2s pre-registration". The two updates below are kept as written; their hours were for (a) to
(h) and are superseded by this one.

**Update 2026-10-21 (operator decision: B5 order and qwen3:14b).** B5 in this window is the two llama3.1:8b cells at 5
seeds, Intel CPU first, then Intel iGPU; qwen3:14b was to follow at 3 seeds on the Intel CPU (its default window
4096) only if the hours left allowed. Computed from measured evo-t2s session minutes ([t2s-r2-session-minutes]) by
`scripts/t2s_window_stage.py qwen_decision`, pasted from [t2s-b5-plan-hours]: "after A5 0.3 h and the llama3.1:8b
cells at 5 seeds (evo-t2s-cpu llama3.1:8b 197.0 h, evo-t2s llama3.1:8b 142.3 h), -4.6 h are left of 335.0 h;
evo-t2s-cpu qwen3:14b at 3 seeds needs 498.1 h (164.9 h/seed): does not fit, not staged; qwen3:14b on Intel is future
work". So both qwen3:14b entries (`t2s_o26_b5_igpu_qwen3_14b`, `t2s_o26_b5_cpu_qwen3_14b`) are dropped from
`queues/t2s_oct26_queue.json` (dry-run passes, fresh build matches the file), and the iGPU llama3.1:8b cell itself is
expected to reach only "4 of 5 seeds fit, cell counts yes (first 3 seeds)". All hours are ESTIMATES. Dated amendment
to the frozen pre-registration: docs/FINDINGS.md "AMENDMENT 2026-10-21 to the B5-on-evo-t2s pre-registration".

**Update 2026-10-22 (B5 is the T2S priority).** Canonical file: `queues/t2s_oct26_queue.json` (one builder,
`scripts/t2s_week_queue.py build-oct26`; its gates point at its own preflight; it is the file `load-window` takes). The
second file `queues/t2s_window_20261026_staged.json` was retired: it staged B5 for llama3.1:8b on the iGPU only at 3
seeds and a different, iGPU-only B6, and its gates named the finished week's `t2s_wk_*` ids. Order of the window:
1. On the controller, first: `py -3.12 harness/t2s_deadline.py open-window --no-launch-after 2026-11-09T04:00:00Z
   --hard-deadline 2026-11-09T04:30:00Z --reason "handover to Zach 2026-11-08; resume only in the next confirmed
   window"` (times are 23:00 and 23:30 EST on 2026-11-08; DST ends 2026-11-01). It runs the handover check on evo-t2s
   (HANDOVER_DONE.json, the pause flag naming its releaser, no process of ours; refuses and changes nothing on a
   failure), widens the results sync to both hosts with a window file that expires at the hard deadline (the
   APU-SyncResults task keeps `--host evo-x2` and is not edited), arms the deadline, waits until the hard deadline, and
   removes the window file in its finally block. So the job that arms the deadline is the job that switches the sync
   and switches it back, and an expired file is ignored and deleted by `scripts/sync_results.py` even if that job died.
2. Deploy (the week's file list plus `window_override_methods.py`, `policy_compare.py` and the router stack under
   `C:\apu\ovn\src`, `C:\apu\ovn\analysis`), then `python t2s_week_queue.py load-window t2s_oct26_queue.json`.
3. The queue: preflight, A5, B5 (two llama3.1:8b cells since 2026-10-21; four cells before), then the B6 pull,
   validations and real runs.
Plan (ESTIMATES, `py -3.12 scripts/t2s_window_stage.py plan`, register row `t2s-b5-plan-hours`): only the two
llama3.1:8b B5 cells are expected to count in this window (CPU 5 seeds, iGPU first 3 or 4 seeds); the qwen3:14b cells
and B6 are not expected to start. Pre-registration: docs/FINDINGS.md "B5 on evo-t2s Intel CPU and Intel iGPU".

The original 2026-10-15 staging note follows (B5 was then pending):

`queues/t2s_oct26_queue.json` (`py -3.12 scripts/t2s_week_queue.py build-oct26`, check with `dry-run-oct26`, all 7
entries pass): t2s_o26_preflight; t2s_o26_b6_model_pull (qwen3:32b, llama3.3:70b); t2s_o26_window_override_methods
(A5); B6 validation on CPU and with OLLAMA_IGPU_ENABLE=1 (negative control at num_ctx 32768, the
x2_r2_llama_val_32k precedent, since 131072 cannot fit llama3.3:70b on this 63.5 GB machine); B6 real at num_ctx 4096
on the CPU and at the iGPU-enabled native window (ollama_default), 5 seeds x 40 turns, gated per model on its own
validation. No timing of these models exists on evo-t2s, so the entries carry no hour estimate. The B5 policy
comparison job is not in the file yet (another agent is building it; `OCT26_PENDING` names where it goes). Deploy at
that window: the week's file list plus `window_override_methods.py`, then
`python t2s_week_queue.py load-window t2s_oct26_queue.json` and a new deadline. Open: whether llama3.3:70b on the CPU
at 32768 (about 42 GB weights plus KV) fits next to the OS; the validation job will show it.
