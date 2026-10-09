# Blade plan: three overnight runs on the NVIDIA platform

The Razer Blade 14 (RTX 4070 Laptop GPU, 8188 MiB VRAM, CUDA) is the NVIDIA platform for the paper and also the
controller laptop. Blade jobs run overnight only, with all local Claude Code work paused while one runs; evo-x2's
remote queue keeps running. No Blade measurement has run yet. Blade night 1 starts only after the operator says
"start Blade night 1".

Code: `harness/blade_common.py` (pins, versions, local Ollama, dry-run budget), `harness/blade_k1.py`,
`harness/blade_r2.py` (reuses `x2_r2_agent` and `x2_r2_mechanism`), `harness/blade_c3.py` (reuses
`blade_spill_sweep`), `harness/blade_queue.py` (the Blade queue), `scripts/blade_night.py` (the night wrapper),
`analysis/blade_hours_estimate.py` (hours from result files), tests in `tests/test_blade_night.py`.

## Versions (read 2026-10-08, read-only)

| component | pinned | found on the Blade | status |
|---|---|---|---|
| hostname | RITZLAPTOP (host_config hw_id `blade`, gpu_vendor nvidia, no ssh_host) | RitzLaptop | entry added |
| GPU | RTX 4070 Laptop, 8188 MiB | NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB, VBIOS 95.06.34.00.cd | ok |
| NVIDIA driver | 610.88 (as in C1/C2) | 610.88; nvidia-smi header "CUDA UMD Version: 13.3" (driver CUDA API) | ok |
| llama.cpp | b10970, commit bfdc32183, asset `llama-b10970-bin-win-cuda-12.4-x64.zip` (sha256 78c878ae...e52dad8) + `cudart-llama-bin-win-cuda-12.4-x64.zip` (sha256 8c79a9b2...e32ae1d6), CUDA runtime 12.4 | both zips in `C:\apu\bin`, sha256 equal to the GitHub release digests; extracted at `C:\apu\bin\llama-b10970-cuda` (`cudart64_12.dll`, `llama-server.exe --version` = build 10970, commit bfdc32183) | ok, nothing to download |
| Ollama | 0.34.4 (same as evo-x2), side by side at `C:\apu\bin\ollama-0.34.4\ollama.exe` | 0.34.1 tray install at `%LOCALAPPDATA%\Programs\Ollama` (bundles `cuda_v12` and `cuda_v13` runners); 0.34.4 being installed by the parent session (2026-10-08) | install in progress |

**Ollama rule (operator, 2026-10-08):** every Blade measurement job uses the side-by-side 0.34.4 binary by explicit
path, `C:\apu\bin\ollama-0.34.4\ollama.exe` (`blade_common.PINNED["ollama_exe"]`, passed as `--ollama-exe` in every
queue entry), never the 0.34.1 tray install. No queued job carries `--allow-version-mismatch`; the jobs refuse that
flag outside a dry run, and a real run refuses to measure on any mismatch with the pins. Every Blade job writes the
full version record (`blade_common.versions_record`: driver, CUDA driver API, Ollama version and exe, Ollama's bundled
CUDA runner dirs, llama-server build and commit, the CUDA runtime DLLs next to it) into its `run_start` row (C3 ignores
the Ollama pin; it does not use Ollama). Which of Ollama's CUDA runners (v12 or v13) actually loads is read from the
server log's "inference compute" line by K1.

Install steps (for the record; the parent session is running them):

```
curl.exe -L -o C:\apu\bin\ollama-windows-amd64-v0.34.4.zip https://github.com/ollama/ollama/releases/download/v0.34.4/ollama-windows-amd64.zip
certutil -hashfile C:\apu\bin\ollama-windows-amd64-v0.34.4.zip SHA256
    (expect 535193f38f3344e5b08f5d1c171c31ce11aa17f0124ff69ae26d8ec7fe06fa62, size 1,461,155,106 bytes)
powershell -NoProfile -Command "Expand-Archive -Path C:\apu\bin\ollama-windows-amd64-v0.34.4.zip -DestinationPath C:\apu\bin\ollama-0.34.4"
C:\apu\bin\ollama-0.34.4\ollama.exe --version
```

The jobs stop every Ollama process (tray app included, or it respawns its server) and start the pinned exe as a hidden
child with its own log; both installs use the same model store (`%USERPROFILE%\.ollama\models`).

**Models:** the Blade store has llama3.1:8b, llama3.2, gemma3:4b, qwen3:4b, qwen3:4b-instruct, qwen3-4b-2507 (blob
85e4a5b7, the same GGUF as evo-x2's). qwen3:8b is being pulled by the parent session. qwen3-4b-2507-tools (K1 context
policy only) can be created after that with evo-x2's own create step:

```
py -3.12 -c "import sys,json; sys.path.insert(0,'harness'); from pathlib import Path; import x2_r2_4b_tools_validate as t; t.ensure_tools_tag(lambda r: print(json.dumps(r, default=str)[:600]), Path('results/blade_setup_4b_tools.jsonl'))"
```

K1 records a missing model and moves on.

## Validation context cap (operator decision 2026-10-08)

Rule: take the largest prompt + generated tokens of any call in evo-x2's R2 validation sessions at num_ctx 131072
(both call-2 variants; the 8192 positive control is excluded because it truncates by design); if it is under 30000
the Blade validation runs at num_ctx 32768, otherwise 65536.

Measured, register row `blade-validation-ctx-cap` (VERIFIED), verbatim: "llama3.1:8b: max prompt+generated 15040
(ollama_ctx_131072 seed 20260901 turn 10); qwen3:8b: max prompt+generated 17567 (ollama_ctx_131072_call2_notools seed
20260902 turn 10); overall max 17567 -> Blade validation cap 32768 (rule: <30000 -> 32768, else 65536)".

Choice: **the Blade's validation arm b (baseline and negative control) runs at num_ctx 32768, not evo-x2's 131072.**
Why it differs: 131072 tokens of f16 KV does not fit in 8 GB, so Ollama would run most layers on the CPU, which makes
the control slow and unlike the tiers it is the control for; 32768 still holds every validation session untruncated
(17567 at most, under 30000). The arm is `ollama_ctx_32768_negative_control` (+ the call-2 suffix), registered by
`blade_r2.py` with num_ctx 32768; x2_r2_agent's validation plan and gates take it as `neg_arm` (default unchanged, so
evo-x2's 131072 plan and gates are untouched). The positive control stays at num_ctx 8192. Every R2 row records
`num_ctx_requested` and the `loaded_context` Ollama reports.

## Plan table

Hours from `py -3.12 analysis/blade_hours_estimate.py --markdown` (register row `blade-night-hours`, PENDING until the
dry-run files exist). Per cell: "measured" = the same work timed on the Blade (C3: C1's own per-ctx segments, load and
thermal waits included); "rates" = the Blade's prefill and decode tok/s from the 60-second dry runs for that
(model, num_ctx), applied to the token workload of the evo-x2 sessions of the same design (register row
`blade-dryrun-rates`); "scaled" = evo-x2's wall time for the same sessions times a stated factor (`FACTORS`: 1.0 for
tiers that fit in 8 GB, 1.5 at 16384, 3.0 at 32768 where Ollama has to put layers on the CPU, 1.5 for mitigation's
per-call render and tokenize, 1.2 for qwen3:8b over llama3.1:8b), labelled "scaled (no Blade rate for <model> at
<ctx>)". No dry-run file exists yet, so every R2 number below is scaled. The Blade default tier is estimated as 4096
until K1 measures it.

| night | job | queue id / output | hours | measured / scaled |
|---|---|---|---|---|
| 1 | (a) K1: default context per model, device and VRAM, overflow at 4K/8K/16K/32K, llama.cpp CUDA defaults; every Ollama process stopped before the version check | `blade_k1_v2` / `results/blade_k1_v2.jsonl`, `.summary.json` | 0.50 | scaled (itemized) |
| 1 | (b) R2 validation and controls, llama3.1:8b, arm b at 32768 with the negative control at 5 sessions (seeds 20260901-05), positive control 8192 (1 session), diagnostic arm as before | `blade_r2_validation_v2` / `results/blade_r2_validation_v2.jsonl` | 0.32 | mixed (rates, scaled) |
| 1 | (b) gate: negative-control canary misses over the 5 sessions, 0 -> branch real, any -> branch mechanism | `blade_r2_gate_v2` / `results/blade_r2_gate_v2.json` | 0.00 | measured (no model) |
| 1 | (b) branch real: R2 tiers default, 4096, 32768, 5 seeds x 40 turns, llama3.1:8b, rules in use from the gate (1, 3, 4 as on evo-x2 if 2 and 5 fail again) | `blade_r2_real_v2` / `results/blade_r2_real_v2.jsonl` | 1.75 | rates |
| 1 | (c) branch real: mitigation, x2_r2_mitigation_v1 design, Blade default (from K1), 4096, 8192, 3 seeds | `blade_r2_mitigation_v2` / `results/blade_r2_mitigation_v2.jsonl` | 1.18 | mixed (rates, scaled) |
| 1 | branch mechanism: mechanism run, llama3.1:8b, 4096 tier, one session | `blade_r2_mechanism_4096_v2` / `results/blade_r2_mechanism_4096_v2.jsonl` | 0.05 | rates |
| 2 | (d) C3 half A (Prefer No Sysmem Fallback, ctx 36864/38912/40960/43008, 1+5 calls) and half B (Driver Default, ctx 40960/43008, 1+3 calls) | `blade_c3_sysmem_fallback_v1` / `results/blade_c3_sysmem_fallback_v1.jsonl` | 3.09 | measured (C1 segments); half A is minutes if the spilled points fail at load |
| 2 | (e) mechanism: render-only validity, message-drop and context-shift logging, llama3.1:8b, one session per tier (default, 32768, 16384, 8192, 4096) | `blade_r2_mechanism_v1` / `results/blade_r2_mechanism_v1.jsonl` | 2.71 | scaled |
| 3 | qwen3:8b R2 validation and controls (same design as night 1), only if it fits | `blade_r2_validation_qwen3_8b_v1` / `results/blade_r2_validation_qwen3_8b_v1.jsonl` | 0.53 | scaled |
| 3 | qwen3:8b R2 tiers default, 4096, 32768, 5 seeds x 40 turns, only if it fits | `blade_r2_real_qwen3_8b_v1` / `results/blade_r2_real_qwen3_8b_v1.jsonl` | 8.29 | scaled (llama3.1:8b workload x1.2; qwen3:8b's own evo-x2 real-run file is not synced) |

Night totals: night 1 3.75 h on the real branch, 0.87 h on the mechanism branch (register `blade-night-hours`, the
night-1 rows above from the same row), night 2 5.80 h (the operator at the keyboard for C3's two gates),
night 3 8.82 h if qwen3:8b fits, minutes if it does not.

### Night 1 rerun (operator 2026-10-08)

Night 1's first run (2026-10-09 03:57Z, `results/blade_night1_summary_20261009T035754Z.json`) left: K1 refused (its
version check ran `ollama --version` while the tray's Ollama 0.34.1 server was up, and that command reports the running
server's version, not the binary's); the validation finished, but its negative control at 32768 had canary misses
(register `blade-r2-validation-run1`), so the real and mitigation jobs were refused by their own validation gates; and
the power log said `restored: false` only because SUB_BUTTONS/LIDACTION is not exposed on this machine.

What changed for the rerun:

- **K1 and R2 version check.** `blade_common.pinned_versions` stops every Ollama process first (tray app
  `ollama app.exe` first, then `ollama.exe` and Ollama's runners, hidden taskkill, re-listed until none is left) and
  only then runs `<pinned exe> --version`; the binary's own "client version" line is the version checked. A process
  that survives the stop, or a server still answering, is a refusal. After the job starts its own server, the server's
  `/api/version` must also equal the pin.
- **Power restore.** A setting the machine does not expose (powercfg prints no AC index, original `None`) is "not
  applicable": never set, never restored, not an error, listed under `not_applicable` in the power log. `restored` is
  false only if an exposed setting failed to restore or read back different from its original.
- **Order.** K1, then the validation rerun at the 32768 cap with the negative control at 5 sessions instead of 3
  (`--neg-sessions 5`, seeds 20260901-05; positive control and diagnostic arm unchanged), then the gate job
  `harness/blade_r2_gate.py`. The gate reads the validation file with `x2_r2_agent.evaluate_gates` and the Blade
  negative-control arm and writes `results/blade_r2_gate_v2.json`. 0 misses across the 5 sessions: branch "real", the
  R2 tiers (default, 4096, 32768; 5 seeds; 40 turns) and the mitigation run for llama3.1:8b, scoring the rules in use
  from the same validation file (rules 1, 3 and 4 if rules 2 and 5 fail the baseline again, as on evo-x2). Any miss:
  branch "mechanism", the real and mitigation jobs are marked `skipped: negative-control canary misses N`, the
  mechanism run at the 4096 tier runs instead (one session, as in night 2's mechanism job), and the decision file
  carries the Blade recall result as an open finding.
- **Mechanical and resumable.** The queue stores the gate's decision in its state entry and in the night summary
  (`gate_decisions`). A conditional job whose branch does not match is `skipped` (terminal, with the reason); while the
  gate has no decision (not run, or errored) the conditional jobs stay pending and run after the gate on the next
  start. The gate exits 2 and writes nothing if the validation did not finish or has fewer than 5 negative-control
  sessions.
- **Clean start.** New ids (`blade_k1_v2`, `blade_r2_validation_v2`, `blade_r2_gate_v2`, `blade_r2_real_v2`,
  `blade_r2_mitigation_v2`, `blade_r2_mechanism_4096_v2`) and `_v2` outputs: the first run's entries in
  `C:\apu\blade\blade_queue_state.json` and its result files stay as they are and are never skipped into or overwritten.
  Night 3 now reads `results/blade_k1_v2.summary.json`.
- **Stub.** `py -3.12 scripts/blade_night.py --night 1 --stub` runs night 1 once per gate branch (0 misses and 2
  misses, through the gate's real decision code on synthetic rows) with the Blade's power layout (LIDACTION not
  exposed): PASS on both branches, power restored on both.
- Register row `blade-r2-validation-run2` reports the rerun once `results/blade_r2_validation_v2.jsonl` is synced
  (PENDING until then).

### Job details

- **K1** (`blade_k1.py`): device and VRAM from nvidia-smi and from Ollama's own log ("inference compute",
  "vram-based default context"); per model a short chat with no num_ctx, then `/api/ps` (context_length, size,
  size_vram; fully on the GPU = size_vram at least 99% of size); a 4000-token calibration at num_ctx 8192; marker
  prompts of 4096/8192/16384/32768 tokens at the default context; num_ctx 4096/8192/16384/32768 each with a 1.5x
  prompt (processed vs the half-window rule floor(n/2)+2, size_vram at that window); then llama-server b10970 CUDA with
  only `-m <Ollama blob> --port --log-file` (n_ctx from /props, layers offloaded, KV and model buffers). Every chat row
  records prompt_eval_count, prompt_eval_duration, eval_count, eval_duration and the derived prefill and decode tok/s.
  The summary (`blade_k1_v1.summary.json`) holds, per model, `fully_on_gpu` at the default context and
  `fully_on_gpu_by_ctx` at each tested window. The 4B tools tag is measured for context policy only; it failed its R2
  validation on evo-x2 (register `R2-install-path-4b-comparison`).
- **R2** (`blade_r2.py`): x2_r2_agent's driver, scoring and gates, v1 protocol (call-2 mode off), seeds
  20260901..20260905 for the tiers, seed-major order. Every call row (x2_r2_agent's) records prompt_eval_count,
  prompt_eval_duration, completion tokens and eval_duration, so prefill and decode rates come from the rows. The tier
  run refuses per model unless the Blade's own validation file passes `validation_preflight_per_model` (with the Blade
  negative-control arm).
- **Mitigation**: `blade_r2.py --mode mitigation --client-trim margin=0.05`, tiers default (resolved to K1's measured
  num_ctx for llama3.1:8b), 4096 and 8192, seeds 20260901..20260903, 40 turns, v1 call-2 mode, OLLAMA_DEBUG=1 with
  the mechanism log parsing. The body is `x2_r2_agent.run_mitigation`, the same function `x2_r2_agent.main --mode
  mitigation` runs on evo-x2. The job parses its equivalent x2_r2_agent argv with x2_r2_agent's own parser and records
  it in `run_start`; the stub night runs the same parse.
- **C3** (`blade_c3.py`): C1's harness per call (qwen3-4b-instruct, f16 KV, `-fa on -ngl 99 -np 1 -t 4`, 90% fill,
  thermal gate, stale-server guard, nvidia-smi, dmon and per-PID Shared Usage), every ctx attempted, half A order
  shuffled with seed 20261008. **Half B (Driver Default) runs in a finally block, with its own confirmation gate, even
  if half A fails or is interrupted**, so the global setting is always put back. After both halves the job writes
  `C:\apu\blade\c3_WAITING_READBACK.txt` (see the read-back step below).
- **Mechanism**: `x2_r2_mechanism.MechanismRuntime` under OLLAMA_DEBUG=1, prompt checks with llama-tokenize against a
  fresh-load prompt_eval_count, a tier citable only with at least 5 checks all within 1%.
- **Night 3 fit decision (automatic):** each night-3 job runs qwen3:8b only if `blade_k1_v1.summary.json` shows it
  fully on the GPU at every context that job uses (validation: the default context, 32768 and 8192; tiers: the default
  context, 4096 and 32768). A missing summary, model or context counts as not fitting. Otherwise the job writes
  `{"record": "skipped", "reason": "skipped: does not fit (...)"}` with the per-context K1 values and exits 0, so night
  3 completes in minutes. qwen3:8b at 32768 is the most likely point to fail this check; K1 decides.

## C3 operator steps (about two minutes per gate)

Night 2 starts with C3, so the operator starts night 2 at the keyboard and stays through both gates (about 3 h in the
worst case, minutes if the spilled points fail at load); the mechanism job runs unattended after that.

Before half A (the job writes `C:\apu\blade\c3_WAITING_A.txt` with these steps and waits):

1. Right-click the desktop, NVIDIA Control Panel, 3D Settings, Manage 3D settings, Global Settings tab.
2. Find "CUDA - Sysmem Fallback Policy", select "Prefer No Sysmem Fallback", press Apply.
3. Close and reopen NVIDIA Control Panel, go back to the same row and read it: it must say "Prefer No Sysmem
   Fallback". nvidia-smi does not expose this setting (C1 recorded it as UNKNOWN for that reason), so this read is the
   verification.
4. Create `C:\apu\blade\c3_confirm_A.flag` whose first line is exactly `Prefer No Sysmem Fallback`, e.g.
   `Set-Content -Path C:\apu\blade\c3_confirm_A.flag -Value "Prefer No Sysmem Fallback"`.

Before half B (`c3_WAITING_B.txt`, shown even if half A failed): the same steps with "Driver Default" and
`c3_confirm_B.flag`, first line exactly `Driver Default`. This restores the setting.

The gate accepts a flag only if it was written after the wait began (a flag left from an earlier run is renamed
`.stale`) and its first line matches exactly; a wrong value is logged as `c3_gate_rejected` and the wait continues.
There is no timeout and no automatic start: if nobody confirms, night 2 waits, and the waiting file says so.

**Read-back after night 2.** Once both halves are done the job writes `c3_WAITING_READBACK.txt`. The operator reopens
NVIDIA Control Panel, reads the "CUDA - Sysmem Fallback Policy" row and creates `C:\apu\blade\c3_readback.flag` whose
first line is the current setting text (expected `Driver Default`). Nothing waits on it (the mechanism job keeps
running); at the end of the night `blade_night.py` reads it into the night summary as `c3_setting_readback`: "read
back: Driver Default", or loudly "!!! SETTING READ BACK AS ..., NOT 'Driver Default' !!!", or, when no flag was written
during the night, "!!! SETTING NOT READ BACK ... Check NVIDIA Control Panel by hand. !!!".

## Night protocol

```
(operator, once all local Claude Code work is paused)
New-Item -ItemType File C:\apu\blade\CLAUDE_CODE_PAUSED.flag
(operator says "start Blade night N", then)
New-Item -ItemType File C:\apu\blade\START_BLADE_NIGHT_<N>.flag
py -3.12 scripts\blade_night.py --night <N>
```

`scripts/blade_night.py`:

1. refuses unless `START_BLADE_NIGHT_<N>.flag` exists (renamed `.used` at start, so it never starts a second time);
2. refuses unless on AC power and the paused condition holds: `CLAUDE_CODE_PAUSED.flag` present and no other python,
   pytest, git or harness llama-server process outside the wrapper's own process tree;
3. writes the current AC standby, AC hibernate and AC lid-close values and the active scheme to
   `C:\apu\blade\logs\night<N>_power_<ts>.json` before changing anything, then sets all three to 0 with
   `powercfg /setacvalueindex SCHEME_CURRENT ...` and `/setactive SCHEME_CURRENT`. If an earlier night's power log never
   reached `restored: true`, its originals are used;
4. runs the night's jobs in order through the Blade queue, with resume;
5. in a finally block restores every original value, reads them back, writes both to the power log, adds the C3
   read-back for night 2, then writes `results/blade_night<N>_summary_<ts>.json`.

The wrapper never touches NVIDIA settings.

### The Blade queue

`harness/blade_queue.py`: its own state file `C:\apu\blade\blade_queue_state.json`, not evo-x2's `queue_state.json`
and not a Windows scheduled task; the wrapper runs it in-process. Nights: 1 = K1, llama3.1:8b validation, tiers,
mitigation; 2 = C3, mechanism; 3 = qwen3:8b validation and tiers (each skipping itself if it does not fit). Each job
runs as a hidden child (`proc_util.popen_hidden`) from the repo root, log `C:\apu\blade\logs\<job_id>.log`. A job left
"running" and an errored job run again on the next start of that night; done jobs do not. Each job resumes inside its
own output file, so a rerun never repeats finished work. Before every job the queue checks the paused condition again.

Nothing here can leave a machine needing a manual resume without saying so: the only waits are C3's two gates, which
by the operator's design wait for a person and say so in `c3_WAITING_<half>.txt`.

## Dry runs

**Stub nights (fakes, no model, no setting changed): PASS for all three nights.** `py -3.12 scripts/blade_night.py
--night <N> --stub` runs the wrapper with a fake powercfg, fake AC, fake paused condition and a fake job runner that
parses each job's argv with that job's own parser (and, for the mitigation job, its equivalent x2_r2_agent argv with
x2_r2_agent's parser; for night 3, the automatic fit decision against the K1 summary that exists, none in a stub, so
"do not run"). Recorded in `results/blade_dryrun/blade_dryrun_night{1,2,3}_stub.json`: the three no-sleep sets and
the setactive come before the first job, the restores after the last, the final values equal the originals, every
output path is prefixed `blade_`; night 2's summary carries the loud "SETTING NOT READ BACK" line (no read-back flag
in a stub). Unit tests (`tests/test_blade_night.py`) cover power restore on a crashed queue and on an interrupted job,
the refusals, queue resume, the C3 gate, half B after a failed or interrupted half A, the read-back states, the night-3
skip, the pinned Ollama path in every job, and the estimator switching a cell from scaled to rates.

**60-second live dry runs: being run by the parent session** (one at a time, after the 0.34.4 install and the qwen3:8b
pull). Each stops itself at the first result row after 60 s, stops every Ollama process and runner in its finally
block and writes only under `results/blade_dryrun/`. `--allow-version-mismatch` stays on these dry-run lines only. After
each, `tasklist | findstr /i "ollama llama-server"` should be empty:

```
py -3.12 harness\blade_k1.py --out results\blade_dryrun\blade_dryrun_k1.jsonl --models llama3.2 --skip-llamacpp --dry-run-seconds 60 --allow-version-mismatch --ollama-exe C:\apu\bin\ollama-0.34.4\ollama.exe
py -3.12 harness\blade_r2.py --mode validation --out results\blade_dryrun\blade_dryrun_r2_validation.jsonl --models llama3.2 --seeds 20260901 --turns 3 --dry-run-seconds 60 --allow-version-mismatch --ollama-exe C:\apu\bin\ollama-0.34.4\ollama.exe
py -3.12 harness\blade_r2.py --mode real --out results\blade_dryrun\blade_dryrun_r2_real.jsonl --models llama3.2 --tiers 4096 --seeds 20260901 --dry-run-seconds 60 --allow-version-mismatch --ollama-exe C:\apu\bin\ollama-0.34.4\ollama.exe
py -3.12 harness\blade_r2.py --mode mitigation --client-trim margin=0.05 --out results\blade_dryrun\blade_dryrun_r2_mitigation.jsonl --models llama3.2 --tiers 4096 --seeds 20260901 --dry-run-seconds 60 --allow-version-mismatch --ollama-exe C:\apu\bin\ollama-0.34.4\ollama.exe
py -3.12 harness\blade_c3.py --out results\blade_dryrun\blade_dryrun_c3_gate.jsonl --dry-run-gate
py -3.12 harness\blade_r2.py --mode mechanism --out results\blade_dryrun\blade_dryrun_r2_mechanism.jsonl --models llama3.2 --tiers 4096 --dry-run-seconds 60 --allow-version-mismatch --ollama-exe C:\apu\bin\ollama-0.34.4\ollama.exe
```

The file names above are the ones the register rows `blade-dryrun-rates` and `blade-night-hours` read. Rates measured
on llama3.2 are reported by model, so they replace the scaled factors only for llama3.2 cells; the plan's llama3.1:8b
and qwen3:8b cells stay "scaled (no Blade rate ...)" unless a dry run uses that model (for example `--models
llama3.1:8b` on the real and validation lines, which gives rates at 4096 and 32768). Earlier history: a first attempt
at the K1 dry run from a subagent session was refused by that session's permission classifier before it started; no
model was loaded then.

| job | launched? | wrote results? | cleaned up? |
|---|---|---|---|
| stub nights 1, 2, 3 (fakes) | yes | yes (`blade_dryrun_night{1,2,3}_stub.json`) | yes (fake power values back to originals) |
| the six 60 s dry runs above | pending (parent session) | pending | pending |

## Open questions

1. qwen3:8b on 8 GB at 32768 (validation arm b and the 32768 tier) is the likely failure point of the night-3 fit
   check; if it fails, night 3 is skipped as a whole by design. Dropping 32768 from night 3 would need an operator
   decision.
2. If K1 shows the Blade default is 4096, the default tier repeats the 4096 tier with no num_ctx sent; kept as asked.
3. The 32768 tier and the mechanism's 16384 and 32768 tiers run with layers on the CPU on 8 GB; their time cost stays
   a scaled guess until a dry run on llama3.1:8b at those windows gives rates.
4. Mitigation needs `C:\apu\bin\llama-b10970\llama-tokenize.exe` (`prompt_token_check.TOKENIZE_EXE`); present on the
   Blade (Vulkan build dir, tokenizing is CPU-only).
5. C3 sets the policy globally (operator decision). A per-program setting for `llama-server.exe` would avoid touching
   other CUDA programs.
6. llama3.1:8b on the Blade: manifest model digest 667b0c19...6a29; whether it equals evo-x2's digest has not been
   checked (no SSH this session). K1 records the digest in its rows via `/api/ps`.

### 60-second live dry runs, run 2026-10-08 (operator-approved), Ollama 0.34.4 side by side

Ollama 0.34.4 installed side by side at `C:\apu\bin\ollama-0.34.4` (zip sha256 equal to the expected value; tray
0.34.1 untouched; `results/blade_install_20261008.json`). qwen3:8b pulled; its model layer digest equals evo-x2's.
Each dry run ran alone (no other local work), on llama3.1:8b with the 0.34.4 binary, stopped itself after 60 s, and
left no ollama or llama-server process; the tray app was restarted (hidden) afterwards. Rates and the per-night hours
are register rows `blade-dryrun-rates` and `blade-night-hours`; user paths in the result files are redacted.

| dry run | launched | wrote results | cleaned up |
|---|---|---|---|
| K1 (default context probe) | yes | yes | yes |
| R2 validation (negative control at the 32768 cap) | yes | yes | yes |
| R2 real tiers (4096) | yes | yes | yes |
| R2 mitigation (4096, client trim) | yes | yes | yes |
| C3 gate (fake operator, no server, no NVIDIA change) | yes | yes | not applicable |
| R2 mechanism (4096, OLLAMA_DEBUG) | yes | yes | yes |

The first validation dry run ran at 131072 because the cap commit had not reached main yet; that file is kept as
`blade_dryrun_r2_validation_precap131072.jsonl` and is not used for rates.

## Validation diagnosis 2026-10-08

Run: `results/blade_r2_validation_v1.jsonl` (night 1, 2026-10-09 03:57Z), llama3.1:8b, Ollama 0.34.4, negative control
at the 32768 cap (3 sessions x 10 turns), positive control at 8192. Register: `blade-r2-validation-run1`.

**a. Effective context and truncation.** The server log (`C:\apu\blade\logs\blade_r2_validation_v1.ollama_serve.log`,
INFO level) shows `n_ctx_slot = 32768` on every negative-control request (8192 only for the positive control), 0
"truncating input" warnings and 0 context shifts. In the failed session (seed 20260903) Ollama's evaluated prompt
tokens grow with the transcript on every call and peak far below the window (largest value in the session's own rows,
turn 10), so no message trimming or token cut was possible. Message-level trimming is logged only at DEBUG, which this
job does not enable; the window margin, not a log line, is the evidence that none occurred. The client's own estimate
is above Ollama's count by the session's calibration ratio (`token_calib_ratio` on each row), an estimate difference, not truncation.

**b. The missed canaries.** The negative control's "2 canary misses" are ONE turn (seed 20260903, turn 5, canary pair
k=1) in which both the system-prompt canary and the history canary were absent. The model's entire final-answer
output on that call was a tool call written as text, not an answer:
`{"name": "lookup_fact", "parameters": {"key":"REC-0005"}}`.
Expected: the k=1 system canary and history tag in a JSON answer (as in the other 29 negative-control turns).
Verdict: **not a recall failure and not a scorer bug.** The model did not attempt the answer on the call where tools are
withheld (v1 protocol) and emitted another tool call as plain text, so no canary could appear; the scorer correctly
reports both canaries absent and counts them as 2 (system + history). The scorer is NOT changed: changing the gate
after seeing this run would make the gate in-sample. The two positive-control misses (turns 10 and 15, over the 8192
window) are the expected truncation detections.

**c. Sampling and model identity.** Every request sets temperature 0, seed 42, num_predict 384 (harness constant,
identical code on evo-x2); top_p and other sampling parameters are not set, so the model's own parameters layer applies.
That layer is identical on both machines: llama3.1:8b model, params and template layer digests on the Blade equal
evo-x2's (model `sha256:667b0c19...6a29`, params `sha256:56bb8bd4...4dcb`, template `sha256:948af274...cf85`), and both
run Ollama 0.34.4. Remaining differences: the backend (CUDA on the Blade, ROCm on evo-x2), the negative-control window
(32768 vs 131072) and therefore the numerics of greedy decoding, which can diverge on a borderline turn.

**Consequence for night 1 (operator plan):** rerun validation with the negative control at 5 sessions; if it has 0
canary misses, run the real tiers and mitigation scoring rules 1, 3 and 4; if not, run the mechanism instead and record
the Blade result (text tool calls on the final-answer call) as an open finding.
