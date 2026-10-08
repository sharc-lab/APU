# Blade plan: two overnight runs on the NVIDIA platform

The Razer Blade 14 (RTX 4070 Laptop GPU, 8188 MiB VRAM, CUDA) is the NVIDIA platform for the paper and also the
controller laptop. Blade jobs run overnight only, with all local Claude Code work paused while one runs; evo-x2's
remote queue keeps running. Nothing in this plan has run yet. Blade night 1 starts only after the operator says
"start Blade night 1".

Code: `harness/blade_common.py` (pins, versions, local Ollama, dry-run budget), `harness/blade_k1.py`,
`harness/blade_r2.py` (reuses `x2_r2_agent` and `x2_r2_mechanism` unchanged), `harness/blade_c3.py` (reuses
`blade_spill_sweep`), `harness/blade_queue.py` (the Blade queue), `scripts/blade_night.py` (the night wrapper),
`analysis/blade_hours_estimate.py` (hours from result files), tests in `tests/test_blade_night.py`.

## Versions (read 2026-10-08, read-only)

| component | pinned | found on the Blade | status |
|---|---|---|---|
| hostname | RITZLAPTOP (host_config hw_id `blade`, gpu_vendor nvidia, no ssh_host) | RitzLaptop | entry added |
| GPU | RTX 4070 Laptop, 8188 MiB | NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB, VBIOS 95.06.34.00.cd | ok |
| NVIDIA driver | 610.88 (as in C1/C2) | 610.88; nvidia-smi header "CUDA UMD Version: 13.3" (driver CUDA API) | ok |
| llama.cpp | b10970, commit bfdc32183, asset `llama-b10970-bin-win-cuda-12.4-x64.zip` (sha256 78c878ae...e52dad8) + `cudart-llama-bin-win-cuda-12.4-x64.zip` (sha256 8c79a9b2...e32ae1d6), CUDA runtime 12.4 | both zips in `C:\apu\bin`, sha256 equal to the GitHub release digests; extracted at `C:\apu\bin\llama-b10970-cuda` (`cudart64_12.dll`, `llama-server.exe --version` = build 10970, commit bfdc32183) | ok, nothing to download |
| Ollama | 0.34.4 (same as evo-x2), side by side at `C:\apu\bin\ollama-0.34.4\ollama.exe` | 0.34.1 tray install at `%LOCALAPPDATA%\Programs\Ollama` (bundles `cuda_v12` and `cuda_v13` runners) | NOT INSTALLED: steps below |

Every Blade job writes the full version record (`blade_common.versions_record`: driver, CUDA driver API, Ollama
version and exe, Ollama's bundled CUDA runner dirs, llama-server build and commit, the CUDA runtime DLLs next to it)
into its `run_start` row, and a real run refuses to measure on any mismatch with the pins (C3 ignores the Ollama pin,
it does not use Ollama). Which of Ollama's CUDA runners (v12 or v13) actually loads is read from the server log's
"inference compute" line by K1.

**Ollama 0.34.4 side by side (operator, not done; about 1.4 GB download):**

```
curl.exe -L -o C:\apu\bin\ollama-windows-amd64-v0.34.4.zip https://github.com/ollama/ollama/releases/download/v0.34.4/ollama-windows-amd64.zip
certutil -hashfile C:\apu\bin\ollama-windows-amd64-v0.34.4.zip SHA256
    (expect 535193f38f3344e5b08f5d1c171c31ce11aa17f0124ff69ae26d8ec7fe06fa62, size 1,461,155,106 bytes)
powershell -NoProfile -Command "Expand-Archive -Path C:\apu\bin\ollama-windows-amd64-v0.34.4.zip -DestinationPath C:\apu\bin\ollama-0.34.4"
C:\apu\bin\ollama-0.34.4\ollama.exe --version
```

The tray install is left as it is. The jobs stop every Ollama process (tray app included, or it respawns its server)
and start the pinned exe as a hidden child with its own log; both use the same model store
(`%USERPROFILE%\.ollama\models`).

**Models (operator, not done):** the Blade store has llama3.1:8b, llama3.2, gemma3:4b, qwen3:4b, qwen3:4b-instruct,
qwen3-4b-2507 (blob 85e4a5b7, the same GGUF as evo-x2's). Missing for night 1: qwen3:8b and qwen3-4b-2507-tools.

```
ollama pull qwen3:8b
py -3.12 -c "import sys,json; sys.path.insert(0,'harness'); from pathlib import Path; import x2_r2_4b_tools_validate as t; t.ensure_tools_tag(lambda r: print(json.dumps(r, default=str)[:600]), Path('results/blade_setup_4b_tools.jsonl'))"
```

The second line reuses evo-x2's create step unchanged (same GGUF digest check, same library template sha256 check
against qwen3:8b's template, LF Modelfile); it needs qwen3:8b pulled first. K1 records a missing model and moves on,
so night 1 can also run without either.

## Plan table

Hours from `py -3.12 analysis/blade_hours_estimate.py --markdown`. "measured" = the same work timed on the Blade
(C1's own per-ctx segments, load and thermal waits included); "scaled" = timed on evo-x2 for the same design times a
stated factor (`FACTORS` in that file: 1.0 for tiers that fit in 8 GB, 1.5 at 16384, 3.0 at 32768 where Ollama has to
put layers on the CPU, 5.0 for validation arm b at 131072, 1.2 for qwen3:8b, 1.5 for mitigation's per-call render
and tokenize). The factors are assumptions; K1 measures the offload they stand for.

| night | job | queue id / output | hours | measured / scaled |
|---|---|---|---|---|
| 1 | (a) K1: default context per model, device and VRAM, overflow at 4K/8K/16K/32K, llama.cpp CUDA defaults | `blade_k1_v1` / `results/blade_k1_v1.jsonl`, `.summary.json` | 0.50 | scaled (itemized) |
| 1 | (b) R2 validation and controls, llama3.1:8b (+ qwen3:8b if it fits) | `blade_r2_validation_v1` / `results/blade_r2_validation_v1.jsonl` | 0.58 | scaled |
| 1 | (b) R2 native, tiers default, 4096, 32768, 5 seeds x 40 turns, llama3.1:8b | `blade_r2_real_v1` / `results/blade_r2_real_v1.jsonl` | 6.90 | scaled |
| 1 | (b) the same for qwen3:8b, only if K1 says it fits | (same job and file) | +9.0 | scaled |
| 1 | (c) mitigation, x2_r2_mitigation_v1 design, Blade default (resolved from K1), 4096, 8192, 3 seeds | `blade_r2_mitigation_v1` / `results/blade_r2_mitigation_v1.jsonl` | 1.50 | scaled |
| 2 | (d) C3 half A: Prefer No Sysmem Fallback, ctx 36864/38912/40960/43008, 1+5 calls | `blade_c3_sysmem_fallback_v1` / `results/blade_c3_sysmem_fallback_v1.jsonl` | up to 2.29 | measured (C1 segments); minutes if the spilled points fail at load |
| 2 | (d) C3 half B: Driver Default restored, ctx 40960/43008, 1+3 calls | (same) | 0.80 | measured (C1 segments, scaled by call count) |
| 2 | (e) mechanism: render-only validity, message-drop and context-shift logging, llama3.1:8b, one session per tier (default, 32768, 16384, 8192, 4096) | `blade_r2_mechanism_v1` / `results/blade_r2_mechanism_v1.jsonl` | 2.71 | scaled |

Night totals: night 1 about 9.5 h with llama3.1:8b alone, about 18.5 h if qwen3:8b fits (it does not fit in one
night; the queue resumes it the next night, see below). Night 2 about 5.8 h, of which C3 needs the operator at the
keyboard for the two gates.

### Job details

- **K1** (`blade_k1.py`): device and VRAM from nvidia-smi and from Ollama's own log ("inference compute",
  "vram-based default context"); per model a short chat with no num_ctx, then `/api/ps` (context_length, size,
  size_vram; fits = size_vram at least 99% of size); a 4000-token calibration at num_ctx 8192; marker prompts of
  4096/8192/16384/32768 tokens at the default context (status, processed vs sent tokens, marker found); num_ctx
  4096/8192/16384/32768 each with a 1.5x prompt (processed vs the half-window rule floor(n/2)+2, size_vram); then
  llama-server b10970 CUDA with only `-m <Ollama blob> --port --log-file` (n_ctx from /props, layers offloaded, KV and
  model buffers, any fit lines). The 4B tools tag is measured for context policy only; it failed its R2 validation on
  evo-x2 (register `R2-install-path-4b-comparison`) and is not in any R2 job here.
- **R2** (`blade_r2.py`): x2_r2_agent's driver, scoring and gates unchanged, v1 protocol (call-2 mode off), seeds
  20260901..20260905 for the real run, seed-major order so an interrupted night leaves every cell with as many seeds
  as possible. qwen3:8b joins validation and the real run only if `blade_k1_v1.summary.json` says it is fully on the GPU
  at its default context; the decision and its reason are a `run_start` field. The real run refuses per model unless
  the Blade's own validation file passes `validation_preflight_per_model`. Validation keeps evo-x2's design exactly
  (arm b num_ctx 131072), which on 8 GB means Ollama runs most layers on the CPU (see open questions).
- **Mitigation** (x2_r2_mitigation_v1's design, pre-registered in FINDINGS 2026-10-08, on main):
  `blade_r2.py --mode mitigation --client-trim margin=0.05`, tiers default (resolved to K1's measured num_ctx for
  llama3.1:8b), 4096 and 8192, seeds 20260901..20260903, 40 turns, v1 call-2 mode, Ollama under OLLAMA_DEBUG=1 with the
  mechanism log parsing. The body is `x2_r2_agent.run_mitigation`, the same function `x2_r2_agent.main --mode
  mitigation` runs on evo-x2 (extracted from main unchanged, with `x2_r2_agent.build_arg_parser()`, so both machines run
  one code path). The job builds the equivalent x2_r2_agent argv, parses it with x2_r2_agent's own parser and
  `x2_r2_client_trim.parse_client_trim`, and records it in `run_start` (`x2_r2_agent_equivalent_argv`); the stub night
  runs the same parse. Only the server differs: the Blade's hidden local Ollama with its own log instead of the WMI
  launch. A K1 default other than x2_r2_agent's fixed tiers (e.g. 40960) is registered as an arm at run time with the
  same fields.
- **C3** (`blade_c3.py`): C1's harness per call (qwen3-4b-instruct, f16 KV, `-fa on -ngl 99 -np 1 -t 4`, 90% fill,
  thermal gate, stale-server guard, nvidia-smi, dmon and per-PID Shared Usage), every ctx attempted (no skip after a
  failure), half A order shuffled with seed 20261008. Each half waits for its confirmation flag; no timeout.
- **Mechanism**: `x2_r2_mechanism.MechanismRuntime` under OLLAMA_DEBUG=1, prompt checks with llama-tokenize against a
  fresh-load prompt_eval_count, a tier citable only with at least 5 checks all within 1%.

## C3 operator steps (about two minutes per half)

Night 2 starts with C3, so the operator starts night 2 at the keyboard and stays through both gates (about 3 h in the
worst case, minutes if the spilled points fail at load), then leaves; the mechanism job runs unattended after that.

Before half A (the job writes `C:\apu\blade\c3_WAITING_A.txt` with these steps and waits):

1. Right-click the desktop, NVIDIA Control Panel, 3D Settings, Manage 3D settings, Global Settings tab.
2. Find "CUDA - Sysmem Fallback Policy", select "Prefer No Sysmem Fallback", press Apply.
3. Close and reopen NVIDIA Control Panel, go back to the same row and read it: it must say "Prefer No Sysmem
   Fallback". This readback is the verification; nvidia-smi does not expose this setting (C1 recorded it as UNKNOWN
   for that reason). If NVIDIA Profile Inspector is installed, its global profile row of the same name is an
   optional second readback; it is not installed and not required.
4. Create `C:\apu\blade\c3_confirm_A.flag` whose first line is exactly `Prefer No Sysmem Fallback`, e.g.
   `Set-Content -Path C:\apu\blade\c3_confirm_A.flag -Value "Prefer No Sysmem Fallback"`.

Before half B (`c3_WAITING_B.txt`): the same steps with "Driver Default" and `c3_confirm_B.flag`, first line exactly
`Driver Default`. This also restores the setting; leave it on Driver Default afterwards.

The gate accepts a flag only if it was written after the wait began (a flag left from an earlier run is renamed
`.stale`) and its first line matches exactly; a wrong value is logged as `c3_gate_rejected` and the wait continues.
There is no timeout and no automatic start: if nobody confirms, night 2 waits, and the waiting file says so.

## Night protocol

```
(operator, once all local Claude Code work is paused)
New-Item -ItemType File C:\apu\blade\CLAUDE_CODE_PAUSED.flag
(operator says "start Blade night 1", then)
New-Item -ItemType File C:\apu\blade\START_BLADE_NIGHT_1.flag
py -3.12 scripts\blade_night.py --night 1
```

`scripts/blade_night.py`:

1. refuses unless `START_BLADE_NIGHT_<N>.flag` exists (renamed `.used` at start, so it never starts a second time);
2. refuses unless on AC power and the paused condition holds: `CLAUDE_CODE_PAUSED.flag` present and no other python,
   pytest, git or harness llama-server process outside the wrapper's own process tree;
3. writes the current AC standby, AC hibernate and AC lid-close values and the active scheme to
   `C:\apu\blade\logs\night<N>_power_<ts>.json` before changing anything, then sets all three to 0 (never sleep, never
   hibernate, lid does nothing) with `powercfg /setacvalueindex SCHEME_CURRENT ...` and `/setactive SCHEME_CURRENT`.
   If an earlier night's power log never reached `restored: true`, its originals are used instead of the values read
   now (which would be this wrapper's own zeros);
4. runs the night's jobs in order through the Blade queue, with resume;
5. in a finally block restores every original value, reads them back, writes both to the power log, then writes
   `results/blade_night<N>_summary_<ts>.json`.

Keep the lid open anyway; the lid setting is only a backstop. The wrapper never touches NVIDIA settings.

### The Blade queue

`harness/blade_queue.py`: its own state file `C:\apu\blade\blade_queue_state.json`, not evo-x2's `queue_state.json`
and not a Windows scheduled task; the wrapper runs it in-process. Each job runs as a hidden child
(`proc_util.popen_hidden`) from the repo root, log `C:\apu\blade\logs\<job_id>.log`. Status pending, running, done,
error or blocked_dependency. A job left "running" (the night was interrupted) and an errored job run again on the
next start of that night; done and blocked jobs do not. Each job resumes inside its own output file (completed R2
sessions and C3 halves are skipped), so a rerun never repeats finished work. Before every job the queue checks the
paused condition again; deleting `CLAUDE_CODE_PAUSED.flag` stops the night before its next job. To continue night 1
on a second night, create `START_BLADE_NIGHT_1.flag` again.

Nothing here can leave a machine needing a manual resume without saying so: the only waits are C3's two gates,
which by the operator's design wait for a person and say so in `c3_WAITING_<half>.txt`.

## Dry runs

**Stub night (fakes, no model, no setting changed): PASS for both nights.** `py -3.12 scripts/blade_night.py --night
1 --stub` and `--night 2 --stub` run the wrapper with a fake powercfg (powercfg's own output format), fake AC, fake
paused condition and a fake job runner that parses each job's argv with that job's own argument parser. Recorded in
`results/blade_dryrun/blade_dryrun_night1_stub.json` and `..._night2_stub.json`: the three no-sleep sets and the
setactive come before the first job, the three restores and the setactive after the last job, the final values equal
the originals, every output path is prefixed `blade_`. Unit tests (`tests/test_blade_night.py`) add restore on a
crashed queue and on a KeyboardInterrupt in a job, refusal without the start flag, on battery and with a heavy
process, the unrestored-log fallback, queue resume, and the C3 gate (stale flag moved, wrong text rejected, exact text
accepted, no timeout of its own over 48 simulated hours).

**60-second live dry runs (operator request of 2026-10-08): NOT RUN.** The first one (K1 on llama3.2) was refused by
the session's permission classifier before it started, so no model was loaded, no Ollama was started and nothing was
cleaned up because nothing ran. The tray Ollama (0.34.1, no model loaded, VRAM 0 MiB) was left as it was. The exact
refused command (worktree root written as `<worktree>`, the LOCALAPPDATA path spelled out in the original):

```
cd <worktree> && date -u +%FT%TZ && timeout 600 py -3.12 harness/blade_k1.py --out results/blade_dryrun/blade_dryrun_k1.jsonl --models llama3.2 --dry-run-seconds 60 --allow-version-mismatch --ollama-exe "%LOCALAPPDATA%/Programs/Ollama/ollama.exe" 2>&1 | tail -30; echo "rc=$?"; date -u +%FT%TZ; tasklist | grep -i -E "ollama|llama"
```

To run them, one at a time, from the repo root (each stops itself at the first result row after 60 s, stops every
Ollama process and runner in its finally block, writes only under `results/blade_dryrun/`; `--allow-version-mismatch`
lets the installed 0.34.1 stand in for the pin and is refused outside a dry run). After each, check
`tasklist | findstr /i "ollama llama-server"` is empty:

```
py -3.12 harness\blade_k1.py --out results\blade_dryrun\blade_dryrun_k1.jsonl --models llama3.2 --skip-llamacpp --dry-run-seconds 60 --allow-version-mismatch --ollama-exe %LOCALAPPDATA%\Programs\Ollama\ollama.exe
py -3.12 harness\blade_r2.py --mode validation --out results\blade_dryrun\blade_dryrun_r2_validation.jsonl --models llama3.2 --seeds 20260901 --turns 3 --dry-run-seconds 60 --allow-version-mismatch --ollama-exe %LOCALAPPDATA%\Programs\Ollama\ollama.exe
py -3.12 harness\blade_r2.py --mode real --out results\blade_dryrun\blade_dryrun_r2_real.jsonl --models llama3.2 --tiers 4096 --seeds 20260901 --dry-run-seconds 60 --allow-version-mismatch --ollama-exe %LOCALAPPDATA%\Programs\Ollama\ollama.exe
py -3.12 harness\blade_r2.py --mode mitigation --client-trim margin=0.05 --out results\blade_dryrun\blade_dryrun_r2_mitigation.jsonl --models llama3.2 --tiers 4096 --seeds 20260901 --dry-run-seconds 60 --allow-version-mismatch --ollama-exe %LOCALAPPDATA%\Programs\Ollama\ollama.exe
py -3.12 harness\blade_c3.py --out results\blade_dryrun\blade_dryrun_c3_gate.jsonl --dry-run-gate
py -3.12 harness\blade_r2.py --mode mechanism --out results\blade_dryrun\blade_dryrun_r2_mechanism.jsonl --models llama3.2 --tiers 4096 --dry-run-seconds 60 --allow-version-mismatch --ollama-exe %LOCALAPPDATA%\Programs\Ollama\ollama.exe
```

The C3 one runs the gate logic with a
fake operator only (no server, no NVIDIA setting). After the last one, restart the tray Ollama from the Start menu if
it is wanted; the dry runs stop it.

| job | launched? | wrote results? | cleaned up? |
|---|---|---|---|
| stub night 1 (all four jobs, fakes) | yes | yes (`blade_dryrun_night1_stub.json`) | yes (fake power values back to originals) |
| stub night 2 (both jobs, fakes) | yes | yes (`blade_dryrun_night2_stub.json`) | yes |
| blade_k1 60 s, llama3.2 | no: refused by the permission classifier | no | nothing to clean |
| blade_r2 validation / real / mitigation / mechanism 60 s | no: not attempted after the refusal (same outcome) | no | nothing to clean |
| blade_c3 gate with a fake | no: not attempted after the refusal | no | nothing to clean |

## Open questions

1. Validation arm b at num_ctx 131072 does not fit in 8 GB (16 GiB of f16 KV for llama3.1:8b), so the no-truncation
   control runs mostly on the CPU (estimate scaled x5). The alternative is a Blade arm b at 32768 (llama3.1:8b
   exceeded a 32768 window only at turn 23 in x2_r2_real_v1, per the MECH_MODEL note in `x2_r2_agent.py`, so a
   10-turn session stays inside it), but that changes the design and
   needs `x2_r2_agent`'s gate code to accept another arm name. Kept as on evo-x2 unless the operator decides otherwise.
2. qwen3:8b may fit at a 4096 default (its Q4_K_M GGUF is 5,027,783,488 bytes per `docs/HARDWARE.md`; Ollama's
   library build was not checked); K1 decides. If it runs, night 1 is about 18.5 h
   and the real run continues on a second night.
3. If K1 shows the Blade default is 4096, the default tier is a repeat of the 4096 tier with no num_ctx sent; kept as
   asked (it tests the request path, not a different window).
4. The 32768 tier and the mechanism's 16384 and 32768 tiers run with layers on the CPU on 8 GB; their time cost is a
   guess until K1 reports size_vram at those windows.
5. Mitigation needs the llama-tokenize exe at `C:\apu\bin\llama-b10970\llama-tokenize.exe`
   (`prompt_token_check.TOKENIZE_EXE`); it is present on the Blade (Vulkan build dir, tokenizing is CPU-only).
6. C3 sets the policy globally (as asked). A per-program setting for `llama-server.exe` would avoid touching other
   CUDA programs; the operator can choose either, the gate text is the same.
7. llama3.1:8b on the Blade: manifest model digest 667b0c19...6a29; whether it equals evo-x2's digest has not been
   checked (no SSH this session). K1 records the digest in its rows via `/api/ps`.
