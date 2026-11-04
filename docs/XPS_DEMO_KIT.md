# XPS demo kit: the four-arm demo comparison on Zach's XPS

Kit prepared 2026-11-03. Nothing has been run with it. The XPS (Dell XPS, 16 GB, Intel, Qwen3-4B) is Zach's machine
and is not configured in this repo (harness/host_config.py knows evo-x2 and evo-t2s only; docs/EXPERIMENTS.md, E1
metadata, says the same). Nothing here reaches the XPS: Zach runs the local step on it, sends the file, and the
operator runs the cloud step on the controller.

## What it measures

The same 90 jobs as the B3 v2 demo: 30 B3 v2 tasks per session (harness/x2_r2_evalset.py, workload evalset, guardrails
off, design v2) x seeds 20260901, 20260902, 20260903. Each job is one task turn, run with the repo's own tool loop
(`x2_r2_evalset.run_turn`, max_tokens 384, temperature 0, thinking off) and graded by `evalset_env.check_task`. Per arm:
correct/90, cloud USD per job, latency per job.

| arm | id | what it does |
|---|---|---|
| (a) | `local_default` | Qwen3-4B on Zach's runtime at its own defaults. The client sets no window (no `num_ctx`). The kit refuses this arm while any override is found (below), unless `--allow-nondefault "<reason>"` is given, and records the reason. |
| (b) | `standard_hybrid` | The standard hybrid: Ollama-style local first, LiteLLM-style cloud fallback, and a prompt-length check against the window. Before each local call, if the estimated prompt (`x2_r2_agent.prompt_tokens_est`, the harness's chars/4 estimate of messages plus tools) is larger than the runtime's window, the task goes to the cloud; otherwise it runs locally. A hard error (HTTP status other than 200, connection error, non-ok outcome) also sends it to the cloud. A silent truncation is not an error, so it stays local. |
| (c) | `full_pipeline` | The Spillway pipeline as in the repo (`analysis/hybrid_full_pipeline.run_pipeline`): route at start (the first 3 turns local, then `SessionGrowthPredictor`, k = 3, against the window), the post-run check on every local task (`src/dse/post_run_check.check_record` with `l3_invariants.CONFIG_TUNED`: L1 recorded call errors, L1 PROXY over-window, L2 PROXY, L3 task rules; on llama-server also the real L1 response class), and on a FAIL a rollback to the pre-task snapshot and a cloud redo (`scripts/cloud_redo.rollback_env` + `live_turn`, as `scripts/post_run_check_cloud.py`). |
| (d) | `all_cloud` | Every task on the cloud model (`scripts/all_cloud_b3.run_cloud_session`). |

Cloud model: claude-sonnet-5-5 (the price file's escalation model), rolling cache breakpoint, as every live cloud job
since 2026-11-02.

Operator decision 2026-11-03: the standard hybrid is the every-call prompt-length check, as here. PITCH-1
(analysis/pitch1_standard_hybrid.py, `STD`) now uses the same check as its primary, so the kit and the PITCH-1 headline
agree; the first-call check is reported there as a sensitivity only.

**Declared approximations (arms b and c).** These are the same as the replay rows. (1) A task sent to the cloud is
deferred in the local step: no local call is made for it, and the local transcript and environment carry on without
it. The cloud step then redoes it from the pre-task state with the full client-side history (`cloud_redo.rebuild`). A
task the post-run check escalates stays in the local transcript. So later local turns never see a cloud answer. (2) For
arm b the fallback is per task, not per call. A check that trips mid-turn redoes the whole task. (3) Latency of a
cloud task = local seconds before it was sent (measured on the XPS) + cloud request seconds (measured from the
controller, not from the XPS).

## Runtime record and the default-settings assertion

Every local run writes `run_start.runtime_config`, which records:

- the runtime (`ollama`, `llamacpp` or `openvino`; for OpenVINO, OpenVINO Model Server with its OpenAI-compatible
  `/v3` endpoint);
- its version: Ollama from `GET /api/version`, llama-server from `GET /props` `build_info`, and from
  `<exe> --version` when `--runtime-exe` is given; the `openvino` / `openvino-genai` package versions where Python can
  see them;
- the default window and where it was read:
  - Ollama: `/api/ps` `context_length` after the warm-up call, with no `num_ctx` sent;
  - llama-server: `/props` `n_ctx`;
  - OpenVINO: declared with `--declared-default-window`, because OVMS does not report it;
- the model;
- every override found:
  - environment variables in the process, user and machine environment (the registry, which is what a tray-started
    runtime inherits) with the prefixes `OLLAMA_` (except `OLLAMA_HOST`, `OLLAMA_MODELS` and `OLLAMA_ORIGINS`, which
    are recorded as benign), `GGML_`, `LLAMA_ARG_`, `LLAMA_`, `OV_`, `OPENVINO_`, `OVMS_`, `SYCL_` and `ZE_`, plus
    `ONEAPI_DEVICE_SELECTOR`, `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` and `IPEX_LLM_*`;
  - Ollama Modelfile parameters (`/api/show` `parameters`, for example `num_ctx`);
  - llama-server launch flags other than model, address, `--jinja` and logging (read from the listening process's
    command line);
  - OVMS config keys that set the device, precision, KV cache or batching (`--runtime-config-file`).

Arm (a) is refused while any override is present. It is also refused if the settings could not be read: a llama-server
command line that cannot be read, an OVMS config file that was not given, or an OpenVINO default window that was not
declared. Pass `--allow-nondefault "<reason>"` to run anyway; the reason is recorded. Arms (b) and (c) are never
refused for an override. Arm (c) may set its own window (`--pipeline-window`).

## Step 1 (Zach, on the XPS, USD 0): the local step

Requirements: Python 3.11 or 3.12, `pip install httpx` (imported by the harness), git, and the runtime with Qwen3-4B
already pulled or loaded. No API key and no cloud access. Run everything from an ordinary PowerShell terminal at
the keyboard.

The tasks need tool calls, so the runtime must return them. Ollama does this as installed. llama-server needs
`--jinja`, which the kit does not count as an override. OpenVINO Model Server needs its tool parser for Qwen3 (for
example `--tool_parser hermes3`). The kit does not count the tool parser as an override, but say which one you used.
Thinking is turned off per request (`think: false` on Ollama, `chat_template_kwargs.enable_thinking: false` on the
other two). A turn where thinking still shows is scored as failed and logged.

1. Clone the repo at the commit the operator names, and run the offline self-test (fake runtime, a few seconds). It
   must print `SELFTEST OK`:

   ```powershell
   git clone https://github.com/sharc-lab/APU.git C:\xps_kit\APU
   cd C:\xps_kit\APU
   git checkout <commit named by the operator>
   py -3.12 harness\xps_demo_kit.py selftest
   ```

2. Start the runtime at its own defaults: no context or KV setting, no extra environment variables. Then record what
   the kit sees (no model call). Use the block for your runtime:

   ```powershell
   # Ollama
   py -3.12 harness\xps_demo_kit.py probe --runtime ollama --model qwen3:4b --runtime-exe "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
   # llama.cpp (start it with only -m, --port and --jinja, for example: llama-server.exe -m C:\models\Qwen3-4B-Q4_K_M.gguf --port 8080 --jinja)
   py -3.12 harness\xps_demo_kit.py probe --runtime llamacpp --model qwen3-4b --runtime-exe C:\llama\llama-server.exe
   # OpenVINO Model Server (give its config file and the default window you read from its docs or config)
   py -3.12 harness\xps_demo_kit.py probe --runtime openvino --model Qwen3-4B --runtime-exe C:\ovms\ovms.exe --runtime-config-file C:\ovms\models\config.json --declared-default-window <N>
   ```

   The last line says whether overrides are present. If it says YES, remove them and restart the runtime. Unset the
   variable (also in System Properties > Environment Variables), drop `-c` and the KV flags, and quit and restart the
   Ollama tray app. Then probe again. If the override is how the machine has to run, keep it and plan to pass
   `--allow-nondefault "<why>"` in step 3.

3. Run arms (a), (b) and (c) at the runtime's defaults. This is 270 task turns plus one warm-up call. It writes one file
   and never overwrites it:

   ```powershell
   py -3.12 harness\xps_demo_kit.py run --runtime ollama --model qwen3:4b --runtime-exe "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --arms a,b,c --out C:\xps_kit\out\xps_demo_local_v1.jsonl --note "<anything worth saying>" *>&1 | Tee-Object C:\xps_kit\out\console_local_v1.txt
   ```

   (Use the same `--runtime`, `--model`, `--runtime-exe` and, for OpenVINO, `--runtime-config-file` and
   `--declared-default-window` as in step 2.)

4. Only if arm (c) should run at a tuned window, which is what the pipeline is allowed to do:
   - Ollama: add `--pipeline-window <N>` to a separate run of arm (c). The kit sends it as `num_ctx`.
   - llama.cpp or OVMS: restart the server with that window, then run arm (c) alone into a second file:

   ```powershell
   py -3.12 harness\xps_demo_kit.py run --runtime llamacpp --model qwen3-4b --arms c --pipeline-window <N> --out C:\xps_kit\out\xps_demo_local_v1_part2.jsonl *>&1 | Tee-Object C:\xps_kit\out\console_local_v1_part2.txt
   ```

   If you do this, run step 3 with `--arms a,b` only, so that arm (c) is in one file.

If a session stops on a connection error that persists after one retry, the kit writes an `r2a_session_incomplete` row
and moves on to the next session. Do not rerun silently. Say so in the note or the email.

## Step 2 (operator, on the controller): intake and the cloud step

1. Put Zach's files in `results/xps_demo/` under these names: `xps_demo_local_v1.jsonl`, plus
   `xps_demo_local_v1_part2.jsonl` if step 4 was used. Commit them as received.
2. Estimate (no key, no network). This gives each arm's cloud task count and how it splits into jobs under USD 2:

   ```powershell
   py -3.12 scripts\xps_demo_cloud.py estimate --local results\xps_demo\xps_demo_local_v1.jsonl
   ```

3. Fill in `results/console_usage_check.json` from the Anthropic Console (`src/cloud/console_gate.py`, TEMPLATE). Then
   run each cloud job by hand. Every job is refused unless it has `--live`, the key in this process,
   `--operator-approval` (always required here, confirmed at the console and written to the ledger before the first
   call), an estimate within the job cap, and a job cap of at most USD 2.00. The Console check must also pass. Nothing
   here raises or skips a cap. Arm (d) runs as one job; arms (b) and (c) run as one or two parts each (`--part 2` when
   the estimate says so):

   ```powershell
   $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')
   $env:CLOUD_CLIENT_HOST_ROLE = 'controller'
   py -3.12 scripts\xps_demo_cloud.py run --arm d --live --local results\xps_demo\xps_demo_local_v1.jsonl --out results\xps_demo\xps_demo_cloud_d.jsonl --ledger <main checkout>\results\cloud_ledger.jsonl --operator-approval "approve xps_demo_cloud_d up to USD <n>: <reason>"
   py -3.12 scripts\xps_demo_cloud.py run --arm b --live --local results\xps_demo\xps_demo_local_v1.jsonl --out results\xps_demo\xps_demo_cloud_b.jsonl --ledger <main checkout>\results\cloud_ledger.jsonl --operator-approval "approve xps_demo_cloud_b up to USD <n>: <reason>"
   py -3.12 scripts\xps_demo_cloud.py run --arm c --live --local results\xps_demo\xps_demo_local_v1.jsonl results\xps_demo\xps_demo_local_v1_part2.jsonl --out results\xps_demo\xps_demo_cloud_c.jsonl --ledger <main checkout>\results\cloud_ledger.jsonl --operator-approval "approve xps_demo_cloud_c up to USD <n>: <reason>"
   Remove-Item Env:\ANTHROPIC_API_KEY
   ```

   A second part writes `..._cloud_<arm>_part2.jsonl`, with job id `xps_demo_cloud_<arm>_part2`.

   Arm (d) is optional. `results/all_cloud_b3_v1.jsonl` already holds a measured claude-sonnet-5-5 all-cloud run on the
   same 90 jobs (row `all-cloud-b3-cost-rolling-cache`). Skipping (d) saves its cost. If you skip it, quote that row
   for (d) and say it was run from a different machine on an earlier date.

4. The table:

   ```powershell
   py -3.12 scripts\xps_demo_cloud.py summarize --local results\xps_demo\xps_demo_local_v1*.jsonl --cloud results\xps_demo\xps_demo_cloud_*.jsonl
   ```

   Register row `xps-demo-comparison` computes the same from these files once they all exist with `run_end`.

## Cost estimate for the cloud step

From register row `xps-demo-cloud-estimate` (computed from `results/all_cloud_b3_v1.jsonl`, the full-history redo files
and the price file, at the rolling cache breakpoint; same inputs as rows `all-cloud-b3-cost-rolling-cache` and
`operating-point-q4-28928-rolling-cache`), pasted:

> ESTIMATE, not run: model claude-sonnet-5-5, rolling cache breakpoint; arm d (all-cloud, 90 jobs) = 90 x USD
> 0.00936/task (all_cloud_b3_v1 repriced, 90/90 correct there; recorded placement 0.01020) = USD 0.8428, one job under
> the USD 2.00 cap; arms b and c, per task sent to the cloud: USD 0.04065 (highest tier-mean full-history redo,
> ollama_default), so 0 to 90 tasks each = USD 0 to 3.6581 per arm, at most 49 tasks per USD 2.00 job (2 jobs if all 90
> go to the cloud); worst case all three arms USD 8.1589; the b and c counts are known only after the local step on the
> XPS (scripts/xps_demo_cloud.py estimate --local)

The per-task figure for arms b and c is deliberately high. It is the highest tier mean, from the default-window tier,
where the histories are longest. The real b and c counts depend on the XPS's default window, which is unknown until the
local step has run. The per-task cap is USD 0.25 for every cloud task of arms b and c.

## File formats

### Local step output (`xps_demo_local_v1*.jsonl`, written by harness/xps_demo_kit.py)

- `run_start`: `harness`, `kit_version`, `host` (hostname, platform, processor, python), `runtime_config` (above), `arms`,
  `seeds`, `design`, `window_default`, `window_pipeline`, `allow_nondefault`, `kit_git_blob`, `evalset_git_blob`,
  `operator_note`.
- `warmup`: the excluded warm-up call.
- `r2a_turn`: one per local task turn. It has the fields of the repo's B3 v2 rows (`calls`, `end_state_ok`,
  `turn_wall_s`, `loaded_context`, `over_loaded_window`, ...) plus `arm`, `arm_id`, `runtime` and `window`. Arm (c)
  rows also carry `prc_check` (decision, layers fired, check seconds) and `local_seconds_with_check`.
- `xps_deferred`: a task sent to the cloud by arm (b) (`reason` `over_window` or `hard_error`, with `detail`) or by arm
  (c) (`route_at_start`, with the projection), plus `local_seconds`.
- `r2a_session`, or `r2a_session_incomplete`: one per session.
- `run_end`.

### Zach's own XPS results (`results/xps_zach/*.jsonl`, read by analysis/xps_intake.py)

There is one JSONL file per kind. The file name is fixed per kind: `ceiling.jsonl`, `tuned_window.jsonl`,
`gpu_share.jsonl`, `loud_failure.jsonl`.

- **First line:** `{"record": "run_start", "kind", "run_id", "ts_utc", "host": {...}, "runtime_config": {"runtime",
  "version", "default_window", "model", "device", "kv_precision", "overrides"}, "tool": {"name", "commit"}}`.
- **Last line:** `{"record": "run_end", "run_id", "ts_utc"}`.
- **Between them:** one record per configuration, task, trial or call. Each record carries the run's `run_id`:

| kind | record | fields | register row(s), PENDING until the file exists with run_end |
|---|---|---|---|
| ceiling | `ceiling_config` | config_id, config, split (fit or heldout), predicted_ceiling_tokens, measured_ceiling_tokens, measured_method | `xps-zach-ceiling-prediction` (held-out mean percent error, held-out within +/-5%); `xps-zach-ceiling-within-5pct` (configs within +/-5% over all configs) |
| tuned_window | `window_task` | config (default or tuned), window_tokens, seed, task_id, ok | `xps-zach-tuned-window` |
| gpu_share | `gpu_share_trial` | gpu_share_pct, trial, ok | `xps-zach-gpu-share` |
| loud_failure | `overflow_call` | prompt_tokens, window_tokens, outcome (error or ok), http_status, error_text, latency_s | `xps-zach-intel-loud-failure-latency` |

Percent error = 100 x |predicted - measured| / measured. The rows compute only from these files. What Zach reported
(5.1% mean error and 4/4 held-out; a tuned window of 18,000 to 40,448 with 30/30; GPU share 50% 6/6 vs 74% 0/6; an
Intel loud-failure latency; within +/-5% over 14 configs) appears only in the row descriptions, labelled as reported.
None of it is a value.

## What to send back (Zach)

1. `C:\xps_kit\out\xps_demo_local_v1.jsonl`, and `xps_demo_local_v1_part2.jsonl` if step 4 was used, unedited.
2. The console transcripts `console_local_v1*.txt` and the output of the step 2 probe.
3. The commit you ran (`git rev-parse HEAD`). Also any local change to the repo (`git status`, `git diff`); there
   should be none.
4. The runtime install details the kit cannot see: how the runtime was installed, the model file or tag and its
   quantization, and for OpenVINO the source of the default window you declared.
5. Any `--allow-nondefault` reason you used, and any session the console showed as incomplete.
6. Your own XPS results as raw files in the format above (`ceiling.jsonl`, `tuned_window.jsonl`, `gpu_share.jsonl`,
   `loud_failure.jsonl`), with the run IDs, and the tool or commit that produced them.
