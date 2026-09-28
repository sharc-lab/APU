# Join key between the evo-t2s rows and Zachary's Axis B files

Purpose: let a row of `results/t2s_overnight_*.jsonl` (Axis A, memory and runtime behavior on evo-t2s) be joined to a
measurement in Zachary's `results/zachary/` files (Axis B, orchestration and host CPU cost) for Fig 6.1. His files were
read and not modified.

## 1. What a row carries

Every call, probe and server-start row has all of these keys (null when not applicable). Definitions:

| field | meaning |
|---|---|
| `hw_id` | machine label, always `evo-t2s` (never `evox2_evo-t2s`) |
| `cpu_name` | exact `Win32_Processor.Name` |
| `gpu_name`, `gpu_driver_version` | `Win32_VideoController` name and driver version |
| `backend` | compute backend of llama-server: `vulkan` or `sycl` |
| `llama_build` | `build_info` from `/props` |
| `model_id`, `model_sha256`, `quant` | model label, SHA-256 of the GGUF, quantization from GGUF metadata |
| `kv_type`, `flash_attn` | always `f16` and `on` in this run |
| `mmap` | true for the build default (`--load-mode auto`, mmap when the device supports it) and for `--load-mode mmap`; false for `--load-mode none`. b10970 has no `--no-mmap` flag |
| `load_mode` | the `--load-mode` value passed to llama-server: `auto`, `mmap` or `none` |
| `n_ctx`, `prompt_tokens` | server context size and measured prompt length in tokens |
| `co_runner` | `none`, `p4` (logical CPUs 0 to 3), `e4` (4 to 7), `nonp12` (4 to 15), `all16` |
| `proc_throttle_max` | active power plan `PROCTHROTTLEMAX` on AC, percent |
| `mem_headroom_gb` | Section C only: free memory after the lock minus (weights + KV + compute), GiB; null otherwise |
| `section` | `0`, `A`, `B`, `C` or `D` for `t2s_overnight_*` files; `AM` for `t2s_amech_*` files; `B1`, `B2`, `B3`, `C1` or `PB` for `t2s_night2_*` files |
| `rep`, `seed` | repetition index within the condition (-1 is the discarded warm-up) and the run's order seed |
| `git_sha`, `script_sha` | git head of the deployed scripts and git blob SHA of `harness/t2s_overnight.py` |
| `load_s` | seconds from server spawn to healthy |
| `ttft_s`, `decode_tok_s`, `e2e_s` | time to first content token, decode tokens per second over the stream, total request time |
| `output`, `score` | probe output text (probes only) and scorer result |
| `igpu_mhz` | median Level Zero Sysman actual frequency, GPU domain, over the call window |
| `pkg_power_w` | median RAPL package power (Windows Energy Meter) over the call window |
| `shared_usage_mib` | maximum GPU Process Memory Shared Usage of the server PID over the window |
| `total_committed_mib` | maximum `\Memory\Committed Bytes` over the window |
| `pages_input_per_s` | maximum `\Memory\Pages Input/sec` over the window |
| `hard_faults_per_s` | maximum `\Memory\Page Reads/sec` over the window (disk reads that satisfied hard faults; Windows has no direct hard-fault rate counter) |
| `error` | verbatim error text, if any |

`t2s_amech_*` rows (section `AM`) add: `phase` (`rope`, `bisect`, `arms`, `fill`), `ngl` (the `-ngl` value passed, null when the flag was left unset), `fit` (the `-fit` value passed, null when unset), `arm` (`a_ngl99`, `b_fit_default`, `c_fit_off`, `last_pass_ngl99`), `cmdline` (the exact llama-server command line), `projected_mib` and `fit_free_mib` (llama.cpp's own projection of device memory and the free device memory it compared it with), `layers_gpu` and `layers_total`, `vk_errors` and `alloc_failed` (the Vulkan result text and failing buffer size from the server log), `fill_used`, `fill_target_90pct` and `fill_capped` (prompt tokens used against 90% of `n_ctx`), and for bisection `bisect_label`. A `bisect_probe` record gives `started` and `props_n_ctx_cap` when the server created its context but `/props` reported a smaller `n_ctx`.

Rows also carry `item_id`, `kind` (`call`, `probe`, `start`), `warmup`, `t_start_utc`, `t_end_utc`, `server_pid`, the
thermal-gate record and, where relevant, `n_reduced` (3 measured calls instead of 5) and `valid`.

`t2s_night2_*` rows add: `cpu_mask` (the hex affinity mask passed to the co-runner, null for `none`), `duty_cycle_pct`
(Section B2 only), `perfboostmode` (Section PB only, the value it was set to), `cpu_p_pct_perf`, `cpu_e_pct_perf`,
`cpu_lpe_pct_perf` (median Windows `% Processor Performance` over the call window for the P, E and LP-E logical CPUs,
grouped by the fixed preflight mapping P=0-3, E=4-11, LP-E=12-15), `cpu_n_cores_read` (how many logical CPUs the
counter returned that sample), `rapl_pp0_w`, `rapl_pp1_w` (best-effort RAPL cores/uncore power, null when the platform
does not expose those Energy Meter instances), `rapl_available` (whether any such instance was found at all). A
`b1_positive_control` record gives the one-shot Sysman throttle-reason check for the `all16` condition.

## 2. Mapping onto Zachary's `replication_remote_search_v3.json`

His file is one aggregate document (5 seeds, 10 sessions each), not per-call rows. Fields found at the top level and
under `setup_ref`, `git`, `env`, `config`, `aggregate`, `per_seed_artifacts`, `audit`.

| our field | his field | notes |
|---|---|---|
| `cpu_name` | `setup_ref.cpu_model` (`Intel(R) Core(TM) Ultra 5 325`); `env.cpu_model` is `unknown` in this file | the only hardware identity he records. Different CPU from ours (`Intel(R) Core(TM) Ultra X7 358H`) |
| logical cores (manifest) | `env.cores_logical` (8), `env.cores_physical` (8) | ours is 16 logical |
| RAM (manifest, 63.49 GB) | `env.ram_gb` (7.56) | different machine class |
| OS | `env.platform` (`Linux ... microsoft-standard-WSL2`) | his run is Linux under WSL2, ours is native Windows 11 |
| `ts_utc` | `generated_utc` | wall-clock join only |
| `seed` | `config.seeds`, `per_seed_artifacts[].config.seed` | his seeds are workload seeds, ours order seeds; not the same quantity |
| `git_sha` | `git.commit`, `measurement_git.commit` | different repositories; join by (repo, commit), never by hash alone |
| `section` | `experiment` (`replication_batch`), `per_seed_artifacts[].experiment` | labels of different experiment families |
| `co_runner` | `config.sessions` (10) and `aggregate.execution_note` (`workers=1; concurrency=10 sessions`) | his concurrency is orchestration sessions, ours is a synthetic spin load. Not equivalent |
| `e2e_s` | `aggregate.batch_host_cpu_ms`, `per_seed_artifacts[].batch_wall_s` | his are host CPU milliseconds and batch wall time, not model request latency |

**Name collision:** his `config.backend` (`openai`) is the LLM API client type, ours `backend` is the compute backend
(`vulkan`, `sycl`). Do not join on it.

## 3. Fields of his with no match in ours

`setup_ref.setup_digest`, `setup_ref.task_suite_digest`, `git.dirty`, `git.dirty_paths`, `env.python`, `env.blas_pin`,
`config.profile`, `config.search_locality`, `config.comparison_type`, `config.allow_dirty`, `config.instr_version`,
`config.llm_median_scale`, every `aggregate.pooled_*_pct` attribution share (tool compute, orchestration, harness,
client HTTP, framework, threadpool, residual), `per_task_host_cpu_ms.*`, `behavior_buckets`, `invariant`, `audit`,
`result_validity`.

## 4. Fields of ours with no match in his

`model_id`, `model_sha256`, `quant`, `kv_type`, `flash_attn`, `mmap`, `n_ctx`, `prompt_tokens`, `mem_headroom_gb`,
`proc_throttle_max`, `ttft_s`, `decode_tok_s`, all GPU and memory telemetry fields, `load_s`.

## 5. What this means for Fig 6.1

The two datasets do not share a machine: his Axis B run is on an Ultra 5 325 with 8 logical cores and 7.56 GB under
WSL2, ours on an Ultra X7 358H with 16 logical cores and 63.49 GB. A per-call join is therefore not possible, and a join
on `cpu_name` would pair unlike hardware. Two joins are defensible:

1. **By platform class, stated as such:** join on `cpu_name` family and OS, labelling the pairing off-machine.
2. **On the same machine (needed for a real joint envelope):** run his harness on evo-t2s and join on
   `hw_id` + `cpu_name` + `cores_logical` + `ram_gb` + `generated_utc` window. He needs to record `hw_id` and the
   `cpu_name` string (his `env.cpu_model` is `unknown`) so the key exists on his side.

The `hw_id` value must be `evo-t2s` on both sides. A committed manifest in this repository once carried
`evox2_evo-t2s`; that is a known label bug (see `docs/RESULT_PROVENANCE.md`) and must not be used as a join value.
