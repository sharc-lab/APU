# Methodology

## Recorded model endpoints.

Model endpoint record/replay is implemented as a backend-level concern so every model invocation goes through the same path.

- Replay key: SHA256 over canonical JSON of `model`, `messages`, `tools`, `temperature`, and `seed`.
- Trace location: `analysis/traces/{model}/{key}.json`.
- `RECORD` mode: always call provider API, then persist trace entry.
- `REPLAY` mode: never call provider API; raise if trace key is missing.
- `AUTO` mode: replay when entry exists; otherwise call API and record.

Each trace entry stores:

- `response`: provider response JSON payload used for downstream parsing.
- `token_counts`: `{prompt_tokens, completion_tokens, total_tokens}` from response usage fields.
- `recorded_latency_ms`: original API latency captured at record time.

Runtime metadata exposes two latency values for each call:

- `recorded_latency_ms`: original provider call latency from the trace.
- `replay_latency_ms`: local cache read latency for the current run.

This separation allows replayed benchmark runs to preserve the original remote latency signal while also quantifying replay overhead.

### Environment controls

- `APU_REPLAY_MODE`: one of `AUTO` (default), `RECORD`, `REPLAY`.
- `APU_TRACES_ROOT`: optional override for trace root directory.

### Backend notes

- `CloudOpenAIBackend` stores traces at `analysis/traces/{model}/{hash}.json`.
- `LocalOllamaBackend` stores traces at `analysis/traces_ollama/{model}/{hash}.json`.
- Both backends emit the same per-turn category keys as the instrumentation enum.

## Router Distillation Flywheel

Each benchmark sweep contributes structured routing decisions and replay-backed model traces.

- Decision logs provide step context, selected backend, budget state, and outcome hints.
- Replay traces provide deterministic output and token metadata at zero additional API cost.
- `analysis/distill_router.py` converts this growing corpus into supervised examples:
	- features: step signals + budget/routing context + trace-derived metadata
	- label: whether local was adequate
- The script trains lightweight classifiers (logistic regression + gradient boosted trees),
	selects the best model, and exports an artifact consumed by `routing/policies/learned_router.py`.

This creates a closed-loop improvement cycle: every run expands training data, and better learned policies can be re-evaluated in replay mode without new model spend.

## Filler Calibration: Cross-Arm Equivalence

The sweep runs on two separate hardware arms that use different token-counting methods for filler calibration:

| Arm | Hardware | Count method | `count_method` value |
|-----|----------|--------------|----------------------|
| Blade 14 | RTX 4070, discrete | Ollama `prompt_eval_count` | `ollama_prompt_eval` |
| evo-t2s | Arrow Lake, Vulkan | llama-server `/tokenize` endpoint | `llamaserver_tokenize` |

Every result row records `count_method` so the calibration path is traceable in analysis without consulting the run manifest.

### Verified equivalence (2026-09-21)

Filler strings were built with the F-NUM template (seed 10000, CHARS_PER_TOKEN=5.03) and measured on both arms. Ollama counts used `prompt_eval_count − 8` (template overhead). `/tokenize` counts used `len(tokens)` with `add_special=False`.

| depth | filler chars | Ollama count | /tokenize count | delta |
|------:|-------------:|-------------:|----------------:|------:|
| 2,000 | 10,060 | 2,000 | 2,000 | 0 |
| 8,000 | 40,240 | 7,996 | 7,996 | 0 |
| 16,000 | 80,480 | 15,996 | 15,996 | 0 |
| 32,000 | 160,960 | 31,999 | 31,999 | 0 |
| 64,000 | 321,919–321,920 | 63,993 | 63,993 | 0 |

Hardware: Blade 14 RTX 4070 (Ollama) / evo-t2s Arrow Lake (llama-server b10970 Vulkan, Qwen3-4B-Q4_K_M.gguf). The delta is 0 at all depths measured (within rounding of the char-heuristic initial slice). Depth labels are directly comparable across arms; no cross-arm correction is needed in analysis.

### BOS token behavior by model

The `/tokenize` endpoint's `add_special` parameter controls whether BOS/EOS tokens are prepended. Results are per model, not per build:

| Model | BOS configured | add_special effect | Empirical test |
|-------|---------------|-------------------|----------------|
| Qwen3-4B-Q4_K_M | No BOS in tokenizer config | No effect regardless of value | Live on evo-t2s b10970: add_special omit/true/false all give identical token lists |
| Llama 3.1 (any quant) | Expected: BOS token 128000 `<|begin_of_text|>` | add_special=True should prepend BOS; False should suppress it | **Not verified via llama-server /tokenize** — no Llama 3.1 GGUF loaded on evo-t2s |
| gpt-oss:120b-cloud | Cloud API model, no local weights | Not applicable | Cannot test |

The implementation always passes `add_special=False`. This is correct for filler calibration regardless of model: filler is injected into the prompt body, not at the token stream start; the chat template adds BOS separately. The Qwen3 case is empirically verified on b10970. The Llama 3.1 case is unverified — the expected behaviour follows from the tokenizer configuration, but the effect via the `/tokenize` endpoint has not been measured with a live GGUF.

## Launching unattended multi-hour sweeps on evo-t2s over SSH

Windows OpenSSH puts every process spawned by an SSH session's PowerShell host into a Job Object with kill-on-job-close semantics. This kills descendant processes when the SSH session's job closes — including on a clean client disconnect — **regardless of how the child was detached**. Confirmed not to survive session end, in order of attempts made during the KV quality/provisioning sweeps (2026-09-23):

1. **`Start-Process ... -WindowStyle Hidden -RedirectStandardOutput/-Error -PassThru`** — process alive at +3s, dead by +15-65s with empty (0-byte) redirect files. Removing `-WindowStyle Hidden` did not help.
2. **`cmd.exe /c "... > log 2>&1"` via `Start-Process`** — same failure mode; the `cmd.exe` wrapper does not escape the job either.
3. **Task Scheduler (`schtasks /create` + `schtasks /run`)** — task never actually ran (`schtasks /query /v` reported `Last Result: 267011` = `SCHED_S_TASK_HAS_NOT_RUN`, `Last Run Time` unset) despite `/run` reporting success. Likely a logon-context restriction (task requires an interactive session the SSH connection does not provide); not resolved further since (4) worked.
4. **`Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=...}` (WMI process creation)** — **works**. The new process is created by the WMI provider host, outside the SSH session's job object entirely. Confirmed alive and accumulating real CPU time (`llama-server` CPU-seconds climbing across repeated checks) at +90s, +150s+, and across full multi-hour unattended runs (kv_quality: 316.5 min; kv_provisioning: ~5.3h estimated for the q4_0/131072 leg).

Usage pattern (PowerShell, run via `ssh ... "..."`, quoting for the outer shell already applied):

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
  -Arguments @{CommandLine='cmd /c cd /d C:\apu && python script.py > run.log 2>&1'}
```

`ReturnValue=0` on the result indicates success; the result's `ProcessId` is `cmd.exe`'s PID, not the actual `python.exe`/`llama-server.exe` PIDs spawned underneath it — check by process name (`Get-Process -Name python,llama-server`) to confirm the real workers are running, not just the wrapper.

Because stdout is not a TTY when redirected this way, Python fully buffers it — `print()` output in the log file does not appear until the buffer fills or the process exits, even though the process is actively working. This is expected and does not indicate a hang; check `Get-Process ... | Select CPU` for rising CPU time as the liveness signal instead of the log file during a run, or use `python -u` to disable buffering if live log tailing is needed.

Before trusting a detached launch for an unattended run, verify survival past at least 60-90s (not just the first few seconds) — job-object teardown on some paths above took up to ~65s to manifest, not immediately.

## Harness mechanics: run_start, per-turn wall time, run_end, server death (2026-10-27)

Scope: the harnesses behind the headline results. B3 v2 is `harness/x2_r2_evalset.py`, R2 is `harness/x2_r2_agent.py`,
B5 is `harness/policy_compare.py`, plus `harness/x2_longtask.py`, `harness/own_compaction.py` (a runtime wrapper driven
by `x2_r2_evalset.py --plan own_compaction_v1`) and `harness/xruntime_llamacpp.py`. Line numbers are at the commit that
added this section. B3 v2, B5, own compaction and the llama-server replication all run their sessions through two
shared functions, so most of the mechanics below are stated once:

- `x2_r2_agent._do_call` (x2_r2_agent.py:1045): every task call of every harness listed here.
- The session loops: `x2_r2_agent.run_session` (x2_r2_agent.py:1134), used by R2 and B5, and
  `x2_r2_evalset._run_session_body` (x2_r2_evalset.py:612), used by B3 v2, own compaction and the llama-server run.

### run_start

Each harness appends one `run_start` row per invocation. A resumed run writes a new one; earlier rows are kept.

| Harness | Written at | When | Contents (main fields) |
|---|---|---|---|
| R2 | x2_r2_agent.py:2024 | after host checks, the plan and the validation preflight; before run_plan starts sessions | mode, plan, models, order, rules in use, call-2 mode and extra, validation preflight, max tokens per call, keep_alive, client_trim, hw_id, interactive session, server env, tiers, seeds, neg_ctx, num_keep, harness file and git blob |
| B3 v2, own compaction | x2_r2_evalset.py:1821 | after host checks and the preflight. If the preflight refuses, the harness returns with a "stopped: refused to start" note and no run_end (x2_r2_evalset.py:1845). The register accepts that by design: numbers_register.py:9362 `_single_complete` | harness, mode, workload, guardrails, models, plan, preflight, rules, num_keep tiers, ctx_policy (own-compaction trigger and target fractions, module blobs), max tokens, keep_alive, hw_id, server env, design, turn protocol, max tool rounds, filler tokens, provenance |
| B5 | policy_compare.py:1693 | after host checks, rule files and preflight; before any server start | platform, models, B1 decision, rules, preflight, seeds, turns, policies, run order, base env, agent and harness blobs, job kind, window cap, deadline, max tokens, keep_alive, the cloud assumptions A-B5-cloud and A-B5-cloud-latency |
| x2_longtask | x2_longtask.py:436 | after host checks and the oracle self-check; refuses to append to a file that already has a run_end (x2_longtask.py:404) | model, think, seeds, arms, tasks, offered tools, step-cap rule, max tokens, keep_alive, stop-after time, reference steps, provenance |
| xruntime_llamacpp | xruntime_llamacpp.py:703 | after host checks, the exe and GGUF existence check and a full GGUF sha256 | plan, arms and n_ctx flags, server command lines, exe, GGUF bytes and sha256 (gguf_ok), port, design, workload, call-2 mode, turn protocol, max tool rounds and tokens, sampling (temperature 0, seed 42), server --help lines, detector plan, ollama loaded models, hw_id, provenance |

### Per-turn wall time (`turn_wall_s`)

Clock: `time.monotonic()` in every harness listed. The per-call `duration_s` also uses it: t2s_k1_ollama.py:345,
x2_r2_agent.py:862 and xruntime_llamacpp.py:255. `ts_utc` is wall-clock UTC (`datetime.now`, x2_r2_agent.py:449),
taken when the row is built, and is used for ordering only.

- **R2 and B5** (x2_r2_agent.py:1151 to 1169):
  - Start: `t0` is taken at the top of the turn loop, before `run_turn` appends the user message and builds the
    prompt (x2_r2_agent.py:1076).
  - End: `turn_wall_s = time.monotonic() - t0` is computed while the row is built, at x2_r2_agent.py:1169.
  - Included:
    - every model call of the turn (call 1, and call 2 if call 1 called a tool);
    - the simulated tool results (in-process);
    - prompt-token estimates and scoring of the turn (`score_calls`, `score_final_answer`);
    - one `runtime.loaded_context` query, which is an `/api/ps` HTTP call on Ollama (x2_r2_agent.py:1157).
  - Model load: the model is unloaded before each session (x2_r2_agent.py:1145). Its reload falls inside turn 1's
    wall time and is recorded per call as `load_duration_s`. These harnesses have no separate warm-up call.
  - B5 policy wrappers run inside `chat`, so their work is inside the turn time: client-trim render and tokenize
    counts, router decisions, and the router's cloud-proxy call. The num_keep measurement runs in `begin_session`,
    outside every turn.
  - B5's reported `wall_min` is the sum of `turn_wall_s` minus the measured proxy time plus the assumed cloud time
    (policy_compare.py:677 to 686). `wall_min_proxy_measured` keeps the raw sum.
- **B3 v2, own compaction, llama-server** (x2_r2_evalset.py:623 to 645):
  - Start: `t0` is taken before the replay filler (replay workload only) and before `run_turn` appends the user
    message (x2_r2_evalset.py:520).
  - End: x2_r2_evalset.py:645, after:
    - all calls of the tool loop (v2: up to MAX_TOOL_ROUNDS tool-bearing calls plus a final call);
    - the in-process environment tool execution (`E.apply_call`);
    - end-state and rule scoring (`score_turn`, `score_final_answer`);
    - `runtime.loaded_context` (x2_r2_evalset.py:632).
  - Wrappers inside `chat`, and so inside the turn time:
    - own compaction: the summarizer calls (own_compaction.py:260 onward), and the wait of up to
      `GIN_WAIT_S` = 3.0 s (x2_r2_mechanism.py:68) for the server's request log line after each call
      (own_compaction.py:217);
    - llama-server: the request guard check, a `LOG_SETTLE_S` = 0.3 s sleep and the server-log read after each call
      (xruntime_llamacpp.py:289 to 307).
  - Model load:
    - Ollama: as for R2, unload at session start and the reload inside turn 1.
    - llama-server: a fresh server per session loads the model in `LlamaServer.start` before the session
      (xruntime_llamacpp.py:339 to 368, `startup_s`), so no load time falls inside a turn.
- **x2_longtask** (x2_longtask.py:87 to 121): the unit is a task, not a turn, and the field is `task_wall_s`.
  - Start: after the system and user messages are built, before the first call.
  - End: after the last call and before grading (`check_long_task`), so grading is excluded.
  - Included: every call and the in-process tool execution (`L.execute`).
  - The excluded warm-up call per arm (x2_longtask.py:327, row `warmup`, `excluded_from_stats`) loads the model
    before any task.
- **Retries inside a turn**: none, from 2026-10-27 (next subsection). Before that, a connection-level error caused one
  recover-and-retry inside `_do_call`, and its time fell inside the turn. No result file has a call row with
  `retried_after_connection_error` true (grep over results/*.jsonl at this commit), so no recorded turn included one.

### run_end

| Harness | Written at | On an exception? |
|---|---|---|
| R2 | x2_r2_agent.py:2129, 2135, 2141, 2157, 2170 (one per mode branch), after the report | No. Any exception goes to `except` (x2_r2_agent.py:2177): note "stopped: ...", no run_end. `finally` (x2_r2_agent.py:2182) stops the Ollama server it started and calls `tq.advance(note)` once |
| B3 v2, own compaction | x2_r2_evalset.py:1915, after the report | No. `except` at x2_r2_evalset.py:1918, `finally` at 1923: server stop, `tq.advance` |
| B5 | policy_compare.py:1880 (also 1716 when no model passed validation, and 1756 for the FA control) | No. `except` at policy_compare.py:1881 (note "stopped", rc 2), `finally` at 1886: server stop, `tq.advance` (queue platforms only) |
| x2_longtask | x2_longtask.py:492 | No. `except` at x2_longtask.py:493, `finally` at 498: file close, server stop, `tq.advance` |
| xruntime_llamacpp | xruntime_llamacpp.py:734 | No. `except` at xruntime_llamacpp.py:736, `finally` at 741: `tq.advance`. Each session's own `finally` (xruntime_llamacpp.py:662) writes `server_stop` and stops that server |

run_end is never written from a `finally` block. A run that raised therefore has no run_end, and
`analysis/numbers_register.py::run_file_incomplete` (numbers_register.py:9372) returns "partial: no run_end in
<file>" for it. Every register row reading that file stays PENDING until a resumed run appends a run_start and a
run_end. Completeness is judged after the file's last run_start (numbers_register.py:9362 to 9369), and multi-part
files by their last part (numbers_register.py:9378 to 9395).

### Server death mid-session

**What happened before 2026-10-27 (all harnesses, through `_do_call`).** A call with no HTTP status covers a
refused connection, a reset or a 900 s timeout (`CALL_TIMEOUT_S`, x2_r2_agent.py:431). Such a call:

1. was logged ("recovering and retrying once");
2. called `runtime.recover()`: on Ollama, wait 60 s, then restart and wait 90 s (x2_r2_agent.py:945); on
   llama-server, restart a fresh server and write a `server_restart` row (xruntime_llamacpp.py:313);
3. retried once.

The first failure's error text was overwritten. Only `retried_after_connection_error: true` on the call row recorded
it. If the restart failed, the error response was scored as a failed turn. The session then ran on to its last turn,
each call failing the same way, and wrote its `r2a_session` row. A dead-server session was therefore counted as a
completed session with wrong turns. So the retry was recorded but easy to miss, and no session was ever marked
incomplete. An exception from a call path was not caught per session. It propagated to `main`, which wrote no
run_end.

**Defined behaviour (2026-10-27)**:

- **No retry and no recover.** `_do_call` raises `x2_r2_agent.SessionIncomplete` (x2_r2_agent.py:1025, raised at
  1052) on the first connection-level failure. The failed call's record is attached to the exception.
- **The session is marked incomplete.** The session loop catches it, writes one `r2a_session_incomplete` row and
  re-raises:
  - where: R2 and B5 at x2_r2_agent.py:1154; B3 v2, own compaction and llama-server at x2_r2_evalset.py:629;
  - row contents (`incomplete_row`, x2_r2_agent.py:1038): the session's base fields and attempt_id,
    `incomplete: true`, `error`, `failed_call`, `n_turns_completed`, `n_turns_planned`, `retried: false`.
  - No `r2a_session` row is written. Every analysis keys sessions on `r2a_session` (`completed_sessions`,
    x2_r2_agent.py; `tasks_real_v2_outcome` and xruntime `_sessions` select turns by the attempt_id of an
    `r2a_session`). So the incomplete session's turns are counted nowhere, and a resume reruns the session.
- **Long tasks**: x2_longtask writes a `longtask_task_incomplete` row (x2_longtask.py:471) and no `longtask_task` row.
  Caveat, unchanged: a resume of x2_longtask reruns every task and does not skip done ones.
- **Own compaction**: a summarizer call with no HTTP status raises the same exception (own_compaction.py:279).
  Before, it was folded in as "summary failed, details lost". No recorded summary call failed
  (results/x2_own_compaction_v1_a.jsonl).
- **The run aborts.** The exception reaches `main`'s `except`. The note is "stopped: SessionIncomplete(...)",
  `finally` stops the server and advances the queue once, and no run_end is written. The register reads the file as
  partial (above) until a resumed run completes it.
- **An HTTP error with a status** (4xx or 5xx, for example llama-server's context-size rejection) is not a server
  death. It is recorded on the call row (`http_status`, `error`, `any_http_error` on the turn) and scored as before.
  Measured semantics for completed sessions are unchanged.
- **The stale-server guard** (`harness/server_guard.py`) applies only to llama-server, here
  `xruntime_llamacpp.py`. The Ollama harnesses do not use it.
  - Before each server start, `assert_port_free` runs (xruntime_llamacpp.py:341). A `GuardError` propagates and the
    run stops.
  - After each start, `assert_server_matches` checks /props and the command line (xruntime_llamacpp.py:364).
  - Before every request, `RequestGuard.check` runs (xruntime_llamacpp.py:289). A server that died between calls has
    no listener on the port, so the check fails before the request is sent: `GuardMismatch` gives a `guard_stop` row,
    the condition (arm) is stopped and the run moves on to the next arm (xruntime_llamacpp.py:658 and 728), and
    `outcome()` marks the run invalid ("request guard stopped a condition", xruntime_llamacpp.py:439).
  - A server that dies during a call gives the `SessionIncomplete` path above.
- Test: `tests/test_server_death_mid_session.py`. Each case uses a fake server that answers N calls and then fails at
  the connection level. They cover:
  - R2 and B3 v2 sessions: exactly N+1 calls, recover() never called, one incomplete row, no `r2a_session`;
  - a 500 with a status, which still completes the session;
  - a llama-server session: no restart and no `server_restart` row, and `server_stop` is still written;
  - an own-compaction summarizer death;
  - x2_longtask `main`: an incomplete task row, no run_end, a "stopped" note and one queue advance;
  - `run_file_incomplete`: "partial" until a resumed run_end.
