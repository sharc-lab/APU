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
