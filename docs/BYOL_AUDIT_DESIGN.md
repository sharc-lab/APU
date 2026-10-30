# "Bring your own logs" audit: design

Status (2026-10-26): replay-from-recorded-logs is implemented (`apu_bench/audit.py`) and tested offline on our own
recorded runs (`tests/test_apu_audit.py`). Live replay is designed below but not implemented. The audit reports no
new measured number. The tests only reproduce counts already in the register.

## Question it answers

Given a customer's recorded agent sessions and a device profile: on this device, how often would the local runtime
have cut the context without telling anyone (hidden failure), and what would the same sessions cost on the cloud?

## Command

    py -3.12 -m apu_bench audit --profile <device.json> --sessions <sessions.jsonl> --out <dir>
        [--server-log WINDOW=PATH ...] [--cloud-model ID] [--calib R]

Writes `audit.json` (`apu-audit-result/1`) and `audit.md`.

## Input format (`apu-audit-session/1`)

JSONL, one request per line, with an optional header line `{"schema": "apu-audit-session/1"}`. The field names
follow an OpenAI-style chat log so a customer can export from their own client or proxy:

- required: `session_id`, `request_idx`
- `messages` (OpenAI chat messages as sent) and `tools`; for a redacted log, `n_messages` and `prompt_chars` instead
- provider usage when present: `prompt_tokens`, `completion_tokens`, `cached_prompt_tokens`, `cache_write_tokens`
- what the client saw: `http_status`, `error`
- grading, optional: `task_id` (groups requests into tasks), `outcome` `{"ok": bool}`
- where it ran, for recorded runs: `ran_at` `{"config", "window"}`, and `recorded_verdict` when a detector v2 verdict
  was written at run time

## Replay against a device profile

Each request is audited at a window from the profile:

1. **server_log** (ground truth): the sessions were served by the runtime with its debug log on (Ollama
   `OLLAMA_DEBUG=1`). The audit cuts the log into one slice per chat request at the `[GIN]` access line, aligns the
   slices to the requests in order (it refuses if the counts differ), and applies detector v2, the same rules as
   `harness/own_compaction.cut_v2_verdict`. A slice counts as cut on any of: Ollama's message-drop line ("truncating
   input messages which exceed context length"), a token cut ("truncating input prompt"), a llama.cpp context
   shift, or an exceed-context error. With redacted messages and no `n_messages`, a message-drop line counts as a
   drop, because Ollama writes it only when it truncates.
2. **recorded_verdict** (ground truth, second hand): our own harness wrote the detector v2 verdict per turn from the
   same log lines at run time and did not keep the log.
3. **estimate** (prediction): no log. The projected tokens (chars/4 x `--calib`) are compared with the window
   through `src/dse/fits_locally.py`, so Zach's ceiling formula can replace the baseline later. The report labels
   these rows PREDICTED.

A cut is **hidden** when the client saw HTTP 200 and no error. A task is a hidden failure when any of its requests
is a hidden cut. The hidden failure rate is hidden-failure tasks divided by tasks. When outcomes are given, wrong
tasks are split into "with a hidden cut" and "without a cut".

## Live replay (not implemented): what it needs

- The device and machine time, under a pre-registration.
- A server per window started by the harness with the debug log on (`harness/demo_servers.py` or `x2_r2_evalset`'s
  own server management) and the stale-server guard (`harness/server_guard.py`: port free, `/props` and cmdline
  check, listener PID check before each request).
- Teacher-forced replay: send each recorded request's messages exactly as recorded (not the local model's own
  earlier answers), so the prompt sizes match the customer's. Then slice the log as above. The local answers can be
  graded only if the customer supplies a grader. Without one the audit reports cuts and cost, not correctness.
- A tokenizer check: the server's own `task.n_tokens` per request, which gives the true prompt size on this device.
- Estimated effort: about 6 h to build on top of `audit.run_audit` (server lifecycle plus sending requests), then
  machine time per customer log.

## Cloud cost

Priced with the committed tables (`src/cloud/prices.py`), per session, on `--cloud-model` (default: the profile's
`cloud_model`):

- **upper bound**: all input at the input rate, plus output at the output rate (no caching);
- **floor**: the last prompt at the input rate and every earlier prompt token at the cache-read rate (as
  `analysis/r2_session_cost.session_cost` "best caching"). It ignores cache-write surcharges;
- **billed**: only when every request carries `cached_prompt_tokens`. Uncached input at the input rate, cached at
  the cache-read rate, cache writes at the cache-write rate, output at the output rate.

Token basis: provider usage when present, else chars/4. The basis is named in the result.

## Report

Per (config, window): ground-truth method, tasks, hidden failures and rate, requests cut, wrong (graded), wrong
with a hidden cut, wrong without a cut, and cloud USD (floor, upper bound, billed). Device and runtime come from the
profile.

## Test on our own recorded runs (offline)

`tests/test_apu_audit.py`:

- `results/demo_detector_check_v2.jsonl` with the committed raw logs `results/demo_live_logs/detector_check_v2/`.
  The audit slices the logs itself and matches the recorded per-call detector v2 verdict on all 20 calls. It
  reproduces register row demo-detector-check-v2: hidden failures on 5/5 tasks at num_ctx 4096 (10 calls cut) and
  0/5 at the runtime default.
- The detector v1 check and the acceptance run logs: the log ground-truth cut counts equal TP + FN of register row
  detector-precision-recall (10 and 12).
- `results/x2_own_compaction_v1_{a,b}.jsonl` (per-turn verdicts recorded, logs not kept): wrong, flagged, silent and
  turns flagged per arm equal register rows own-compaction-4096 and own-compaction-8192.

Which recorded runs kept server logs: only the demo runs (`results/demo_live_logs/`: detector_check,
detector_check_v2, acceptance). The B3 v2 real run (`results/x2_evalset_tasks_real_v2.jsonl`) kept no server logs.
The own-compaction and own-config-probe runs recorded detector v2 verdicts per turn but did not commit the logs.

## One-page report (MVP, added 2026-10-29; due 2026-12-15)

`apu_bench/onepage.py`, from the audit result: one page in Markdown and HTML with the hidden failure rate per setting
(ground truth from the runtime's own cut-off log, or PREDICTED), the cloud cost of the same sessions and the
escalation cost per 100 tasks, and a recommended setting (window and KV type) chosen by a stated rule: among settings
with log ground truth and within an optional cost bound on escalation cost per 100 tasks, the lowest hidden failure
rate; ties by lower wrong rate, then lower escalation cost, then larger window. The page prints "wrong without a cut"
next to every rate, because a hidden failure rate of 0 does not mean the tasks were right.

    py -3.12 -m apu_bench audit --profile <device.json> --sessions <s.jsonl> --out <dir> --onepage [--cost-bound USD]
    py -3.12 -m apu_bench examples --out reports/apu_audit_onepage

Tested on our own recorded runs (`tests/test_apu_audit_onepage.py`; generated pages in `reports/apu_audit_onepage/`):
the demo detector check v2 (raw logs replayed) and the own-compaction runs (recorded verdicts). On the
own-compaction runs the rule picks own_compaction_8192 (0 hidden failures, tied with client_trim_8192 and broken by
wrong 24/150 against 30/150), and the page shows that every one of those wrong tasks had no cut: the rate does not
catch them.

Remaining for 2026-12-15 (estimates):

| item | hours |
|---|---|
| KV type recorded per setting: read it from the server log (Ollama prints the KV type) instead of "not recorded" | 3 |
| Completion tokens in the cost: the adapters record none, so cost is input only; read them where the logs have them | 2 |
| A device-profile field for local cost (machine hours) so the rule can compare local plus escalation against all-cloud | 4 |
| A run with outcomes and logs together on more than one window and KV setting (the demo has 2 windows, 5 tasks each; the own-compaction runs kept no logs) | 6 offline after a run exists; the run itself needs machine time and a pre-registration |
| An external-format adapter (OpenAI-style request logs and a plain Ollama debug log with no harness records) | 6 |
| Page layout check in a browser and print to one sheet; wording review | 2 |
| total | 23 |
