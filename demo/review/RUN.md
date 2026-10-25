# Running the live demo (one page)

The screen runs on the controller laptop. The local model runs on evo-x2 (qwen3:14b), reached over SSH. The cloud
calls are made from the controller, the only machine that holds the key. Every number on the screen is tagged LIVE,
REPLAYED or MEASURED. What to click and say: demo/review/DEMO_SCRIPT.md.

## Before the day (operator)

1. **Usage check for the cloud gate:** fill in `results/console_usage_check.json` from the Anthropic Console less than
   24 h before the showing (src/cloud/console_gate.py). Without it the cloud parts show REPLAYED with the reason; the
   demo still runs. This is the one step that cannot be automated.
2. **Hold the machine free:** queue the rolling hold (below) once the window is known. It keeps evo-x2 free from 30
   minutes before the showing until its end and releases itself; it never interrupts a running job.

## Start (controller, PowerShell, from the repo root)

```
.\demo\live\start.ps1                       # LIVE; then open http://127.0.0.1:8800/ (browser bookmark)
.\demo\live\start.ps1 -Mode replay -Port 8801   # REPLAYED spare screen, no network
```

`start.ps1` reads the key from the user environment into that process only and removes it when the screen stops. The
controller starts the evo-x2 side by itself over SSH (`py -3.12 C:\apu\ovn\demo\demo_live_x2.py --mode agent
--results-dir C:\apu\ovn\results`, ConnectTimeout 10 s); that agent starts one Ollama server per configuration on its
own port (11541 to 11544, OLLAMA_DEBUG=1 for that process only) and stops them when the run ends or the connection
drops. **No manual fallback:** if evo-x2 does not answer within about 10 s, or a lane writes nothing for 5 minutes,
the screen continues with the recorded run (results/demo_live_run_20261023.jsonl) and says so; F5 rebuilds a run in
progress after a browser freeze.

**Budget.** Cloud calls go to the demo ledger (`demo/ledger/cloud_ledger_demo.jsonl`), capped at USD 10, with a
session limit of USD 3 and the Console gate before the first live call. A cloud task that would not fit is REPLAYED.

**Cut-off signal on screen.** Ollama's own truncation and context-shift log lines (detector v2), the runtime's own
signal. Its re-check on evo-x2 passed: 5/5 picks flagged at num_ctx 4096, 0/5 at the default window (register row
demo-detector-check-v2). Detector v1, the token-count check, is not used: false alarms at default (see
detector-precision-recall). The acceptance run was made under v1: in the Intel scenario, Ours sent two tasks
(cal_cancel, file_write_standing) to the cloud on a v1 token-count flag although the server showed no cut. The
recordings were re-made on the v2 screen (2026-10-24); the caption says the replay was recorded under v1, and those
two lanes read "Sent to the cloud by detector v1 when this run was recorded (a token-count flag; the server showed no
cut). v1 is not used now."

## Holding evo-x2 free (rolling hold; queue only with the operator's times)

The queue has no time gate, so the hold is a queue entry inserted directly after the running job. Each time it is
launched it decides (harness/x2_demo_hold.py `decide`): if the next pending job fits entirely before the free-from
time (window start minus the warm-up) by its **high** estimate (the "high X h" in its queue note, else 1.5 x
projected_duration_h), the hold re-inserts itself right after that job and lets it run; otherwise it holds the queue
until the release time and then releases with its own tq.advance(). The release time is in the entry's command and
note. Times are ISO 8601 with offset (PDT is -07:00 on 2026-10-27).

```
py -3.12 scripts/deploy_evo.py 'C:\apu\ovn\demo' harness/x2_demo_hold.py --host evo-x2
py -3.12 scripts/deploy_evo.py 'C:\apu\ovn\qfill' harness/x2_queue_fill.py --host evo-x2
# verify blobs (as for every deploy): run_provenance.verify_deployed_blobs on C:\apu\ovn\demo and C:\apu\ovn\qfill
ssh evo-x2: cd C:\apu\ovn; py -3.12 demo\x2_demo_hold.py --start-utc [START] --release-utc [END] --warmup-min 30 --check-plan
ssh evo-x2: cd C:\apu\ovn; py -3.12 qfill\x2_queue_fill.py --hold-start-utc [START] --hold-release-utc [END] --hold-warmup-min 30 --hold-rolling --hold-id x2_demo_hold_tue --insert-after [RUNNING_JOB_ID] x2_demo_hold_tue
```

`--check-plan` reads the queue only and prints which jobs still run before the window (by high estimates), when the
hold engages at the latest, and a warning if the job already running could overrun the window (the hold cannot
interrupt it).

## Live rehearsal (one end-to-end run in a test slot before Tuesday; operator runs it)

- **Slot:** the same rolling hold with the slot's times and no warm-up, e.g. `--hold-id x2_demo_hold_rehearsal
  --hold-start-utc [SLOT_START] --hold-release-utc [SLOT_START + 45 min]`, inserted after the running job;
  `--check-plan` with the same times gives the projected start.
- **Command:** `.\demo\live\start.ps1`, then Run 5 tasks on each scenario.
- **Duration:** about 3 min per scenario plus about 30 s agent start; plan 15 minutes for both with margin.
- **Spend (demo ledger, billed):** about USD 0.16 for amd8192 and under USD 0.01 for intel, from row demo-live-run;
  upper bound without prompt caching about USD 0.47 for both. Session limit USD 3.

## The acceptance run: one controller step (done 2026-10-23)

```
py -3.12 -m demo.live.complete_run --local results/demo_live_run_local_v1.jsonl --out results/demo_live_run_20261023.jsonl
```

Recordings: `demo/review/demo_live_amd8192.webm` (main, real speed) and `demo/review/demo_live_intel.webm` (backup;
"2x speed" shown on stretches where nothing changes).
