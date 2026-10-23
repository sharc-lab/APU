# Running the live demo (one page)

The screen runs on the controller laptop. The local model runs on evo-x2 (qwen3:14b), reached over SSH. The cloud
calls are made from the controller, the only machine that holds the key. Every number on the screen is tagged LIVE,
REPLAYED or MEASURED.

## Before you start (about 5 minutes)

1. **evo-x2 must be idle.** Never start a live run while a queue job is running. Check from the controller:
   `C:\Windows\System32\OpenSSH\ssh.exe -o BatchMode=yes Ritz@100.118.33.76 "type C:\apu\ovn\queue_state.json"`.
   No entry should have status "running", unless it is the demo hold job agreed with the operator.
2. **Deployed files.** The demo files sit in `C:\apu\ovn\demo` on evo-x2 (deployed with `scripts/deploy_evo.py
   --host evo-x2`, blobs verified). qwen3:14b must be in evo-x2's model store.
3. **Ports.** 11541 to 11544 must be free on evo-x2. The agent refuses a held port; it never reuses it.

## Start (controller, PowerShell, from the repo root)

```
$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')
$env:CLOUD_CLIENT_HOST_ROLE = 'controller'; $env:DEMO_CLOUD_LIVE = '1'
py -3.12 -m demo.live --mode live --port 8800
```

Open http://127.0.0.1:8800/ full screen (1600 x 1000). Pick the scenario ("AMD workstation, constrained memory
(8192)" is the headline) and press "Run 5 tasks". When you are done, close the terminal; that also removes the key
from the process.

The controller starts the evo-x2 side by itself over SSH: `py -3.12 C:\apu\ovn\demo\demo_live_x2.py --mode agent
--results-dir C:\apu\ovn\results`. That agent starts one Ollama server per configuration on its own port, with its
own log and OLLAMA_DEBUG=1 for that process only. It stops them when the run ends or the connection drops. It never
touches the Ollama tray app or system-wide settings.

**Budget.** Cloud calls go to the demo ledger (`demo/ledger/cloud_ledger_demo.jsonl`), capped at USD 10, with a
session limit of USD 3. A cloud task that would not fit is shown REPLAYED instead.

## If evo-x2 is unreachable (or busy)

Use the recorded acceptance run. Every number is then labelled REPLAYED:

```
py -3.12 -m demo.live --mode replay --replay results/demo_live_run_<date>.jsonl --port 8800
```

The screen shows "REPLAYED" in the mode badge. The run plays back with its recorded timings.

The backup recording is `demo/review/demo_live_run.webm`.

## What to say if something falls back

- **A cloud part shows REPLAYED:** "No live cloud call for this one (budget or connection). This is the recorded
  result of the same task on the same model."
- **The whole run is REPLAYED:** "The workstation is busy or not reachable right now. This is the recorded run from
  <date>, played back at its real speed."
