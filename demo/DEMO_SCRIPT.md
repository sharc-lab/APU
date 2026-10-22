# Demo script, 2026-10-28 (5 minutes)

Four views on one page, top to bottom: A Silent failure, B Device setting, C Cost, D Live escalation. Every number on
screen is a link: click it to see the register row (docs/NUMBERS_REGISTER.md) or the result-file line it came from.
Every claim carries a label: measured, projected, simulated, priced, not executed, or not measured. Say the label out
loud when you read a number.

The numbers quoted below are the register values at commit time, with their row ids. Read them off the screen on the
day: the screen is rebuilt from the register, this file is not.

## Before the audience arrives (5 minutes, not part of the talk)

1. From the repo root on the controller laptop, in the terminal you will present from:

   ```
   py -3.12 -m demo.dashboard.build_cache
   py -3.12 -m pytest tests/test_dashboard.py tests/test_dashboard_views.py -q
   ```

2. Start the server. Stub only (no cloud call; view D shows this task's recorded cloud redo from
   results/cloud_redo_v1.jsonl, labelled measured since that run executed):

   ```
   py -3.12 -m demo.dashboard
   ```

   With the one live call for view D (key loaded into this process only, never printed):

   ```
   $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')
   $env:CLOUD_CLIENT_HOST_ROLE = 'controller'; $env:DEMO_CLOUD_LIVE = '1'
   py -3.12 -m demo.dashboard
   ```

   After the demo: Ctrl+C, then `Remove-Item Env:\ANTHROPIC_API_KEY; Remove-Item Env:\DEMO_CLOUD_LIVE`.

3. Open http://127.0.0.1:8765/ and check the header: four views load, no "Error" text.

Cloud spend rules for view D: one provider request per server start (a second click shows the stub), through
src/cloud/client.py with a USD 10 cap (DEMO_CLOUD_CAP_USD may lower it, never raise it) and its own ledger,
demo/ledger/cloud_ledger_demo.jsonl (not committed). Any failure (no key, host refused, cap, provider error) falls back
to the stub with the reason on screen. Do not restart the server to get a second live call during the demo.

## 0:00 to 1:30, view A: silent failure

Click: nothing; it is the top of the page. Point at the two cards.

Say:
- "Same machine, evo-t2s. Same two models, same 40-turn agent sessions, Ollama defaults, nothing configured. The
  only difference is which device Ollama picked."
- Left card: "On the Intel CPU Ollama loads a 4096-token window. Both models silently lost a rule in 5 of 5 sessions,
  all after the window filled." Measured, rows t2s-r2-real-cpu.
- Right card: "On the iGPU it loads 32768. llama3.1:8b lost nothing; qwen3:14b had 1 of 5, and that one also happens
  inside the window on its validation run (baseline-type), so 0 after the window filled." Measured, row
  t2s-r2-real-igpu.
- Point at the small line under each card: "Calls that surfaced an error to the client: 0 of 800. Nothing told the
  user." Measured, counted over every call in results/t2s_r2_real_cpu_v1.jsonl and results/t2s_r2_real_igpu_v1.jsonl.
- Bar chart: "This is what it does to work. qwen3:14b on evo-x2, graded task results: 11.3% wrong at the default window
  (17 of 150), 42.0% at 4096 (63 of 150). Every wrong result came back as a normal answer: 0 surfaced." Measured, row
  x2-evalset-tasks-real-v2-wrong-rate; the excess at 4096 is +30.7 points, session-level 95% CI +26.0 to +34.7, row
  x2-evalset-tasks-real-v2-ci-units.
- Optional, if asked about other models: open "Rule loss per model and window" (row R2-v1b-gated-per-model, 5 seeds,
  the strengthened run v1 + v1b).

Do not say: "every model fails at 4096". qwen3:8b was 1 of 5 at 4096 (R2-v1b-gated-per-model) and llama3.3:70b
resisted rule loss (H-scale-verdict).

## 1:30 to 2:45, view B: the setting for this device

Click: Device "evo-x2", Model "qwen3:14b", Workload "Multi-turn agent session". Then change Device to "evo-t2s, Intel
CPU" and Workload stays the same. Then "Blade 14".

Say:
- evo-x2 / qwen3:14b: "The fastest measured setting from the policy comparison: window 40960, KV cache q8_0 with flash
  attention on, no num_keep, full history. Same rule and history scores as the default, 75.4 minutes per session
  instead of 214.8." The minutes are measured (rows b5-policy-metrics, cost-cut-config-time); the KV and flash
  attention tiles are labelled "pending flash-attention control".
- Read the caveat: "We switched flash attention on together with q8_0, so we do not yet know which of the two gives the
  speedup. The control run that separates them has not read yet; when it does, this view recommends whichever one it
  shows." Row b5-fa-control: VERIFIED, but its run is incomplete ("reading: not read (incomplete)"). Same for
  llama3.1:8b (q4_0). Do not say "q8_0 makes it faster".
- evo-t2s CPU / qwen3:14b: "Here nothing we measured holds the rules: the default loses them, and client-side trimming
  did not recover qwen3:14b. So the tool says not measured and does not invent a setting." Rows t2s-r2-real-cpu,
  t2s-r2-mitigation-cpu. (Switch the model to llama3.1:8b to show the one that trimming did recover.)
- Blade: "Not measured. Every field says so."
- If asked about the timeout floor: Workload "Single-shot calls", evo-x2, llama3.1:8b: llama-server is excluded at the
  5% timeout share (11.8% of calls timed out); Ollama is kept at 2.9%. Row x2-v3-usability.

## 2:45 to 3:45, view C: what it costs

Click: nothing; scroll to C.

Say:
- Read the lead line exactly as the screen shows it. At commit time: "Fixing a wrong answer in the cloud costs about
  USD 0.02 (mean USD 0.0201, median USD 0.0122); the cost is that the failure is silent." Label: measured. Row
  cloud-redo-cost-per-wrong-task: all 197 wrong tasks re-run on claude-sonnet-5-5 (mode executed), mean USD 0.0201,
  median USD 0.0122, mean 6.89 s per task.
- Cost only. Do not quote how many of the redone tasks the cloud got right: that figure is under diagnosis and is not
  on screen (operator, 2026-10-21). If asked: "we are still checking the grading of the cloud redo".
- "Measured first: how often it is wrong and silent, and the time the right setting saves (37.1 minutes per session for
  llama3.1:8b, 139.4 for qwen3:14b)." Measured, rows x2-evalset-tasks-real-v2-wrong-rate, cost-cut-config-time.
- "Then the projections, which depend on an assumed team: USD 5.47 to 132.31 per day at 40 sessions a day." Label:
  projected (cloud part measured, volume assumed), row impact-v2-P1. Say "assumed volume" every time.
- Router: "We do not show a router saving. Our own comparison found the router dominated on evo-x2." Row b5-verdict:
  refuted.
- Cost documents: docs/IMPACT.md is quoted verbatim. docs/OCT29_COST_SUMMARY.md shows "not present yet" until it lands;
  the panel reads it on page load, no rebuild or restart needed.

## 3:45 to 4:45, view D: send one wrong task to the cloud

Click: "Send to the cloud" once.

Say, before clicking:
- "This is one real wrong answer from the qwen3:14b run at a 4096 window: book a meeting with the default length from
  the standing instructions. The local model booked 60 minutes; the instructions say 50. HTTP 200, no error." Measured,
  results/x2_evalset_tasks_real_v2.jsonl (the line is one click away).
- Right card, before any click: "In the cloud redo run this exact task cost USD 0.0122 and took 4.8 s; here is its
  answer." Measured, results/cloud_redo_v1.jsonl (the line is one click away). Read the answer; do not call it right or
  wrong (no graded cloud result is shown while the redo's grading is under diagnosis).

After clicking, read the two tiles and the answer:
- Live (label measured): "What it cost, how long it took, and what the cloud answered." This is one call, n = 1, with
  tools offered and no tool round trips: say "one call".
- Not sent live (label measured, "recorded redo"): read the reason line ("live calls are off", "no key", or the
  provider error); the tiles then show the recorded redo of this task (USD, seconds, answer). That is the expected
  fallback, not a failure of the demo.

## 4:45 to 5:00, close

Say: "A wrong window is silent on every device we measured; the right setting depends on the device and the model, we
show it only where we measured it, and fixing one answer in the cloud is cheap. The expensive part is not knowing."

## Which claims are measured

| Claim on screen | Label | Source |
|---|---|---|
| CPU loads 4096, iGPU 32768, same models (evo-t2s, Ollama default) | measured | t2s-r2-real-cpu, t2s-r2-real-igpu |
| CPU: 5/5 sessions per model silently lost a rule after the window filled | measured | t2s-r2-real-cpu |
| 0 of 800 calls surfaced an error, per device | measured | results/t2s_r2_real_cpu_v1.jsonl, results/t2s_r2_real_igpu_v1.jsonl |
| qwen3:14b wrong-task rate 11.3% default to 42.0% at 4096, 0 surfaced | measured | x2-evalset-tasks-real-v2-wrong-rate |
| Excess +30.7 points at 4096 (95% CI +26.0 to +34.7) | measured | x2-evalset-tasks-real-v2-ci-units |
| Rule loss per model and window, evo-x2, 5 seeds | measured | R2-v1b-gated-per-model |
| evo-x2 qwen3:14b: 40960, 75.4 vs 214.8 min per session | measured | b5-policy-metrics, cost-cut-config-time |
| evo-x2: q8_0 (qwen3:14b) / q4_0 (llama3.1:8b) with flash attention as the setting | pending flash-attention control | b5-fa-control (not read yet) |
| Flash attention vs KV precision attribution | not measured (no reading yet) | b5-fa-control |
| evo-t2s CPU qwen3:14b: no setting holds the rules | measured (what failed); setting not measured | t2s-r2-real-cpu, t2s-r2-mitigation-cpu |
| Blade: every setting | not measured | none |
| 5% timeout floor exclusions | measured | x2-v3-usability |
| About USD 0.02 (mean USD 0.0201, median USD 0.0122) to fix a wrong answer in the cloud | measured (executed redo) | cloud-redo-cost-per-wrong-task |
| How many redone tasks the cloud got right | not shown (under diagnosis) | cloud-redo-cost-per-wrong-task |
| USD 5.47 to 132.31 per day for one team | projected (assumed volume; cloud part measured) | impact-v2-P1 |
| Router savings | not shown (b5-verdict refuted) | b5-verdict |
| View D live answer, USD, seconds | measured, n = 1, this run only | demo/ledger/cloud_ledger_demo.jsonl (not committed) |
| View D recorded redo of the task (USD 0.0122, 4.8 s, its answer) | measured | results/cloud_redo_v1.jsonl |
