# Demo script, 2026-10-28 (5 minutes, story mode)

The dashboard opens in story mode: five steps, about one minute each, with Next and Back buttons (or the arrow keys)
and a step indicator. After step 5, "Finish: explore freely" (or "Explore freely" at any time) shows the four full
views (A Silent failure, B Device setting, C Cost, D Live escalation) and the older frontier and replay screens.
"Story mode" in the header starts the story again.

Every number on screen is a link: click it to see the register row (docs/NUMBERS_REGISTER.md) or the result-file line
it came from. Every claim carries a label: measured, priced, not executed, projected, being measured, or not measured.
Say the label out loud when you read a number.

The numbers quoted below are the register values at commit time. Read them off the screen on the day: the screen is
rebuilt from the register, this file is not. Steps 1 to 4 run entirely from the offline cache; only step 5 can use
the network, and without it the task's recorded full-history redo is shown.

Never quote a figure from the first cloud redo (results/cloud_redo_v1.jsonl): it was a broken protocol run (thinking
consumed the output budget; docs/FINDINGS.md, cloud redo diagnosis). The cost to fix a wrong answer now comes from the
full-history redo (row cloud-redo-full-context), which ran with up to 2048 output tokens per call against the local
run's 384; say so if asked. Quote its fix rate only with its 95% CI and the inherited-actions caveat.

Backup: if the laptop or the browser fails, play demo/review/demo_story.webm (3 minutes, no audio, no live call,
step 5 shows the recorded full-history redo) and talk over it with the same script. The two-page leave-behind is
demo/review/LEAVE_BEHIND.pdf.

## Before the audience arrives (5 minutes, not part of the talk)

1. From the repo root on the controller laptop, in the terminal you will present from:

   ```
   py -3.12 -m demo.dashboard.build_cache
   py -3.12 -m pytest tests/test_dashboard.py tests/test_dashboard_views.py -q
   ```

2. Start the server. Without a live call (step 5 shows the task, the local answer and the recorded full-history
   redo of the same task):

   ```
   py -3.12 -m demo.dashboard
   ```

   With the one live call for step 5 (key loaded into this process only, never printed):

   ```
   $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')
   $env:CLOUD_CLIENT_HOST_ROLE = 'controller'; $env:DEMO_CLOUD_LIVE = '1'
   py -3.12 -m demo.dashboard
   ```

   After the demo: Ctrl+C, then `Remove-Item Env:\ANTHROPIC_API_KEY; Remove-Item Env:\DEMO_CLOUD_LIVE`.

3. Open http://127.0.0.1:8765/ : step 1 of 5 shows, no "Error" text. (`?step=3` opens a given step; `?explore=1`
   skips the story.)

Cloud spend rules for step 5: one provider request per server start (a second click sends nothing), through
src/cloud/client.py with a USD 10 cap (DEMO_CLOUD_CAP_USD may lower it, never raise it) and its own ledger,
demo/ledger/cloud_ledger_demo.jsonl (not committed). Any failure (no key, host refused, cap, provider error) shows
the recorded full-history redo with the reason on screen. Do not restart the server to get a second live call during the demo.

## Step 1 (0:00 to 1:00): the problem

Click: nothing; step 1 is the opening screen.

Say:
- "Same machine, evo-t2s. Same two models, same 40-turn agent sessions, Ollama defaults, nothing configured. The only
  difference is which device Ollama picked."
- "On the Intel CPU Ollama loads a 4096-token window. Both models silently lost a rule in 5 of 5 sessions, all after
  the window filled. On the iGPU it loads 32768: llama3.1:8b lost nothing; qwen3:14b had 1 of 5, and that one also
  happens inside the window on its validation run, so 0 after the window filled." Measured (t2s-r2-real-cpu,
  t2s-r2-real-igpu).
- "Calls that surfaced an error to the client: 0 of 800 on each device." Measured (the two result files).
- "What it does to work: qwen3:14b task results go from 11.3% wrong at the default window to 42.0% at 4096, and none of
  the 63 wrong results surfaced an error." Measured (x2-evalset-tasks-real-v2-wrong-rate; excess +30.7 points,
  session-level 95% CI +26.0 to +34.7, x2-evalset-tasks-real-v2-ci-units).

Do not say: "every model fails at 4096" (step 2 shows why).

## Step 2 (1:00 to 2:00): why it is hard

Click: Next.

Say:
- "The one-line fix, num_keep, which keeps the system prompt: 30.0% wrong at 4096. Better than 42.0%, still far from
  11.3%." Measured (x2-evalset-tasks-real-v2-wrong-rate).
- "Client-side trimming has a side effect: with trimming, qwen3:14b called the logging tool in 24, 25 and 25 of 40
  turns per seed; without it, 40 of 40. Trimming removes the earlier turns that showed the two-call pattern the model
  imitates." Measured counts (R2-qwen3-14b-4096-reconciliation); the interpretation is the one in docs/FINDINGS.md.
- "And it depends on the model: at 4096, llama3.1:8b lost rules in 5 of 5 sessions, qwen3:14b 4 of 5, qwen3:8b 1 of 5;
  qwen3:32b 2 of 3; llama3.3:70b and mistral-small3.2:24b resisted rule loss, 0 of 3." Measured (R2-v1b-gated-per-model,
  H-scale-verdict, H-scale-family-verdict).

## Step 3 (2:00 to 3:00): the tool, a decision at session start

Click: Next. The picker opens on evo-x2 / qwen3:14b / multi-turn agent session. Then change Device to "evo-t2s, Intel
CPU". If time allows, "Blade 14".

Say:
- "The rule: route local only if local accuracy at the recommended setting is at least all-cloud accuracy minus a
  tolerance; otherwise the whole session goes to the cloud from the start. At matched quality, tolerance 0, the
  measured answer for every measured device is cloud from the start." Measured (hybrid-route-at-start).
- "evo-x2 would route local only if you accept 7.8 points less accuracy than claude-haiku-5-5, or 11.1 points less than
  claude-sonnet-5-5 (B3 task accuracy). The evo-t2s iGPU: 16.7 and 20.0 points, measured by R2 session pass rate, not
  B3 task accuracy." Measured (hybrid-route-at-start).
- evo-x2 / qwen3:14b, the local setting if you do run locally: "Window 40960, KV cache q8_0. Same rule and history
  scores as the default, 75.4 minutes per session instead of 214.8. The flash-attention control says the speedup is
  the KV precision, not flash attention." Measured (b5-policy-metrics, cost-cut-config-time, b5-fa-control). Same for
  llama3.1:8b with q4_0 (its route at start is not measured).
- evo-t2s CPU / qwen3:14b: "Here nothing we measured holds the rules: the default loses them and trimming did not
  recover qwen3:14b, so the decision is the whole session to the cloud from the start." Point at "kept local, measured
  by R2 session pass rate (not B3 task accuracy): 0/5"; on the iGPU it is 5/5 (llama3.1:8b) and 4/5 (qwen3:14b).
  Measured (t2s-r2-real-cpu, t2s-r2-mitigation-cpu, t2s-config-comparison); the savings beside it are projected.
- Blade: "Not measured. Every field says so."

## Step 4 (3:00 to 4:00): the money

Click: Next.

Say:
- Read the first line as the screen shows it (the prepared text from docs/IMPACT.md, row hybrid-savings-tradeoff,
  measured): "Against claude-sonnet-5-5 at 1,000 tasks/day: all local on evo-x2 saves USD 306.10 a month at 80/90
  (88.9%) correct versus 90/90 (100.0%) all-cloud, and 10 wrong answers are silent. The only measured detector, window
  overflow, escalates 0 tasks at this window."
- Point at the chart: "The dotted line is a ceiling, no such detector: we do not claim a verifier for in-window wrong
  answers." Never quote the ceiling figure as a saving.
- "Fix after the fact: about 2 cents (mean USD 0.0227 per task), 84.8% fixed (95% Wilson CI 79.1% to 89.1%), but
  executed actions cannot be undone; decide up front instead. 28 of the 30 unfixed inherited the local model's earlier
  executed actions." Measured (cloud-redo-full-context; full history, up to 2048 output tokens per call against the
  local run's 384).
- "On the Intel CPU at 4096 the window-flag hybrid escalates 88 of 90 tasks and costs more than all-cloud against
  claude-sonnet-5-5 (USD 359.80 a month more at 1,000 tasks/day): escalating with the full history costs more per task
  than a fresh cloud run." Projected (from evo-x2's 4096 tier).
- "And escalating late, with the full history, costs 1.76 times a fresh cloud run of the same task (1.64 times on the
  median; 2.22 and 1.87 unmatched)." Measured (late-escalation-cost-vs-fresh, claude-sonnet-5-5, 119 matched rows).
- "Against claude-haiku-5-5 the escalation cost is interim: the full-history redo ran on Sonnet only."
- "One long agent task sent whole to the cloud: USD 2.53 at the median on claude-sonnet-5-5." Priced, not executed
  (cloud-cost-per-real-task, recalibrated).
- Do not quote a router saving: the router was dominated in our own comparison on evo-x2 (b5-verdict: refuted). Do not
  quote the P1 cost per day (still being measured).

## Step 5 (4:00 to 5:00): one live escalation

Click: Next, then "Send to the cloud" once.

Say, before clicking:
- "One real wrong answer from qwen3:14b at a 4096 window: book a meeting with the default length from the standing
  instructions. The local model set 60 minutes; the instruction gives 50. HTTP 200, no error." Measured
  (results/x2_evalset_tasks_real_v2.jsonl, one click away).
- Right card: "In the full-history redo this exact task cost USD 0.0243 and took 4.2 s; here is its answer." Measured
  (results/cloud_redo_full_context_v1_part3.jsonl, one click away).

After clicking:
- Live (label measured): "What it cost, how long it took, and what the cloud answered." One call, n = 1, tools offered,
  no tool round trips: say "one call", not "this is what fixing costs".
- Not sent live (label measured, "recorded redo"): read the reason line ("live calls are off", "no key", or the
  provider error); the tiles show the recorded full-history redo of this task. That is the expected fallback.

Close (10 seconds): "A wrong window is silent on every device we measured. Fixing after the fact is cheap but cannot
undo executed actions, so the tool decides at session start, per device, and shows the setting only where we measured
it." Then click "Finish: explore freely" for questions.

## Which claims are measured

| Claim on screen | Label | Source |
|---|---|---|
| CPU loads 4096, iGPU 32768, same models (evo-t2s, Ollama default) | measured | t2s-r2-real-cpu, t2s-r2-real-igpu |
| CPU: 5/5 sessions per model silently lost a rule after the window filled | measured | t2s-r2-real-cpu |
| 0 of 800 calls surfaced an error, per device | measured | results/t2s_r2_real_cpu_v1.jsonl, results/t2s_r2_real_igpu_v1.jsonl |
| qwen3:14b wrong-task rate 11.3% default to 42.0% at 4096, 0 surfaced | measured | x2-evalset-tasks-real-v2-wrong-rate |
| Excess +30.7 points at 4096 (95% CI +26.0 to +34.7) | measured | x2-evalset-tasks-real-v2-ci-units |
| num_keep at 4096: 30.0% wrong | measured | x2-evalset-tasks-real-v2-wrong-rate |
| Trimming: logging tool 24/25/25 of 40 vs 40/40/40 | measured (interpretation from docs/FINDINGS.md) | R2-qwen3-14b-4096-reconciliation |
| Rule loss at 4096 per model, 8B to 70B | measured | R2-v1b-gated-per-model, H-scale-verdict, H-scale-family-verdict |
| evo-x2 qwen3:14b: 40960, 75.4 vs 214.8 min per session | measured | b5-policy-metrics, cost-cut-config-time |
| evo-x2: q8_0 (qwen3:14b) / q4_0 (llama3.1:8b); the speedup is KV precision, not flash attention | measured | b5-fa-control |
| evo-t2s CPU qwen3:14b: no setting holds the rules; whole session to the cloud | measured (what failed); setting not measured | t2s-r2-real-cpu, t2s-r2-mitigation-cpu |
| evo-t2s kept local, measured by R2 session pass rate (not B3 task accuracy) | measured | t2s-config-comparison |
| At matched quality every measured device routes the whole session to the cloud from the start; local only at tolerance >= 7.8 / 11.1 points (evo-x2 vs Haiku / Sonnet) and 16.7 / 20.0 (evo-t2s iGPU, R2 session pass rate) | measured | hybrid-route-at-start |
| Blade: every setting | not measured | none |
| evo-x2 vs Sonnet at 1,000 tasks/day: local saves USD 306.10/month at 80/90, 10 wrong answers silent; flag escalates 0 | measured | hybrid-savings-tradeoff |
| Detector ceiling | side note only: "ceiling, no such detector" | hybrid-savings-tradeoff |
| Intel CPU at 4096: flag hybrid costs USD 359.80/month more than all-cloud vs Sonnet | projected | hybrid-savings-tradeoff |
| Late full-history escalation costs 1.76x a fresh cloud run of the same task (1.64x median; 2.22x / 1.87x unmatched) | measured | late-escalation-cost-vs-fresh |
| Fix after the fact: about 2 cents (mean USD 0.0227), 84.8% fixed (95% Wilson CI 79.1% to 89.1%), 28 of 30 unfixed inherited earlier actions | measured (full history, max_tokens 2048 vs local 384) | cloud-redo-full-context |
| One long agent task in the cloud: USD 2.53 median (Sonnet, 70% cache hits assumed) | priced, not executed | cloud-cost-per-real-task |
| P1 cost per day for one team, and the problem ranking | being measured (rows still use the first redo) | none until re-computed |
| Any figure from the first redo (results/cloud_redo_v1.jsonl) | never shown (broken protocol run) | withheld |
| Router savings | not shown (b5-verdict refuted) | b5-verdict |
| Step 5 live answer, USD, seconds | measured, n = 1, this run only | demo/ledger/cloud_ledger_demo.jsonl (not committed) |
| Step 5 recorded full-history redo of the task (USD 0.0243, 4.2 s, its answer) | measured | results/cloud_redo_full_context_v1_part3.jsonl |
