# Demo script, 2026-10-28 (5 minutes, story mode)

The dashboard opens in story mode: five steps, about one minute each, with Next and Back buttons (or the arrow keys)
and a step indicator. After step 5, "Finish: explore freely" (or "Explore freely" at any time) shows the four full
views (A Silent failure, B Device setting, C Cost, D Live escalation) and the older frontier and replay screens.
"Story mode" in the header starts the story again.

Every number on screen is a link: click it to see the register row (docs/NUMBERS_REGISTER.md) or the result-file line
it came from. Every claim carries a label: measured, priced, not executed, projected, pending flash-attention control,
or not measured. Say the label out loud when you read a number.

The numbers quoted below are the register values at commit time. Read them off the screen on the day: the screen is
rebuilt from the register, this file is not. Steps 1 to 4 run entirely from the offline cache; only step 5 can use
the network, and without it nothing is sent and the cost reads "being measured".

Cost to fix a wrong answer: being measured. The earlier cloud redo (results/cloud_redo_v1.jsonl) was a broken
protocol run (thinking consumed the output budget; docs/FINDINGS.md, cloud redo diagnosis), so none of its figures or
answers are shown or quoted anywhere (operator, 2026-10-22). The corrected full-history redo replaces it in the final
pass.

Backup: if the laptop or the browser fails, play demo/review/demo_story.webm (3 minutes, no audio) and talk over it
with the same script. The two-page leave-behind is demo/review/LEAVE_BEHIND.pdf. Both still show the earlier redo
figure and are regenerated in the final pass; until then do not use them in front of an audience.

## Before the audience arrives (5 minutes, not part of the talk)

1. From the repo root on the controller laptop, in the terminal you will present from:

   ```
   py -3.12 -m demo.dashboard.build_cache
   py -3.12 -m pytest tests/test_dashboard.py tests/test_dashboard_views.py -q
   ```

2. Start the server. Without a live call (step 5 shows the task and the local answer; the cloud cost reads "being
   measured"):

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
"being measured" with the reason on screen. Do not restart the server to get a second live call during the demo.

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

## Step 3 (2:00 to 3:00): the tool

Click: Next. The picker opens on evo-x2 / qwen3:14b / multi-turn agent session. Then change Device to "evo-t2s, Intel
CPU". If time allows, "Blade 14".

Say:
- evo-x2 / qwen3:14b: "Window 40960; KV cache q8_0 with flash attention. Same rule and history scores as the default,
  75.4 minutes per session instead of 214.8." The minutes are measured (b5-policy-metrics, cost-cut-config-time).
- Point at the yellow labels: "The KV and flash attention tiles say pending flash-attention control: we switched both
  on together, so we do not yet know which one gives the speedup. When the control run reads, this view recommends
  whichever one it shows." Row b5-fa-control is VERIFIED but its run is incomplete ("reading: not read
  (incomplete)"). Same for llama3.1:8b (q4_0). Do not say "q8_0 makes it faster".
- evo-t2s CPU / qwen3:14b: "Here nothing we measured holds the rules: the default loses them and trimming did not
  recover qwen3:14b, so the tool says not measured instead of inventing a setting." (t2s-r2-real-cpu,
  t2s-r2-mitigation-cpu). Switch the model to llama3.1:8b to show the case trimming did recover.
- Blade: "Not measured. Every field says so."

## Step 4 (3:00 to 4:00): the money

Click: Next.

Say:
- "Sending one long agent task whole to the cloud, a 62-step agent task at the median (OpenHands traces): about USD 1.54 on
  claude-sonnet-5-5, USD 2.63 at the 90th percentile, assuming a 70% cache hit rate; about USD 0.08 on
  claude-haiku-5-5." Label: priced, not executed (cloud-cost-per-real-task; token counts from traces, prices from the
  price file, no API call).
- Read the line exactly as the screen shows it: "Cost to fix a wrong answer: being measured." Say: "Our first cloud
  redo had a protocol fault, so we are re-measuring it; we show no figure until the corrected run is in."
- "Sending every long task to the cloud is not cheap; escalating only the wrong step needs to know which step went
  wrong, and the failure is silent."
- Do not quote any figure from the earlier redo (cost, seconds, how many it got right) or the P1 cost per day built on
  it. Do not quote a router saving: the router was dominated in our own comparison on evo-x2 (b5-verdict: refuted).

## Step 5 (4:00 to 5:00): one live escalation

Click: Next, then "Send to the cloud" once.

Say, before clicking:
- "One real wrong answer from qwen3:14b at a 4096 window: book a meeting with the default length from the standing
  instructions. The local model set 60 minutes; the instruction gives 50. HTTP 200, no error." Measured
  (results/x2_evalset_tasks_real_v2.jsonl, one click away).
- Right card: "Cost to fix a wrong answer: being measured." Do not quote the earlier redo of this task.

After clicking:
- Live (label measured): "What it cost, how long it took, and what the cloud answered." One call, n = 1, tools offered,
  no tool round trips: say "one call", not "this is what fixing costs".
- Not sent live (label "being measured"): read the reason line ("live calls are off", "no key", or the provider error);
  the tiles read "being measured". That is the expected fallback.

Close (10 seconds): "A wrong window is silent on every device we measured; the right setting depends on the device and
the model, and we show it only where we measured it. The expensive part is not knowing." Then click "Finish: explore
freely" for questions.

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
| evo-x2: q8_0 (qwen3:14b) / q4_0 (llama3.1:8b) with flash attention as the setting | pending flash-attention control | b5-fa-control (not read yet) |
| evo-t2s CPU qwen3:14b: no setting holds the rules | measured (what failed); setting not measured | t2s-r2-real-cpu, t2s-r2-mitigation-cpu |
| Blade: every setting | not measured | none |
| One long agent task in the cloud: USD 1.54 median, 2.63 p90 (sonnet, 70% cache hits assumed) | priced, not executed | cloud-cost-per-real-task |
| Cost to fix a wrong answer in the cloud | being measured (no figure; the earlier redo was a broken protocol run) | none until cloud-redo-full-context |
| P1 cost per day for one team, and the problem ranking | being measured (both include the redo cost) | none until re-computed |
| Router savings | not shown (b5-verdict refuted) | b5-verdict |
| Step 5 live answer, USD, seconds | measured, n = 1, this run only | demo/ledger/cloud_ledger_demo.jsonl (not committed) |
| Step 5 fallback when not sent live | being measured | none |
