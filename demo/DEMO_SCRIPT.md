# Demo script, 2026-10-28 (5 minutes, story mode)

The dashboard opens in story mode, written for a non-expert audience: five steps, one idea and one big number each,
plain words, presenter-size type, Next and Back (or the arrow keys). Each number carries one small tag (measured or
priced). Every source (register row, result file, exact counts, confidence intervals) is in the step's closed
"Details" panel; open it only if asked. The "Engineering view" link (corner, and on the step bar; or `?explore=1`)
opens the full views (A Silent failure, B Device setting, C Cost, D Live escalation, and the frontier and replay
screens). "Finish" on step 5 also opens it.

The rehearsal sheet with what to say per step, the one number, and the likely question is
demo/review/STEP_SCRIPT.md. The numbers below are the register values at commit time; the screen is rebuilt from the
register, so read them off the screen on the day.

Never quote a figure from the first cloud redo (results/cloud_redo_v1.jsonl): it was a broken protocol run. Quote the
cloud fix rate only with its 95% interval and the inherited-actions caveat. Never quote a router saving or the
detector ceiling as a headline.

Backup: demo/review/demo_story.webm (3 minutes, no audio, no live call). Leave-behind: demo/review/LEAVE_BEHIND.pdf.

## Before the audience arrives (5 minutes, not part of the talk)

1. From the repo root on the controller laptop:

   ```
   py -3.12 -m demo.dashboard.build_cache
   py -3.12 -m pytest tests/test_dashboard.py tests/test_dashboard_views.py -q
   py -3.12 -m demo.dashboard
   ```

2. Open http://127.0.0.1:8765/ : step 1 of 5 shows, no "Error" text. (`?step=4&tol=12` opens step 4 with the slider
   at 12; `?step=5&machine=Intel%20laptop` opens step 5 on the Intel laptop.)

The live cloud call is only in the engineering view (D Live escalation): one call per server start through
src/cloud/client.py with a USD 10 cap and its own ledger; to enable it, start the server with the key loaded into that
process only:

```
$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')
$env:CLOUD_CLIENT_HOST_ROLE = 'controller'; $env:DEMO_CLOUD_LIVE = '1'
py -3.12 -m demo.dashboard
```

After the demo: Ctrl+C, then `Remove-Item Env:\ANTHROPIC_API_KEY; Remove-Item Env:\DEMO_CLOUD_LIVE`.

## Step 1 (0:00 to 1:00): Same laptop, two settings

Say: "Same Intel laptop, same open 8B model. Default setting: 5 of 5 agent sessions broke their own rules, and no error
was shown. Right setting: 5 of 5 kept their rules." Measured (t2s-r2-real-cpu, t2s-r2-real-igpu; the Intel laptop's
CPU device loads a 4096 memory setting, its graphics chip 32768).

## Step 2 (1:00 to 2:00): How often answers are silently wrong

Say: "On real tasks a 14B model was wrong 11% of the time with enough memory, 35% when tight, 42% when tighter. Error
messages shown: zero." Measured (x2-evalset-tasks-real-v2-wrong-rate: 11.3%, 35.3%, 42.0%, shown rounded; 0 of 197
wrong answers showed an error).

## Step 3 (2:00 to 3:00): Why not just change one setting?

Say: "A bigger memory setting can be up to 16 times slower, or will not start. The quick fix still leaves 30% wrong.
Trimming old history breaks a different rule." Measured (impact-session-time-by-window: 202.3 vs 12.2 minutes per
session; fh1-amd-boundary-qwen3-32b: first failed to start at a memory setting of 367360; x2-evalset-tasks-real-v2-
wrong-rate: 30.0% with num_keep; R2-qwen3-14b-4096-reconciliation: the logging rule held in 24, 25 and 25 of 40 turns
with trimming against 40 of 40 without).

## Step 4 (3:00 to 4:00): What it costs

Say: "A whole long task in a top cloud model costs about $2.53. Fixing one wrong answer later costs about 2 cents, but
actions already taken cannot be undone. How much accuracy gap you accept decides where it runs; we measure that gap
per device." Move the slider to 12, then back to 5. Priced: cloud-cost-per-real-task (Sonnet, realistic caching).
Measured: cloud-redo-full-context (mean USD 0.0227; 84.8% fixed, 95% Wilson CI 79.1% to 89.1%; 28 of 30 unfixed
inherited earlier actions) and hybrid-route-at-start (the slider).

## Step 5 (4:00 to 5:00): What our tool does

Pick "AMD workstation", then "Intel laptop", then "NVIDIA laptop". Say: "At the gap you accept, the tool says run
locally with this setting, or send the whole session to the cloud from the start, with the accuracy to expect and what
it saves per month. Where we have no data, it says not measured." Measured (hybrid-route-at-start; the Intel laptop's
local accuracy is measured by R2 session pass rate, not task accuracy). NVIDIA laptop: not measured.

Close: "The failure is silent, the right setting depends on the machine, and we measure the accuracy gap per device so
you can decide where it runs." Then "Finish" for questions in the engineering view.

## Which claims are measured

| Claim on screen | Tag | Source |
|---|---|---|
| Default setting: 5 of 5 sessions broke their own rules, no error shown | measured | t2s-r2-real-cpu |
| Right setting: 5 of 5 kept their rules | measured | t2s-r2-real-igpu |
| 11% / 35% / 42% wrong (rounded), error messages shown 0 | measured | x2-evalset-tasks-real-v2-wrong-rate |
| Bigger memory setting up to 16x slower, or will not start | measured | impact-session-time-by-window, fh1-amd-boundary-qwen3-32b |
| The quick fix: still 30% wrong | measured | x2-evalset-tasks-real-v2-wrong-rate |
| Trimming old history breaks a different rule | measured | R2-qwen3-14b-4096-reconciliation |
| $2.53 per long task in the cloud | priced | cloud-cost-per-real-task |
| Fixing a wrong answer later: about 2 cents | measured | cloud-redo-full-context |
| Slider and tool card: route, expected accuracy, savings per month | measured | hybrid-route-at-start |
| NVIDIA laptop | not measured | none |
