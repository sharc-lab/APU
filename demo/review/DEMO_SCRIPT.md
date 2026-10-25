# Live demo script (one page, Tue 2026-10-27)

**Before the audience arrives (30 min warm-up, evo-x2 held free by the queued hold job):** in a PowerShell window at
the repo root run `.\demo\live\start.ps1`, then open the bookmark `http://127.0.0.1:8800/` in the browser and press
**F11** (full screen). Press **Run 5 tasks** once on the AMD scenario as
a warm-up (live cloud calls, about 16 cents on the demo ledger), then press **F5** to get a clean screen.
<!-- 16 cents: register row demo-live-run, amd8192 ours 0.021041 + cloud 0.010656 USD per task, x 5 tasks, billed -->
 (Spare window, already open: `.\demo\live\start.ps1 -Mode replay
-Port 8801`, see "If something fails".)

## 1. The screen (before pressing anything)
- **Click:** nothing yet. The scenario box shows "AMD workstation, constrained memory (8192)".
- **Say:** "Three ways to run the same assistant task. On the left, the computer does it alone with its standard
  settings. On the right, everything goes to a paid cloud model. In the middle is our tool: before each task it checks
  whether the conversation fits in this computer's memory, and sends it to the cloud only if it does not."

## 2. Run
- **Click:** **Run 5 tasks**.
- **Say, while task 1 runs:** "Each task is one step of a long conversation. The earlier part of the conversation is
  replayed from a recording; this step runs now. Watch the bar: it is how much of the conversation the computer
  could keep in memory."
- **When a lane turns red ("Cut off"):** "Here the computer had to drop part of the conversation, and it gives no
  warning. That is when answers go wrong without anyone noticing. The red line comes from the computer's own log,
  not from a guess."
- **When the middle lane says "On the cloud":** "Our tool saw before starting that this task would not fit, so it sent
  it to the cloud from the first step."

## 3. The three numbers (bottom of each card, "recorded headline", 90 tasks)
- **Say:** "On its own, at this memory size, the computer got 57 of 90 tasks right."
  <!-- 57/90: register row hybrid-route-at-start-measured (naive local, ollama_ctx_8192) -->
- **Say:** "With our tool deciding at the start of each task: 88 of 90 right, at under one cent per task."
  <!-- 88/90 and USD 0.00819 per task: register row hybrid-route-at-start-measured (hybrid, ollama_ctx_8192 vs claude-sonnet-5-5) -->
- **Say:** "Sending everything to the cloud: 90 of 90, at about one cent per task, and nothing stays on the computer."
  <!-- 90/90 and USD 0.01020 per task: register row hybrid-route-at-start-measured (all-cloud, claude-sonnet-5-5) -->
- **Then:** "Today's five tasks are a live sample; the 90-task figures are the measured result."

## 4. Second scenario (optional, 3 minutes)
- **Click:** the scenario box, choose "Intel laptop default, simulated on AMD", then **Run 5 tasks**.
- **Say:** "A typical laptop keeps much less of the conversation by default. Our tool picks the laptop's graphics
  chip setting, which we measured holds the whole conversation, so the work can stay on the laptop. Here the laptop
  is simulated on the workstation, as the screen says."

## If something fails (keep talking; nothing needs the keyboard except where stated)
- **The workstation does not answer:** nothing to do. Within about 10 seconds the screen says "The workstation did not
  answer" and continues with the recorded run; the badge turns **REPLAYED**. Say: "The workstation is not reachable,
  so this is the recorded run from Friday, at its real speed."
- **The cloud is refused (budget or the usage check):** nothing to do. Those cloud results show **REPLAYED** with the
  reason. Say: "The cloud part is the recorded result of the same task."
- **A lane stalls:** nothing to do. After 5 minutes without progress the screen switches to the recorded run by itself.
  To switch at once: press **Alt+Tab** to the spare window (REPLAYED screen on port 8801) and click **Run 5 tasks**.
- **The browser freezes:** press **F5**. The page rebuilds the run in progress from the server. If the browser itself
  is stuck, press **Ctrl+Shift+Esc**, end the browser, open `http://127.0.0.1:8800/` again.
- **Nothing works:** play the recorded clip: open `demo\review\demo_live_amd8192.webm` (double-click in File Explorer,
  press **F11** for full screen). Say: "This is the same demo recorded on Friday."
