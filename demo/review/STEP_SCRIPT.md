# What each step says (rehearsal sheet, one page)

Five steps, about a minute each. Read the big number, say two or three sentences, then Next. Every number has a small
tag (measured or priced); the sources are in each step's closed Details panel. The engineering view is the link in the
corner.

## 1. Same laptop, two settings
- **Number:** 5 of 5 (measured).
- **Say:** "Same Intel laptop, same open 8B model, same agent task. With the default setting, 5 of 5 sessions broke
  their own rules, and no error was shown. With the right setting, 5 of 5 kept their rules."
- **Likely question:** "Did the right setting get everything right?" **Answer:** "It kept every rule we check; how
  well it remembered older facts is a separate measurement, in Details."

## 2. How often answers are silently wrong
- **Number:** 42% (measured).
- **Say:** "On real tasks, a 14B model gave 11% wrong answers with enough memory, 35% when tight, 42% when tighter.
  Error messages shown: zero."
- **Likely question:** "Are those exact?" **Answer:** "Rounded to whole percent; the exact counts are in Details,
  150 tasks per setting."

## 3. Why not just change one setting?
- **Number:** 16.6x (measured).
- **Say:** "At the default memory setting a session is 16.6 times slower than at the tightest setting, and the
  tightest setting is the one that is 42% wrong. The quick fix still leaves 30% wrong. Trimming old history breaks a
  different rule."
- **Likely question:** "16.6 times slower than what?" **Answer:** "A 40-turn session took 202.3 minutes at the
  default memory setting against 12.2 at the 4096 setting, on the AMD workstation (row impact-session-time-by-window).
  The fast arm is the broken one, so this is not a like-for-like speedup."

## 4. What it costs
- **Number:** $2.53 (priced) and 1.76x (measured).
- **Say:** "Sending a whole long task to a top cloud model costs about $2.53. Fixing one wrong answer later costs
  1.76 times a fresh cloud run of the same task, and actions the agent already took cannot be undone. How much accuracy gap you accept decides where
  it runs; we measure that gap per device." Move the slider to 12, then back to 5. The dollar figures beside it are
  projected: measured cost per task times an assumed 1,000 tasks a day.
- **Live button:** click "Fix one wrong answer now (live)" once. Live (tag measured): "That is a real wrong answer
  sent to the cloud just now: here is the fix, what it cost in cents and how long it took." If it falls back (tag
  recorded): "No live connection here, so this is the recorded fix of the same answer from our test run: same task,
  same protocol." A second click shows the same result; there is one live call per demo run.
- **Likely question:** "Does the cloud fix every wrong answer?" **Answer:** "No: 84.8% (95% confidence interval 79.1%
  to 89.1%); most of the rest inherited actions the local model had already taken."

## 5. What our tool does
- **Number:** expected accuracy and monthly savings for the picked machine (measured).
- **Say:** "Pick the machine. At the gap you accept, the tool says run locally with this setting, or send the whole
  session to the cloud from the start, with the accuracy to expect and what it saves. Where we have no data, it says
  not measured."
- **Likely question:** "Why does it say cloud at 5 points?" **Answer:** "Measured accuracy locally is 80 of 90 against
  90 of 90 in the cloud, a gap of 11.1 points on the workstation; at 11.1 points or more it runs locally."
- **Intel laptop saving:** USD 169.89 a month at 1,000 tasks a day against claude-sonnet-5-5 (row
  t2s-config-comparison, Intel iGPU, qwen3:14b), not $306: the $306 is the naive Intel CPU column, another setting.
