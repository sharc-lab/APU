# APU audit: demo detector check v2 (evo-x2, qwen3:14b)

- **Logs:** results/demo_detector_check_v2.jsonl with results/demo_live_logs/detector_check_v2/serve_*.log (replayed)
- **Device:** AMD Ryzen AI Max+ 395
- **Runtime:** ollama 0.34.4
- **Model:** qwen3:14b
- **Cloud model priced:** claude-haiku-5-5
- **Token basis for cost:** server count
- **Cost bound:** none

**Recommended setting.** window 40960, KV not recorded (config ollama_default): lowest hidden failure rate (0.0%); hidden failure rate 0.0%, wrong 0/5, escalation USD 0.0000 per 100 tasks

Rule: Among settings with runtime cut-off ground truth (server log, or verdict recorded from it), within the cost bound on escalation cost per 100 tasks if one is given: lowest hidden failure rate; ties by lower wrong rate, then lower escalation cost, then larger window, then config name.

| config | window | KV | ground truth | tasks | hidden failures | hidden rate | wrong | wrong with hidden cut | wrong without a cut | cloud USD floor | cloud USD upper | cloud USD billed | escalation USD per 100 tasks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ollama_ctx_4096 | 4096 | not recorded | server_log | 5 | 5 | 100.0% | 2/5 | 2 | 0 | 0.0022 | 0.0040 | n/a | 0.0441 |
| ollama_default | 40960 | not recorded | server_log | 5 | 0 | 0.0% | 0/5 | 0 | 0 | 0.0041 | 0.0073 | n/a | 0.0000 |

- Hidden failure: the runtime cut the context (its own cut-off log line) while the client saw HTTP 200 and no error; a task is a hidden failure when any of its requests is. PREDICTED rows come from the chars/4 estimate, not a log, and are never recommended.
- A hidden failure rate of 0 does not mean the tasks were right: see "wrong without a cut".
- Cloud cost: the same sessions priced on the committed price tables (floor = best caching, upper = no caching; billed only when provider usage was recorded). Escalation cost = hidden failure rate x mean floor cost per task x 100. Where completion tokens were not recorded, cost is input only.
