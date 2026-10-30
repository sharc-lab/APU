# APU audit: own compaction and client trim (evo-x2, qwen3:14b)

- **Logs:** results/x2_own_compaction_v1_a.jsonl and _b.jsonl (detector v2 verdicts recorded per turn; logs not kept)
- **Device:** AMD Ryzen AI Max+ 395
- **Runtime:** ollama 0.34.4
- **Model:** qwen3:14b
- **Cloud model priced:** claude-haiku-5-5
- **Token basis for cost:** calibrated estimate of the final call as sent
- **Cost bound:** none

**Recommended setting.** window 8192, KV not recorded (config own_compaction_8192): lowest hidden failure rate (0.0%), tied with 1 other setting(s), broken by the rule's tie order; hidden failure rate 0.0%, wrong 24/150, escalation USD 0.0000 per 100 tasks

Rule: Among settings with runtime cut-off ground truth (server log, or verdict recorded from it), within the cost bound on escalation cost per 100 tasks if one is given: lowest hidden failure rate; ties by lower wrong rate, then lower escalation cost, then larger window, then config name.

| config | window | KV | ground truth | tasks | hidden failures | hidden rate | wrong | wrong with hidden cut | wrong without a cut | cloud USD floor | cloud USD upper | cloud USD billed | escalation USD per 100 tasks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| client_trim_4096 | 4096 | not recorded | recorded_verdict | 150 | 8 | 5.3% | 36/150 | 6 | 30 | 0.0261 | 0.1730 | n/a | 0.0009 |
| own_compaction_4096 | 4096 | not recorded | recorded_verdict | 150 | 8 | 5.3% | 35/150 | 6 | 29 | 0.0261 | 0.1729 | n/a | 0.0009 |
| client_trim_8192 | 8192 | not recorded | recorded_verdict | 150 | 0 | 0.0% | 30/150 | 0 | 30 | 0.0252 | 0.1684 | n/a | 0.0000 |
| own_compaction_8192 | 8192 | not recorded | recorded_verdict | 150 | 0 | 0.0% | 24/150 | 0 | 24 | 0.0253 | 0.1691 | n/a | 0.0000 |

- Hidden failure: the runtime cut the context (its own cut-off log line) while the client saw HTTP 200 and no error; a task is a hidden failure when any of its requests is. PREDICTED rows come from the chars/4 estimate, not a log, and are never recommended.
- A hidden failure rate of 0 does not mean the tasks were right: see "wrong without a cut".
- Cloud cost: the same sessions priced on the committed price tables (floor = best caching, upper = no caching; billed only when provider usage was recorded). Escalation cost = hidden failure rate x mean floor cost per task x 100. Where completion tokens were not recorded, cost is input only.
