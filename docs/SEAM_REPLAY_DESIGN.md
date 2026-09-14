# SEAM replay design and holdout results

## Design

`tools/fdr_replay.py` replays sealed local/cloud ledgers under:

- `cloud_only`
- `agnostic_default` (DERIVED from sealed X-2 `cb781dbf-3486-4fbc-a69a-34026f801abe`)
- `slo_escalate`
- `emission_escalate`

## Outputs (`derived/d1_replay/`)

| File | Role |
|---|---|
| `configs.json` | Config grid |
| `pareto.json` | Domination / frontier |
| `H1_PREDICTIONS.md` | Predicted $ / quality (run_ids cited) |
| `X2_REPLAY_CHECK.md` | Holdout / consistency |

Holdout details and ASSUMED tags: see those files. Cloud $ fit tags
ASSUMED(fit=20 entries) in `H1_PREDICTIONS.md`.
