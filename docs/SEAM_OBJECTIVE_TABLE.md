# SEAM axis-by-objective table

Each cell cites a `run_id` (or ASSUMED). Resolve payloads in
`zjohnson2005/gnn-hls-accel` via `derived/SEAM_SEALED_INDEX.md`.

| Axis / objective | What was measured | run_id / seal | Notes |
|---|---|---|---|
| Topology (P vs LP-E map) | Logical CPU clustering | `fb5cd2d5-e850-4de1-9b90-d368b5aa9994` | M1 ACCEPTED; not a performance ratio |
| C-2 TTFT limits (unguarded canary era) | turn-1 TTFT limit per KV | `62395fdb-1899-415f-b708-6adc81a24dda` | see RESULT note on unguarded |
| C-2 TTFT (canary present, never armed) | u8/u4 limits 9750 | `c647f0c7-5cc9-47bb-a491-3450533c34d1` | caveat: canary never armed |
| W-3 quality int4 | multi-turn BFCL quality | `6225d6e1-4e0a-41c9-90bb-695ecc5fbe0a` | H-1 entry pin |
| W-3 quality int8 | multi-turn BFCL quality | `1d8db970-4c18-4bcf-824d-d9c141b6eb22` | paired weight |
| X-2 feasibility / R1 source | cpu-p NON_RESIDENT wall | `cb781dbf-3486-4fbc-a69a-34026f801abe` | DERIVED R1 scale source |
| RESIDENT delta-prefill | SD-001 | `41e419bd-f3e9-43b1-8364-0ebd89fa086b` | H-1 input |
| Decode BW+c fit | gpu_only decode | `a784f5ec-5615-4fea-a680-a07874426ae4` | cpu-p decode ASSUMED from gpu fit |
| Commit intercept | — | `83127e1b-9d6e-4103-bee6-2a63c00f479f` | H-1 input |
| H-1 predicted cloud $ | FDR replay predictions | `derived/d1_replay/H1_PREDICTIONS.md` | cites input run_ids above |
| Pareto / domination | config frontier | `derived/d1_replay/pareto.json` | replay over sealed configs |
| X-2 replay holdout check | replay consistency | `derived/d1_replay/X2_REPLAY_CHECK.md` | holdout results |
