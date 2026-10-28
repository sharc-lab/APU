# apu_bench: format plan against a typical vendor benchmark RFP

Status (2026-10-26): harness built and tested offline with the fake runtime only. It has not been run on any
machine. No number in this document is a measurement.

## What it is

One command:

    py -3.12 -m apu_bench run --profile apu_bench/profiles/evo-x2-qwen3-14b.json --subset b3v2-small --out <dir>
        [--runtime fake|live] [--execute] [--rules-from <validation.jsonl>]

Inputs: a device profile (`apu-device-profile/1`, `apu_bench/profile.py`) and a B3 v2 task subset
(`apu_bench/taskset.py`: `b3v2-small` = seed 20260901, 30 task turns; `b3v2-3seed` = seeds 20260901-03, 90 task turns).

Outputs in `<dir>`:

- `result.json` (`apu-bench-result/1`): the device profile and its sha256, the missing hardware tags, the task set id,
  seeds and content sha256, git head and dirty flag, Python and OS, the blob ids of the harness files, the server
  command and env, the client command line of each step, the run rules, the records file sha256, and per window:
  tasks, wrong, wrong rate, wrong silent, wrong flagged, flagged, seconds per task (mean, median, p90).
- `report.md`: the same, readable (device table, task set, results table, a "Reproduce" section).
- `records.jsonl`: the per-turn records the summary was computed from (fake mode: generated; live mode: what
  `harness/x2_r2_evalset.py` wrote).

Runtimes:

- `fake`: records generated offline from the task definitions. The loaded window is the profile's, the prompt is the
  task-definition projection, flagged means the projection is over the window (`src/dse/fits_locally.py` baseline),
  and the outcome and seconds come from a seeded hash. Every fake report says "FAKE RUNTIME: generated records, not a
  measurement" at the top.
- `live`: composes the two `x2_r2_evalset.py` steps (validation, then real on the profile's tiers and the subset's
  seeds). Without `--execute` it only writes `plan.json`. With `--execute` it runs them in-process through
  `x2_r2_evalset.main(argv, advance=False)`, so no queue is advanced and no pause is set. It is meant for the device
  itself. Under this repo's rules a real run needs a pre-registration first.

## Mapping to typical RFP requirements

MLPerf Inference run rules and submission format are the reference here: a fixed and versioned benchmark, declared
system description fields, fixed seeds and scenarios, an accuracy gate, logs submitted with results, and audit or
reproduction by a third party.

| RFP requirement | What apu_bench does | Gap |
|---|---|---|
| Versioned benchmark | `bench_version`, `schema` on every output; task set identified by a sha256 of the exact content sent (system prompt, tool schema, every user message, item ids and kinds) | No release process or tags yet; a schema change has no migration notes |
| Fixed workload | Subsets are fixed seed lists over a generated task set; the hash proves two runs used the same tasks | Only B3 v2 (short tasks). No long-task or multi-model subset |
| Reproducibility | git head + dirty flag, harness file blob ids, exact client argv, server command and env, seeds, records sha256 | Blob ids are of the working-tree bytes (on Windows with autocrlf they differ from git's blob). No container or lockfile for Python or runtime |
| Hardware tagging | Profile `hardware`: hostname, cpu, gpu, memory_gb, os, driver; `runtime` name and version; `model` name, quant, digest. Missing tags are listed in the report | Tags are self-declared, not probed. The evo-x2 profile has gpu, memory, os, driver, quant and digest null (not in any committed file) |
| Run rules | Recorded in `run_rules`: unit = task turn of a completed session, one session per seed per window, no retries, scoring by `end_state_ok`, flag = window overflow or detector v2 cut | No warm-up rule (the repo convention of 1 warm-up + N measured is not applied to agent sessions), no thermal or power state rule, no minimum run duration, no repeat count or variance requirement |
| Accuracy gate | `x2_r2_evalset` validation step and preflight gates (live mode) | Not surfaced as a pass/fail field in `result.json` |
| Latency metric | Seconds per task (turn wall clock): mean, median, p90 | No time to first token, no tokens per second, no energy |
| Silent failure reporting | Wrong split into silent vs flagged per window | Flag sources are window overflow and detector v2 only, so an Ollama-specific log parser. Other runtimes need their own cut detector |
| Result submission format | One JSON file per run plus the records; schema version inside | No JSON Schema file published, no signature, no submitter or organisation field, no division (closed vs open) |
| Third-party audit | Records and logs are kept, and the result can be recomputed from `records.jsonl` (`bench.summarize`) | No audit script that re-runs a subset and compares. Detector v2 logs are not copied into the output directory yet |
| Multiple runtimes | Profile has `runtime.name` | Live mode drives Ollama through `x2_r2_evalset` only. llama-server or vLLM need a runtime adapter |

## Gaps to close before an external submission (planned, not done)

1. Probe hardware tags on the device instead of declaring them (about 2 h: OS, CPU, GPU, driver and memory from
   standard system calls, compared with the profile).
2. Publish a JSON Schema for `apu-device-profile/1` and `apu-bench-result/1` and validate against it in the tests
   (about 2 h).
3. Run rules: decide warm-up, repeats and variance reporting, and write them into the pre-registration (about 1 h to
   write, then a real run).
4. Copy the server debug logs and the validation file into the output directory with their hashes (about 1 h).
5. First live run on one device under a pre-registration (machine time; not scheduled).
6. A runtime adapter for llama-server (about 4 h, reusing `harness/server_guard.py`).
