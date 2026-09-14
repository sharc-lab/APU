# Canary drift guard

A runtime-agnostic protocol for catching instrument drift during a long
measurement run. Written to be reimplemented, not ported: the reference
implementation is PowerShell bound to OpenVINO delta-prefill cells
(`tools/run_delta_prefill_matrix.ps1`), but nothing about the protocol is
specific to that stack.

---

## The problem it solves

A free-memory check before each cell is necessary and not sufficient. Two
failures get past it.

**Headroom that is never released between cells.** A dry-run gate passed at
7,536 MB available, and every cell in that session then ran at roughly 4,800 MB,
because memory freed by one cell had not returned before the next started. The
gate was checked once at launch and never again.

**Drift a threshold cannot see at all.** Identical cells, identical prompt
hashes, roughly one hour after boot against roughly twenty-two hours:

```
n_cached = 2,000     1.097 s  ->   2.393 s
n_cached = 8,000     7.090 s  ->  18.137 s     2.56x
```

Available memory was *higher* on the slow day. Clocks were flat. No thermal
signal. Nothing in a memory threshold, a temperature reading or a frequency
counter would have refused either run.

The only thing that detects this is re-measuring a known quantity partway
through and comparing it against itself.

---

## Protocol

### 1. One fixed cell

Pick a single configuration and never vary it. Everything about it stays
constant for the life of the run: model, arm, context length, delta, residency
mode, generation settings.

Reference implementation uses `gpu_only_f16`, `n_cached=4000`, `delta=200`,
`RESIDENT`. The specific values do not matter. What matters is that it sits in
the operating range of the real cells, so it drifts when they drift, and that it
is cheap enough to run often.

### 2. Fire it every N cells

N is derived, not chosen. Take the shortest observed interval between a
last-good and a first-bad cell in a prior degraded session, and divide by the
mean wall time per cell:

```
N = floor(657 s / 51.34 s) = 12
```

from session `7f569929`. That guarantees at least one canary lands inside the
shortest degradation onset you have actually observed. If you have no such
session yet, run without a canary once, find the onset, then derive N from it.

### 3. Calibrate on the first C canaries

The first `C` successful canaries establish the baseline. `C = 3` in the
reference; the implementation refuses `C < 2`, since you cannot derive variance
from one point.

```
ref_t1        = median(turn1_prefill_s) over the C calibration canaries
ref_t2        = median(turn2_prefill_s) over the same
early_max_t1  = max_i |t1_i - ref_t1| / ref_t1
early_max_t2  = max_i |t2_i - ref_t2| / ref_t2
```

`early_max` is the instrument's own observed noise, measured on the machine that
is about to do the work, in the state it is about to do it in.

**The guard does not arm until calibration completes.** Canaries 1 through C
cannot trip it.

### 4. Threshold derived from that noise

```
threshold_t1 = max(2 * early_max_t1, rel_drift_floor)
threshold_t2 = max(2 * early_max_t2, rel_drift_floor)

rel_drift_floor = 0.05
```

The `2x` allows as much additional deviation as calibration already showed. The
floor covers the case where `early_max` comes out near zero and the threshold
would otherwise collapse to something that trips on rounding.

**This is the load-bearing part.** The threshold is not 2x, not 5x, not any
round number picked before the run. It is derived from measurements taken during
the run, on the machine doing the measuring. A pre-chosen threshold is a guess
about an instrument you have not characterized.

### 5. Abort, do not degrade

Every canary after calibration is compared against `ref` on both turns. If
either exceeds its threshold, the session stops immediately with status
`FAIL_CANARY_DRIFT`.

An aborted run is a successful guard, not a failed run. A run that silently
degrades 2.5x and finishes is worse than one that stops, because it produces
numbers that look valid and are not.

---

## What to record

Per canary: the raw measurements, the index of the matrix cell it followed, and
available memory at its start.

Per session: the fixed cell configuration, `N` with its derivation, `C`,
`rel_drift_floor`, the computed `ref_t1` / `ref_t2` and `early_max_t1` /
`early_max_t2`, and the resulting thresholds. If the run aborts, the canary that
tripped it and by how much.

Recording the derivation alongside the number is the point. A threshold whose
provenance cannot be named is indistinguishable from one someone made up.

---

## Reimplementing on another runtime

Substitutions for a non-OpenVINO stack:

| reference | substitute |
|---|---|
| `turn1_prefill_s`, `turn2_prefill_s` | any two stable per-cell timings you already record; prefill and delta-prefill work because they are deterministic given a fixed prompt |
| `gpu_only_f16` arm | whatever your equivalent fixed configuration is |
| PowerShell orchestrator | anything that can interleave a fixed cell into the run queue |

Two properties the substitute metric must have. It has to be deterministic given
a fixed input, so variation means the machine changed rather than the workload.
And it has to live in the operating range of the real cells, so it degrades when
they degrade. A metric that is too cheap will not drift with the thing you care
about.

Generated-token counts and anything sampling-dependent are poor choices. Prefill
time is a good one because the same prompt does the same work every time.

---

## Interaction with the memory floor

The canary replaces neither the floor nor the tier-1 process check. All three
address different failures.

```
tier-1 refuse      operator software resident during a measurement
memory floor       insufficient headroom, checked BEFORE EVERY CELL
canary             drift the other two cannot see
```

The floor being checked per cell rather than once at launch is not a detail. It
is the specific failure that let a session pass its gate at 7,536 MB and then
run every cell at 4,800.
