# FLEET-FRONTIER: registration (H4, supporting result)

Registered Fri Oct 23 2026, about 1:45 PM ET (Task BO), before any fleet result was computed.
Mac only, $0. Supporting result under the PLAN LOCK (docs/PLAN_LOCK_2026-10-23.md, addendum).
Predictions: derived/fleet/FLEET_PREDICTIONS.json (DERIVED; tools/fleet_predict.py). Extraction:
tools/fleet_cdf.py. Tests: tests/test_cc_bo.py.

## Question
What share of real agent steps fits the usable context of each installed memory tier:
- (a) at runtime defaults;
- (b) at the law's memory ceiling, L_mem = B / (k + a), with B set by the Windows graphics-memory
  formula?

## Inputs (pinned)
- **Demand: TraceLab v0.0.2 (UW-SyFI/TraceLab).**
  - Revision 7256bbf77e76d5cb83f4ddbd5ad31d8faee9dc7a, file data/v0.0.2/syfi_coding_trace.jsonl.gz,
    sha256 11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65, CC BY 4.0.
  - 665,453 rounds; the paper analyses 4,265 sessions, so this version is named in every result.
  - Per-step context = input_tokens_total.
  - Stored in derived/fleet/raw/ and kept out of git (gitignored; re-fetch by revision + hash).
  - Power-user selection bias is noted.
- **Copilot traces 2608.00101:** the reported median of 68K tokens per call is a cross-check only;
  there is no raw data.
- **Fleet: the Steam Hardware Survey, September 2026, RAM tier mix.** It is fetched and pinned (URL,
  date, sha256 of the page) before any result. It is gamer-skewed; this is stated.
- **Defaults: Ollama tiers** (docs.ollama.com/context-length): 4,096 below 24 GiB of VRAM, 32,768
  for 24-48 GiB, 262,144 at 48 GiB and above. The iGPU graphics tier stands in for VRAM.

## Method (fixed now)
- **Tier capacity:**
  - S(M) = MIN(0.80 M, MAX(M - 16 GiB, 0.50 M)), M = OS-visible RAM (installed - 0.5 GiB,
    ASSUMED);
  - B(M) = S(M) - (W + c0), where W + c0 = S_XPS - B_XPS(fit). S_XPS is replaced by the A19
    measurement with no refit;
  - L_mem uncapped = B / (k + a), with a bounded step B / k; L* = min(L_mem, 40,960, default
    window).
- **Fit fraction per tier** = the CDF of input_tokens_total at L*, from tools/fleet_cdf.py. It is
  weighted by the Steam tier mix for the fleet number.
- **Computed only after the Nov 6 gate**, with the gate-passing law parameters. Before that only the
  schema check runs (done 2026-10-23: 100 records, no missing fields).

## Predictions (DERIVED; FLEET_PREDICTIONS.json)
- **Per-tier capacities, Qwen3-4B INT4:**
  - 8 GiB does not load (B < 0);
  - 16 GiB: 15,000 / 26,500 uncapped (in-sample, by construction);
  - 32 GiB: about 50K f16, so window-bound;
  - 64 GiB: window-bound, with the default 32K (graphics tier 47.5 GiB < 48).
- **Fit fractions:**
  - at defaults, <= 2% fleet-weighted;
  - at the 16 GiB measured ceilings, within [0%, 20%].
  - The award report's lognormal figures are recorded verbatim. The TraceLab CDF replaces them.
