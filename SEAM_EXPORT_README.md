# SEAM characterization export

Reviewable export for `sharc-lab/APU` (`seam/characterization`).

Canonical evidence: `zjohnson2005/gnn-hls-accel`. Start at `docs/SEAM_PROVENANCE.md`.
Adds only paths new relative to `main` on `sharc-lab/APU` (no overwrites of lab files).

## Refresh LAB-2 (2026-10-29; filtered export approved by Zach, Thu Oct 29 2026 evening ET)

Source commit: `zjohnson2005/gnn-hls-accel@1d18b4f137d884af9eabd91df3edbcd6ef95791a` (all files read from that commit's git objects).
Base: `3e6bad2` (tip of `seam/characterization` on `sharc-lab/APU` before this commit). Added on top as one
fast-forward commit.

What was added or refreshed:
- `docs/registrations/`, `docs/amendments/`, `docs/PLAN_LOCK_2026-10-23.md`,
  `docs/EXPERIMENT_REGISTRY.md`, `docs/CANARY_PROTOCOL.md`: verbatim copies of the
  canonical files, except that host names, tailnet IPs, ssh targets and user paths are
  redacted (each redacted file and its canonical sha256 is listed in `EXPORT_VERIFY.json`).
  Numbers inside them carry their own run_id / DERIVED / ASSUMED tags as written in the
  canonical repo; no number was edited here.
- Tools and configs to reproduce CC / INV-1 / XRT-1 (incl. `tools/xrt_llamacpp.py`),
  found by following Python imports and PowerShell references from the runners named
  in the CC / INV-1 / XRT-1 queue scripts. The queue scripts themselves live outside the
  canonical repo and are not exported.
- Predictions (DERIVED, registered before the runs): `derived/br/`, `derived/bs/`,
  `derived/cc/*.json`, and Task BT `derived/cost1/` (COST-1) and `derived/route1/` (ROUTE-1 and the
  DUET addendum). The COST-1 registration, ROUTE-1 amendment and DUET addendum carry their own
  sha256 headers (self-hash computed with the value replaced by 64 zeros).
- Task BT simulators: `seam/route1/` and `seam/duet/` (pure Python, synthetic sessions only;
  device and TTFT constants are PLACEHOLDER until the Nov 22 freeze) with their tests in
  `tests/bt/` (run from the export root: `python -m pytest tests/bt`).
- CC closeouts: `analysis/cc/`, `analysis/cc_20261026/` (scores cite run_ids).
- CC figures: `analysis/cc_figs_20261026/` (DERIVED from `analysis/cc_20261026/`); source paths
  and sha256 in `EXPORT_VERIFY.json`.
- Sealed CC summaries: `derived/cc_sealed_summaries/<run_id>.summary.json` (70 files,
  summary.json only, no raw traces). Host names and user paths are redacted in these
  copies, so they are NOT byte-identical to the sealed files; `EXPORT_VERIFY.json`
  lists each canonical path and canonical sha256 for verification against `zjohnson2005/gnn-hls-accel`.
- INV-1 / XRT-1 scoring notes: not committed at the source commit, so not included.
- Filtered export: host names, tailnet IPs, ssh targets and user paths are also redacted in
  the files kept from earlier exports (38 files; listed in `EXPORT_VERIFY.json`).
- Public by default (approved by Zach, Thu Oct 29 2026 evening ET). Private spans are replaced by
  `[private]` in 12 files; each is listed in `EXPORT_VERIFY.json` with line, category
  and the sha256 of the unredacted canonical file, so hashed registrations stay verifiable against the
  private canonical copy. Files left out entirely: meeting notes, asks lists, coordination requests,
  research reports, paper abstracts and drafts, agent reports.

Files from the Sep 14 export that are not listed above are unchanged and may be older
than the source commit.
