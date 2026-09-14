# SEAM Provenance (export)

## Canonical evidence repo

All sealed measurement evidence for SEAM lives in
**[zjohnson2005/gnn-hls-accel](https://github.com/zjohnson2005/gnn-hls-accel)**.
That repository is **canonical for provenance**. This `seam/characterization`
branch on `sharc-lab/APU` is a **reviewable export** of code, configs, and
derived summaries only. It does not move `raw/` payloads.

## How numbers resolve

Every numeric claim in the export docs either:

1. cites a `run_id` that resolves to a sealed tree under `raw/` or
   `derived/` in `zjohnson2005/gnn-hls-accel`, or
2. is explicitly tagged **ASSUMED**.

Use `derived/SEAM_SEALED_INDEX.md` and `derived/d1_replay/` to look up
run_ids. To inspect bytes, clone the canonical repo and open the path in
the index `source_path` column.

## GIT_SHA_MAP rewrite (AM-039)

History was rewritten once with `git filter-repo` to strip GitHub-rejected
oversized blobs so the evidence branch could push. Sealed manifests still
cite pre-rewrite commit SHAs via `git_sha` fields; those sealed files are
byte-identical.

The auditable old→new map is `docs/GIT_SHA_MAP.md` (AM-039). Cited SHAs
are **translatable**: rewritten trees differ from pre-rewrite trees only by
deletion of oversized paths that were never inputs to sealed runs. Evidence
tree digest over `raw/`+`derived/` is unchanged across the rewrite.

Pre-rewrite archive (out of repo): local tag `prerewrite/875dc74`, bundle
SHA-256 `9878b87630586b7a43714aab1644ba95e03575968bb452fac2495680f6ff1792`.

## Requesting or mirroring raw artifacts

1. Clone `zjohnson2005/gnn-hls-accel` (canonical).
2. Locate the sealed directory via `derived/SEAM_SEALED_INDEX.md`.
3. Verify with the sealed `tree_sha256` / `.sealed` marker.
4. For a partial mirror, request specific `run_id`s from the SEAM maintainer;
   do not invent substitute numbers.

## What this export deliberately omits

`raw/` run dirs, large derived artifacts, corpus, figures, analysis probe
dumps, censor trees, and the prerewrite bundle.
