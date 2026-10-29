# Approvals recorded 2026-10-21 (Task BE)

Status: REGISTERED 2026-10-21. Zach approved the items below in chat on
2026-10-21 at 9:45 PM ET ("download and go ahead on all decisions"), as written in
~/seam_closeout_1013/papers_20261015/TASK_BE_PROMPT.md. This file records approvals
only. It edits no registered file, and runners do not read it.

## Earlier 2026-10-21 approvals (recorded here for completeness)
- Task BC: "write it" for DEMO2_ARMS_REVISED_2026-10-21.md (17a102e); "yes" for
  DEMO2_PER_DEVICE_2026-10-21.md (bad1bbd).

## 1. Model downloads: APPROVED
- Qwen3-8B INT4 on both hosts: OpenVINO/Qwen3-8B-int4-ov @
  5c47abf4b8e12ebe8e99745bb0c1ec17e0c0abcc (pin configs/models/Qwen3-8B-int4-ov.v2.yaml,
  ir_sha256 cd97874b...).
- Qwen3-8B INT8 on the T2S: OpenVINO/Qwen3-8B-int8-ov @
  f82c978d714abc9c398efa361827a76d3cae5473 (new pin configs/models/Qwen3-8B-int8-ov.yaml,
  ir_sha256 eac56228...).
- Qwen3-4B INT8 on the XPS: OpenVINO/Qwen3-4B-int8-ov @
  fd9c3cc8adf942151e48904b7341a57eae58d8d5 (pin configs/models/Qwen3-4B-int8-ov.yaml,
  ir_sha256 f7d323d0...).
- All three are official OpenVINO IRs of the hybrid Qwen3 (base_model Qwen/Qwen3-8B or
  Qwen3-4B, not 2507). No export is needed.
- Zach runs ~/seam_closeout_1013/xps/download_models_{xps,t2s}.sh. Claude runs nothing on
  the hosts.

## 2. Cloud spend: APPROVED
- Calibration cloud-only log (DEMO2_CALIB_30): ~$5-20, cap $20.
- Held-out cloud-only log (DEMO2_HELDOUT_60, once, after the freeze): ~$5-20, cap $20.
- P7 and the 1-replicate held-out arms stay approved within the registered caps.
  2 replicates are NOT approved.
- Ceiling $275 unchanged. DEMO-2 cloud spend so far: $0.

## 3. Arm A re-registered as RESIDENT, with shared corrections in EVERY arm: APPROVED
- One persistent pipeline per run, with tool-set prefixes warm (--resident-pipeline
  --warm-toolset-prefix) and the run order grouped by tool set (--order-by-toolset).
- Prompt caching on (--caching-policy ephemeral) and cloud-side tool-output retention
  (K 1,024 h70t30; --retention-cloud-only for arms without local retention).
- Applied identically to every arm and reported separately from any arm's gain.
- New arm A predictions are DERIVED (derived/h1_hybrid/demo2/DEMO2_BE_PREDICTIONS.json).

## 4. Device-provisioned arm P: APPROVED
- XPS P = Qwen3-8B INT4, resident with pre-warmed tool-set prefixes. Fallback P' = Qwen3-4B
  INT8 if the 8B trips the step-0 10 s trigger more often than A (rule in
  DEMO2_ISO_QUALITY_2026-10-21.md, fixed now).
- T2S P = Qwen3-8B INT8, resident.
- T2S thinking-with-budget is a separately registered secondary arm only if a harness exists
  in time; else NOT_RUN. The harness does not exist (2026-10-21), so it is NOT_RUN.
- Label everywhere: "device-provisioned (model tier/precision), not model-fixed".
