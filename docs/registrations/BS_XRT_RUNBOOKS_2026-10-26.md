# XRT on Rithwik's machines: runbooks, E4 deviation, E5a/E5b split, optional T2S evening

Registered Mon Oct 26 2026, about 8:55 PM ET (Task BS), before any of these runs. Mac only, $0.
Dated note to docs/registrations/BR_XRT_2026-10-26.md, which stays unchanged. Basis: Rithwik's reply
of Mon Oct 26 evening and Zach's Task BS instruction (docs/amendments/APPROVALS_2026-10-26.md).
Predictions: derived/bs/BS_XRT_PREDICTIONS.json. Runners (the runbooks) read neither file.

## 1. Who runs what (operator record)
| Part | Machine | Operator | How | Our access |
|---|---|---|---|---|
| E3 | Rithwik's personal RTX 4070 laptop (8 GB VRAM), Windows | Rithwik | runs run.ps1 himself, sends the zip + screenshots | none |
| E5a / E5b | Strix Halo evo-x2, Windows | Rithwik's job queue | one queued job (run.ps1, env vars, no prompts) | none |
| T2S-XRT (optional) | EVO-T2S 64 GB | Zach's queue over ssh, in an evening Rithwik picks | as the v3 T2S queue | ssh, his OK per evening |
- OS build, GPU driver, driver-reported GPU memory, Shared GPU Memory / VGM, llama.cpp version line and
  the operator name are written by the runbooks (run_meta.json, system_at_start.txt, collect.ps1's
  system_record.txt and dxdiag.txt). We never had access to the 4070 or the evo-x2.
- The record is the logs + screenshots as sent. On receipt: the zip is stored as received under
  derived/xrt_rithwik/<machine>_<date>/ with RECEIPT.json (zip sha256 we compute vs the sha256 collect.ps1
  printed, and a check of the MANIFEST.sha256 inside), then sealed; nothing in it is edited.

## 2. Deviation: E4 (vLLM) NOT RUN
vLLM needs Linux (WSL or Docker) on these Windows machines; on the evo-x2 that means admin changes and
a reboot on his queue machine, which Rithwik did not approve. E4 is recorded as NOT RUN with that reason.
No substitute cell is added. P-XRT-7 (E3) and P-XRT-8 (E5, now E5b) are unchanged.

## 3. Runbooks (pinned; ~/seam_closeout_1013/rithwik_runbooks/, sources tools/rithwik_runbooks/)
- Same tag b11559 as XRT-1. New pins in configs/xrt/llamacpp_pin.yaml `rithwik_assets` (GitHub release
  digests, nothing downloaded): win CUDA 12.4 build zip 265,207,128 B (sha256 4a885b80...ecd7), its cudart
  zip 391,443,627 B (8c79a9b2...e1d6); optional win ROCm zip 257,958,782 B (5511af01...f9ab). The Vulkan
  zip and the Qwen3-4B Q4_K_M GGUF (rev bc640142) are the XRT-1 pins. CUDA 12.4 was chosen because its
  minimum Windows driver (551.61, NVIDIA CUDA 12.4 release notes, LITERATURE) suits a 40-series laptop;
  the script stops below 528.33 and asks below 551.61.
- Server flags, prompt rule (n - 64 tokens from one /tokenize call, tiled), hold rule and bisection are
  those of tools/xrt_llamacpp.py; tests/test_rithwik_runbooks.py checks the PowerShell bisection against
  bisect_plan and the argv against build_server_cmd.
- Additions over XRT-1 (recorded, not predictions): one repeat at the largest hold and the first fail
  (ADDENDUM 1 robust-ceiling rule); per-trial failure cause from the log (memory / max_buffer /
  refused_fa / other); on the 4070, `exceeds_vram` = CUDA buffers > nvidia-smi memory.total (a sign that
  Sysmem Fallback is on) and the operator's typed confirmation of "Prefer No Sysmem Fallback".
- E3 cells exactly as registered: FA off at ubatch 128 / 512 / 2048 with KV f16 (range 4,096-65,536),
  then FA on at ubatch 512 with f16 (4,096-65,536) and q4_0 (8,192-196,608). Step 1,024. At most 52 trials.

## 4. E5 split
- **E5a (default share only; vendor replication of the law).** llama.cpp b11559 Vulkan, KV f16 and q4_0,
  FA on, ubatch 512, step 1,024, at whatever GPU share the machine already has:
  - L cells (largest -c that loads; llama.cpp allocates the KV and compute buffers at load):
    f16 4,096-4,194,304, q4_0 8,192-4,194,304.
  - P cells (largest -c that loads AND prefills n - 64 tokens, the E3 rule): f16 4,096-131,072,
    q4_0 8,192-131,072. Above 131,072 (Qwen3-4B with YaRN) the window and the prefill time bind.
  - At most 54 trials; job timeout 6 h; ROCm build only with E5_ROCM=1 and only if it already works there.
- **E5b (3+ share points)** ONLY if Rithwik says share changes are OK (VGM/BIOS change + reboot per point
  on his queue machine). The script refuses unless E5B_SHARE_OK=1 and E5B_SHARE_GB (typed by the
  operator). Same cells per point. P-XRT-8 unchanged and scored on the P cells as registered; if every
  P cell holds at 131,072, P-XRT-8 is UNRESOLVED (window binds) and the L cells are reported against the
  same slope, descriptively.

## 5. E5a predictions (fixed now; the law with the machine's own reported memory)
Rule, no refit: L_pred(kv) = floor((F - W - C) / k_kv), with k f16 147,456 and q4_0 41,472 B/token
(P-XRT-1), and F, W, C read from the FIRST HELD trial log of that L cell, Vulkan device only:
F = "MiB free" on the "using device Vulkan0" line, W = Vulkan0 model buffer, C = Vulkan0 compute +
output buffers (all x 2^20 B). Nothing else is estimated. The FA-on non-KV term is flat in n_ctx
(the chunked case of the law: a x min(L, cap) with cap = ubatch).
- Source note (llama.cpp b11559, ggml-vulkan.cpp, device memory query): for an integrated GPU the
  reported free memory is the sum of the budgets of ALL memory heaps (VGM plus shared), not the
  device-local heap alone. This is the "advertised size" question of INV-1 H1 on a third vendor.
- P-E5a-1: every loaded trial's KV buffer = n_ctx x k / 2^20 within 1%.
- P-E5a-2: Vulkan0 compute buffer flat across loaded trials with n_ctx <= 131,072 (max/min <= 1.05).
- P-E5a-3 (primary): L-cell largest hold within +-10% of L_pred, per KV type. If L_pred > 4,194,304
  the predicted status is above_range. A boundary whose failure cause is max_buffer (a single-allocation
  limit, not the budget) is UNRESOLVED, not HIT or MISS.
- P-E5a-4: q4_0 / f16 L-cell ratio >= 3.0 (3.56 if the non-KV terms are equal).
- P-E5a-5: if L_pred >= 132,096 the P cell holds at 131,072 (above_range); otherwise the P-cell largest
  hold is within 1,024 of the L cell's (no allocation beyond load during an FA-on prefill).
- Descriptive, not scored: the same rule with F = the device-local heap only (vulkaninfo, or dxdiag
  Dedicated Memory).

## 6. Optional T2S evening (CONDITIONAL on Rithwik's OK for the evening and the folder install)
- E2-style llama.cpp boundary on the 64 GB T2S at its default share: the E5a L and P cells (f16, q4_0,
  FA on, ubatch 512), SYCL and Vulkan builds (the XRT-1 pins), installed under
  C:\Users\<user>\seam_tools\llamacpp\ only. Plus a read-only A19 record (DXGI budget) before and after.
- Same no-system-change rules as run_queue_t2s_cc_v3.sh: no reboot, no slider or setting change, nothing
  of his touched; guard before every cell (his processes, banned processes, watchdog launch-check); stop
  and hand back clean at the agreed time.
- Predictions: P-E5a-1..5 with the same substitution rule (device SYCL0 / Vulkan0). From A19 (shared
  48,634 MiB) the P cells are expected to hold at 131,072 (window binds, as OpenVINO's 40,448 3/3).
- No queue script yet: it is written (new file) once Rithwik picks the evening.
