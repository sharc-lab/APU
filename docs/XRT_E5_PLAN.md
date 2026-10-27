# XRT E5 on evo-x2: GPU-share sweep, PLAN ONLY (2026-10-26)

**Status: PLAN. Nothing is queued, no setting has been changed, no reboot, no SSH.** This answers item E5 of Zach's
"Cross-vendor run specs (E3, E4, E5), Oct 26 2026". The spec is unpublished and not in the repo. The registration is
Zach's (his `docs/registrations/BR_XRT_2026-10-26.md`). Every hour figure below is my own ESTIMATE unless it is
pasted from a register row. Before any of it runs, it needs the operator's approval and Zach's registration.

What the spec asks for E5, briefly:

- **Share points.** 3 to 6 GPU-share points. On Windows the share is the VGM setting in AMD Software, with one reboot
  per point. Record the OS-reported GPU memory at each point.
- **At each point.** llama.cpp b11559 Vulkan (and ROCm if present), KV f16 and q4_0, `-fa on`, ubatch 512. Run the
  same boundary search as E3: the largest `-c` that loads and prefills N - 64 tokens. Use `--fit off` and record the
  loaded n_ctx.
- **Context range.** The range must be large enough that memory binds, not the 40,960 window. If needed, use a larger
  model; Zach registers it.
- **Prediction (registered).** The ceiling is linear in the share, with slope 1/k (R^2 >= 0.98).
- **Time.** The spec says "About 3-4 h". The first three points are due by Sun Nov 8 (Gate B), the rest by Nov 15.

## What evo-x2 has now (from repo docs, not re-checked live)

- **llama.cpp.** Only `C:\apu\bin\llama-b10970\llama-server.exe`: build 10970, commit bfdc32183, Vulkan
  (docs/HARDWARE.md). **b11559 is not on evo-x2.** There is no llama.cpp ROCm build. ROCm is present only inside
  Ollama (gfx1151; register row x2-device-detect-mechanism).
- **b11559 release assets** (GitHub release b11559, published 2026-10-11, digests from the release page):
  - `llama-b11559-bin-win-vulkan-x64.zip`: 32,084,119 B, sha256
    bf2fb2a7fc5d7669ccabbb6de3f6dd6b54058f213b32e731abc62b590f8566dc.
  - `llama-b11559-bin-win-rocm-10.0-x64.zip`: 257,958,782 B, sha256
    5511af01bbcaa17ebdb21dd7ab8fc8841903f25a3a08c8441ee5a43ec5afb9ab.
  - Not verified: whether the ROCm zip runs on gfx1151 without a separately installed HIP SDK.
- **Deploying b11559.** A new build means unzipping one zip into one new folder, for example
  `C:\apu\bin\llama-b11559-vulkan`, after checking the sha256 against the release digest. It replaces nothing.
- **Current share.** BIOS/VGM dedicated reservation of 64 GB. dxdiag: "Dedicated 65,360 MB, Shared 32,587 MB".
  Vulkan total grew from 98,123 to 114,326 MiB after a reboot (docs/X2_CHANGELOG.md 2026-09-29, not fully
  explained). Register rows fh1-amd-boundary-*: "dedicated 65536 MiB, Vulkan device-local heap 76218 MiB, Vulkan
  total 114327 MiB".

## Design caveat to settle with Zach first: on Windows the boundary is a silent spill, not a failure

On this machine llama-server Vulkan does not fail when it passes the dedicated share. It spills silently into Windows
shared memory. Pasted from register row fh1-amd-boundary-llama-3.3-70b: "MEASURED spill boundary (GPU shared usage
> +1024 MiB, unclamped): last no-spill n_ctx 65536 (spill_stats), first spill n_ctx 100096 (spill_stats); cost across
it: decode ratio 0.983, TTFT ratio 1.004". So "loads and prefills" keeps holding past the share. A hard failure comes
only near the Vulkan total, which is dedicated plus shared. That matters for the prediction:

- **Arithmetic** (mine; Qwen3-4B: 36 layers, 8 KV heads x 128). f16 KV costs 144 KiB per token, q4_0 40.5 KiB.
- **Qwen3-4B at the current share.** The hard-fail ceiling (Vulkan total) is above 500k tokens for f16. Even the
  spill onset (dedicated 64 GiB) is about 450k. Both are far past the 40,960 window, so with Qwen3-4B the window binds,
  not memory.
- **Llama-3.3-70B** (already on evo-x2, 131,072 window, 320 KiB per token f16). The spill onset is measured below the
  window (65,536 to 100,096 above). The hard-fail ceiling is about (114,327 - ~40,500 MiB weights) / 320 KiB, roughly
  236k tokens, above the window. So under the spec's "loads and prefills" criterion it is window-bound too.
- **Qwen3-32B** (with YaRN to 131,072, 256 KiB per token f16). The hard fail is about 378k by the same arithmetic, so
  it is also window-bound. The measured HARD_FAIL start at 367,360 (row fh1-amd-boundary-qwen3-32b) is consistent with
  that figure. But that row also records "CLAMPED req 290000 served 40960", so the loaded n_ctx must always be logged.
- **Lower shares on Windows** (option (a)) shrink the dedicated pool, but by my reading the shared pool grows as the
  dedicated share shrinks. So a hard-fail ceiling that depends on the share below the window is not expected at any
  share on Windows. What tracks the share is the spill onset.

So before any hours are spent, Zach needs to choose one of these:

- **(i) Spill onset** as the boundary on Windows. Measure GPU shared usage at each trial, as the fh1 harness did.
  `xrt_llamacpp.py` does not do this today; it would need a small addition or a wrapper.
- **(ii) Hard-fail boundary.** Accept that on Windows it sits near the Vulkan total, not the share. The slope-1/k
  prediction then tests a different quantity.
- **(iii) Run E5 under Linux** with `ttm.pages_limit`, as the spec's own Linux route says. evo-x2 runs Windows; that
  is out of scope here.

Model choice for (i) is the operator's and Zach's:

- **Qwen3-4B** binds f16 below 40,960 only if the dedicated share is under about 8 GB (`(D - 2.4 GB) / 144 KiB <
  40,960`), and q4_0 only under about 4 GB. Whether VGM offers such low points has not been checked.
- **Llama-3.3-70B** binds f16 below its window at shares from about 48 GB to about 80 GB (my arithmetic: about 26k tokens at 48 GB, about 78k at 64 GB, consistent with the measured 65,536 to 100,096). q4_0 (90 KiB per token) does not bind below
  131,072 at 64 GB.
- Zach registers any model other than Qwen3-4B with its sha256. Recording the sha256 of the 70B GGUF already on
  evo-x2 is a read-only step.

## Option (a): full sweep, 3 to 6 share points

- **One scheduled block.** The operator gives explicit approval for each reboot, and someone must be able to reach
  the machine physically or at the console in case SSH or Tailscale does not come back.
- **Record the original share first** and restore it at the end. Record it in AMD Software (screenshot) plus dxdiag,
  `vulkaninfo --summary` heaps and budgets, and `llama-server --list-devices`. Restoring it costs one more reboot.
- **Points.** For example 3 or 6 of the VGM presets. The presets are not yet read off the machine, so read them first
  with no change. Pick the points so the chosen model binds below its window at every point (see the model choice
  above).
- **At each point:**
  1. Change VGM in AMD Software and reboot (approved). Run the post-reboot gate: dxdiag Dedicated/Shared, the
     vulkaninfo heaps and budgets, `llama-server --list-devices`, driver version, `rocm-smi` if present.
  2. Run the boundary cells: Vulkan f16 and q4_0, `-fa on`, ub 512, `--fit off`, logging the loaded n_ctx. Add ROCm
     only if the ROCm zip is shown to run.
- **Queue handling, with an automatic release.** Two ways:
  - **Recommended: no pause flag.** Put the block in the queue as entries behind the running job, never preempting
    it. Each point is a self-checking job (`x2_e5_point_<k>`). It reads the OS-reported GPU memory, writes a "share
    not as planned" row and advances if the share does not match, else runs the cells, then calls `tq.advance()`.
    Between points sits a gate job (`x2_e5_share_gate_<k>`). It waits for the operator's VGM change and reboot, up to
    a stated deadline. On timeout it writes "share <k> not set" and calls `tq.advance()`, which is the automatic
    release. The reboot kills the gate job, and the watchdog (SYSTEM task every 10 min) then launches the next entry.
    Still to check: that this expected kill does not count toward the watchdog's crash circuit.
  - **If a pause flag is used.** `t2s_queue.set_pause` has no expiry today. It would need a small change before the
    block: an optional `release_after_utc` in the pause record that `queue_watchdog.tick()` honors, with a test. The
    reason text must then name the release, for example: "E5 share sweep, attended by <operator>; auto-released by the
    watchdog at <UTC time> (release_after_utc) or earlier by the block's final restore step".
- **The evo-x2 ask-first list, step by step:**
  - VGM change, at every point and at the restore: **BIOS/UEFI changes.** VGM sets the UMA frame-buffer reservation
    that firmware applies at boot (docs/HARDWARE.md), so I treat it as BIOS-level. Ask first every time.
  - Reboot: not on the list, but this plan needs explicit operator approval for each one.
  - Deploying the b11559 Vulkan zip into a new folder: not on the list. These are user-mode binaries; no driver
    change.
  - The b11559 ROCm zip: not on the list if it is self-contained. If it needs a HIP SDK or runtime install, treat that
    as **GPU driver changes** and ask first.
  - Pulling a larger model: allowed after a free-disk check. Deleting model files is ask-first and not needed.
  - No step touches sshd, Tailscale, the port-22 rule or accounts.
- **Hours (my ESTIMATES).** Per point, about 0.25 h for the VGM change, reboot and gate, plus the cells.
  - Qwen3-4B at small shares: prefills under 40,960 tokens, about 9 trials per cell, about 2 min per trial, so about
    0.6 h for two cells and **about 0.85 h per point**. 3 points plus the restore reboot come to **about 3 h**; 6
    points to **about 5.5 h**. Add about 0.5 h for reading presets, deploying and recording the original state.
  - Llama-3.3-70B at shares of about 48 to 80 GB: hold trials prefill up to about 100k tokens. The measured evo-x2 prompt
    rate is 127 to 175 tok/s (row x2-scale-probe-hours, Ollama ROCm, pasted: "ctx_4096 probe pe 155.4", "ctx_32768
    probe pe 174.6", "default probe pe 127.2"). So a hold takes 10 to 25 min, about 2.5 h per cell, and **about 5 h
    per point**. 3 points come to **about 16 h**, 6 points to **about 31 h**.
  - Adding ROCm roughly doubles the cell time.
  - The spec's "About 3-4 h" matches only the Qwen3-4B, 3-point case.

## Option (b): boundary search at the current default share only

- **No reboot, no setting change.** A normal queued job, self-checking and safe to re-run, that calls `tq.advance()`
  on exit, so no pause flag is needed. It goes behind the running job.
- **Steps:**
  1. Deploy the b11559 Vulkan zip, sha256 checked, into one new folder.
  2. Record dxdiag, `vulkaninfo --summary`, `--list-devices` and the driver version.
  3. Run the boundary search: KV f16 and q4_0, `-fa on`, ub 512, `--fit off`, recording the loaded n_ctx per trial.
     That is `xrt_llamacpp.py boundary` behind a queue wrapper; the tool itself uses a bare `subprocess.Popen` and is
     meant for interactive use.
  4. Add ROCm only if present and self-contained.
- **What it can show.** By the caveat above, at the current 64 GB share neither Qwen3-4B nor the 70B binds memory
  below its window under "loads and prefills". Under (i), spill onset, the 70B f16 has a measured spill between 65,536
  and 100,096, and (b) would refine it at 1,024. The model is the operator's choice, registered by Zach.
- **Hours (my ESTIMATES).**
  - Deploy and record: about 0.3 h.
  - 70B f16: range 32,768 to 131,072 at 1,024, about 9 trials, with holds of 10 to 25 min at the measured 127 to
    175 tok/s. **About 2 to 3 h.**
  - 70B q4_0: window-bound by arithmetic, so the low and high trials only. **About 0.7 to 1 h.**
  - Qwen3-4B at the current share: window-bound, so low and high only, each a prefill of 40k or less. **About 0.3 h.**
  - Total **about 3.5 to 4.5 h** with the 70B, or **about 0.6 h** with Qwen3-4B alone, in which case it only confirms
    the window bound. Double the cell time if ROCm is added.
  - No register row has a measured llama-server Vulkan load time for these models on evo-x2. The 70B TTFT in
    fh1-amd-boundary-llama-3.3-70b ("TTFT 22810 ms") is a request time, not a load time, so load time is folded into
    the per-trial estimates above.

## Related (not E5)

- **E3** (RTX 4070 laptop): the kit is outside the repo, in `C:\xrt` on the controller laptop, and so is its RUN_E3.md
  (operator-run).
- **E4** (vLLM on that laptop): no upstream native Windows path. Only a community fork with cp312 wheels; see the
  operator report of 2026-10-26.
- **Optional evo-t2s item:** docs/T2S_WEEK_PLAN.md, staged section.
