# Figures generated from committed/pulled result JSONL (no new analysis beyond docs/CLAIMS_LEDGER.md)

Regenerate with `py -3.12 analysis/make_paper_figures.py`.

- **fig_power_effect_b1_b3.png** -- TTFT ratio vs `none`, across `none/p4/e4/nonp12` co-runner core sets, 6 models (qwen3-4b-2507, qwen3-8b from B1; llama31-8b, qwen3-14b, qwen3-30b-a3b-2507, qwen3-32b from B3). hw=evo-t2s (Intel Arc B390, Vulkan). Source: `night2_final.jsonl`.
- **fig_b2_duty_cycle.png** -- TTFT and decode ratio vs the nonp12 co-runner's duty cycle (0/25/50/75/100%), qwen3-4b-2507 and qwen3-8b. hw=evo-t2s (Intel Arc B390, Vulkan). Source: `night2_final.jsonl`.
- **fig_b4_ecore_threshold.png** -- TTFT ratio vs active E-core count across the 9 B4 conditions, qwen3-8b, n=1 model, 5 calls/condition. hw=evo-t2s (Intel Arc B390, Vulkan). Verified real L2-cache cluster layout (P0-3 individual, E4-7/E8-11/LP-E12-15 each shared). Source: `night3_results.jsonl`.
- **fig_c1_memory_lock.png** -- Start outcome (ok vs ggml_vulkan device-lost crash) and responsiveness (resp_max_s) by memory headroom (0/-1/-2 GB), qwen3-8b, mmap on/off x 3 reps = 6 cells/headroom, n=18 cells total. hw=evo-t2s (Intel Arc B390, Vulkan). Source: `night3_results.jsonl`.
- **fig_a24_budget_boundary.png** -- Last-ok/first-fail llama.cpp-projected device memory at the n_ctx bisection boundary (256-token step), 4 models (qwen3-32b, qwen3-8b, qwen3-30b-a3b-2507, llama31-8b), n=1 bisection each (32B repeated 3-4x, stable). Budget line = vulkaninfo heap budget (47865 MiB). hw=evo-t2s (Intel Arc B390, Vulkan). Source: `a24_amech_results.jsonl`.
- **fig_blade_c1_spill.png** -- TTFT ratio vs context size, 5 context points (8192, 16384, 24576, 32768, 40960). hw=blade_rtx4070 (discrete RTX 4070 Laptop GPU, OFF-TARGET). Onset of the driver's silent Sysmem Fallback spillover (A-20). Source: `blade_m1_vram_spill_20260925T041415Z.jsonl`.
