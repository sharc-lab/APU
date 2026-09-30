"""Prints the failure-mode map (docs/FAILURE_MAP.md) from a hardcoded evidence table.

This is deliberately NOT an automated scan of results/*.jsonl -- the underlying claims (which runtime/fit-mode
config, which files, which outcome category) require reading the actual harness scripts and CLAIMS_LEDGER.md
write-ups to attribute correctly (e.g. distinguishing "-ngl 99 pinned" from "default fit" needs the actual llama-server
command line, not just a JSONL row), which is exactly the kind of judgment call that should not be silently
automated over existing, hand-verified claims. This script is the single source that both docs/FAILURE_MAP.md's
table and any downstream code (e.g. a future paper-figure script) should read from, so the two never drift apart --
update EVIDENCE below and re-run this script to check docs/FAILURE_MAP.md still matches, rather than editing the
markdown table by hand.

Usage: py -3.12 analysis/make_failure_map.py
"""

ROWS = ["T2S (Intel, Vulkan)", "X2 (AMD, Vulkan)", "Blade (NVIDIA, CUDA)"]
COLS = ["llama.cpp -ngl 99", "llama.cpp default fit", "llama.cpp -fit off", "Ollama default", "Ollama num_ctx fixed"]

# outcome, one-line evidence summary, evidence file(s), n
EVIDENCE = {
    ("T2S (Intel, Vulkan)", "llama.cpp -ngl 99"): (
        "HARD_FAIL", "vkAllocateMemory ErrorOutOfDeviceMemory at the 47,866 MiB heap budget, holds within "
        "0.6-0.8% across 5 models 8B-70B (claim A-24)",
        ["results/t2s_amech_20260926T181456Z.jsonl", "results/t2s_night2_20260929T034014Z.jsonl"], "n>=3/model x 5 models"),
    ("T2S (Intel, Vulkan)", "llama.cpp default fit"): (
        "SILENT_SPILL, then CRASH one step higher", "32B: 3/65 layers to CPU at 17% decode cost, no error; "
        "crashes outright at the next context step (claim A-25)",
        ["results/t2s_overnight_20260926T011744Z.jsonl", "results/t2s_amech_20260926T181456Z.jsonl"],
        "n=1 model, 1 step each side (UNVERIFIED at full KV fill)"),
    ("T2S (Intel, Vulkan)", "llama.cpp -fit off"): ("NOT_MEASURED", None, [], None),
    ("T2S (Intel, Vulkan)", "Ollama default"): (
        "NOT_MEASURED", "K1 v2's 40,960 is a model-native-context cap (qwen3:8b), not a memory-shortfall result "
        "(claim A-28); K1 v3 queued to get the real signal",
        ["results/t2s_k1_ollama_evo-x2_20260930T022035Z.jsonl (x2 side, same caveat)"], None),
    ("T2S (Intel, Vulkan)", "Ollama num_ctx fixed"): ("NOT_MEASURED", None, [], None),
    ("X2 (AMD, Vulkan)", "llama.cpp -ngl 99"): ("NOT_MEASURED", "amech bisection has only targeted evo-t2s", [], None),
    ("X2 (AMD, Vulkan)", "llama.cpp default fit"): ("NOT_MEASURED", None, [], None),
    ("X2 (AMD, Vulkan)", "llama.cpp -fit off"): ("NOT_MEASURED", None, [], None),
    ("X2 (AMD, Vulkan)", "Ollama default"): (
        "NOT_MEASURED", "same caveat as T2S: K1 v2 measured a model-native cap, not shortfall behavior; K1 v3 queued", [], None),
    ("X2 (AMD, Vulkan)", "Ollama num_ctx fixed"): ("NOT_MEASURED", None, [], None),
    ("Blade (NVIDIA, CUDA)", "llama.cpp -ngl 99"): (
        "SILENT_SPILL", "driver-default Sysmem Fallback Policy: silent decode slowdown (0.08 + 74.6x excess "
        "fraction, R2=0.998), sharp onset ctx 36,864->38,912, 10/10 correctness probes unchanged at 14.2% spill "
        "(claim A-20); open question: does 'Prefer No Sysmem Fallback' turn this into HARD_FAIL",
        ["results/blade_m1_vram_spill_20260925T041415Z.jsonl", "results/blade_c1_spill_sweep_20260925T053651Z.jsonl",
         "results/blade_c2_spill_correctness_20260925T110739Z.jsonl"], "n=5 spill points, 10 correctness probes"),
    ("Blade (NVIDIA, CUDA)", "llama.cpp default fit"): ("NOT_MEASURED", "all Blade M1 runs pin -ngl 99", [], None),
    ("Blade (NVIDIA, CUDA)", "llama.cpp -fit off"): ("NOT_MEASURED", None, [], None),
    ("Blade (NVIDIA, CUDA)", "Ollama default"): (
        "NOT_MEASURED", "an Ollama run exists (stage_c_20260818T040408Z.jsonl) but tests position-pressure/quality "
        "at normal context, not memory exhaustion", ["results/stage_c_20260818T040408Z.jsonl"], None),
    ("Blade (NVIDIA, CUDA)", "Ollama num_ctx fixed"): ("NOT_MEASURED", None, [], None),
}

GAPS_HOURS = {
    "T2S, -fit off": ("amech-style bisection, -fit off added as a new fit-arm", "4-8h"),
    "T2S, Ollama default (real shortfall)": ("K1 v3 (queued: t2s_k1_tier_v3)", "~16-17h"),
    "T2S, Ollama num_ctx fixed": ("K1 v3 harness, +1 forced-num_ctx variant call", "+1-2h on top of K1 v3"),
    "X2, all 5 columns": ("port amech to evo-x2; K1 v3 (queued: x2_k1_tier_v3) covers the two Ollama columns",
                           "amech port 4-8h; K1 v3 ~16-17h"),
    "Blade, default fit / -fit off": ("re-run blade_m1_vram_spill.py-style sweep with -ngl unset, then -fit off", "2-4h each"),
    "Blade, Ollama shortfall": ("Ollama-driven version of the M1 ctx ladder, model native ceiling > VRAM", "2-4h"),
    "Blade, Prefer No Sysmem Fallback variant": ("re-run M1's ctx ladder once with that driver setting", "2-3h"),
}


def print_table():
    header = "| |" + "|".join(COLS) + "|"
    print(header)
    print("|" + "---|" * (len(COLS) + 1))
    for row in ROWS:
        cells = []
        for col in COLS:
            outcome, _detail, _files, n = EVIDENCE[(row, col)]
            cells.append(f"{outcome}" + (f" ({n})" if n else ""))
        print(f"| **{row}** | " + " | ".join(cells) + " |")


def print_gaps():
    print("\nMinimum runs to fill each NOT_MEASURED gap:")
    for gap, (run, hours) in GAPS_HOURS.items():
        print(f"  {gap}: {run} -- est. {hours}")


def not_measured_count():
    return sum(1 for v in EVIDENCE.values() if v[0] == "NOT_MEASURED")


if __name__ == "__main__":
    print_table()
    print_gaps()
    print(f"\n{not_measured_count()} / {len(EVIDENCE)} cells are NOT_MEASURED.")
