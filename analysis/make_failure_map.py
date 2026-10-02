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
    ("T2S (Intel, Vulkan)", "llama.cpp -fit off"): (
        "HARD_FAIL", "llama-3.3-70b pushed to n_ctx=23552 (the A-24 budget boundary's own first_fail point, "
        "hi) with -fit off: server never starts, exit code 1, vkAllocateMemory ErrorOutOfDeviceMemory "
        "(233.8 MB alloc request) at 'failed to allocate compute pp buffers' during context init -- a hard, "
        "immediate failure with no partial start, same failure signature (ErrorOutOfDeviceMemory) as the "
        "-ngl 99 arm at the same boundary, confirming -fit off does not change the outcome class at this "
        "point, only removes the automatic fit attempt",
        ["results/t2s_night2_20260929T034014Z.jsonl"], "n=1 model, 1 point (the A-24 hi boundary, not "
        "independently re-bisected for this arm)"),
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
    "T2S, -fit off": ("minimal design: 1 model, 1 ctx just past A-24's already-known budget wall (47,866 MiB), "
                      "3 probes, no bisection needed (the boundary is already measured)", "0.3h"),
    "T2S, Ollama default (real shortfall)": ("minimal design: llama3.1:8b (already pulled), 1 prompt length past "
                                             "its own live-measured tier (4096, found during R2), 3 probes", "0.3h"),
    "T2S, Ollama num_ctx fixed": ("minimal design: 1 model (llama3.1:8b, already pulled), 1 num_ctx forced past "
                                  "the runtime's own chosen tier, 3 probe calls", "0.3h"),
    "X2, all 5 columns": ("minimal design per cell: 1 model, 1 length/context past the memory limit, 3 probes. "
                          "-ngl 99/default fit/-fit off need the boundary located first (X2 has no A-24-equivalent "
                          "measured boundary yet, unlike T2S); the two Ollama columns reuse llama3.1:8b, already "
                          "pulled, at the tier already measured live during R2 (131072)",
                          "~0.5-1h per llama.cpp column (3 columns), ~0.3h per Ollama column (2 columns), "
                          "~2.4-3.9h total for all 5"),
    "Blade, default fit / -fit off": ("minimal design: 1 model, 1 ctx just past A-20's already-known onset "
                                      "(36,864-38,912), 3 probes, -ngl unset then -fit off", "0.3h each"),
    "Blade, Ollama shortfall": ("minimal design: 1 model, Ollama num_ctx forced past VRAM, 3 probes", "0.3h"),
    "Blade, Prefer No Sysmem Fallback variant": ("re-run the same A-20 onset point once with that driver setting, "
                                                  "3 probes (boundary already known, no bisection needed)", "0.3h"),
}

# 2026-09-30 re-estimate: the original numbers above (K1 v3's own ~16-17h estimate, amech-port guesses of 4-8h)
# assumed the FULL original design (multiple models x a 5-length probe sweep, or a full bisection ladder) for every
# gap. The minimal design that actually answers "what happens when memory runs out under this runtime/policy" needs
# far less: ONE model already pushed past its memory limit ONCE (not bisected -- for T2S and Blade the boundary is
# already known from A-24/A-20, so "past the limit" is a single known value, not something to search for), 3 probe
# calls to confirm the outcome is stable, not a one-off. This drops the total from roughly 35-50h to roughly 3.9-6.4h:
# 0.3h x 6 single-known-boundary columns (T2S -fit off, T2S Ollama default, T2S Ollama fixed, all 3 Blade gaps) plus
# 2.4-3.9h for X2's 5 columns, whose boundary is not pre-located (no A-24-equivalent measurement exists on X2 yet).
TOTAL_HOURS_MINIMAL_DESIGN = "roughly 3.9-6.4h total (0.3h x 6 known-boundary columns + 2.4-3.9h for X2's 5 columns)"


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
    print("\nMinimum runs to fill each NOT_MEASURED gap (minimal design: 1 model pushed past its memory limit "
         "once per cell, 3 probes):")
    for gap, (run, hours) in GAPS_HOURS.items():
        print(f"  {gap}: {run} -- est. {hours}")
    print(f"\nTotal: {TOTAL_HOURS_MINIMAL_DESIGN}")


def not_measured_count():
    return sum(1 for v in EVIDENCE.values() if v[0] == "NOT_MEASURED")


if __name__ == "__main__":
    print_table()
    print_gaps()
    print(f"\n{not_measured_count()} / {len(EVIDENCE)} cells are NOT_MEASURED.")
