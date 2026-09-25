"""Backfill run manifests for the two committed fig61 sweep runs.

191031Z: cache_prompt absent (server default = cached), streaming usage unavailable.
203557Z: cache_prompt=false, /tokenize PC, stream_options honored.
"""
import json
from pathlib import Path

RESULTS = Path("C:/Users/rithw/OneDrive/Documents/GitHub/APU/results")
GGUF_SHA = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
GATE_DISAGREEMENTS = [
    {"probe_id": "sea_01", "arm": "LATE",
     "stage_c_mean": 0.0, "evo_t2s": 1.0, "output": "A9"}
]
GATE_FILE = "fig61_stagec_gate_20260922T190239Z.jsonl"
GATE_REF  = "stage_c_20260818T040408Z.jsonl"

def load_rows(fname):
    return [json.loads(l) for l in (RESULTS / fname).read_text(encoding="utf-8").splitlines() if l.strip()]

def filler_tokens_from_rows(rows):
    return rows[0]["filler_tokens"]

def pc_stats(rows, tokens_field):
    pass_n = sum(1 for r in rows if r.get("positive_control_ok", False))
    fail_n = len(rows) - pass_n
    return pass_n, fail_n, round(pass_n / len(rows) * 100, 1)

# ── 191031Z ────────────────────────────────────────────────────────────────────
rows_old = load_rows("fig61_stagec_full_20260922T191031Z.jsonl")
ft_old   = filler_tokens_from_rows(rows_old)
pc_pass_old, pc_fail_old, pc_rate_old = pc_stats(rows_old, "n_prompt_tokens_actual")

manifest_191031 = {
    "schema_version":    1,
    "run_tag":           "stagec_full_20260922T191031Z",
    "result_file":       "fig61_stagec_full_20260922T191031Z.jsonl",
    "validity_notes":    "SCORES VALID. TIMING INVALID (prefix KV cache active; "
                         "rep TTFT varied 70x on identical prompt). "
                         "PC UNVERIFIED (n_prompt_tokens_actual=0 for all rows; "
                         "streaming returned no usage; positive_control_ok=False for all). "
                         "Superseded for timing/PC by 20260922T203557Z.",
    "n_rows":            len(rows_old),
    "platform":          "evo-t2s",
    "hardware_config":   "evox2_evo-t2s",
    "memory_architecture": "unified",
    "server": {
        "binary":          "C:\\apu\\bin\\llama-b10970\\llama-server.exe",
        "build_id":        "b10970-bfdc32183",
        "ctx_size":        8192,
        "n_parallel":      1,
        "port":            8383,
        "reasoning_flags": "none",
    },
    "model": {
        "alias":       "qwen3:4b-instruct",
        "gguf_path":   "C:\\apu\\models\\qwen3-4b-instruct-85e4a5b7.gguf",
        "gguf_sha256": GGUF_SHA,
    },
    "generation": {
        "max_tokens":                   128,
        "temperature":                  0,
        "cache_prompt":                 "UNKNOWN (not set; server default applied KV cache)",
        "stream_options_include_usage": "not sent",
    },
    "filler": {
        "target_tokens": 4000,
        "actual_tokens": ft_old,
        "seed":          42,
        "variant":       "F-NUM",
        "count_fn":      "llamaserver_tokenize",
    },
    "truncation": {
        "method":                    "left_char",
        "chars_per_token_heuristic": 5.03,
    },
    "n_prompt_tokens_source": "UNKNOWN (streaming returned no usage; all rows = 0)",
    "sweep": {
        "n_probes":        11,
        "n_budget_ratios": 6,
        "budget_ratios":   [1.20, 1.00, 0.85, 0.70, 0.55, 0.40],
        "n_arms":          2,
        "n_reps":          3,
        "total_calls":     396,
    },
    "quality_gate": {
        "gate_file":         GATE_FILE,
        "reference_file":    GATE_REF,
        "n_cells_checked":   22,
        "n_cells_disagree":  1,
        "disagreements":     GATE_DISAGREEMENTS,
    },
    "positive_control": {
        "tolerance_pct": 5.0,
        "n_pass":        pc_pass_old,
        "n_fail":        pc_fail_old,
        "pass_rate_pct": pc_rate_old,
        "note":          "All fail because n_prompt_tokens_actual=0 (streaming usage unavailable). "
                         "positive_control_ok fields in rows are unreliable for this run.",
    },
}

# ── 203557Z ────────────────────────────────────────────────────────────────────
rows_new = load_rows("fig61_stagec_full_20260922T203557Z.jsonl")
ft_new   = filler_tokens_from_rows(rows_new)
pc_pass_new, pc_fail_new, pc_rate_new = pc_stats(rows_new, "n_prompt_tokens_actual")

pc_fail_details = [
    {"probe_id": r["probe_id"], "arm": r["arm"], "budget_ratio": r["budget_ratio"],
     "actual": r["n_prompt_tokens_actual"], "intended": r["intended_budget_tokens"],
     "err_pct": round(abs(r["n_prompt_tokens_actual"] - r["intended_budget_tokens"])
                      / max(r["intended_budget_tokens"], 1) * 100, 1)}
    for r in rows_new if not r["positive_control_ok"]
]

manifest_203557 = {
    "schema_version":    1,
    "run_tag":           "stagec_full_20260922T203557Z",
    "result_file":       "fig61_stagec_full_20260922T203557Z.jsonl",
    "validity_notes":    "FULLY VALID. Scores match 191031Z cell-for-cell (0 differences). "
                         "cache_prompt=false verified (7640ms vs 5846ms at startup, ratio=1.3x). "
                         "PC: 387/396 pass; 9 marginal failures at r=0.40 LATE sea probes (5.0-5.3%).",
    "n_rows":            len(rows_new),
    "platform":          "evo-t2s",
    "hardware_config":   "evox2_evo-t2s",
    "memory_architecture": "unified",
    "server": {
        "binary":          "C:\\apu\\bin\\llama-b10970\\llama-server.exe",
        "build_id":        "b10970-bfdc32183",
        "ctx_size":        8192,
        "n_parallel":      1,
        "port":            8383,
        "reasoning_flags": "none",
    },
    "model": {
        "alias":       "qwen3:4b-instruct",
        "gguf_path":   "C:\\apu\\models\\qwen3-4b-instruct-85e4a5b7.gguf",
        "gguf_sha256": GGUF_SHA,
    },
    "generation": {
        "max_tokens":                   128,
        "temperature":                  0,
        "cache_prompt":                 False,
        "stream_options_include_usage": True,
    },
    "filler": {
        "target_tokens": 4000,
        "actual_tokens": ft_new,
        "seed":          42,
        "variant":       "F-NUM",
        "count_fn":      "llamaserver_tokenize",
    },
    "truncation": {
        "method":                    "left_char",
        "chars_per_token_heuristic": 5.03,
    },
    "n_prompt_tokens_source": "llamaserver_tokenize",
    "sweep": {
        "n_probes":        11,
        "n_budget_ratios": 6,
        "budget_ratios":   [1.20, 1.00, 0.85, 0.70, 0.55, 0.40],
        "n_arms":          2,
        "n_reps":          3,
        "total_calls":     396,
    },
    "quality_gate": {
        "gate_file":         GATE_FILE,
        "reference_file":    GATE_REF,
        "n_cells_checked":   22,
        "n_cells_disagree":  1,
        "disagreements":     GATE_DISAGREEMENTS,
    },
    "positive_control": {
        "tolerance_pct": 5.0,
        "n_pass":        pc_pass_new,
        "n_fail":        pc_fail_new,
        "pass_rate_pct": pc_rate_new,
        "failures":      pc_fail_details,
        "note":          "9 failures at r=0.40 LATE sea probes: char-truncation rounding "
                         "over-delivers 5.0-5.3% tokens. Conservative (more context than intended). "
                         "See THREATS.md section 20.",
    },
}

for tag, manifest in [("191031Z", manifest_191031), ("203557Z", manifest_203557)]:
    out = RESULTS / f"fig61_stagec_manifest_20260922T{tag}.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out.name}")
