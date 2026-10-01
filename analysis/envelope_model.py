"""Predictive envelope model: feasibility, latency, effective context, quality regime, and failure-silence,
as a function of (machine, runtime_policy, model_id, context_length).

Every number in this file is either (a) pulled directly from docs/CLAIMS_LEDGER.md / docs/FINDINGS.md /
docs/PRIOR_ART.md with the exact citation given in the constant's comment, (b) computed from a real row range
in a results/*.jsonl file committed to this repo (extraction script embedded in the comment above the constant
so it can be re-run), or (c) imported directly from analysis/make_failure_map.py's EVIDENCE table (never forked).
Nothing here is guessed. Where no real data exists for a cell, the function says so explicitly (ASSUMED flag or
a NOT_MEASURED-shaped return) rather than interpolating a number that was never measured.

Hardware scope actually available in this repo as of this commit (2026-09-30):
  - evo-t2s (Intel Arc B390 iGPU, Vulkan, unified LPDDR5X): real per-model memory-budget-boundary data (A-24),
    real per-context TTFT data near that boundary (t2s_amech), real co-runner TTFT multiplier (A-23).
  - evo-x2 (AMD Strix Halo, Vulkan): real R1b truncation-cliff quality numbers exist in docs/FINDINGS.md's
    "R1b evaluation audit" section (quoted directly below), but as of this commit **no results/*.jsonl row
    anywhere in this repo carries hw_id "evo-x2" with a ttft_s value** (docs/FINDINGS.md, PX2 section: "no
    evo-x2 per-call timing to anchor on ... no R1, A70 or P70 phase has produced rows on any machine yet").
    So: X2 has real quality-cliff data but zero real latency data. This asymmetry is reported plainly wherever
    it matters (predict_latency, validate_ttft_cross_machine) rather than papered over.
  - Blade RTX4070 (off-target arm): real silent-spill data (A-20), used only in predict_failure_silence via
    make_failure_map's EVIDENCE table.
  - PX2 (the pre-registered evo-x2 co-runner experiment meant to supply the H2 multiplier) RAN on 2026-09-30 and
    completed (docs/FINDINGS.md, "PX2 real results" subsection, and docs/CLAIMS_LEDGER.md claim A-23's
    2026-09-30 update). **2026-10-01 refit, done in this change:** results/t2s_night2_20260930T135145Z.jsonl IS
    present in this repo as of this commit (the "absent" status earlier text here described is stale -- the file
    was pulled from evo-x2 and committed). The H2 co-runner term is now modeled as two separate, per-machine
    quantities rather than one shared TTFT scalar: H2_T2S_TTFT_MULTIPLIER (evo-t2s, the only metric ever measured
    there is TTFT, from a CPU co-runner, A-23) and H2_X2_DECODE_MULTIPLIER / H2_X2_TTFT_MULTIPLIER (evo-x2, real
    per-model ratios computed directly from the raw PX2 call rows across all 5 models -- see the extraction
    script in the comment above H2_X2_PER_MODEL). On evo-x2 both the bandwidth hog (B4) and the compute/power hog
    (S4) raise TTFT substantially vs baseline (~1.02x-1.12x across the 5 models) but move it almost identically
    (within 2-3% of each other), so TTFT cannot separate the two hog types there; the real, criterion-worthy
    separation is on **decode throughput**: B4 cuts decode to 0.905x-0.928x of baseline across all 5 models while
    S4 leaves it at 0.987x-1.000x. predict_latency's evo-x2 branch now applies the decode multiplier (keyed by
    co_runner_kind) as the modeled co-runner effect, and reports the TTFT multiplier for transparency without
    using it as the discriminating signal. evo-t2s's term remains TTFT-only because no decode-throughput
    isolation under a co-runner has ever been attempted there -- the two machines' co-runner terms are not
    assumed to be the same underlying quantity, and are never merged into one scalar.

Run as a script to print every prediction for a small demo grid:
    py -3.12 analysis/envelope_model.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_failure_map import EVIDENCE, ROWS, COLS  # noqa: E402 -- reused directly, not forked, per task instructions

# ---------------------------------------------------------------------------------------------------------------
# Section 1: constants sourced from committed claims/results. Every block below states its source file/claim.
# ---------------------------------------------------------------------------------------------------------------

# A-24 (docs/CLAIMS_LEDGER.md): "The budget is 47,866 MiB (vulkaninfo heap budget; llama-server free figure
# 47,865 MiB)". We use the llama-server free figure (47,865 MiB) because that is the number the runtime itself
# checks allocations against, which is what predict_feasibility needs.
T2S_VULKAN_BUDGET_MIB = 47865.0

# A-24 bisect_result rows, results/t2s_amech_20260926T181456Z.jsonl, phase=="bisect". Extracted with:
#   python -c "
#   import json
#   rows=[json.loads(l) for l in open('results/t2s_amech_20260926T181456Z.jsonl') if l.strip()]
#   bis=[r for r in rows if r.get('phase')=='bisect']
#   from collections import defaultdict
#   bym=defaultdict(set)
#   for r in bis: bym[r['model_id']].add((r['n_ctx'], r['projected_mib']))
#   for k,v in bym.items(): print(k, sorted(v))"
# Each list is real (n_ctx, llama.cpp-projected MiB) points straddling the budget boundary for that model.
# These are the actual GPU-memory-projection points the amech bisection logged, not derived or reconstructed.
T2S_BUDGET_BISECT_POINTS = {
    "qwen3-32b": [(111104, 46494.0), (113152, 47008.0), (115200, 47522.0), (115712, 47650.0),
                  (115968, 47714.0), (116224, 47779.0), (117248, 48036.0)],
    "qwen3-8b": [(4096, 5123.0), (48384, 11394.0), (113920, 20674.0), (294144, 46194.0), (302336, 47354.0),
                 (304384, 47644.0), (304896, 47717.0), (305152, 47753.0), (305408, 47789.0), (306432, 47934.0)],
    "qwen3-30b-a3b-2507": [(4096, 17996.0), (308224, 46805.0), (316416, 47581.0), (318464, 47775.0),
                            (319488, 47872.0), (320000, 47920.0), (320256, 47945.0), (320512, 47969.0)],
    "llama31-8b": [(4096, 5019.0), (340992, 47460.0), (343040, 47718.0), (343552, 47783.0),
                    (343808, 47815.0), (344064, 47847.0), (345088, 47976.0)],
}
# A-24 "Update (2026-09-29, A70, llama-3.3-70b)" -- results/t2s_night2_20260929T034014Z.jsonl, NOT present in
# this worktree's results/ directory (confirmed by listing). The only numbers available are the ones quoted in
# prose in docs/CLAIMS_LEDGER.md: last-OK n_ctx=23,296 with llama.cpp-projected 47,482 MiB (and server-logged
# 48,100.63 MiB); first-fail n_ctx=23,552 with no projected-MiB figure quoted. We keep only the one real point
# usable for a linear fit (there is no second projected-MiB point to pair with it), and the held-out check below
# uses this single point only for its final held-out-vs-predicted comparison, never for fitting.
T2S_A70_LAST_OK_POINT = (23296, 47482.0)  # (n_ctx, projected_mib), llama-3.3-70b, A-24 update text

# A-23 (docs/CLAIMS_LEDGER.md): the H1 co-runner slowdown replicated across models on evo-t2s (CPU co-runner,
# TTFT-only). PX2, the evo-x2 experiment meant to supply an equivalent multiplier for X2, RAN and completed
# (2026-09-30; docs/FINDINGS.md "PX2 real results", docs/CLAIMS_LEDGER.md claim A-23's 2026-09-30 update) and its
# real result is NOT a TTFT multiplier at all -- the clean, vendor-specific, criterion-worthy separation on evo-x2
# is a DECODE-throughput effect (bandwidth hog only), not TTFT (the two hog types are statistically
# indistinguishable on TTFT there). The 2026-10-01 refit below (H2_T2S_TTFT_MULTIPLIER / H2_X2_DECODE_MULTIPLIER /
# H2_X2_TTFT_MULTIPLIER) replaces the single shared H2_CORUNNER_TTFT_MULTIPLIER scalar that used to sit here (an
# earlier pass substituted A-23's evo-t2s TTFT ratio for the evo-x2 cell, which this refit removes). An alias is
# kept at the bottom of this block, pointing at the evo-t2s-only term, so any external caller still importing the
# old name gets the real evo-t2s number rather than an ImportError, but every production code path in this file
# now uses the machine-specific names.

# --- 2026-10-01 refit: real PX2 evo-x2 per-machine co-runner terms (results/t2s_night2_20260930T135145Z.jsonl) ---
# docs/FINDINGS.md's "PX2 real results" section and docs/CLAIMS_LEDGER.md claim A-23's 2026-09-30 update state the
# finding in prose/worked-example form (qwen3-8b only). The numbers below are computed directly from the raw,
# committed call rows for all 5 PX2 models, not transcribed from prose, with the exact extraction used:
#   python -c "
#   import json, statistics
#   from collections import defaultdict
#   rows=[json.loads(l) for l in open('results/t2s_night2_20260930T135145Z.jsonl') if l.strip()]
#   calls=[r for r in rows if r.get('record') is None and r.get('item_id','').startswith('PX2') and not r.get('warmup')]
#   by=defaultdict(list)
#   for r in calls: by[(r['model_id'], r['item_id'].split('_')[-1])].append(r)
#   def med(cond_rows, field):
#       vals=[x[field] for x in cond_rows if x.get(field) is not None]
#       return statistics.median(vals) if vals else None
#   for m in sorted(set(k[0] for k in by)):
#       n0 = by[(m,'N0')]
#       for cond in ('B4','S4'):
#           c = by[(m,cond)]
#           print(m, cond, 'ttft_ratio=', med(c,'ttft_s')/med(n0,'ttft_s'), 'decode_ratio=', med(c,'decode_tok_s')/med(n0,'decode_tok_s'))"
# This is the real, vendor-specific split the module docstring and A-23's 2026-09-30 update call for: on evo-x2
# the bandwidth hog (B4) and the compute/power hog (S4) are statistically indistinguishable on TTFT (both ~1.05x-
# 1.12x of N0) but cleanly separable on DECODE throughput (B4 cuts it to 0.91x-0.93x of N0; S4 leaves it at
# 0.99x-1.00x of N0) -- the opposite of evo-t2s, where the only real co-runner effect ever measured (A-23, CPU
# co-runner) is a TTFT multiplier and decode throughput under a co-runner has never been isolated on evo-t2s at
# all. The two machines' co-runner terms are therefore NOT the same quantity and must not be merged into one
# shared scalar; each is kept as its own per-machine dict below, used by predict_latency's per-machine branches.
H2_X2_PER_MODEL = {
    # model_id: {"ttft_ratio_b4": ..., "ttft_ratio_s4": ..., "decode_ratio_b4": ..., "decode_ratio_s4": ..., "n": ...}
    "qwen3-8b":        {"ttft_ratio_b4": 1.1241, "ttft_ratio_s4": 1.0980, "decode_ratio_b4": 0.9270, "decode_ratio_s4": 0.9943, "n": 5},
    "qwen3-14b":       {"ttft_ratio_b4": 1.0980, "ttft_ratio_s4": 1.0774, "decode_ratio_b4": 0.9093, "decode_ratio_s4": 0.9869, "n": 5},
    "qwen3-32b":       {"ttft_ratio_b4": 1.0776, "ttft_ratio_s4": 1.0508, "decode_ratio_b4": 0.9269, "decode_ratio_s4": 0.9997, "n": 5},
    "llama31-8b":      {"ttft_ratio_b4": 1.1116, "ttft_ratio_s4": 1.0877, "decode_ratio_b4": 0.9282, "decode_ratio_s4": 0.9934, "n": 5},
    "llama33-70b":     {"ttft_ratio_b4": 1.0451, "ttft_ratio_s4": 1.0244, "decode_ratio_b4": 0.9053, "decode_ratio_s4": 0.9999, "n": 3},
}
H2_X2_DECODE_MULTIPLIER = {
    # the real, criterion-worthy, vendor-specific separation on evo-x2: decode throughput, bandwidth-hog-specific.
    "bandwidth_hog_b4": {
        "value_range": (0.9053, 0.9282),
        "point_estimate": sum(v["decode_ratio_b4"] for v in H2_X2_PER_MODEL.values()) / len(H2_X2_PER_MODEL),
        "source": "PX2 (evo-x2, B4 bandwidth hog), results/t2s_night2_20260930T135145Z.jsonl, 5 models, "
                  "n=5 calls/condition (n=3 for llama-3.3-70b)",
    },
    "power_hog_s4": {
        "value_range": (0.9869, 0.9999),
        "point_estimate": sum(v["decode_ratio_s4"] for v in H2_X2_PER_MODEL.values()) / len(H2_X2_PER_MODEL),
        "source": "PX2 (evo-x2, S4 compute/power hog), results/t2s_night2_20260930T135145Z.jsonl, same rows",
    },
}
H2_X2_TTFT_MULTIPLIER = {
    # Real (not borrowed) evo-x2 TTFT ratios. Reported for completeness/transparency only -- predict_latency's
    # evo-x2 branch does NOT use this as its primary co-runner effect, because B4 and S4 move TTFT almost
    # identically here (within 2-3% of each other across all 5 models per A-23's 2026-09-30 update), so TTFT is
    # not the axis that separates the two hog types on this machine. The decode multiplier above is.
    "bandwidth_hog_b4": {"value_range": (1.0451, 1.1241),
                         "point_estimate": sum(v["ttft_ratio_b4"] for v in H2_X2_PER_MODEL.values()) / len(H2_X2_PER_MODEL),
                         "source": "PX2 (evo-x2, B4), same extraction as H2_X2_DECODE_MULTIPLIER"},
    "power_hog_s4": {"value_range": (1.0244, 1.0980),
                     "point_estimate": sum(v["ttft_ratio_s4"] for v in H2_X2_PER_MODEL.values()) / len(H2_X2_PER_MODEL),
                     "source": "PX2 (evo-x2, S4), same extraction as H2_X2_DECODE_MULTIPLIER"},
}
# evo-t2s's real co-runner term is a TTFT multiplier (CPU co-runner only; no decode-throughput isolation exists
# for evo-t2s in this repo). Renamed from H2_CORUNNER_TTFT_MULTIPLIER (kept below as an alias for back-compat)
# to make the per-machine split explicit at the name level, per the 2026-10-01 refit.
H2_T2S_TTFT_MULTIPLIER = {
    "value_range": (1.39, 1.43),
    "point_estimate": 1.42,
    "source": "A-23 (evo-t2s, CPU co-runner, nonp12/all16 condition) -- the only co-runner metric ever measured "
              "on evo-t2s is TTFT; decode throughput under a co-runner has not been isolated on evo-t2s",
}
H2_CORUNNER_TTFT_MULTIPLIER = H2_T2S_TTFT_MULTIPLIER  # back-compat alias; evo-t2s-only, see comment above

# --- K1 effective-context: real measured values where they exist, ASSUMED fallback elsewhere ---
# results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl, record=="tier" rows, field ollama_default_ctx (source
# "api_ps" -- read back from Ollama's own /api/ps after load, not a static config value). Extracted with:
#   python -c "
#   import json
#   rows=[json.loads(l) for l in open('results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl') if l.strip()]
#   for r in rows:
#       if r.get('record')=='tier': print(r['model_tag'], r['ollama_default_ctx'], r['ollama_default_ctx_source'])"
# This is a REAL measured effective-context ceiling for evo-x2 under Ollama's own default auto-fit policy, for
# the 3 model tags this K1 run actually loaded -- not the OLLAMA_VRAM_CONTEXT_TIERS documented-default fallback.
K1_X2_EFFECTIVE_CTX = {
    "qwen3-4b-2507": 262144,  # model_tag qwen3-4b-2507, capped=False (native 262144, not capped by VRAM)
    "llama31-8b": 131072,     # model_tag llama3.1:8b, capped=False (native 131072, not capped by VRAM)
    "qwen3-8b": 40960,        # model_tag qwen3:8b, capped=True -- matches MODEL_NATIVE_CTX's A-28 figure exactly
}
K1_X2_EFFECTIVE_CTX_SOURCE = "MEASURED (results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl, record=='tier', " \
    "ollama_default_ctx via api_ps -- real post-load context, not a config default)"
# No t2s_k1_ollama_*.jsonl or t2s_r2_*.jsonl file is committed anywhere in this repo for evo-t2s (confirmed by
# listing results/) -- so there is no evo-t2s equivalent of K1_X2_EFFECTIVE_CTX. This asymmetry is intentional
# and reported, not papered over with an empty dict pretending to be data:
K1_T2S_EFFECTIVE_CTX = {}  # NOT_MEASURED: no K1/R2 result JSONL exists for evo-t2s in this repo as of this commit
K1_T2S_EFFECTIVE_CTX_SOURCE = "NOT_MEASURED (no t2s_k1_ollama_*.jsonl or t2s_r2_*.jsonl file is committed for " \
    "evo-t2s; predict_effective_context falls back to the OLLAMA_VRAM_CONTEXT_TIERS ASSUMED default for this " \
    "machine)"

# docs/PRIOR_ART.md lines ~304-305: "Ollama shipped exactly this behavior in v0.15.5: tiers of 4,096 tokens
# below 24 GiB VRAM, 32,768 tokens from 24-48 GiB, 262,144 tokens at 48 GiB+"
OLLAMA_VRAM_CONTEXT_TIERS = (
    (24.0, 4096),   # < 24 GiB -> 4096
    (48.0, 32768),  # 24-48 GiB -> 32768
    (float("inf"), 262144),  # >= 48 GiB -> 262144
)

# claim A-28 (cited verbatim in analysis/make_failure_map.py's EVIDENCE table): "K1 v2's 40,960 is a
# model-native-context cap (qwen3:8b)". This is a real measured native-context ceiling on Ollama for qwen3:8b.
# Other entries are public model-card context lengths, NOT measured in this repo -- flagged accordingly in
# MODEL_NATIVE_CTX_SOURCE.
MODEL_NATIVE_CTX = {
    "qwen3-8b": 40960,
    "qwen3-32b": 40960,
    "qwen3-30b-a3b-2507": 262144,
    "llama31-8b": 131072,
    "llama33-70b": 131072,
}
MODEL_NATIVE_CTX_SOURCE = {
    "qwen3-8b": "MEASURED (claim A-28, K1 v2, Ollama default context cap)",
    "qwen3-32b": "ASSUMED (public Qwen3 dense model-card context length, not independently measured in this repo)",
    "qwen3-30b-a3b-2507": "ASSUMED (public Qwen3-30B-A3B-2507 model-card context length, not independently measured)",
    "llama31-8b": "ASSUMED (public Llama-3.1-8B model-card context length, not independently measured)",
    "llama33-70b": "ASSUMED (public Llama-3.3-70B model-card context length, not independently measured)",
}

# T2S TTFT regression source data: results/t2s_amech_20260926T181456Z.jsonl, rows with ttft_s not None and
# model_id=="qwen3-32b" (phase in {"arms","fill","fillmatch"}). Extracted with:
#   python -c "
#   import json
#   rows=[json.loads(l) for l in open('results/t2s_amech_20260926T181456Z.jsonl') if l.strip()]
#   pts=[(r['prompt_tokens'], r['ttft_s']) for r in rows
#        if r.get('ttft_s') is not None and r.get('model_id')=='qwen3-32b' and r.get('prompt_tokens')]
#   print(pts)"
# n=39 real (prompt_tokens, ttft_s) pairs. IMPORTANT CAVEAT (stated here and repeated in predict_latency's
# docstring): every one of these calls ran at n_ctx between 111,104 and 117,248 -- i.e. near the A-24 memory
# budget boundary, where the KV cache allocation itself is already large. This fit is NOT a clean "TTFT vs
# prompt length at a fixed small context" relation; it is the real TTFT-vs-prompt-length relation *at
# near-budget n_ctx specifically*, which is the only real per-call TTFT series this repo has for more than one
# prompt length on T2S. This is stated as a limitation, not smoothed over.
T2S_TTFT_FIT_POINTS = [
    (514, 2.600526800029911), (514, 2.642418000032194), (514, 2.648057699901983), (514, 2.781623200047761),
    (514, 2.872182699968107), (514, 2.872429799987003), (514, 2.875089199980721), (514, 2.9053367000306025),
    (514, 2.906083900015801), (514, 2.906380500062369), (514, 2.911578700062819), (514, 2.928240899927914),
    (514, 2.963508400018327), (514, 2.981565000023693), (514, 2.986843300051987),
    (2000, 9.32279639999615), (2000, 9.782105299993418),
    (12098, 188.13261680002324), (12098, 188.62706710002385), (12098, 188.66619119996903), (12098, 188.7270532999537),
    (12101, 211.0422306000255), (12101, 211.54313320002984), (12102, 211.52209069998935), (12102, 211.74191179999616),
    (12102, 211.7851812999579), (12801, 228.42300959996646), (12801, 228.8854775999789),
    (12802, 209.96516599995084), (12802, 210.07183910009917), (12802, 210.23560830007773), (12802, 210.28150430007372),
    (12802, 212.2203387999907), (12802, 212.23153849999653), (12802, 212.27270289999433), (12802, 212.37671109999064),
    (12802, 228.84521549998317), (12802, 228.88740700000199), (12802, 229.01966330001596),
]
T2S_DECODE_FIT_POINTS = [
    # same rows, decode_tok_s field -- used for a companion decode regression (weaker relation, reported honestly)
    (514, 4.708593770167235), (514, 5.065640733458869), (514, 5.062194971101545), (514, 5.153678858782956),
    (514, 5.110), (514, 5.201965484232663), (514, 5.058645966717677), (514, 5.048679325971566),
    (514, 5.109560866724809), (514, 5.02), (514, 5.11), (514, 5.06), (514, 5.17), (514, 4.97), (514, 5.08),
    (2000, 5.068388463939333),
    (12098, 3.356575451094482), (12098, 3.3507336049965066), (12098, 3.352380025629329), (12098, 3.349905094025295),
    (12101, 4.177130378270901), (12101, 3.4657068641211275), (12102, 4.0107061790432645), (12102, 4.457313699761265),
    (12801, 3.401810879173178), (12802, 3.2603151344150993), (12802, 3.630161999527045), (12802, 2.8147145248796246),
    (12802, 3.2443536144678404), (12802, 3.2511729252792567), (12802, 3.250692014105972), (12802, 3.250599834213779),
    (12802, 2.716882249347446), (12802, 2.712708504063808), (12802, 2.7135406266971414), (12802, 2.711441397313109),
]

# R1b truncation-cliff quality data, docs/FINDINGS.md "R1b evaluation audit, both machines, from the live
# Oct-1-cut runs (2026-09-30)". Underlying JSONL (t2s_night2_20260929T202603Z.jsonl / ...T205109Z.jsonl) is not
# present in this worktree's results/ directory (confirmed by listing); numbers below are transcribed verbatim
# from the prose write-up, which is itself the citable source per that section's own header. Baseline arm:
# "every probe scores 0.0 at ratios 0.4 and 0.85 ... and 1.0 at ratio 1.2 ... Ratio 0.98 is the only ratio where
# probes split: art_08/art_09/art_10 score 1.0 ... while art_01-art_07 score 0.0" -- i.e. 3/10 probes pass at
# ratio 0.98, same split on both machines. Self-report arm (corrected scorer): "t2s 0.950 and x2 0.900 at ratio
# 1.2 ... Ratio 0.98 goes from 0.000 to 0.222 (t2s) / 0.194 (x2)"; ratios 0.4/0.85 are 0.000 on both arms, both
# machines ("real truncation failures, not artifacts").
R1B_BASELINE_SCORE_BY_RATIO = {0.4: 0.0, 0.85: 0.0, 0.98: 0.3, 1.2: 1.0}  # 3/10 probes at 0.98, both machines
R1B_SELFREPORT_SCORE_BY_RATIO = {
    "evo-t2s": {0.4: 0.0, 0.85: 0.0, 0.98: 0.222, 1.2: 0.950},
    "evo-x2": {0.4: 0.0, 0.85: 0.0, 0.98: 0.194, 1.2: 0.900},
}
# results/partial_truncation.json, analysis["extinction_ratios"] -- the fine-grained ratio at which each probe's
# answer content is fully removed by truncation (mean of the 5 probes, qwen3:4b-instruct, Blade off-target):
R1B_EXTINCTION_RATIOS = {"rag_01": 0.975, "rag_02": 0.975, "rag_05": 0.979, "sea_01": 0.972, "sea_04": 0.971}
R1B_MEAN_EXTINCTION_RATIO = sum(R1B_EXTINCTION_RATIOS.values()) / len(R1B_EXTINCTION_RATIOS)

# Map (machine, runtime_policy) inputs onto make_failure_map's exact row/col strings.
MACHINE_TO_ROW = {
    "evo-t2s": "T2S (Intel, Vulkan)",
    "evo-x2": "X2 (AMD, Vulkan)",
    "blade_rtx4070": "Blade (NVIDIA, CUDA)",
}
RUNTIME_POLICY_TO_COL = {
    "llama_ngl99": "llama.cpp -ngl 99",
    "llama_default_fit": "llama.cpp default fit",
    "llama_fit_off": "llama.cpp -fit off",
    "ollama_default": "Ollama default",
    "ollama_num_ctx_fixed": "Ollama num_ctx fixed",
}


def _linreg(points):
    """Ordinary least squares y = a + b*x on a list of (x, y) real pairs. Returns (intercept, slope, r2)."""
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] ** 2 for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    ybar = sy / n
    ss_tot = sum((p[1] - ybar) ** 2 for p in points)
    ss_res = sum((p[1] - (a + b * p[0])) ** 2 for p in points)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a, b, r2


def fit_budget_boundary(model_points: dict) -> dict:
    """Fits, per model, a linear projected_mib = intercept + slope * n_ctx line on the real A-24 bisect points
    in T2S_BUDGET_BISECT_POINTS (see that constant's comment for the exact extraction command). slope is real
    llama.cpp-projected MiB-per-context-token (close to, but not identical to, the architectural KV bytes/token,
    since projected_mib also includes the model's own weight+compute buffers, which are context-independent and
    fold into the intercept). Returns {model_id: {"intercept_mib": a, "slope_mib_per_ctx": b, "r2": r2}}."""
    return {mid: dict(zip(("intercept_mib", "slope_mib_per_ctx", "r2"), _linreg(pts)))
            for mid, pts in model_points.items()}


T2S_BUDGET_FITS = fit_budget_boundary(T2S_BUDGET_BISECT_POINTS)
T2S_TTFT_FIT = dict(zip(("intercept_s", "slope_s_per_token", "r2"), _linreg(T2S_TTFT_FIT_POINTS)))
T2S_DECODE_FIT = dict(zip(("intercept_toks", "slope_toks_per_token", "r2"), _linreg(T2S_DECODE_FIT_POINTS)))


# ---------------------------------------------------------------------------------------------------------------
# Section 2: the five predict_* functions plus the top-level predict().
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Prediction:
    feasibility: dict
    latency: dict
    effective_context: dict
    quality_regime: dict
    failure_silence: dict


def predict_feasibility(machine: str, model_id: str, context_length: int, runtime_policy: str = "llama_ngl99") -> dict:
    """FITS / SILENT_SPILL / HARD_FAIL / NOT_MEASURED with a margin in MiB.

    Source: claim A-24 (docs/CLAIMS_LEDGER.md) for the 47,865 MiB evo-t2s Vulkan budget, fit per-model from the
    real bisect points in T2S_BUDGET_BISECT_POINTS (see that constant's docstring for the exact extraction). The
    outcome label at a negative margin is read from analysis/make_failure_map.EVIDENCE for the (machine,
    runtime_policy) cell (claims A-24/A-25: HARD_FAIL under -ngl 99, SILENT_SPILL-then-CRASH under default fit),
    not invented here. System-memory headroom margin (a distinct axis from the Vulkan device-heap margin computed
    here) is measured separately by claim A-22/c1b: at or above zero headroom responsiveness stays near-instant
    (median 0.016-0.031s); below zero headroom the crash boundary is bimodal (near-instant despite crash, or a
    two-to-three-order-of-magnitude freeze, on the 2 clean below-zero cells). This function returns the device-
    heap margin only; a caller wanting the headroom axis too should read A-22/c1b's numbers directly
    (docs/CLAIMS_LEDGER.md claim A-22, "2026-09-29 night3/c1b" update) rather than have the two silently merged,
    since they were measured on different axes (Vulkan device allocation vs. Windows-level system memory lock).
    For evo-x2, no A-24-equivalent memory-boundary bisection exists anywhere in this repo
    (make_failure_map's own EVIDENCE table: "amech bisection has only targeted evo-t2s") -- returns NOT_MEASURED
    rather than reusing T2S's fit, since the two machines' memory architectures are not assumed equivalent.
    """
    if machine != "evo-t2s":
        return {"status": "NOT_MEASURED", "margin_mib": None, "measured": False,
                "reason": f"no memory-budget-boundary bisection (A-24-equivalent) exists for machine={machine!r} "
                          f"in this repo; make_failure_map's EVIDENCE table confirms the amech bisection has only "
                          f"ever targeted evo-t2s"}
    if model_id not in T2S_BUDGET_FITS:
        return {"status": "NOT_MEASURED", "margin_mib": None, "measured": False,
                "reason": f"model_id={model_id!r} was not one of the 4 models bisected in "
                          f"results/t2s_amech_20260926T181456Z.jsonl (qwen3-32b, qwen3-8b, qwen3-30b-a3b-2507, "
                          f"llama31-8b); llama33-70b has only the single A-24-update point T2S_A70_LAST_OK_POINT, "
                          f"insufficient to fit its own slope -- see validate_budget_boundary_loo for how that "
                          f"point is used instead (held-out prediction, not its own fit)"}
    fit = T2S_BUDGET_FITS[model_id]
    projected_mib = fit["intercept_mib"] + fit["slope_mib_per_ctx"] * context_length
    margin_mib = T2S_VULKAN_BUDGET_MIB - projected_mib
    row = MACHINE_TO_ROW[machine]
    col = RUNTIME_POLICY_TO_COL.get(runtime_policy, "llama.cpp -ngl 99")
    if margin_mib >= 0:
        return {"status": "FITS", "margin_mib": round(margin_mib, 1), "measured": True,
                "projected_mib": round(projected_mib, 1), "budget_mib": T2S_VULKAN_BUDGET_MIB,
                "source": "claim A-24 budget + per-model linear fit on real bisect points"}
    outcome, detail, files, n = EVIDENCE.get((row, col), ("NOT_MEASURED", None, [], None))
    return {"status": outcome, "margin_mib": round(margin_mib, 1), "measured": outcome != "NOT_MEASURED",
            "projected_mib": round(projected_mib, 1), "budget_mib": T2S_VULKAN_BUDGET_MIB,
            "source": f"claim A-24 budget + per-model linear fit; outcome label from make_failure_map EVIDENCE "
                      f"[{row!r}, {col!r}]: {detail}"}


def predict_latency(machine: str, runtime_policy: str, prompt_tokens: int, co_runner: bool = False,
                    co_runner_kind: str = "bandwidth") -> dict:
    """Predicted TTFT and decode tok/s, with a PER-MACHINE co-runner term (2026-10-01 refit).

    Functional form: simple linear-in-prompt-length OLS, ttft_s = intercept + slope * prompt_tokens, fit
    separately per machine. Chosen because it is the simplest form the real data supports (R2=0.993 on T2S's own
    39-point series, see T2S_TTFT_FIT_POINTS) -- no evidence in this repo motivates a log-linear or quadratic
    form, and a 2-parameter linear fit is the most defensible choice given n=39 real points at 4 distinct prompt
    lengths. CAVEAT (repeated from T2S_TTFT_FIT_POINTS's own docstring): this fit's real data all ran near the
    A-24 memory-budget boundary (n_ctx 111,104-117,248), so it reflects near-budget TTFT, not TTFT at a small
    fixed context. It is the only real multi-length TTFT series this repo has for evo-t2s.

    evo-x2: NO real TTFT-vs-prompt-length series exists in this repo for evo-x2 (docs/FINDINGS.md PX2 section:
    "no evo-x2 per-call timing to anchor on ... no results/*.jsonl row carries hw_id 'evo-x2' [with a ttft]" --
    still true for a *multi-length* series; PX2 itself did measure real evo-x2 TTFT and decode values, but all at
    one fixed prompt length, so it cannot supply its own length-vs-latency slope). This returns measured=False
    for the base (no-co_runner) prediction on evo-x2 and labels it a cross-machine extrapolation of T2S's fit.

    H2 CO-RUNNER TERM, now per-machine (2026-10-01 refit -- this used to be one shared TTFT scalar borrowed from
    evo-t2s for both machines; it no longer is):
      - machine=="evo-t2s": applies H2_T2S_TTFT_MULTIPLIER (the real, measured evo-t2s effect -- a TTFT penalty
        from a CPU co-runner; A-23). co_runner_kind is accepted but has no effect here, since evo-t2s's A-23 data
        does not distinguish hog types.
      - machine=="evo-x2": applies H2_X2_DECODE_MULTIPLIER[co_runner_kind] to decode_tok_s, NOT a TTFT multiplier,
        because PX2's real, criterion-worthy separation on evo-x2 is a decode-throughput effect that is specific
        to the bandwidth hog (co_runner_kind="bandwidth" -> B4, ~0.91-0.93x decode) and nearly absent for the
        power/compute hog (co_runner_kind="power" -> S4, ~0.99-1.00x decode). TTFT on evo-x2 does also rise under
        either hog (H2_X2_TTFT_MULTIPLIER, real PX2 data) but the two hog types are statistically indistinguishable
        on that axis there, so TTFT is reported but not used as the discriminating co-runner signal for evo-x2.
      - any other machine: co_runner has no modeled effect (returns the base prediction unchanged; no H2 data
        exists for machine=="blade_rtx4070" at all).
    """
    if machine == "evo-t2s":
        ttft = T2S_TTFT_FIT["intercept_s"] + T2S_TTFT_FIT["slope_s_per_token"] * prompt_tokens
        decode = T2S_DECODE_FIT["intercept_toks"] + T2S_DECODE_FIT["slope_toks_per_token"] * prompt_tokens
        measured = True
    else:
        ttft = T2S_TTFT_FIT["intercept_s"] + T2S_TTFT_FIT["slope_s_per_token"] * prompt_tokens
        decode = T2S_DECODE_FIT["intercept_toks"] + T2S_DECODE_FIT["slope_toks_per_token"] * prompt_tokens
        measured = False
    ttft = max(ttft, 0.01)
    decode = max(decode, 0.01)

    ttft_mult, ttft_mult_source = 1.0, None
    decode_mult, decode_mult_source = 1.0, None
    if co_runner:
        if machine == "evo-t2s":
            ttft_mult = H2_T2S_TTFT_MULTIPLIER["point_estimate"]
            ttft_mult_source = H2_T2S_TTFT_MULTIPLIER["source"]
        elif machine == "evo-x2":
            key = "bandwidth_hog_b4" if co_runner_kind == "bandwidth" else "power_hog_s4"
            decode_mult = H2_X2_DECODE_MULTIPLIER[key]["point_estimate"]
            decode_mult_source = H2_X2_DECODE_MULTIPLIER[key]["source"]
            ttft_mult = H2_X2_TTFT_MULTIPLIER[key]["point_estimate"]
            ttft_mult_source = (H2_X2_TTFT_MULTIPLIER[key]["source"] +
                                " (reported, not used as the discriminating signal -- see predict_latency docstring)")
        ttft *= ttft_mult
        decode *= decode_mult

    return {"ttft_s": round(ttft, 3), "decode_tok_s": round(decode, 3), "measured": measured and not co_runner,
            "fit_form": "linear: ttft_s = a + b * prompt_tokens (OLS)",
            "fit_coefficients": {"t2s": T2S_TTFT_FIT},
            "co_runner_kind": co_runner_kind if co_runner else None,
            "ttft_co_runner_multiplier": ttft_mult if co_runner else None,
            "ttft_co_runner_multiplier_source": ttft_mult_source,
            "decode_co_runner_multiplier": decode_mult if co_runner else None,
            "decode_co_runner_multiplier_source": decode_mult_source,
            "note": None if machine == "evo-t2s" else
                    "evo-x2 has no real TTFT-vs-prompt-length series in this repo; the base TTFT prediction is "
                    "the T2S fit applied to evo-x2 as a cross-machine extrapolation, not a measurement. The "
                    "co-runner multiplier itself (if co_runner=True) IS real evo-x2 PX2 data -- see "
                    "H2_X2_DECODE_MULTIPLIER/H2_X2_TTFT_MULTIPLIER -- only the base, no-co-runner TTFT curve is "
                    "extrapolated; see validate_ttft_cross_machine"}


def predict_effective_context(machine: str, runtime_policy: str, model_id: str) -> dict:
    """Effective context length actually usable, with an explicit MEASURED/ASSUMED flag.

    2026-10-01: evo-x2 now has REAL measured effective-context values for 3 model tags (K1_X2_EFFECTIVE_CTX,
    results/t2s_k1_ollama_evo-x2_20260930T205515Z.jsonl, see that constant's docstring) -- for an (evo-x2, ollama
    runtime_policy, model_id) cell that matches one of those 3, this returns the real measured value with
    flag=="MEASURED" instead of falling back to the documented default. evo-t2s has no K1/R2 equivalent anywhere
    in this repo (confirmed by listing results/ -- harness/t2s_k1_ollama.py and harness/t2s_r2_session_growth.py
    exist as code, but no t2s_k1_ollama_*.jsonl or t2s_r2_*.jsonl file is committed for evo-t2s), so every evo-t2s
    return, and every evo-x2 return for a model_id not in K1_X2_EFFECTIVE_CTX, remains ASSUMED: it falls back to
    the documented Ollama VRAM tiers (docs/PRIOR_ART.md, OLLAMA_VRAM_CONTEXT_TIERS) capped by the model's own
    native context (MODEL_NATIVE_CTX, MODEL_NATIVE_CTX_SOURCE has per-model measured/assumed provenance for the
    native-context figure itself). This MEASURED-for-3-cells/ASSUMED-elsewhere split is reported explicitly in
    the return rather than blended into one confidence level.

    The "vram_gib" used to pick a tier (ASSUMED path only) is a caller-supplied machine memory figure; if not
    given, machine-specific defaults are used: evo-t2s and evo-x2 are both unified-memory parts reported elsewhere
    in this repo at 64 GiB (evo-t2s, docs/PAPER_OUTLINE.md Fig 6.1 spec) and 128 GiB (evo-x2, docs/FINDINGS.md
    PX2 section) respectively.
    """
    if machine == "evo-x2" and runtime_policy.startswith("ollama") and model_id in K1_X2_EFFECTIVE_CTX:
        return {"effective_context": K1_X2_EFFECTIVE_CTX[model_id], "measured": True,
                "flag": "MEASURED",
                "basis": f"real post-load ollama_default_ctx (api_ps) for model_tag={model_id!r}",
                "native_ctx_source": K1_X2_EFFECTIVE_CTX_SOURCE,
                "note": "real K1 measurement, not the OLLAMA_VRAM_CONTEXT_TIERS ASSUMED fallback"}
    machine_vram_gib = {"evo-t2s": 64.0, "evo-x2": 128.0, "blade_rtx4070": 8.0}.get(machine, 64.0)
    if runtime_policy.startswith("ollama"):
        tier_ctx = next(ctx for cap, ctx in OLLAMA_VRAM_CONTEXT_TIERS if machine_vram_gib < cap)
    else:
        tier_ctx = None  # non-Ollama runtimes don't apply this tiering at all
    native = MODEL_NATIVE_CTX.get(model_id)
    if tier_ctx is None:
        value = native
        basis = "model native context (non-Ollama runtime_policy; no VRAM tiering applies)"
    else:
        value = min(tier_ctx, native) if native else tier_ctx
        basis = f"min(Ollama VRAM tier {tier_ctx} @ {machine_vram_gib} GiB, model native {native})"
    return {"effective_context": value, "measured": False,
            "flag": "ASSUMED",
            "basis": basis,
            "native_ctx_source": MODEL_NATIVE_CTX_SOURCE.get(model_id, "unknown model_id"),
            "note": "no K1/R2 result JSONL present locally for this (machine, runtime_policy, model) cell; "
                    "this is the documented-default fallback, not a measurement"}


def predict_quality_regime(prompt_tokens: int, full_prompt_tokens: int, effective_context: int) -> dict:
    """"full" if the prompt fits in effective_context; otherwise the R1b truncation-cliff regime.

    Cliff data: R1B_BASELINE_SCORE_BY_RATIO and R1B_SELFREPORT_SCORE_BY_RATIO, transcribed verbatim from
    docs/FINDINGS.md's "R1b evaluation audit, both machines, from the live Oct-1-cut runs (2026-09-30)" section
    (the underlying JSONL is not present in this worktree; the prose write-up is the section's own citable
    source). ratio = prompt_tokens / full_prompt_tokens (how much of the intended prompt survives); the nearest
    measured ratio bucket's score is used (no interpolation invented between real buckets -- R1B_MEAN_EXTINCTION_RATIO
    from the finer-grained real partial_truncation.json sweep marks where the cliff edge actually sits, ~0.974).
    """
    if full_prompt_tokens <= 0:
        ratio = 1.0
    else:
        ratio = min(prompt_tokens / full_prompt_tokens, 1.2)
    if prompt_tokens <= effective_context:
        return {"regime": "full", "ratio": round(ratio, 3), "predicted_score": 1.0,
                "source": "prompt_tokens <= effective_context"}
    buckets = sorted(R1B_BASELINE_SCORE_BY_RATIO)
    nearest = min(buckets, key=lambda b: abs(b - ratio))
    score = R1B_BASELINE_SCORE_BY_RATIO[nearest]
    return {"regime": "truncation_cliff", "ratio": round(ratio, 3), "nearest_measured_ratio": nearest,
            "predicted_score": score, "extinction_ratio_mean": round(R1B_MEAN_EXTINCTION_RATIO, 4),
            "source": "R1b evaluation audit (docs/FINDINGS.md, 2026-09-30) baseline-arm score at nearest "
                      "measured ratio bucket; cliff edge cross-checked against partial_truncation.json's "
                      "extinction ratios (mean 0.974, 5 probes, qwen3:4b-instruct off-target)"}


def predict_failure_silence(machine: str, runtime_policy: str) -> dict:
    """HARD_FAIL / SILENT_SPILL / SILENT_TRUNCATION / CRASH / HANG / NOT_MEASURED for (machine, runtime_policy).

    Reuses analysis/make_failure_map.EVIDENCE directly (imported, not copied) -- the exact same table that
    generates docs/FAILURE_MAP.md.
    """
    row = MACHINE_TO_ROW.get(machine)
    col = RUNTIME_POLICY_TO_COL.get(runtime_policy)
    if row is None or col is None or (row, col) not in EVIDENCE:
        return {"outcome": "NOT_MEASURED", "detail": f"unrecognized (machine={machine!r}, "
                f"runtime_policy={runtime_policy!r}); known machines: {list(MACHINE_TO_ROW)}, "
                f"known runtime policies: {list(RUNTIME_POLICY_TO_COL)}", "files": [], "n": None}
    outcome, detail, files, n = EVIDENCE[(row, col)]
    return {"outcome": outcome, "detail": detail, "files": files, "n": n, "row": row, "col": col}


def predict(machine: str, runtime_policy: str, model_id: str, context_length: int,
            prompt_tokens: Optional[int] = None, full_prompt_tokens: Optional[int] = None,
            co_runner: bool = False, co_runner_kind: str = "bandwidth") -> Prediction:
    """Top-level combiner. prompt_tokens/full_prompt_tokens default to context_length (i.e. "prompt fills the
    requested context exactly") when not given -- callers doing the trace join pass their own real values.
    co_runner_kind ("bandwidth" or "power") only matters for machine=="evo-x2" (see predict_latency)."""
    prompt_tokens = context_length if prompt_tokens is None else prompt_tokens
    full_prompt_tokens = context_length if full_prompt_tokens is None else full_prompt_tokens
    feas = predict_feasibility(machine, model_id, context_length, runtime_policy)
    lat = predict_latency(machine, runtime_policy, prompt_tokens, co_runner=co_runner, co_runner_kind=co_runner_kind)
    eff = predict_effective_context(machine, runtime_policy, model_id)
    qual = predict_quality_regime(prompt_tokens, full_prompt_tokens,
                                   eff["effective_context"] if eff["effective_context"] else context_length)
    fail = predict_failure_silence(machine, runtime_policy)
    return Prediction(feasibility=feas, latency=lat, effective_context=eff, quality_regime=qual,
                       failure_silence=fail)


if __name__ == "__main__":
    demo = [
        ("evo-t2s", "llama_ngl99", "qwen3-32b", 100000),
        ("evo-t2s", "llama_ngl99", "qwen3-32b", 120000),
        ("evo-t2s", "llama_default_fit", "qwen3-32b", 120000),
        ("evo-x2", "ollama_default", "qwen3-8b", 8192),
    ]
    for machine, policy, model_id, ctx in demo:
        p = predict(machine, policy, model_id, ctx)
        print(f"\n== {machine} / {policy} / {model_id} @ ctx={ctx} ==")
        print(" feasibility:", p.feasibility)
        print(" latency:", p.latency)
        print(" effective_context:", p.effective_context)
        print(" quality_regime:", p.quality_regime)
        print(" failure_silence:", p.failure_silence)
