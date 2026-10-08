"""Writes src/dse/envelope_data.json, the router's envelope inputs, from numbers-register rows only.

Every number in that file is the `detail` of a register compute function in analysis/numbers_register.py (the same
function that writes the row in docs/NUMBERS_REGISTER.md), and each block names its claim_id and carries the row's
value string verbatim, so a reader can match any router input to its register row. Nothing in the JSON is typed by
hand; tests/test_dse_router.py rebuilds it in memory and fails if the committed file differs.

Register rows used (claim_id -> what the router takes from it):
  T2S-vs-X2-default-ctx      Ollama default context per model on evo-t2s and evo-x2 (K1 v3)
  K1v3-X2-table              Ollama default context per model on evo-x2 (K1 v3)
  R2-real-v1-loaded-ctx      context Ollama loaded in the R2 real run, per tier x model (evo-x2)
  R2-real-v1-token-calib     per-model calibration ratio of the chars/4 token estimate (R2 harness)
  ttft-physical-fit-per-machine  TTFT fit ttft_s = a*n + b*n^2 per machine x model
  decode-rate-per-machine    median decode tok/s per machine x model, no co-runner
  B3-corunner-6model         evo-t2s CPU co-runner (nonp12) TTFT ratio per model
  PX2-full-ratio-table       evo-x2 co-runner TTFT and decode ratio per model (condition B4)
  A-24-budget-boundary       evo-t2s llama-server (Vulkan) last n_ctx that started, per model
  ollama-overflow-keeps-half Ollama overflow keeps num_ctx/2 + 2 tokens (cited in reasons)
  R2-mechanism-lowlevel      what overflow does per tier (cited in reasons)

Model names: register rows use two spellings (Ollama tags "llama3.1:8b", llama-server ids "llama31-8b"); both are
stored under one key, norm_model(): lowercase, "." removed, ":" -> "-".

Usage: py -3.12 analysis/build_envelope_data.py   (writes src/dse/envelope_data.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import numbers_register as nr  # noqa: E402

OUT_PATH = REPO / "src" / "dse" / "envelope_data.json"

# The evo-x2 co-runner condition the router applies: PX2's B4 (the bandwidth co-runner, the only PX2 condition with a
# decode penalty, register PX2-decode-ratio). evo-t2s: B3's nonp12 CPU co-runner (TTFT only; no decode ratio row).
X2_CORUNNER_CONDITION = "B4"


def norm_model(name: str) -> str:
    return name.strip().lower().replace(".", "").replace(":", "-")


def _entry(claim_id):
    for e in nr.NUMBER_ENTRIES:
        if e["claim_id"] == claim_id:
            return e
    raise KeyError(claim_id)


def _row(claim_id, sources):
    """Computes one register row and records its citation in `sources`. Returns the compute dict."""
    e = _entry(claim_id)
    res = e["compute"](REPO)
    sources[claim_id] = {"register_value": res["value"], "script_function": e["script_function"],
                         "data_files": e["data_files"]}
    return res


def _merge_ctx(dst: dict, model: str, ctx: int, claim_id: str):
    key = norm_model(model)
    if key in dst and dst[key]["ctx"] != ctx:
        raise ValueError(f"default ctx conflict for {key}: {dst[key]} vs {ctx} ({claim_id})")
    dst.setdefault(key, {"ctx": int(ctx), "claim_ids": []})["claim_ids"].append(claim_id)


def build() -> dict:
    sources: dict = {}
    machines = {"evo-t2s": {}, "evo-x2": {}}

    # ── effective context, Ollama default, per machine x model
    t2s_ctx, x2_ctx = {}, {}
    for m, d in _row("T2S-vs-X2-default-ctx", sources)["detail"].items():
        _merge_ctx(t2s_ctx, m, d["t2s_default_ctx"], "T2S-vs-X2-default-ctx")
        _merge_ctx(x2_ctx, m, d["x2_default_ctx"], "T2S-vs-X2-default-ctx")
    for m, d in _row("K1v3-X2-table", sources)["detail"].items():
        _merge_ctx(x2_ctx, m, d["ollama_default_ctx"], "K1v3-X2-table")
    loaded = _row("R2-real-v1-loaded-ctx", sources)["detail"]
    for m, ctx in loaded.get("ollama_default", {}).items():
        _merge_ctx(x2_ctx, m, ctx, "R2-real-v1-loaded-ctx")
    machines["evo-t2s"]["ollama_default_ctx"] = dict(sorted(t2s_ctx.items()))
    machines["evo-x2"]["ollama_default_ctx"] = dict(sorted(x2_ctx.items()))

    # ── latency: TTFT fit and decode rate
    fit = _row("ttft-physical-fit-per-machine", sources)["detail"]
    dec = _row("decode-rate-per-machine", sources)["detail"]
    for hw in machines:
        machines[hw]["ttft_fit"] = {norm_model(m): {"a": v["a"], "b": v["b"], "r2": v["r2"]}
                                    for m, v in sorted(fit.get(hw, {}).items())}
        machines[hw]["decode_tok_s"] = {norm_model(m): v["median"] for m, v in sorted(dec.get(hw, {}).items())}

    # ── co-runner term per machine
    b3 = _row("B3-corunner-6model", sources)["detail"]
    machines["evo-t2s"]["co_runner"] = {
        "condition": "nonp12 (CPU co-runner)", "claim_id": "B3-corunner-6model",
        "ttft_ratio": {norm_model(m): r for m, r in sorted(b3.items())}, "decode_ratio": {}}
    px2 = _row("PX2-full-ratio-table", sources)["detail"]
    x2_ttft, x2_dec = {}, {}
    for key, v in sorted(px2.items()):
        m, cond = key.rsplit("@", 1)
        if cond == X2_CORUNNER_CONDITION:
            x2_ttft[norm_model(m)] = v["ttft_ratio"]
            x2_dec[norm_model(m)] = v["decode_ratio"]
    machines["evo-x2"]["co_runner"] = {"condition": f"{X2_CORUNNER_CONDITION} (PX2 bandwidth co-runner)",
                                       "claim_id": "PX2-full-ratio-table", "ttft_ratio": x2_ttft,
                                       "decode_ratio": x2_dec}

    # ── memory budget: evo-t2s llama-server measured boundary; evo-x2 has none measured
    a24 = _row("A-24-budget-boundary", sources)["detail"]
    machines["evo-t2s"]["llama_server_memory"] = {
        "claim_id": "A-24-budget-boundary",
        "last_ok_n_ctx": {norm_model(m): v["last_ok_n_ctx"] for m, v in sorted(a24.items())},
        "first_fail_n_ctx": {norm_model(m): v["first_fail_n_ctx"] for m, v in sorted(a24.items())}}
    machines["evo-x2"]["llama_server_memory"] = None

    # ── token calibration (R2 harness chars/4 estimate)
    calib = _row("R2-real-v1-token-calib", sources)["detail"]
    token_calib = {norm_model(m): v["median"] for m, v in sorted(calib.items())}

    # ── overflow mechanism citations (used verbatim in docs, not as router inputs)
    keep = _row("ollama-overflow-keeps-half", sources)["detail"]
    _row("R2-mechanism-lowlevel", sources)

    return {
        "generated_by": "analysis/build_envelope_data.py",
        "note": "derived from numbers-register rows only; regenerate, never edit by hand",
        "machines": machines,
        "token_calib_ratio": token_calib,
        "ollama_overflow_processed": {str(v["num_ctx"]): v["processed_after_overflow"]
                                      for _, v in sorted(keep.items())},
        "sources": dict(sorted(sources.items())),
    }


def dumps(data: dict) -> str:
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def main():
    data = build()
    OUT_PATH.write_text(dumps(data), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO)}: {len(data['sources'])} register rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
