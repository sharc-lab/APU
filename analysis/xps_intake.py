"""Intake for Zach's XPS results (Dell XPS, 16 GB, Intel, Qwen3-4B): his raw files become register rows with run IDs.

Nothing Zach has reported is a number in this repo until his files are here. The register rows built on this module
(analysis/numbers_register.py, ids xps-zach-*) are PENDING while their file is missing and compute only from the file;
their descriptions say what he reported, never as a measured value.

FILE FORMAT (one JSONL file per kind, in results/xps_zach/, the file name fixed per kind, KINDS below). Every line is
one JSON object with a "record" field. Every file:
  first line  {"record": "run_start", "kind": <kind>, "run_id": <string, unique per run>, "ts_utc": <ISO time>,
               "host": {"hostname", "model" (e.g. "Dell XPS ..."), "ram_gb", "cpu", "gpu"},
               "runtime_config": {"runtime" (ollama | llamacpp | openvino | ...), "version", "default_window",
                                  "model" (e.g. Qwen3-4B and its build: quantization / weights precision),
                                  "device", "kv_precision", "overrides" (list; empty when at defaults)},
               "tool": {"name", "commit" or "version"}}
  last line   {"record": "run_end", "run_id": <same>, "ts_utc": <ISO time>}
  between     records of the kind's type, each carrying the same "run_id" and the fields in REQUIRED[kind].
Several runs may share one file (each with its own run_start / run_end and run_id); every record's run_id must belong
to a run_start in the file.

Kinds and their records:
  ceiling        "ceiling_config": config_id (str), config (dict: the settings of this configuration), split ("fit" or
                 "heldout"), predicted_ceiling_tokens (int), measured_ceiling_tokens (int), measured_method (str).
                 Rows: xps-zach-ceiling-prediction (mean absolute percent error on held-out configs, and held-out within
                 +/-5%), xps-zach-ceiling-within-5pct (configs within +/-5% over all configs).
  tuned_window   "window_task": config ("default" or "tuned"), window_tokens (int), seed (int), task_id (str),
                 ok (bool).
                 Row: xps-zach-tuned-window (window per config and correct / n).
  gpu_share      "gpu_share_trial": gpu_share_pct (number), trial (int), ok (bool). Row: xps-zach-gpu-share (ok / n per
                 share).
  loud_failure   "overflow_call": prompt_tokens (int), window_tokens (int), outcome ("error" or "ok"), http_status
                 (int or null), error_text (str or null), latency_s (number: wall time from request to the response or
                 the error). Row: xps-zach-intel-loud-failure-latency (calls over the window: how many failed loudly and
                 how fast; calls within it for comparison).

Percent error convention: 100 x |predicted - measured| / measured. "Within +/-5%" means that error <= 5.0.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

DIR = "results/xps_zach"
KINDS = {"ceiling": "ceiling.jsonl", "tuned_window": "tuned_window.jsonl", "gpu_share": "gpu_share.jsonl",
         "loud_failure": "loud_failure.jsonl"}
RECORD = {"ceiling": "ceiling_config", "tuned_window": "window_task", "gpu_share": "gpu_share_trial",
          "loud_failure": "overflow_call"}
REQUIRED = {
    "ceiling": {"config_id": str, "config": dict, "split": str, "predicted_ceiling_tokens": int,
                "measured_ceiling_tokens": int, "measured_method": str},
    "tuned_window": {"config": str, "window_tokens": int, "seed": int, "task_id": str, "ok": bool},
    "gpu_share": {"gpu_share_pct": (int, float), "trial": int, "ok": bool},
    "loud_failure": {"prompt_tokens": int, "window_tokens": int, "outcome": str, "latency_s": (int, float)},
}
START_FIELDS = ("kind", "run_id", "ts_utc", "host", "runtime_config", "tool")
RUNTIME_FIELDS = ("runtime", "version", "default_window", "model")
TOL_PCT = 5.0


class IntakeError(ValueError):
    """The file does not follow the format; the message names the line and the problem."""


def path_of(repo, kind: str) -> Path:
    return Path(repo) / DIR / KINDS[kind]


def _type_ok(v, t) -> bool:
    if t is int or t == (int, float):
        if isinstance(v, bool):
            return False
    return isinstance(v, t)


def load(path, kind: str) -> dict:
    """Parse and validate one file. {"runs": {run_id: run_start}, "records": [...]}; IntakeError on any problem."""
    if kind not in KINDS:
        raise IntakeError(f"unknown kind {kind!r}; one of {sorted(KINDS)}")
    runs, ended, recs = {}, set(), []
    with open(path, encoding="utf-8") as fh:
        lines = [(i, ln) for i, ln in enumerate(fh, 1) if ln.strip()]
    if not lines:
        raise IntakeError(f"{Path(path).name}: empty")
    for i, ln in lines:
        try:
            r = json.loads(ln)
        except ValueError as e:
            raise IntakeError(f"line {i}: not JSON ({e})") from e
        rec = r.get("record")
        if rec == "run_start":
            miss = [f for f in START_FIELDS if f not in r]
            if miss:
                raise IntakeError(f"line {i}: run_start lacks {miss}")
            if r["kind"] != kind:
                raise IntakeError(f"line {i}: run_start kind {r['kind']!r}, this file is {kind!r}")
            rmiss = [f for f in RUNTIME_FIELDS if f not in (r["runtime_config"] or {})]
            if rmiss:
                raise IntakeError(f"line {i}: runtime_config lacks {rmiss}")
            if r["run_id"] in runs:
                raise IntakeError(f"line {i}: run_id {r['run_id']!r} started twice")
            runs[r["run_id"]] = r
        elif rec == "run_end":
            if r.get("run_id") not in runs:
                raise IntakeError(f"line {i}: run_end for unknown run_id {r.get('run_id')!r}")
            ended.add(r["run_id"])
        elif rec == RECORD[kind]:
            if r.get("run_id") not in runs or r["run_id"] in ended:
                raise IntakeError(f"line {i}: record outside an open run (run_id {r.get('run_id')!r})")
            for f, t in REQUIRED[kind].items():
                if f not in r or not _type_ok(r[f], t):
                    raise IntakeError(f"line {i}: {rec} field {f!r} missing or not {t}")
            recs.append(r)
        else:
            raise IntakeError(f"line {i}: unexpected record {rec!r} in a {kind} file")
    open_runs = sorted(set(runs) - ended)
    if open_runs:
        raise IntakeError(f"{Path(path).name}: no run_end for run(s) {open_runs}")
    if not recs:
        raise IntakeError(f"{Path(path).name}: no {RECORD[kind]} records")
    _kind_checks(kind, recs)
    return {"runs": runs, "records": recs}


def _kind_checks(kind, recs):
    if kind == "ceiling":
        for r in recs:
            if r["split"] not in ("fit", "heldout"):
                raise IntakeError(f"config {r['config_id']}: split must be fit or heldout")
            if r["measured_ceiling_tokens"] <= 0:
                raise IntakeError(f"config {r['config_id']}: measured ceiling must be positive")
    if kind == "tuned_window" and any(r["config"] not in ("default", "tuned") for r in recs):
        raise IntakeError("window_task config must be default or tuned")
    if kind == "loud_failure" and any(r["outcome"] not in ("error", "ok") for r in recs):
        raise IntakeError("overflow_call outcome must be error or ok")


def _ids(d) -> str:
    return ", ".join(sorted(d["runs"]))


def _runtime(d) -> str:
    out = []
    for rid in sorted(d["runs"]):
        rc = d["runs"][rid]["runtime_config"]
        out.append(f"{rc.get('runtime')} {rc.get('version')}, default window {rc.get('default_window')}, "
                   f"model {rc.get('model')}")
    return "; ".join(sorted(set(out)))


def pct_err(r) -> float:
    return 100.0 * abs(r["predicted_ceiling_tokens"] - r["measured_ceiling_tokens"]) / r["measured_ceiling_tokens"]


def _q(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] if s else None


# ── the row computations ───────────────────────────────────────────────────────────────────────────────────

def ceiling_prediction(d: dict) -> dict:
    held = [r for r in d["records"] if r["split"] == "heldout"]
    fit = [r for r in d["records"] if r["split"] == "fit"]
    if not held:
        raise IntakeError("no heldout configs")
    he = [pct_err(r) for r in held]
    within = sum(1 for e in he if e <= TOL_PCT)
    fit_txt = (f"; fit configs: mean |error| {statistics.fmean([pct_err(r) for r in fit]):.1f}% over {len(fit)}"
               if fit else "; no fit configs in the file")
    return {"value": f"run_id {_ids(d)} ({_runtime(d)}): held-out configs: mean |error| {statistics.fmean(he):.1f}% "
                     f"(max {max(he):.1f}%) over {len(held)}, within +/-{TOL_PCT:g}%: {within}/{len(held)}" + fit_txt
                     + "; error = 100 x |predicted - measured| / measured ceiling tokens",
            "n": f"{len(held)} held-out configs", "mean_heldout_pct": statistics.fmean(he), "heldout_within": within,
            "heldout_n": len(held)}


def ceiling_within(d: dict) -> dict:
    errs = [pct_err(r) for r in d["records"]]
    within = sum(1 for e in errs if e <= TOL_PCT)
    return {"value": f"run_id {_ids(d)} ({_runtime(d)}): {within}/{len(errs)} configs within +/-{TOL_PCT:g}% of the "
                     f"measured ceiling (fit and held-out together); max |error| {max(errs):.1f}%, mean "
                     f"{statistics.fmean(errs):.1f}%",
            "n": f"{len(errs)} configs", "within": within, "configs": len(errs)}


def tuned_window(d: dict) -> dict:
    parts, out = [], {}
    for cfg in ("default", "tuned"):
        rs = [r for r in d["records"] if r["config"] == cfg]
        if not rs:
            raise IntakeError(f"no {cfg} window_task records")
        wins = sorted({r["window_tokens"] for r in rs})
        ok = sum(r["ok"] for r in rs)
        out[cfg] = {"windows": wins, "ok": ok, "n": len(rs)}
        parts.append(f"{cfg} window {'/'.join(f'{w:,}' for w in wins)}: {ok}/{len(rs)} correct")
    return {"value": f"run_id {_ids(d)} ({_runtime(d)}): " + "; ".join(parts),
            "n": f"{out['default']['n'] + out['tuned']['n']} task runs", **out}


def gpu_share(d: dict) -> dict:
    by = {}
    for r in d["records"]:
        by.setdefault(float(r["gpu_share_pct"]), []).append(r["ok"])
    parts = [f"GPU share {k:g}%: {sum(v)}/{len(v)} ok" for k, v in sorted(by.items())]
    return {"value": f"run_id {_ids(d)} ({_runtime(d)}): " + "; ".join(parts), "n": f"{len(d['records'])} trials",
            "by_share": {k: [sum(v), len(v)] for k, v in by.items()}}


def loud_failure(d: dict) -> dict:
    over = [r for r in d["records"] if r["prompt_tokens"] > r["window_tokens"]]
    within = [r for r in d["records"] if r["prompt_tokens"] <= r["window_tokens"]]
    if not over:
        raise IntakeError("no overflow_call with prompt_tokens above window_tokens")
    loud = [r for r in over if r["outcome"] == "error"]
    lat = [float(r["latency_s"]) for r in loud]
    txt = (f"run_id {_ids(d)} ({_runtime(d)}): over the window {len(loud)}/{len(over)} calls failed loudly (an error "
           f"returned)")
    if lat:
        txt += f", time to the error median {statistics.median(lat):.2f} s, p90 {_q(lat, 0.9):.2f} s"
    silent = len(over) - len(loud)
    txt += f"; {silent} over-window call(s) returned ok"
    if within:
        wl = [float(r["latency_s"]) for r in within if r["outcome"] == "ok"]
        txt += (f"; within the window {sum(r['outcome'] == 'ok' for r in within)}/{len(within)} ok"
                + (f", latency median {statistics.median(wl):.2f} s" if wl else ""))
    return {"value": txt, "n": f"{len(d['records'])} calls", "over": len(over), "loud": len(loud)}


ROWS = {"ceiling_prediction": ("ceiling", ceiling_prediction), "ceiling_within": ("ceiling", ceiling_within),
        "tuned_window": ("tuned_window", tuned_window), "gpu_share": ("gpu_share", gpu_share),
        "loud_failure": ("loud_failure", loud_failure)}


def compute(repo, row: str) -> dict:
    kind, fn = ROWS[row]
    return fn(load(path_of(repo, kind), kind))
