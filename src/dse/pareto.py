"""Quality vs cost Pareto frontier and config recommender over measured outcome-table rows (2026-10-08).

Public API (the demo dashboard builds against exactly this; do not rename):
  load_points(files, cloud_source="stub") -> list[Point]
  frontier(points, machine=None) -> list[Point]
  recommend(points, budget_usd, quality_floor, latency_target_ms, hardware) -> Recommendation

INPUTS ARE DISCOVERED, NEVER COUNTED IN ADVANCE. `files` is a list of (machine_label, path) pairs. Every
(machine, runtime/config, model) group found in the rows becomes one Point, so more models, more configs, a T2S
file or a Blade file are picked up by passing them in; nothing here knows how many there are.

ONE POINT = one (machine, runtime, model) cell:
  * rows: outcome rows (record "outcome_row", or any row with item_id/model_id/config/score) that pass
    harness/x2_outcome_table.py::row_is_valid; the latest row per (item, model, config) wins, across files of the
    same machine in the order given (so a later file supersedes an earlier one).
  * quality: trace-weighted mean score, weights from results/workload_pack/item_weights_trace_weighted.json
    renormalized over the items present in that cell (same rule as the register's x2-v3-scores row). Error rows
    that row_is_valid keeps (context_overflow, other) are real outcomes and count as score 0. timeout_latency rows
    (harness/x2_outcome_table.py::analysis_outcome, cause timeout; 2026-10-08 operator decision) are EXCLUDED from
    the quality denominator: a call that hit the per-call timeout is a usability outcome, not a wrong answer. They
    stay in n_rows and error_causes, and the point carries the flag "timeout_latency_excluded=<n>".
    quality_n = number of scored items (timeout_latency excluded); weight_coverage = sum of those items' weights
    (share of the trace-weighted pack the cell has a quality measurement for). If no scored item has a weight, the
    plain mean is used and flagged "unweighted".
  * latency_p50_ms / latency_p90_ms: over the cell's rows that completed (classify_error_cause == "none"), from
    the rows' latency_s (end-to-end call time). Linear-interpolated percentiles.
  * usd_per_1k_steps: one step = one workload-pack item call. Local cells: 0 marginal API cost, plus an optional
    amortized hardware cost (hw_usd_per_hour[machine] x mean completed-call latency). No hardware cost is applied
    by default: configs/hardware/evox2_*.yaml and evo_t2s.yaml carry bom_cost_usd null.
  * template_source: chat_template_source per row (harness/chat_template_source.py), the row's own field, else
    derived from (model, config) with the probe facts of the matching host (results/*chat_template_sources*.jsonl,
    host compared to the machine label ignoring case and punctuation). A cell measured through an Ollama tag
    created from a bare GGUF is flagged "bare_template": an install-path result, not model capability, so frontier()
    and recommend() leave it out unless include_flagged=True (docs/FINDINGS.md 2026-10-08, install path).

CLOUD POINTS (machine "cloud").
  * cloud_source="stub" (default, and what "auto" falls back to): one point per cloud model in
    analysis/cloud_budget_forecast.py (CHEAP, MID), Standard tier, cost = trace-weighted mean per-item cost over the
    outcome table's step set (pack minus r2_sessions, harness/x2_outcome_table.py::load_items_trace_weighted),
    priced from results/cloud_prices_20261008.json with that module's o200k input token counts and per-family
    output medians. The stub answers nothing real, so quality and latency are None (not measured): a stub point
    sits on no frontier and is never recommended, it only prices the all-cloud baseline. Every stub point has
    stub=True, "STUB" in flags, in config_id and in label(). Pass stub_cloud_quality to give stub points an
    assumed quality (flag "stub_assumed_quality"); that number is an assumption, not a measurement.
  * Real cloud outcome rows: pass the file with machine label "cloud" in `files`. Quality and latency come from the
    rows exactly as for a local cell; cost from each row's cost_usd if present, else its token counts priced from
    the price file (input_tokens/output_tokens, falling back to sent_tokens/completion_tokens). A model with real
    rows never also gets a stub point.
  * cloud_source=<path to a ledger jsonl> (src/cloud/client.py's results/cloud_ledger.jsonl format): real cost per
    call per model (mean cost_usd x 1000), stub=False, quality None unless a real cloud outcome file covers the
    same model (then the outcome file's quality is kept and the ledger supplies the cost).
  * cloud_source="auto": the ledger at results/cloud_ledger.jsonl if it has call rows, else the stub.
  * cloud_source="none": no cloud points beyond real cloud outcome rows.

RECOMMENDATION. Candidates: points on an allowed machine (hardware list, matched ignoring case and punctuation,
prefix either way, so "evo-x2" matches "evox2_strix_halo_128gb") plus cloud points, with a measured quality >=
quality_floor, a measured latency (p50 by default) <= latency_target_ms, and projected spend for n_steps (default
1000) <= budget_usd. Chosen = cheapest, then highest quality, then lowest latency. savings_vs_all_cloud_usd_per_1k =
all-cloud reference cost - chosen cost, where the reference is the cheapest cloud point that meets the floor, or,
when no cloud quality is measured (stub), the cheapest cloud point (the smallest, i.e. most conservative, saving).
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "harness", REPO / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CLOUD_MACHINE = "cloud"
STUB = "STUB"
BARE_FLAG = "bare_template"
WEIGHTS_FILE = "results/workload_pack/item_weights_trace_weighted.json"
PRICES_FILE = "results/cloud_prices_20261008.json"
DEFAULT_LEDGER = "results/cloud_ledger.jsonl"
TEMPLATE_FACTS_GLOB = "*chat_template_sources*.jsonl"
SMALL_N = 30            # quality_n below this is flagged "small_n"
LOW_COVERAGE = 0.5      # weight_coverage below this is flagged "low_weight_coverage"
STUB_TIER = "standard"


# ------------------------------------------------------------------------------------------------- data types
@dataclass
class Point:
    machine: str
    runtime: str
    model: str
    config_id: str
    quality: Optional[float]
    quality_n: int
    usd_per_1k_steps: float
    latency_p50_ms: Optional[float]
    latency_p90_ms: Optional[float]
    stub: bool
    template_source: Optional[str]
    flags: list[str] = field(default_factory=list)
    # extras (not part of the frozen API, safe to ignore)
    weight_coverage: Optional[float] = None
    n_rows: int = 0
    error_causes: dict[str, int] = field(default_factory=dict)
    cost_basis: str = ""
    source_files: list[str] = field(default_factory=list)

    @property
    def is_cloud(self) -> bool:
        return self.machine == CLOUD_MACHINE

    @property
    def bare_template(self) -> bool:
        return BARE_FLAG in self.flags

    def label(self) -> str:
        s = f"{self.model} {self.runtime}"
        if self.stub:
            s += f" [{STUB}]"
        if self.bare_template:
            s += " [bare template]"
        return s

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label()
        return d


@dataclass
class Recommendation:
    chosen: Optional[Point]
    reason: str
    savings_vs_all_cloud_usd_per_1k: Optional[float]
    alternatives: list[Point] = field(default_factory=list)
    all_cloud_reference: Optional[Point] = None
    stub: bool = False  # True when any number in this recommendation comes from a STUB point
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen.to_dict() if self.chosen else None, "reason": self.reason,
                "savings_vs_all_cloud_usd_per_1k": self.savings_vs_all_cloud_usd_per_1k,
                "alternatives": [p.to_dict() for p in self.alternatives],
                "all_cloud_reference": self.all_cloud_reference.to_dict() if self.all_cloud_reference else None,
                "stub": self.stub, "inputs": self.inputs}


# ------------------------------------------------------------------------------------------------- helpers
def _x2():
    import x2_outcome_table as x2
    return x2


def _cts():
    import chat_template_source as cts
    return cts


def _jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def machine_matches(machine: str, option: str) -> bool:
    a, b = _norm(machine), _norm(option)
    return bool(a and b) and (a == b or a.startswith(b) or b.startswith(a))


def percentile(values: list[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (numpy's default method), q in [0, 100]."""
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _is_outcome_row(r: dict) -> bool:
    if r.get("record") not in (None, "outcome_row"):
        return False
    return all(r.get(k) is not None for k in ("item_id", "model_id", "config")) and "score" in r


def load_weights(repo: Path = REPO) -> dict[str, float]:
    return json.loads((Path(repo) / WEIGHTS_FILE).read_text(encoding="utf-8"))


def load_prices(repo: Path = REPO) -> dict:
    return json.loads((Path(repo) / PRICES_FILE).read_text(encoding="utf-8"))


def _template_facts(repo: Path) -> dict[str, dict]:
    """{normalized host: facts} from every results/*chat_template_sources*.jsonl probe file."""
    cts = _cts()
    by_host: dict[str, list[dict]] = {}
    for f in sorted((Path(repo) / "results").glob(TEMPLATE_FACTS_GLOB)):
        for r in _jsonl(f):
            if r.get("host"):
                by_host.setdefault(_norm(r["host"]), []).append(r)
    return {h: cts.facts_from_probe(rows) for h, rows in by_host.items()}


def _template_fn(machine: str, facts_by_host: dict[str, dict]):
    x2, cts = _x2(), _cts()
    facts = next((f for h, f in facts_by_host.items() if machine_matches(machine, h)), None)

    def src(r: dict) -> Optional[str]:
        if r.get("chat_template_source"):
            return r["chat_template_source"]
        if facts is None:
            return None
        return cts.source_for_row(r, x2.MODEL_MAP, facts, x2.LEGACY_OLLAMA_TAG)["chat_template_source"]
    return src


def _row_cost_usd(r: dict, prices: dict) -> Optional[float]:
    if r.get("cost_usd") is not None:
        return float(r["cost_usd"])
    model = r.get("model_id")
    tin = r.get("input_tokens", r.get("sent_tokens"))
    tout = r.get("output_tokens", r.get("completion_tokens"))
    if tin is None or tout is None:
        return None
    if model in prices.get("models", {}):
        import cloud_budget_forecast as cbf
        return cbf.call_cost(prices, model, STUB_TIER, int(tin), int(tout))
    try:
        from src.cloud.client import estimate_cost_usd
        return estimate_cost_usd(model, int(tin), int(tout))
    except Exception:
        return None


def _weighted_mean(vals: dict[str, float], weights: dict[str, float]) -> tuple[Optional[float], bool]:
    """(mean, weighted?) over item ids; plain mean if no item carries a weight."""
    if not vals:
        return None, True
    wsum = sum(weights.get(i, 0.0) for i in vals)
    if wsum > 0:
        return sum(weights.get(i, 0.0) * v for i, v in vals.items()) / wsum, True
    return sum(vals.values()) / len(vals), False


# ------------------------------------------------------------------------------------------------- stub cloud
@lru_cache(maxsize=4)
def _stub_token_counts(repo_s: str) -> dict[str, tuple[int, float]]:
    """{item_id: (o200k input tokens, expected output tokens)} from analysis/cloud_budget_forecast.py."""
    import cloud_budget_forecast as cbf
    items, _fams, _outs = cbf.pack_summary(Path(repo_s))
    return {it["item_id"]: (it["input_tokens"], it["output_tokens"]) for it in items}


def _step_item_ids(repo: Path) -> list[str]:
    return [it["item_id"] for it in _x2().load_items_trace_weighted(Path(repo))]


def stub_cloud_points(repo: Path = REPO, weights: Optional[dict] = None, prices: Optional[dict] = None,
                      token_counts: Optional[dict] = None, step_items: Optional[Iterable[str]] = None,
                      models: Optional[Iterable[str]] = None, stub_cloud_quality: Optional[float] = None
                      ) -> list[Point]:
    """One STUB point per cloud model: trace-weighted mean priced cost per step x 1000; quality/latency None."""
    import cloud_budget_forecast as cbf
    weights = weights if weights is not None else load_weights(repo)
    prices = prices if prices is not None else load_prices(repo)
    token_counts = token_counts if token_counts is not None else _stub_token_counts(str(repo))
    step_items = list(step_items) if step_items is not None else _step_item_ids(repo)
    models = list(models) if models is not None else [cbf.CHEAP, cbf.MID]
    ids = [i for i in step_items if i in token_counts]
    out = []
    for m in models:
        per_item = {i: cbf.call_cost(prices, m, STUB_TIER, int(token_counts[i][0]), token_counts[i][1]) for i in ids}
        mean, weighted = _weighted_mean(per_item, weights)
        flags = [STUB, "stub_quality_unmeasured" if stub_cloud_quality is None else "stub_assumed_quality",
                 "stub_latency_unmeasured"]
        if not weighted:
            flags.append("unweighted")
        out.append(Point(machine=CLOUD_MACHINE, runtime=f"api_{STUB_TIER}", model=m,
                         config_id=f"{CLOUD_MACHINE}:{STUB}:{m}:{STUB_TIER}", quality=stub_cloud_quality,
                         quality_n=0, usd_per_1k_steps=(mean or 0.0) * 1000, latency_p50_ms=None,
                         latency_p90_ms=None, stub=True, template_source=None, flags=flags,
                         weight_coverage=None, n_rows=0, cost_basis=(
                             f"{STUB}: {PRICES_FILE} {STUB_TIER} rates x o200k token counts "
                             f"(analysis/cloud_budget_forecast.py) over {len(ids)} step items"),
                         source_files=[PRICES_FILE, WEIGHTS_FILE]))
    return out


def ledger_points(path, existing: Optional[dict[str, Point]] = None) -> list[Point]:
    """Real cloud cost per model from a ledger (one row per real call, rows with cost_usd and model_id)."""
    by_model: dict[str, list[float]] = {}
    for r in _jsonl(path):
        if r.get("record") == "alert" or r.get("cost_usd") is None or not r.get("model_id"):
            continue
        if str(r["model_id"]).startswith("stub-") or r.get("stub"):
            continue
        by_model.setdefault(r["model_id"], []).append(float(r["cost_usd"]))
    out = []
    for m, costs in sorted(by_model.items()):
        usd = sum(costs) / len(costs) * 1000
        prior = (existing or {}).get(m)
        if prior is not None:
            prior.usd_per_1k_steps = usd
            prior.cost_basis = f"ledger {Path(path).name}: mean of {len(costs)} real calls"
            prior.source_files.append(str(path))
            continue
        out.append(Point(machine=CLOUD_MACHINE, runtime="api", model=m, config_id=f"{CLOUD_MACHINE}:api:{m}",
                         quality=None, quality_n=0, usd_per_1k_steps=usd, latency_p50_ms=None,
                         latency_p90_ms=None, stub=False, template_source=None,
                         flags=["quality_unmeasured", "latency_unmeasured"], n_rows=len(costs),
                         cost_basis=f"ledger {Path(path).name}: mean of {len(costs)} real calls",
                         source_files=[str(path)]))
    return out


# ------------------------------------------------------------------------------------------------- loading
def load_points(files: list[tuple[str, str]], cloud_source: str = "stub", *, repo: Path = REPO,
                hw_usd_per_hour: Optional[dict[str, float]] = None, stub_cloud_quality: Optional[float] = None,
                weights: Optional[dict] = None, prices: Optional[dict] = None,
                stub_token_counts: Optional[dict] = None, step_items: Optional[Iterable[str]] = None,
                template_facts: Optional[dict] = None) -> list[Point]:
    """Points from outcome files [(machine_label, path), ...]; machine label "cloud" marks real cloud rows.
    Keyword-only arguments exist for tests and what-if runs; defaults read the committed files under `repo`."""
    repo = Path(repo)
    x2 = _x2()
    weights = weights if weights is not None else load_weights(repo)
    prices = prices if prices is not None else load_prices(repo)
    facts_by_host = template_facts if template_facts is not None else _template_facts(repo)
    hw = hw_usd_per_hour or {}

    latest: dict[str, dict[tuple, dict]] = {}
    sources: dict[str, list[str]] = {}
    for machine, path in files:
        sources.setdefault(machine, []).append(str(path))
        bucket = latest.setdefault(machine, {})
        for r in _jsonl(Path(path) if Path(path).is_absolute() else repo / path):
            if _is_outcome_row(r) and x2.row_is_valid(r):
                bucket[(r["item_id"], r["model_id"], r["config"])] = r

    points: list[Point] = []
    for machine, rows in latest.items():
        src = _template_fn(machine, facts_by_host)
        cells: dict[tuple, dict[str, dict]] = {}
        for (iid, m, c), r in rows.items():
            cells.setdefault((m, c), {})[iid] = r
        for (m, c), items in sorted(cells.items()):
            scored = {i: r for i, r in items.items() if x2.counts_toward_accuracy(r)}
            n_timeout_latency = len(items) - len(scored)
            scores = {i: float(r.get("score") or 0.0) for i, r in scored.items()}
            quality, weighted = _weighted_mean(scores, weights)
            coverage = sum(weights.get(i, 0.0) for i in scored)
            causes: dict[str, int] = {}
            lat = []
            for r in items.values():
                k = x2.classify_error_cause(r)
                causes[k] = causes.get(k, 0) + 1
                if k == "none" and r.get("latency_s") is not None:
                    lat.append(float(r["latency_s"]) * 1000)
            tsrc = sorted({str(src(r)) for r in items.values()})
            template = tsrc[0] if len(tsrc) == 1 else "mixed:" + ",".join(tsrc)
            if template == "None":
                template = None
            flags = []
            if any(src(r) == _cts().BARE for r in items.values()):
                flags.append(BARE_FLAG)
            if template is None:
                flags.append("template_unknown")
            elif template.startswith("mixed:"):
                flags.append("template_mixed")
            if not weighted:
                flags.append("unweighted")
            if len(scored) < SMALL_N:
                flags.append("small_n")
            if n_timeout_latency:
                flags.append(f"timeout_latency_excluded={n_timeout_latency}")
            if weighted and coverage < LOW_COVERAGE:
                flags.append("low_weight_coverage")
            for k, v in sorted(causes.items()):
                if k != "none":
                    flags.append(f"errors:{k}={v}")
            p50, p90 = percentile(lat, 50), percentile(lat, 90)
            if machine == CLOUD_MACHINE:
                costs = {i: _row_cost_usd(r, prices) for i, r in items.items()}
                costs = {i: v for i, v in costs.items() if v is not None}
                mean_cost, _w = _weighted_mean(costs, weights)
                usd = (mean_cost or 0.0) * 1000
                basis = f"real cloud rows: trace-weighted mean of {len(costs)} per-row costs"
                if not costs:
                    flags.append("cost_unmeasured")
            else:
                rate = next((v for k, v in hw.items() if machine_matches(machine, k)), 0.0)
                mean_lat_s = (sum(lat) / len(lat) / 1000) if lat else 0.0
                usd = rate * mean_lat_s * 1000 / 3600
                basis = "local: 0 marginal API cost" + (
                    f" + amortized hardware USD {rate}/h x mean call {mean_lat_s:.1f} s" if rate else "")
                if rate:
                    flags.append("amortized_hw_cost")
            points.append(Point(machine=machine, runtime=c, model=m, config_id=f"{machine}:{c}:{m}",
                                quality=quality, quality_n=len(scored), usd_per_1k_steps=usd,
                                latency_p50_ms=p50, latency_p90_ms=p90, stub=False, template_source=template,
                                flags=flags, weight_coverage=coverage if weighted else None, n_rows=len(items),
                                error_causes=causes, cost_basis=basis, source_files=list(sources[machine])))

    real_cloud = {p.model: p for p in points if p.is_cloud}
    use_ledger = None
    if cloud_source not in ("stub", "none", "auto"):
        use_ledger = Path(cloud_source) if Path(cloud_source).is_absolute() else repo / cloud_source
    elif cloud_source == "auto":
        led = repo / DEFAULT_LEDGER
        if led.exists() and ledger_points(led):
            use_ledger = led
    if use_ledger is not None:
        points.extend(ledger_points(use_ledger, existing=real_cloud))
    elif cloud_source in ("stub", "auto"):
        stubs = stub_cloud_points(repo, weights=weights, prices=prices, token_counts=stub_token_counts,
                                  step_items=step_items, stub_cloud_quality=stub_cloud_quality)
        points.extend(p for p in stubs if p.model not in real_cloud)
    return points


# ------------------------------------------------------------------------------------------------- frontier
def _eligible(p: Point, include_flagged: bool) -> bool:
    return p.quality is not None and (include_flagged or not p.bare_template)


def _lat(p: Point, stat: str) -> Optional[float]:
    return p.latency_p90_ms if stat == "p90" else p.latency_p50_ms


def frontier(points: list[Point], machine: Optional[str] = None, *, include_flagged: bool = False,
             latency_target_ms: Optional[float] = None, latency_stat: str = "p50") -> list[Point]:
    """Pareto-optimal points (min usd_per_1k_steps, max quality), cheapest first. machine=None: across all points;
    a machine label: that machine's points only (use "cloud" for cloud points). Points with no measured quality
    (STUB cloud) and bare-template cells (unless include_flagged) are never on a frontier. With latency_target_ms,
    points whose latency is unmeasured or above the target are dropped first. Ties (same cost and quality) keep
    the lower-latency point."""
    cand = [p for p in points if _eligible(p, include_flagged) and (machine is None or p.machine == machine)]
    if latency_target_ms is not None:
        cand = [p for p in cand if _lat(p, latency_stat) is not None and _lat(p, latency_stat) <= latency_target_ms]
    big = float("inf")
    cand.sort(key=lambda p: (p.usd_per_1k_steps, -p.quality, _lat(p, latency_stat) or big, p.config_id))
    out, best = [], -1.0
    for p in cand:
        if p.quality > best + 1e-12:
            out.append(p)
            best = p.quality
    return out


# ------------------------------------------------------------------------------------------------- recommend
def all_cloud_reference(points: list[Point], quality_floor: float) -> Optional[Point]:
    cloud = [p for p in points if p.is_cloud]
    if not cloud:
        return None
    meeting = [p for p in cloud if p.quality is not None and p.quality >= quality_floor]
    pool = meeting or cloud
    return min(pool, key=lambda p: (p.usd_per_1k_steps, p.config_id))


def recommend(points: list[Point], budget_usd: float, quality_floor: float, latency_target_ms: float,
              hardware: list[str], *, n_steps: int = 1000, latency_stat: str = "p50",
              include_flagged: bool = False, n_alternatives: int = 5) -> Recommendation:
    """Cheapest point meeting the quality floor, the latency target and the budget for n_steps; see module doc."""
    inputs = {"budget_usd": budget_usd, "quality_floor": quality_floor, "latency_target_ms": latency_target_ms,
              "hardware": list(hardware), "n_steps": n_steps, "latency_stat": latency_stat,
              "include_flagged": include_flagged}
    allowed = [p for p in points if p.is_cloud or any(machine_matches(p.machine, h) for h in hardware)]
    ref = all_cloud_reference(points, quality_floor)
    rejected = {"no_quality": 0, "bare_template": 0, "below_floor": 0, "latency": 0, "budget": 0}
    ok = []
    for p in allowed:
        lat = _lat(p, latency_stat)
        if p.quality is None:
            rejected["no_quality"] += 1
        elif p.bare_template and not include_flagged:
            rejected["bare_template"] += 1
        elif p.quality < quality_floor:
            rejected["below_floor"] += 1
        elif lat is None or lat > latency_target_ms:
            rejected["latency"] += 1
        elif p.usd_per_1k_steps * n_steps / 1000 > budget_usd:
            rejected["budget"] += 1
        else:
            ok.append(p)
    big = float("inf")
    ok.sort(key=lambda p: (p.usd_per_1k_steps, -p.quality, _lat(p, latency_stat) or big, p.config_id))
    chosen = ok[0] if ok else None
    savings = None
    if chosen is not None and ref is not None:
        savings = ref.usd_per_1k_steps - chosen.usd_per_1k_steps
    stub = bool((chosen and chosen.stub) or (ref and ref.stub))
    ref_s = "none" if ref is None else (f"{ref.model} USD {ref.usd_per_1k_steps:.2f}/1k steps"
                                        + (f" [{STUB}]" if ref.stub else ""))
    rej_s = ", ".join(f"{k} {v}" for k, v in rejected.items() if v)
    if chosen is None:
        reason = (f"no config meets quality >= {quality_floor}, {latency_stat} latency <= {latency_target_ms:.0f} ms "
                  f"and budget USD {budget_usd} on {', '.join(hardware) or 'no hardware'} (rejected: {rej_s or 'none'})"
                  f"; all-cloud reference {ref_s}")
    else:
        reason = (f"{chosen.machine} {chosen.label()}: cheapest of {len(ok)} configs meeting quality >= "
                  f"{quality_floor} (has {chosen.quality:.3f}, n={chosen.quality_n}) and {latency_stat} "
                  f"<= {latency_target_ms:.0f} ms (has {_lat(chosen, latency_stat):.0f} ms) at USD "
                  f"{chosen.usd_per_1k_steps:.2f}/1k steps; vs all-cloud {ref_s}")
    return Recommendation(chosen=chosen, reason=reason, savings_vs_all_cloud_usd_per_1k=savings,
                          alternatives=ok[1:1 + n_alternatives], all_cloud_reference=ref, stub=stub, inputs=inputs)


# ------------------------------------------------------------------------------------------------- serialization
def points_table(points: list[Point]) -> dict[str, Any]:
    """JSON-ready table: every point, the frontier per machine (config_ids), and the frontier rule."""
    machines = sorted({p.machine for p in points})
    return {"points": [p.to_dict() for p in points],
            "frontier_by_machine": {m: [p.config_id for p in frontier(points, m)] for m in machines},
            "frontier_rule": "min usd_per_1k_steps, max quality; STUB (quality unmeasured) and bare-template "
                             "points excluded",
            "stub_points": [p.config_id for p in points if p.stub]}


def format_table(points: list[Point]) -> list[str]:
    on = {p.config_id for m in {q.machine for q in points} for p in frontier(points, m)}
    out = ["| machine | config | quality | n | USD/1k steps | p50 ms | p90 ms | template | frontier | flags |",
           "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    f = lambda v, fmt: "n/a" if v is None else format(v, fmt)  # noqa: E731
    for p in sorted(points, key=lambda p: (p.machine, p.usd_per_1k_steps, -(p.quality or -1), p.config_id)):
        out.append(f"| {p.machine} | {p.label()} | {f(p.quality, '.3f')} | {p.quality_n} | "
                   f"{p.usd_per_1k_steps:.2f} | {f(p.latency_p50_ms, '.0f')} | {f(p.latency_p90_ms, '.0f')} | "
                   f"{p.template_source or 'n/a'} | {'yes' if p.config_id in on else ''} | {', '.join(p.flags)} |")
    return out
