"""Stand-ins for src/dse/router.py and src/dse/pareto.py with exactly the agreed signatures.

Used by the tests always, and by the running dashboard only while the real modules are not on main (data.py
picks; the UI then shows a "fake backend" banner on every screen that uses them). Nothing here invents a
number: every value is computed from the rows passed in (outcome-table rows, R2 turn rows) or from
src/cloud/client.py's pricing table (cloud cost, always marked stub). Where the fake has no basis for a value it
returns None and the UI prints "n/a".

Agreed interfaces (from the task brief; the real modules are built in parallel):
  Router(envelope, budget_usd, quality_floor, latency_target_ms, hardware, runtime, model, cloud_client=None)
    .decide(step) -> RouteDecision
    .replay_session(rows) -> list[RouteDecision]
  load_points(files, cloud_source="stub") -> list[Point]
  frontier(points, machine=None) -> list[Point]
  recommend(points, budget_usd, quality_floor, latency_target_ms, hardware) -> Recommendation
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]

# ── router ──────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class RouteDecision:
    target: str                      # local | local_trimmed | cloud
    machine: Optional[str]
    runtime: Optional[str]
    model: Optional[str]
    num_ctx: Optional[int]
    tokens: dict                     # {system, kept_history, answer_budget, total}
    kept_turns: int
    dropped_turns: int
    predicted_latency_ms: Optional[float]
    est_cost_usd: float
    budget_remaining_usd: float
    stub: bool
    reason: str


def _turn_tokens(rows: list[dict]) -> list[int]:
    """Per-turn transcript growth (calibrated tokens): turn 1 is its transcript minus the system prompt."""
    out, prev = [], None
    for r in rows:
        cur = int(r.get("transcript_tokens_calibrated") or 0)
        if prev is None:
            out.append(max(0, cur - int(r.get("system_prompt_tokens_calibrated") or 0)))
        else:
            out.append(max(0, cur - prev))
        prev = cur
    return out


class FakeRouter:
    """Context-fit router with the agreed signature: keep the system prompt and the current turn, drop the
    oldest history turns until the request fits num_ctx, escalate to the (stub) cloud only if even that does not
    fit. It never predicts latency (None) because it has no latency model."""

    CLOUD_MODEL = "gpt-4o-mini"

    def __init__(self, envelope, budget_usd, quality_floor, latency_target_ms, hardware, runtime, model,
                 cloud_client=None):
        self.envelope = envelope
        self.budget_usd = float(budget_usd)
        self.spent = 0.0
        self.quality_floor = quality_floor
        self.latency_target_ms = latency_target_ms
        self.hardware = hardware
        self.runtime = runtime
        self.model = model
        self.cloud_client = cloud_client
        self._history: list[int] = []

    def decide(self, step: dict) -> RouteDecision:
        num_ctx = step.get("num_ctx_requested") or step.get("loaded_context")
        system = int(step.get("system_prompt_tokens_calibrated") or 0)
        answer = int(step.get("answer_budget_tokens") or 0)
        current = int(step.get("turn_tokens_calibrated") or 0)
        hist = list(self._history)
        self._history.append(current)
        full = system + sum(hist) + current + answer
        if num_ctx is None or full <= num_ctx:
            return self._local("local", num_ctx, system, hist, current, answer, 0,
                               f"fits: {full} tokens <= num_ctx {num_ctx}, sent as is")
        room = num_ctx - system - current - answer
        if room >= 0:
            kept, used = [], 0
            for t in reversed(hist):
                if used + t > room:
                    break
                kept.append(t)
                used += t
            dropped = len(hist) - len(kept)
            return self._local("local_trimmed", num_ctx, system, kept, current, answer, dropped,
                               f"{full} tokens > num_ctx {num_ctx}: dropped {dropped} oldest turns, "
                               f"system prompt kept")
        from src.cloud.client import estimate_cost_usd
        cost = estimate_cost_usd(self.CLOUD_MODEL, system + sum(hist) + current, answer)
        self.spent += cost
        return RouteDecision("cloud", None, "cloud", self.CLOUD_MODEL, None,
                             {"system": system, "kept_history": sum(hist), "answer_budget": answer,
                              "total": system + sum(hist) + current + answer},
                             len(hist), 0, None, cost, self.budget_usd - self.spent, True,
                             f"system + current turn + answer exceed num_ctx {num_ctx}: cloud (STUB)")

    def _local(self, target, num_ctx, system, kept, current, answer, dropped, reason):
        total = system + sum(kept) + current + answer
        return RouteDecision(target, self.hardware, self.runtime, self.model, num_ctx,
                             {"system": system, "kept_history": sum(kept), "answer_budget": answer,
                              "total": total},
                             len(kept), dropped, None, 0.0, self.budget_usd - self.spent, False, reason)

    def replay_session(self, rows: list[dict]) -> list[RouteDecision]:
        self._history = []
        rows = sorted(rows, key=lambda r: r["turn_idx"])
        sizes = _turn_tokens(rows)
        out = []
        for r, n in zip(rows, sizes):
            step = dict(r)
            step.setdefault("turn_tokens_calibrated", n)
            out.append(self.decide(step))
        return out


# ── pareto ──────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Point:
    machine: str
    runtime: str
    model: str
    config_id: str
    quality: Optional[float]
    quality_n: int
    usd_per_1k_steps: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p90_ms: Optional[float]
    stub: bool
    template_source: Optional[str]
    flags: list = field(default_factory=list)


@dataclass
class Recommendation:
    chosen: Optional[Point]
    reason: str
    savings_vs_all_cloud_usd_per_1k: Optional[float]
    alternatives: list = field(default_factory=list)


_MACHINE_BY_FILE_PREFIX = {"x2_": "evo-x2", "t2s_": "evo-t2s"}


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


def _template_sources() -> dict:
    """(model, config) -> chat template source, parsed from register row x2-chat-template-source."""
    import re
    p = REPO / "docs" / "NUMBERS_REGISTER.md"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("| x2-chat-template-source |"):
                for m in re.finditer(r"([\w.:-]+)/(\w+) (?:\(tag [^)]*\) )?(\w+) sha256", line):
                    out[(m.group(1), m.group(2))] = m.group(3)
    return out


def load_points(files, cloud_source="stub") -> list[Point]:
    """One local point per (machine, config, model) from outcome_row records: quality = mean score with
    non-200 rows counted as 0, latency percentiles over 200 rows. Cloud points carry no quality (stub has no
    measured quality) and a stub cost from the published pricing table at the local rows' mean token count."""
    templates = _template_sources()
    groups: dict[tuple, list[dict]] = {}
    for f in files:
        f = Path(f)
        machine = next((m for p, m in _MACHINE_BY_FILE_PREFIX.items() if f.name.startswith(p)), f.stem)
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("record") != "outcome_row":
                continue
            groups.setdefault((machine, r["config"], r["model_id"]), []).append(r)
    pts, all_sent = [], []
    for (machine, config, model), rows in sorted(groups.items()):
        scores = [(_f(r.get("score")) or 0.0) if str(r.get("http_status")) == "200" else 0.0 for r in rows]
        lat = sorted(_f(r.get("latency_s")) * 1000 for r in rows
                     if str(r.get("http_status")) == "200" and _f(r.get("latency_s")) is not None)
        all_sent += [_f(r.get("sent_tokens")) for r in rows if _f(r.get("sent_tokens")) is not None]
        runtime = "llama_server" if config.startswith("llama_server") else "ollama"
        tsrc = templates.get((model, config)) if machine == "evo-x2" else None
        flags = ["bare template"] if tsrc and tsrc.startswith("bare") else []
        pts.append(Point(machine, runtime, model, config, statistics.fmean(scores) if scores else None,
                         len(scores), 0.0, statistics.median(lat) if lat else None,
                         lat[min(len(lat) - 1, int(0.9 * len(lat)))] if lat else None, False, tsrc, flags))
    if cloud_source == "stub" and all_sent:
        from src.cloud.client import CLOUD_MODELS, estimate_cost_usd
        mean_in = statistics.fmean(all_sent)
        for tier, mid in CLOUD_MODELS.items():
            pts.append(Point("cloud", "cloud", mid, f"cloud_{tier}", None, 0,
                             1000 * estimate_cost_usd(mid, int(mean_in), 384), None, None, True, None, ["STUB"]))
    return pts


def frontier(points, machine=None) -> list[Point]:
    """Non-dominated local points (higher quality, lower p50 latency) for one machine or all."""
    cand = [p for p in points if not p.stub and p.quality is not None and p.latency_p50_ms is not None
            and (machine is None or p.machine == machine)]
    out = [p for p in cand if not any(
        (q.quality >= p.quality and q.latency_p50_ms <= p.latency_p50_ms)
        and (q.quality > p.quality or q.latency_p50_ms < p.latency_p50_ms) for q in cand)]
    return sorted(out, key=lambda p: p.latency_p50_ms)


def recommend(points, budget_usd, quality_floor, latency_target_ms, hardware) -> Recommendation:
    hw = set(hardware or [])
    ok = [p for p in points if not p.stub and p.machine in hw and p.quality is not None
          and p.quality >= quality_floor and p.latency_p50_ms is not None
          and p.latency_p50_ms <= latency_target_ms]
    cloud = [p for p in points if p.stub and p.usd_per_1k_steps is not None]
    cloud_cost = min((p.usd_per_1k_steps for p in cloud), default=None)
    if not ok:
        return Recommendation(None, "no local config on the selected hardware meets the quality floor and "
                                    "latency target; route to cloud (STUB cost)", 0.0 if cloud else None, [])
    ok.sort(key=lambda p: (-p.quality, p.latency_p50_ms))
    best = ok[0]
    sav = None if cloud_cost is None else cloud_cost - (best.usd_per_1k_steps or 0.0)
    return Recommendation(best, f"highest quality local config meeting floor {quality_floor} and p50 <= "
                                f"{latency_target_ms} ms on {', '.join(sorted(hw))}", sav, ok[1:4])
