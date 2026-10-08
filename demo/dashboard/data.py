"""The one adapter between the dashboard and the DSE modules, plus the computations behind each endpoint.

Integration point: `load_backend()` is the only place that imports src/dse/router.py and src/dse/pareto.py. It
uses them when they expose the agreed interface (Router with decide/replay_session; pareto with load_points,
frontier, recommend) and falls back to demo/dashboard/fakes.py otherwise. Set DASHBOARD_BACKEND=fake to force the
fakes (the tests do). Every response says which backend produced it, and the UI shows a banner when it is the fake.

Everything else here reads the offline snapshot in demo/dashboard/cache/ (built by build_cache.py) and returns
plain dicts. Each number carries a `src`: a register row id ({"register": id}) or a result file location
({"file": path, "line": n} or {"file": path, "match": {...}}).
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from demo.dashboard import fakes, naive

REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).resolve().parent / "cache"
FAKE_SRC = "fake: demo/dashboard/fakes.py (src/dse module not on main yet)"


@dataclass
class Backend:
    router_cls: Any
    pareto: Any
    router_source: str
    pareto_source: str
    envelope: Any = None

    @property
    def is_fake(self) -> bool:
        return self.router_source.startswith("fake") or self.pareto_source.startswith("fake")


def _real_router():
    mod = importlib.import_module("src.dse.router")
    cls = getattr(mod, "Router", None)
    if cls is None or not all(hasattr(cls, a) for a in ("decide", "replay_session")):
        raise ImportError("src/dse/router.py has no Router with decide/replay_session")
    env = None
    for name in ("load_envelope", "default_envelope", "build_envelope"):
        if callable(getattr(mod, name, None)):
            env = getattr(mod, name)()
            break
    return cls, env


def _real_pareto():
    mod = importlib.import_module("src.dse.pareto")
    if not all(callable(getattr(mod, a, None)) for a in ("load_points", "frontier", "recommend")):
        raise ImportError("src/dse/pareto.py lacks load_points/frontier/recommend")
    return mod


def load_backend(force_fake: bool | None = None) -> Backend:
    if force_fake is None:
        force_fake = os.environ.get("DASHBOARD_BACKEND", "").lower() == "fake"
    router_cls, env, rsrc = fakes.FakeRouter, None, FAKE_SRC
    pareto, psrc = fakes, FAKE_SRC
    if not force_fake:
        try:
            router_cls, env = _real_router()
            rsrc = "real: src/dse/router.py"
        except Exception:
            pass
        try:
            pareto = _real_pareto()
            psrc = "real: src/dse/pareto.py"
        except Exception:
            pass
    return Backend(router_cls, pareto, rsrc, psrc, env)


def to_dict(obj) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        return {k: to_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


class Cache:
    NAMES = ("manifest", "register", "hardware", "workloads", "r2_sessions", "points", "mitigation")

    def __init__(self, directory: Path = CACHE_DIR):
        self.dir = Path(directory)
        for n in self.NAMES:
            p = self.dir / f"{n}.json"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run `py -3.12 -m demo.dashboard.build_cache` first")
            setattr(self, n, json.loads(p.read_text(encoding="utf-8")))


# ── points / frontier / recommendation ─────────────────────────────────────────────────────────────


def point_id(p: dict) -> str:
    return "|".join(str(p.get(k)) for k in ("machine", "runtime", "model", "config_id"))


def _point_src(p: dict, files: list[str]) -> dict:
    if p.get("stub"):
        return {"file": "src/cloud/client.py", "note": "STUB: pricing table, no measured cloud quality"}
    pref = {"evo-x2": "x2_", "evo-t2s": "t2s_"}.get(p.get("machine"), "")
    f = next((f for f in files if Path(f).name.startswith(pref)), files[0] if files else None)
    return {"file": f, "match": {"record": "outcome_row", "config": p.get("config_id"),
                                 "model_id": p.get("model")}}


def _points(cache: Cache, backend: Backend) -> tuple[list, list[dict]]:
    """Rebuild point objects of the backend's own Point type from the cached dicts (so the real recommend()
    receives its own type), and return the decorated dicts for the UI."""
    cls = getattr(backend.pareto, "Point", None)
    objs, dicts = [], []
    for d in cache.points["points"]:
        if cls is not None and dataclasses.is_dataclass(cls):
            names = {f.name for f in dataclasses.fields(cls)}
            obj = cls(**{k: v for k, v in d.items() if k in names})
        else:
            obj = SimpleNamespace(**d)
        objs.append(obj)
        dicts.append({**d, "id": point_id(d), "src": _point_src(d, cache.points["files"])})
    return objs, dicts


def machines(cache: Cache) -> list[str]:
    return sorted({p["machine"] for p in cache.points["points"] if not p.get("stub")})


def frontier_view(cache: Cache, backend: Backend, selected: list[str] | None = None) -> dict:
    objs, dicts = _points(cache, backend)
    sel = selected or machines(cache)
    per = {}
    for m in sel:
        fr = {point_id(to_dict(p)) for p in backend.pareto.frontier(objs, machine=m)}
        pts = [dict(d, on_frontier=d["id"] in fr) for d in dicts if d["machine"] == m]
        per[m] = {"points": pts, "frontier_ids": sorted(fr)}
    return {"machines": per, "cloud": [d for d in dicts if d.get("stub")], "files": cache.points["files"],
            "backend": cache.points["backend"]}


def recommendation(cache: Cache, backend: Backend, budget_usd: float, quality_floor: float,
                   latency_target_ms: float, hardware: list[str]) -> dict:
    objs, dicts = _points(cache, backend)
    by_id = {d["id"]: d for d in dicts}
    rec = to_dict(backend.pareto.recommend(objs, budget_usd, quality_floor, latency_target_ms, hardware))

    def deco(p):
        return None if p is None else by_id.get(point_id(p), dict(p, id=point_id(p)))

    cloud = [d for d in dicts if d.get("stub")]
    return {"chosen": deco(rec.get("chosen")), "reason": rec.get("reason"),
            "savings_vs_all_cloud_usd_per_1k": rec.get("savings_vs_all_cloud_usd_per_1k"),
            "savings_stub": True,  # all-cloud side is the stub pricing until real cloud rows exist
            "savings_src": {"file": "src/cloud/client.py", "note": "STUB cloud cost from the pricing table"},
            "alternatives": [deco(a) for a in rec.get("alternatives") or []],
            "all_cloud": cloud, "inputs": {"budget_usd": budget_usd, "quality_floor": quality_floor,
                                           "latency_target_ms": latency_target_ms, "hardware": hardware},
            "backend": backend.pareto_source}


# ── side-by-side scenario ──────────────────────────────────────────────────────────────────────────


def scenario_list(cache: Cache) -> dict:
    r2 = cache.r2_sessions
    reg = cache.register
    return {"source": r2["source"], "arm": r2["arm"],
            "sessions": [{"key": s["key"], "model": s["model"], "seed": s["seed"], "num_ctx": s["num_ctx"],
                          "n_turns": len(s["turns"]),
                          "first_over_window_turn": s["summary"]["first_over_window_turn"],
                          "truncation_detected_turn": s["summary"]["truncation_detected_turn"],
                          "first_error_turn": s["summary"]["first_error_turn"],
                          "src": {"file": r2["source"], "line": s["summary_line"]}} for s in r2["sessions"]],
            "register": {i: reg[i] for i in cache.manifest["scenario_register_ids"] if i in reg},
            "mitigation": cache.mitigation}


def _session(cache: Cache, key: str) -> dict:
    for s in cache.r2_sessions["sessions"]:
        if s["key"] == key:
            return s
    raise KeyError(key)


def replay(cache: Cache, backend: Backend, key: str, budget_usd: float = 50.0, quality_floor: float = 0.9,
           latency_target_ms: float = 30000.0) -> dict:
    s = _session(cache, key)
    src_file = cache.r2_sessions["source"]
    router = backend.router_cls(backend.envelope, budget_usd, quality_floor, latency_target_ms, "evo-x2",
                                "ollama", s["model"], None)
    decisions = [to_dict(d) for d in router.replay_session([dict(t) for t in s["turns"]])]
    adv = s.get("advertised_ctx") or {}
    mit = cache.mitigation
    steps, n_cost, o_cost = [], 0.0, 0.0
    held = canary_ok = canary_n = 0
    for i, (t, dec) in enumerate(zip(s["turns"], decisions), 1):
        nv = naive.decide(t, adv.get("ctx"))
        n_cost += nv["est_cost_usd"]
        o_cost += float(dec.get("est_cost_usd") or 0.0)
        rules = {r: t.get(r) for r in t["rules_in_use"]}
        applicable = [v for v in rules.values() if v is not None]
        all_held = all(applicable)
        held += 1 if all_held else 0
        if t.get("canary_check"):
            canary_n += 1
            canary_ok += 1 if (t.get("canary_sys_ok") and t.get("canary_hist_ok")) else 0
        statuses = [c.get("http_status") for c in t.get("calls") or []]
        errors = [c.get("error") for c in t.get("calls") or [] if c.get("error")]
        row_src = {"file": src_file, "line": t["line"]}
        naive_side = {**nv, "running_cost_usd": n_cost, "src": row_src,
                      "advertised_ctx": adv.get("ctx"),
                      "advertised_src": {"file": src_file, "line": adv.get("line")}}
        if nv["target"] == "local":
            naive_side["outcome"] = {
                "recorded": True, "http_status": statuses, "errors": errors, "rules": rules,
                "rules_held": f"{sum(1 for v in applicable if v)}/{len(applicable)}",
                "canary_check": t.get("canary_check"), "canary_sys_ok": t.get("canary_sys_ok"),
                "canary_hist_ok": t.get("canary_hist_ok"), "recall_ok": t.get("recall_ok"),
                "failures": t.get("turn_failures"), "over_loaded_window": t.get("over_loaded_window"),
                "loaded_context": t.get("loaded_context"),
                "transcript_tokens": t.get("transcript_tokens_calibrated"),
                "silent": bool(t.get("turn_failures") or t.get("canary_sys_ok") is False
                               or t.get("canary_hist_ok") is False) and not errors
                          and all(str(x) == "200" for x in statuses)}
            naive_side["running_quality"] = {"turns_all_rules_held": held, "turns": i,
                                             "canary_checks_passed": canary_ok, "canary_checks": canary_n}
        else:
            naive_side["outcome"] = {"recorded": False, "note": "routed to cloud: no recorded outcome (STUB)"}
        ours = {"decision": dec, "running_cost_usd": o_cost, "stub_cost": bool(dec.get("stub")),
                "src": row_src, "system_prompt_in_request": (dec.get("tokens") or {}).get("system", 0) > 0
                and dec.get("target") in ("local", "local_trimmed"),
                "quality": ({"label": mit["label"], "measured": False} if not mit.get("measured") else
                            {"label": "measured", "measured": True,
                             "register_ids": mit.get("register_ids")})}
        steps.append({"turn": t["turn_idx"], "task_text": t.get("task_text"), "naive": naive_side, "ours": ours})
    return {"key": key, "model": s["model"], "seed": s["seed"], "num_ctx": s["num_ctx"],
            "router_backend": backend.router_source, "summary": s["summary"],
            "summary_src": {"file": src_file, "line": s["summary_line"]}, "steps": steps,
            "naive_params": {"difficulty_words": naive.DIFFICULTY_WORDS,
                             "difficulty_threshold": naive.DIFFICULTY_THRESHOLD}}


# ── source viewer ──────────────────────────────────────────────────────────────────────────────────


def allowed_sources(cache: Cache) -> set[str]:
    return set(cache.manifest["sources"]) | {"src/cloud/client.py"} | set(cache.points["files"])


def source(cache: Cache, file: str, line: int | None = None, match: dict | None = None,
           limit: int = 50) -> dict:
    """A whitelisted source file's sha256 (as of the cache build) and, if asked, one raw line or the line
    numbers of rows matching every key=value in `match`. Reads the local repo copy; never anything else."""
    if file not in allowed_sources(cache):
        raise PermissionError(file)
    p = REPO / file
    meta = cache.manifest["sources"].get(file, {})
    out = {"file": file, "sha256_at_build": meta.get("sha256"), "exists": p.exists()}
    if not p.exists():
        return out
    if line is not None:
        with open(p, encoding="utf-8") as f:
            for i, l in enumerate(f, 1):
                if i == line:
                    out["line"] = i
                    out["text"] = l.rstrip("\n")
                    break
    elif match:
        hits, n = [], 0
        with open(p, encoding="utf-8") as f:
            for i, l in enumerate(f, 1):
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if all(str(r.get(k)) == str(v) for k, v in match.items()):
                    n += 1
                    if len(hits) < limit:
                        hits.append(i)
        out.update({"match": match, "n_matching_rows": n, "lines": hits})
    return out
