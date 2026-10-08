"""Build demo/dashboard/cache/ from committed files so the dashboard runs fully offline.

    py -3.12 -m demo.dashboard.build_cache

Reads only committed inputs: docs/NUMBERS_REGISTER.md, configs/hardware/*.yaml, results/workload_pack/items,
the outcome tables, results/x2_r2_real_v1.jsonl (and results/x2_r2_mitigation_v1.jsonl if it has been synced).
Writes JSON snapshots plus a manifest with each source file's sha256, so a number on screen can be traced to the
exact bytes it came from. No network, no model call, no subprocess except `git rev-parse` through
harness/proc_util.py (hidden window).
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache"

REGISTER = "docs/NUMBERS_REGISTER.md"
R2_FILE = "results/x2_r2_real_v1.jsonl"
MITIGATION_FILE = "results/x2_r2_mitigation_v1.jsonl"
OUTCOME_FILES = ["results/x2_outcome_table_v3.jsonl", "results/t2s_outcome_table_full.jsonl"]
WORKLOAD_DIR = "results/workload_pack/items"
SCENARIO_ARM = "ollama_ctx_4096_call2_notools"
DEFAULT_ARM = "ollama_default_call2_notools"
# Register rows the scenario screen cites. Read verbatim from the register at build time.
SCENARIO_REGISTER_IDS = ["R2-real-v1-first-events", "R2-real-v1-gated-kill", "R2-real-v1-kill-criterion",
                         "R2-real-v1-survival", "R2-mechanism-verdict", "R2-mechanism-lowlevel"]

# configs/hardware/<name>.yaml -> the hw_id that result files use. evo-x2 is the 128 GB Strix Halo
# (docs/HARDWARE.md: 64 GB BIOS-reserved for the iGPU + 63.6 GB Windows-visible).
MACHINE_OF_HARDWARE = {"evo_t2s": "evo-t2s", "evox2_strix_halo_128gb": "evo-x2"}

# Workload pack family -> outcome-table family / scenario it maps to (the form's example picker).
WORKLOAD_FAMILIES = {
    "a_longdoc_qa.jsonl": ("Long-document QA", "longdoc_qa"),
    "b_function_calling.jsonl": ("Function calling", "function_calling"),
    "c_gsm8k.jsonl": ("Grade-school math", "gsm8k"),
    "d_r2_sessions.jsonl": ("Multi-turn agent session (R2)", "r2_session"),
    "e_trace_mix.jsonl": ("Agent trace length mix", "trace_length_mix"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    sys.path.insert(0, str(REPO / "harness"))
    try:
        from proc_util import check_output_hidden
        return check_output_hidden(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def parse_register(path: Path) -> dict:
    cols = ["id", "status", "value", "n", "reported", "files", "script", "commit", "date"]
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| claim id") or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split(" | ")]
        if len(parts) != len(cols):
            continue
        row = dict(zip(cols, parts))
        row["files"] = [f.strip() for f in row["files"].split(";") if f.strip()]
        out[row["id"]] = row
    return out


def hardware() -> list[dict]:
    out = []
    for p in sorted((REPO / "configs" / "hardware").glob("*.yaml")):
        y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        name = y.get("name", p.stem)
        out.append({"name": name, "machine": MACHINE_OF_HARDWARE.get(name),
                    "memory_gb": y.get("memory_gb"), "memory_architecture": y.get("memory_architecture"),
                    "bom_cost_usd": y.get("bom_cost_usd"), "src": p.relative_to(REPO).as_posix()})
    return out


def workloads() -> list[dict]:
    """One example per workload-pack family: its first item's prompt (shortened) is the description."""
    out = []
    for fname, (label, family) in WORKLOAD_FAMILIES.items():
        p = REPO / WORKLOAD_DIR / fname
        if not p.exists():
            continue
        first = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
        text = next((first[k] for k in ("description", "question", "prompt", "user_text", "task")
                     if isinstance(first.get(k), str)), None)
        if text is None:
            text = json.dumps({k: v for k, v in first.items() if isinstance(v, (str, int))})[:300]
        out.append({"id": family, "label": label, "item_id": first.get("item_id") or first.get("id"),
                    "description": " ".join(text.split())[:280], "src": f"{WORKLOAD_DIR}/{fname}",
                    "scenario": family == "r2_session",
                    "replay_src": R2_FILE if family == "r2_session" else None})
    return out


def _r2_harness():
    """Import the R2 harness quietly (its session generator prints progress) to rebuild each turn's task text."""
    sys.path.insert(0, str(REPO / "harness"))
    sys.path.insert(0, str(REPO))
    with contextlib.redirect_stdout(io.StringIO()):
        import x2_r2_agent
    return x2_r2_agent


# Fields the router may need; long text fields (final_text, call content) are dropped from the snapshot.
_DROP = {"final_text"}


def _trim_row(r: dict) -> dict:
    out = {k: v for k, v in r.items() if k not in _DROP}
    if isinstance(out.get("calls"), list):
        out["calls"] = [{k: v for k, v in c.items() if k not in ("content", "native_tool_calls")}
                        for c in out["calls"]]
    return out


def r2_sessions() -> dict:
    path = REPO / R2_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [(i + 1, json.loads(l)) for i, l in enumerate(lines)]
    start = next(r for _, r in rows if r.get("record") == "run_start")
    a = _r2_harness()
    advertised = {}
    for ln, r in rows:
        if r.get("record") == "r2a_session" and r["arm_id"] == DEFAULT_ARM:
            advertised.setdefault(r["model_id"], {"ctx": max(r["loaded_context_values"]), "line": ln})
    sessions = []
    with contextlib.redirect_stdout(io.StringIO()):
        for ln, s in rows:
            if s.get("record") != "r2a_session" or s["arm_id"] != SCENARIO_ARM:
                continue
            ags = a.build_agent_session(s["seed"], s["n_turns"])
            sys_tokens_est = a.est_tokens(ags.system_prompt)
            turns = []
            for tl, t in rows:
                if not (t.get("record") == "r2a_turn" and t["attempt_id"] == s["attempt_id"]):
                    continue
                spec_turn = ags.spec.turns[t["turn_idx"] - 1]
                row = _trim_row(t)
                row.update({
                    "line": tl,
                    "task_text": spec_turn.user_text,
                    "task_words": len(spec_turn.user_text.split()),
                    # derived: harness est_tokens(system prompt) x the session's own calibration ratio
                    "system_prompt_tokens_calibrated": round(sys_tokens_est * t["token_calib_ratio"]),
                    "answer_budget_tokens": start["max_tokens_per_call"],
                    "rules_in_use": start["rules_in_use"][t["model_id"]],
                    "turn_failures": a.turn_failures(t, start["rules_in_use"]),
                })
                turns.append(row)
            turns.sort(key=lambda r: r["turn_idx"])
            sessions.append({"key": f"{s['model_id']}|{s['seed']}", "model": s["model_id"], "seed": s["seed"],
                             "arm_id": s["arm_id"], "num_ctx": s["num_ctx_requested"], "summary": s,
                             "summary_line": ln, "turns": turns,
                             "advertised_ctx": advertised.get(s["model_id"])})
    return {"source": R2_FILE, "arm": SCENARIO_ARM, "rules_in_use": start["rules_in_use"],
            "max_tokens_per_call": start["max_tokens_per_call"], "run_start_line": next(
                ln for ln, r in rows if r.get("record") == "run_start"), "sessions": sessions}


def mitigation(register: dict) -> dict:
    rows = {k: v for k, v in register.items() if "mitigation" in k.lower()}
    synced = (REPO / MITIGATION_FILE).exists()
    measured = synced and any(v["status"] not in ("PENDING", "UNSUPPORTED") for v in rows.values())
    return {"file": MITIGATION_FILE, "synced": synced, "measured": measured, "register_ids": sorted(rows),
            "label": None if measured else "projected, pending mitigation run (x2_r2_mitigation_v1)"}


def _repo_rel(path: str) -> str:
    try:
        return Path(path).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def points(backend) -> dict:
    files = [REPO / f for f in OUTCOME_FILES if (REPO / f).exists()]
    # The agreed load_points interface takes (machine, path) pairs; the machine comes from the file-name prefix.
    from demo.dashboard.fakes import machine_for_file
    pts = backend.pareto.load_points([(machine_for_file(f), f) for f in files], cloud_source="stub")
    as_dict = [dataclasses.asdict(p) if dataclasses.is_dataclass(p) else dict(vars(p)) for p in pts]
    for d in as_dict:  # repo-relative, so the committed cache carries no local user path
        if d.get("source_files"):
            d["source_files"] = [_repo_rel(s) for s in d["source_files"]]
    return {"files": [f.relative_to(REPO).as_posix() for f in files], "backend": backend.pareto_source,
            "points": as_dict}


def build(out_dir: Path = CACHE, backend=None) -> dict:
    from demo.dashboard import data
    backend = backend or data.load_backend()
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = parse_register(REPO / REGISTER)
    snap = {
        "register.json": {k: reg[k] for k in sorted(reg)},
        "hardware.json": hardware(),
        "workloads.json": workloads(),
        "r2_sessions.json": r2_sessions(),
        "points.json": points(backend),
        "mitigation.json": mitigation(reg),
    }
    sources = [REGISTER, R2_FILE, *OUTCOME_FILES, "results/x2_chat_template_sources.jsonl"]
    sources += [h["src"] for h in snap["hardware.json"]] + [w["src"] for w in snap["workloads.json"]]
    if (REPO / MITIGATION_FILE).exists():
        sources.append(MITIGATION_FILE)
    manifest = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "backend": {"router": backend.router_source, "pareto": backend.pareto_source},
                "sources": {s: {"sha256": sha256(REPO / s), "bytes": (REPO / s).stat().st_size}
                            for s in dict.fromkeys(sources) if (REPO / s).exists()},
                "scenario_register_ids": SCENARIO_REGISTER_IDS}
    snap["manifest.json"] = manifest
    for name, obj in snap.items():
        (out_dir / name).write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build()
    print(f"cache written to {CACHE} (router={m['backend']['router']}, pareto={m['backend']['pareto']}, "
          f"{len(m['sources'])} source files)")
