"""apu_bench run: a device profile plus a B3 v2 subset -> result.json (apu-bench-result/1), report.md, records.jsonl.

Runtimes:
  fake  records generated offline from the task definitions (no model, no server): loaded window = the profile's,
        prompt = the task-definition projection, flagged = the projection is over the window, outcome and seconds
        from a seeded hash. Every number in a fake report is labelled FAKE and is not a measurement; the mode exists
        so the whole pipeline (schema, provenance, report) is testable offline.
  live  composes the two x2_r2_evalset.py steps (validation, then real on the profile's tiers and the subset's seeds)
        and, only with --execute, runs them in this process through x2_r2_evalset.main(argv, advance=False) (no queue
        advance, no pause), then reads the records it wrote. Never run so far. It is meant for the device itself
        (the harness starts and stops its own Ollama server); a real run needs a pre-registration first.

Metrics per window (task turns of completed sessions only, as analysis/impact_model.b3_v2_task_outcomes):
  wrong       end_state_ok is not True
  flagged     the turn's prompt exceeded the loaded window (over_loaded_window, else final_call_prompt_tokens_calibrated
              > loaded_context, analysis/impact_model.turn_over_window) or detector v2 flagged a runtime cut
              (cut_v2_flagged, when the run recorded it)
  silent      wrong and not flagged; flagged wrong = wrong and flagged
  seconds     turn_wall_s of the task turn: mean, median, p90 (statistics.quantiles n=10 inclusive)
"""
from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from apu_bench import BENCH_VERSION, RESULT_SCHEMA
from apu_bench import profile as P
from apu_bench import taskset as TS

REPO = Path(__file__).resolve().parents[1]
FAKE_LABEL = "FAKE RUNTIME: generated records, not a measurement"
LIVE_LABEL = "LIVE: measured on the device named in the profile"
HARNESS_FILES = ("harness/x2_r2_evalset.py", "harness/x2_r2_agent.py", "harness/evalset_env.py",
                 "apu_bench/bench.py", "apu_bench/taskset.py", "apu_bench/profile.py", "src/dse/fits_locally.py")


# ── provenance ───────────────────────────────────────────────────────────────────────────────────────

def git_blob(path: Path) -> str | None:
    """The git blob id of a file's bytes (what `git hash-object` prints), computed without git."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git_state(repo=REPO) -> dict:
    sys.path.insert(0, str(Path(repo) / "harness"))
    from proc_util import run_hidden
    try:
        head = run_hidden(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        dirty = run_hidden(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=30)
        return {"git_head": head.stdout.strip() or None, "git_dirty": bool(dirty.stdout.strip())}
    except (OSError, ValueError) as e:
        return {"git_head": None, "git_dirty": None, "git_error": repr(e)}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── commands ─────────────────────────────────────────────────────────────────────────────────────────

def live_steps(prof: dict, ts: dict, out_dir: Path, rules_from: str | None = None) -> list[dict]:
    """The x2_r2_evalset.py argv of each step (validation unless rules_from is given, then real)."""
    common = ["--design", ts["design"], "--workload", "evalset", "--guardrails", "off", "--models", prof["model"]["name"]]
    env = [x for k, v in sorted((prof["runtime"].get("server_env") or {}).items()) for x in ("--server-env", f"{k}={v}")]
    steps = []
    val = rules_from
    if val is None:
        val = str(Path(out_dir) / "validation.jsonl")
        steps.append({"step": "validation", "argv": ["--mode", "validation", *common, "--out", val, *env]})
    steps.append({"step": "real", "argv": ["--mode", "real", *common, "--tiers", ",".join(w["tier"] for w in prof["windows"]),
                                           "--seeds", ",".join(map(str, ts["seeds"])), "--rules-from", val,
                                           "--out", str(Path(out_dir) / "records.jsonl"), *env]})
    for s in steps:
        s["cmdline"] = ["py", "-3.12", "harness/x2_r2_evalset.py", *s["argv"]]
    return steps


def server_command(prof: dict) -> dict:
    rt = prof["runtime"]
    return {"cmd": rt.get("server_cmd") or [rt["name"], "serve"], "env": rt.get("server_env") or {},
            "note": "x2_r2_evalset starts its own server per job (OLLAMA_DEBUG=1, its own log); this is the profile's "
                    "declared command"}


# ── fake runtime ─────────────────────────────────────────────────────────────────────────────────────

def _u(*key) -> float:
    h = hashlib.sha256(json.dumps(key).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def fake_records(prof: dict, ts: dict) -> list[dict]:
    """Records in the r2a_turn / r2a_session shape x2_r2_evalset writes, generated offline (FAKE)."""
    from src.dse.fits_locally import fits_locally
    model = prof["model"]["name"]
    rows = [{"record": "run_start", "harness": "apu_bench.fake", "fake": True, "plan_name": ts["id"],
             "models": [model]},
            {"record": "runtime", "fake": True, "runtime": prof["runtime"]["name"],
             "version": prof["runtime"].get("version")}]
    for w in prof["windows"]:
        for s in ts["sessions"]:
            att = hashlib.sha256(f"{prof['profile_id']}|{w['tier']}|{s['seed']}".encode()).hexdigest()[:12]
            arm = f"{w['tier']}_call2_notools"
            for t in s["turns"]:
                if t["kind"] != "task":
                    continue
                f = fits_locally(t["projected_prompt_tokens"], w["window"])
                u = _u(prof["profile_id"], w["tier"], s["seed"], t["item_id"])
                ok = u >= (0.10 if f["fits"] else 0.60)
                rows.append({"record": "r2a_turn", "fake": True, "mode": "fake", "model_id": model, "arm_id": arm,
                             "attempt_id": att, "seed": s["seed"], "turn_idx": t["turn_idx"], "task_id": t["item_id"],
                             "item_kind": "task", "task_type": t["task_type"], "end_state_ok": ok,
                             "loaded_context": w["window"],
                             "final_call_prompt_tokens_calibrated": t["projected_prompt_tokens"],
                             "over_loaded_window": not f["fits"], "fits_source": f["source"],
                             "turn_wall_s": round(4.0 + 20.0 * _u("s", w["tier"], s["seed"], t["item_id"]), 3)})
            rows.append({"record": "r2a_session", "fake": True, "mode": "fake", "model_id": model, "arm_id": arm,
                         "attempt_id": att, "seed": s["seed"]})
    rows.append({"record": "run_end", "fake": True})
    return rows


# ── summary (shared by fake and live records) ─────────────────────────────────────────────────────────

def _p90(xs):
    xs = sorted(xs)
    return xs[0] if len(xs) == 1 else statistics.quantiles(xs, n=10, method="inclusive")[8]


def turn_flagged(r: dict) -> bool:
    if r.get("cut_v2_flagged"):
        return True
    if r.get("over_loaded_window") is not None:
        return bool(r["over_loaded_window"])
    p, w = r.get("final_call_prompt_tokens_calibrated"), r.get("loaded_context")
    return p is not None and w is not None and p > w


def summarize(records: list[dict], prof: dict, seeds) -> list[dict]:
    done = {r.get("attempt_id") for r in records if r.get("record") == "r2a_session"}
    seeds = set(seeds)
    out = []
    for w in prof["windows"]:
        arm = f"{w['tier']}_call2_notools"
        turns = [r for r in records if r.get("record") == "r2a_turn" and r.get("arm_id") == arm
                 and r.get("item_kind") == "task" and r.get("end_state_ok") is not None
                 and r.get("attempt_id") in done and r.get("seed") in seeds]
        n = len(turns)
        wrong = [r for r in turns if r["end_state_ok"] is not True]
        flagged = [r for r in turns if turn_flagged(r)]
        secs = [float(r["turn_wall_s"]) for r in turns if r.get("turn_wall_s") is not None]
        out.append({"tier": w["tier"], "window": w["window"], "n_tasks": n,
                    "loaded_windows": sorted({r.get("loaded_context") for r in turns if r.get("loaded_context")}),
                    "correct": n - len(wrong), "wrong": len(wrong), "wrong_rate": len(wrong) / n if n else None,
                    "flagged": len(flagged), "wrong_flagged": sum(1 for r in wrong if turn_flagged(r)),
                    "wrong_silent": sum(1 for r in wrong if not turn_flagged(r)),
                    "seconds_per_task": ({"mean": statistics.fmean(secs), "median": statistics.median(secs),
                                          "p90": _p90(secs)} if secs else None)})
    return out


# ── report ───────────────────────────────────────────────────────────────────────────────────────────

def report_md(res: dict) -> str:
    p, ts, pv = res["device_profile"], res["task_set"], res["provenance"]
    hw = p["hardware"]
    L = [f"# APU bench report: {p['profile_id']}", ""]
    L += [f"**{res['label']}**", ""]
    L += ["## Device", "", "| tag | value |", "|---|---|"]
    L += [f"| {k} | {hw.get(k) if hw.get(k) not in (None, '') else 'not recorded'} |" for k in P.HARDWARE_TAGS]
    L += [f"| runtime | {p['runtime']['name']} {p['runtime'].get('version') or '(version not recorded)'} |",
          f"| model | {p['model']['name']} (quant {p['model'].get('quant') or 'not recorded'}) |", ""]
    if res["missing_tags"]:
        L += [f"Missing tags (a gap for a submission): {', '.join(res['missing_tags'])}", ""]
    L += ["## Task set", "", f"{ts['id']}: {ts['description']}; design {ts['design']}; seeds "
          f"{', '.join(map(str, ts['seeds']))}; sha256 `{ts['sha256']}`", ""]
    L += ["## Results", "",
          "| window | tier | tasks | wrong | wrong rate | wrong: silent | wrong: flagged | flagged | s/task mean | s/task median | s/task p90 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["results"]:
        s = r["seconds_per_task"] or {}
        wr = f"{100 * r['wrong_rate']:.1f}%" if r["wrong_rate"] is not None else "n/a"
        L.append(f"| {r['window']} | {r['tier']} | {r['n_tasks']} | {r['wrong']} | {wr} | {r['wrong_silent']} | "
                 f"{r['wrong_flagged']} | {r['flagged']} | {s.get('mean', float('nan')):.1f} | "
                 f"{s.get('median', float('nan')):.1f} | {s.get('p90', float('nan')):.1f} |")
    L += ["", "Silent = wrong with no signal (no window overflow, no runtime cut line); flagged = wrong and the "
          "prompt was over the loaded window or the runtime logged a cut.", ""]
    L += ["## Reproduce", "", f"- git head `{pv['git_head']}` (dirty: {pv['git_dirty']})",
          f"- bench {res['bench_version']}, schema {res['schema']}, mode {res['mode']}",
          f"- profile sha256 `{res['device_profile_sha256']}`",
          f"- records sha256 `{res['records_sha256']}`",
          f"- server: `{' '.join(res['commands']['server']['cmd'])}` env {json.dumps(res['commands']['server']['env'])}"]
    for s in res["commands"]["client"]:
        L.append(f"- {s['step']}: `{' '.join(s['cmdline'])}`")
    L += [f"- harness blobs: " + ", ".join(f"{k} {v[:10] if v else 'missing'}" for k, v in pv["harness_blobs"].items()),
          ""]
    return "\n".join(L)


# ── entry ────────────────────────────────────────────────────────────────────────────────────────────

def run(profile_path, subset: str, out_dir, runtime: str = "fake", execute: bool = False, rules_from=None,
        argv=None, now=None) -> dict:
    prof, prof_sha, missing = P.load(profile_path)
    ts = TS.build(subset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = live_steps(prof, ts, out, rules_from)
    rec_path = out / "records.jsonl"
    if runtime == "fake":
        recs = fake_records(prof, ts)
        rec_path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        label = FAKE_LABEL
    elif runtime == "live":
        if not execute:
            plan = {"plan_only": True, "steps": steps, "server": server_command(prof),
                    "note": "not executed: pass --execute on the device itself, after a pre-registration"}
            (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
            return plan
        ev = TS._ev()
        for s in steps:
            rc = ev.main(s["argv"], advance=False)
            if rc not in (0, None):
                raise RuntimeError(f"x2_r2_evalset {s['step']} returned {rc}")
        recs = [json.loads(x) for x in rec_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        label = LIVE_LABEL
    else:
        raise ValueError(f"runtime must be fake or live, got {runtime!r}")
    res = {"schema": RESULT_SCHEMA, "bench_version": BENCH_VERSION,
           "created_utc": (now or datetime.now(timezone.utc)).isoformat(), "mode": runtime, "label": label,
           "device_profile": prof, "device_profile_sha256": prof_sha, "missing_tags": missing,
           "task_set": {k: v for k, v in ts.items() if k != "sessions"},
           "provenance": {**git_state(), "python": sys.version.split()[0], "platform": platform.platform(),
                          "argv": list(argv) if argv is not None else None,
                          "harness_blobs": {f: git_blob(REPO / f) for f in HARNESS_FILES}},
           "commands": {"server": server_command(prof), "client": steps},
           "run_rules": {"unit": "task turn of a completed session", "sessions_per_seed_per_window": 1,
                         "retries": 0, "scoring": "end_state_ok (x2_r2_evalset environment end state and answer)",
                         "flag": "over_loaded_window or detector v2 cut (cut_v2_flagged)"},
           "records_file": rec_path.name, "records_sha256": sha256_file(rec_path),
           "results": summarize(recs, prof, ts["seeds"])}
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report_md(res) + "\n", encoding="utf-8")
    return res
