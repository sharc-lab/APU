"""The Blade's own job queue: a local, in-order runner for one night's jobs, with resume. Not evo-x2's queue
(t2s_queue / queue_state.json) and not a Windows scheduled task: scripts/blade_night.py starts it, in-process.

State: C:\\apu\\blade\\blade_queue_state.json, {"nights": {"1": {"jobs": {job_id: {"status", "rc", "attempts",
"started_utc", "ended_utc", "log"}}}}}. Status: pending -> running -> done | error | blocked_dependency. A job found
"running" at start (the night was interrupted) is run again; every job resumes inside its own output file (completed
sessions / halves are skipped), so a rerun never repeats finished work. done and blocked_dependency are not rerun;
error is rerun on the next start of that night.

Paused condition (checked before the night and again before every job): the operator's flag
C:\\apu\\blade\\CLAUDE_CODE_PAUSED.flag exists (all local Claude Code work paused), and no other heavy process runs
(python / pytest / git / llama-server outside this process tree). Deleting the flag mid-night stops the queue before
its next job (a graceful stop; the running job finishes).

Each job runs as `<python> <argv...>` from the repo root via proc_util.popen_hidden (no console window), with stdout
and stderr in C:\\apu\\blade\\logs\\<job_id>.log.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import blade_common as bc  # noqa: E402

STATE_FILE = bc.BLADE_DIR / "blade_queue_state.json"
PAUSED_FLAG = bc.BLADE_DIR / "CLAUDE_CODE_PAUSED.flag"
HEAVY_NAMES = {"python.exe", "pythonw.exe", "py.exe", "pytest.exe", "git.exe", "llama-server.exe"}
EXIT_BLOCKED = 3

K1_SUMMARY = "results/blade_k1_v1.summary.json"
R2_VALIDATION = "results/blade_r2_validation_v1.jsonl"
R2_VALIDATION_QWEN = "results/blade_r2_validation_qwen3_8b_v1.jsonl"
# Every measurement job names the side-by-side Ollama 0.34.4 explicitly (never the tray install); no real job carries
# --allow-version-mismatch (that flag is for dry runs only, and the jobs refuse it outside one).
OLLAMA = ["--ollama-exe", bc.PINNED["ollama_exe"]]
_LLAMA = ["--models", "llama3.1:8b", *OLLAMA]
_QWEN = ["--models", "", "--models-if-fit", "qwen3:8b", "--k1-summary", K1_SUMMARY, *OLLAMA]

NIGHTS = {
    # night 1: K1, then llama3.1:8b R2 (validation and controls, the three tiers), then the mitigation
    1: [
        {"id": "blade_k1_v1", "argv": ["harness/blade_k1.py", "--out", "results/blade_k1_v1.jsonl", *OLLAMA],
         "outputs": ["results/blade_k1_v1.jsonl", K1_SUMMARY]},
        {"id": "blade_r2_validation_v1",
         "argv": ["harness/blade_r2.py", "--mode", "validation", "--out", R2_VALIDATION, *_LLAMA],
         "outputs": [R2_VALIDATION]},
        {"id": "blade_r2_real_v1",
         "argv": ["harness/blade_r2.py", "--mode", "real", "--out", "results/blade_r2_real_v1.jsonl", *_LLAMA,
                  "--tiers", "default,4096,32768", "--rules-from", R2_VALIDATION, "--require-validation-gates"],
         "outputs": ["results/blade_r2_real_v1.jsonl"]},
        {"id": "blade_r2_mitigation_v1",
         "argv": ["harness/blade_r2.py", "--mode", "mitigation", "--out", "results/blade_r2_mitigation_v1.jsonl",
                  *_LLAMA, "--k1-summary", K1_SUMMARY, "--tiers", "default,4096,8192", "--client-trim", "margin=0.05",
                  "--rules-from", R2_VALIDATION, "--require-validation-gates"],
         "outputs": ["results/blade_r2_mitigation_v1.jsonl"]},
    ],
    # night 2: C3 (both halves, operator-gated, half B always) then the mechanism run
    2: [
        {"id": "blade_c3_sysmem_fallback_v1",
         "argv": ["harness/blade_c3.py", "--out", "results/blade_c3_sysmem_fallback_v1.jsonl"],
         "outputs": ["results/blade_c3_sysmem_fallback_v1.jsonl"], "operator_gated": True},
        {"id": "blade_r2_mechanism_v1",
         "argv": ["harness/blade_r2.py", "--mode", "mechanism", "--out", "results/blade_r2_mechanism_v1.jsonl",
                  *_LLAMA, "--tiers", "default,32768,16384,8192,4096"],
         "outputs": ["results/blade_r2_mechanism_v1.jsonl"]},
    ],
    # night 3: qwen3:8b R2, only if K1 shows it fully on the GPU at every context the job uses; otherwise each job
    # writes "skipped: does not fit" and exits 0
    3: [
        {"id": "blade_r2_validation_qwen3_8b_v1",
         "argv": ["harness/blade_r2.py", "--mode", "validation", "--out", R2_VALIDATION_QWEN, *_QWEN],
         "outputs": [R2_VALIDATION_QWEN]},
        {"id": "blade_r2_real_qwen3_8b_v1",
         "argv": ["harness/blade_r2.py", "--mode", "real", "--out", "results/blade_r2_real_qwen3_8b_v1.jsonl", *_QWEN,
                  "--tiers", "default,4096,32768", "--rules-from", R2_VALIDATION_QWEN, "--require-validation-gates"],
         "outputs": ["results/blade_r2_real_qwen3_8b_v1.jsonl"]},
    ],
}


# ── state ───────────────────────────────────────────────────────────────────────────────────────────

def load_state(path=None) -> dict:
    path = Path(path or STATE_FILE)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"nights": {}}


def save_state(state: dict, path=None):
    path = Path(path or STATE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def init_night(state: dict, night: int) -> dict:
    jobs = state.setdefault("nights", {}).setdefault(str(night), {}).setdefault("jobs", {})
    for j in NIGHTS[night]:
        jobs.setdefault(j["id"], {"status": "pending", "rc": None, "attempts": 0})
    return state


# ── paused condition ────────────────────────────────────────────────────────────────────────────────

def own_tree(procs: list[dict], pid: int) -> set:
    """This process, its ancestors (e.g. the py.exe launcher) and its descendants (the night's job children)."""
    by = {p["pid"]: p for p in procs}
    down = {pid}
    changed = True
    while changed:
        changed = False
        for p in procs:
            if p.get("ppid") in down and p["pid"] not in down:
                down.add(p["pid"])
                changed = True
    up = set()
    cur = by.get(pid)
    while cur is not None and cur.get("ppid") in by and cur["ppid"] not in up | down:
        up.add(cur["ppid"])
        cur = by.get(cur["ppid"])
    return up | down


def heavy_processes(procs: list[dict], my_pid: int) -> list[dict]:
    """Heavy = python / pytest / git / a harness llama-server, outside this process's own ancestors and children."""
    mine = own_tree(procs, my_pid)
    out = []
    for p in procs:
        if p["pid"] in mine:
            continue
        n = p["name"].lower()
        if n in HEAVY_NAMES and not bc.is_ollama_process(p):
            out.append({"pid": p["pid"], "name": p["name"], "cmd": (p.get("cmd") or "")[:200]})
    return out


def paused_condition(flag=None, procs_fn=None, my_pid=None) -> dict:
    flag = Path(flag or PAUSED_FLAG)
    procs = (procs_fn or bc.list_processes)()
    heavy = heavy_processes(procs, my_pid or os.getpid())
    reasons = []
    if not flag.exists():
        reasons.append(f"no {flag} (create it once all local Claude Code work is paused)")
    if heavy:
        reasons.append("other heavy processes running: " + ", ".join(f"{h['name']}({h['pid']})" for h in heavy))
    return {"ok": not reasons, "reasons": reasons, "heavy": heavy, "flag": str(flag)}


# ── runner ──────────────────────────────────────────────────────────────────────────────────────────

def run_job_subprocess(job: dict, log_path: Path, python=None, popen=None) -> int:
    import subprocess
    if popen is None:
        from proc_util import popen_hidden as popen
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as fh:
        p = popen([python or sys.executable, *job["argv"]], cwd=str(bc.REPO), stdout=fh, stderr=subprocess.STDOUT,
                  stdin=subprocess.DEVNULL)
        return p.wait()


def run_night(night: int, runner=None, state_path=None, paused_fn=None, log=print, now=bc.utc_iso) -> dict:
    """Runs the night's jobs in order with resume. runner(job, log_path) -> return code. Returns the summary."""
    runner = runner or run_job_subprocess
    paused_fn = paused_fn or paused_condition
    state = init_night(load_state(state_path), night)
    save_state(state, state_path)
    jobs_state = state["nights"][str(night)]["jobs"]
    summary = {"night": night, "jobs": [], "stopped_early": None}
    for job in NIGHTS[night]:
        js = jobs_state[job["id"]]
        if js["status"] in ("done", "blocked_dependency"):
            log(f"{job['id']}: {js['status']} earlier, skipped")
            summary["jobs"].append({"id": job["id"], "status": js["status"], "skipped": True})
            continue
        cond = paused_fn()
        if not cond["ok"]:
            summary["stopped_early"] = {"before": job["id"], "reasons": cond["reasons"]}
            log(f"queue stopped before {job['id']}: {cond['reasons']}")
            break
        log_path = bc.LOG_DIR / f"{job['id']}.log"
        js.update({"status": "running", "attempts": js.get("attempts", 0) + 1, "started_utc": now(),
                   "log": str(log_path)})
        save_state(state, state_path)
        log(f"{job['id']}: start (attempt {js['attempts']})")
        try:
            rc = runner(job, log_path)
        except Exception as e:
            rc = f"runner exception {e!r}"[:200]
        js["rc"] = rc
        js["status"] = "done" if rc == 0 else ("blocked_dependency" if rc == EXIT_BLOCKED else "error")
        js["ended_utc"] = now()
        save_state(state, state_path)
        log(f"{job['id']}: {js['status']} (rc {rc})")
        summary["jobs"].append({"id": job["id"], "status": js["status"], "rc": rc, "outputs": job["outputs"],
                                "log": str(log_path)})
    return summary
