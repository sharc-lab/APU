"""R2 window result at 10 seeds on the v1b harness blob (docs/FINDINGS.md 2026-10-22, "PRE-REGISTRATION: R2 window
result at 10 seeds on the v1b harness blob ... with a reproduction check first"). Queue id x2_r2_v1b_seeds_06_10.

Deployed into C:\\apu\\ovn\\r2v1b next to the frozen agent (harness/frozen_v1b/x2_r2_agent.py, git blob a4c4a928, the
bytes v1b ran). This launcher:
  1. puts its own directory first and C:\\apu\\ovn second on sys.path and refuses (no model call) unless the side copy
     of x2_r2_agent.py has blob V1B_AGENT_BLOB and every module in DEP_BLOBS resolves to C:\\apu\\ovn with that blob;
  2. stage 1, reproduction check: runs the agent's own main() on one v1b session (REPRO: llama3.1:8b,
     ollama_ctx_4096_call2_notools, seed 20260904, 40 turns) into results/x2_r2_v1b_repro_check.jsonl and compares
     final_text turn by turn with the same session in results/x2_r2_real_v1b.jsonl; any difference, missing turn,
     error or refusal fails it, and the job stops there (a repro_check record, note "stopped: reproduction check
     failed ...", no new seed);
  3. stage 2: the agent's own main() with strong_plan narrowed to TIERS x NEW_SEEDS x 40 turns for llama3.1:8b and
     qwen3:14b into results/x2_r2_v1b_seeds_06_10.jsonl;
  4. calls t2s_queue.advance() once, in its finally block (the agent is called with advance=False).
The agent itself is not modified. Two things are added around it: strong_plan is replaced by a narrowed version (the
blob has no --tiers/--seeds), and the run_start record of both output files gets the provenance fields (agent and
launcher blobs, module blobs) through a json shim in the agent's namespace that touches no other record.

    python r2v1b\\x2_r2_v1b_seeds.py               the job
    python r2v1b\\x2_r2_v1b_seeds.py --check       provenance check, both plans and both argvs, v1b reference session
                                                  found; no server, no model call, no queue access
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVN = HERE.parent
RESULTS = OVN / "results"

V1B_AGENT_BLOB = "a4c4a928fc7a9b6ac985d473b789b41b9a4b1c12"
# Blobs of the modules the agent imports from C:\apu\ovn, as the 2026-10-12 provenance manifest recorded them
# (results/t2s_k2_pause_resume_20261012T021512Z_manifest.json, the first job after v1b and the mitigation run).
DEP_BLOBS = {
    "t2s_r2_session_growth": "878952c7d7d2e45fa602c25c34d7a09502b20262",
    "host_config": "3e542ba8e270ced2a2fe8222c27402d9ceaa259c",
    "x2_r2_client_trim": "c96930d7e1ea7f8fea83d53ad23b126623875dfa",
    "x2_r2_mechanism": "a80926e3cce616caff2eda6e57d7520e4986ea2f",
    "t2s_queue": "22398580630cadd0248513660b2ceeafc7ad7dea",
    "proc_util": "74cfea98f47a3e8bf289cae4b8f6216d711b2751",
    "t2s_k1_ollama": "5857352ac205e21dd283bbb6ce1805a6d6830b1b",
}
SFX = "_call2_notools"
MODELS = ("llama3.1:8b", "qwen3:14b")
TIERS = ("ollama_default", "ollama_ctx_16384", "ollama_ctx_8192", "ollama_ctx_4096")
NEW_SEEDS = (20260906, 20260907, 20260908, 20260909, 20260910)
TURNS = 40
REPRO = {"model": "llama3.1:8b", "arm": "ollama_ctx_4096" + SFX, "seed": 20260904}
V1B_FILE = "x2_r2_real_v1b.jsonl"
REPRO_OUT = "x2_r2_v1b_repro_check.jsonl"
OUT = "x2_r2_v1b_seeds_06_10.jsonl"
VALIDATION = "x2_r2_validation_v2.jsonl"


def git_blob_sha1(path) -> str | None:
    try:
        b = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def setup_path(here: Path = HERE) -> None:
    front = [str(here), str(here.parent)]
    sys.path[:] = front + [p for p in sys.path if p not in front]


def provenance(ag, here: Path = HERE, ovn: Path = OVN, dep_blobs=None, agent_blob=V1B_AGENT_BLOB) -> dict:
    """Blobs of the side agent and of each dependency (imported now); problems lists every mismatch."""
    import importlib
    dep_blobs = DEP_BLOBS if dep_blobs is None else dep_blobs
    problems = []
    a_file = Path(ag.__file__).resolve()
    a_blob = git_blob_sha1(a_file)
    if a_file.parent != here.resolve():
        problems.append(f"x2_r2_agent imported from {a_file}, expected {here}")
    if a_blob != agent_blob:
        problems.append(f"x2_r2_agent blob {a_blob} != {agent_blob}")
    mods = {}
    for name, want in dep_blobs.items():
        m = importlib.import_module(name)
        f = Path(m.__file__).resolve()
        b = git_blob_sha1(f)
        mods[name] = {"file": str(f), "blob": b}
        if f.parent != ovn.resolve():
            problems.append(f"{name} imported from {f}, expected {ovn}")
        if b != want:
            problems.append(f"{name} blob {b} != {want}")
    return {"agent_file": str(a_file), "agent_blob": a_blob, "launcher_file": str(Path(__file__).resolve()),
            "launcher_blob": git_blob_sha1(__file__), "modules": mods, "problems": problems,
            "harness": "x2_r2_agent at the v1b blob via x2_r2_v1b_seeds"}


class _JsonShim(types.SimpleNamespace):
    """The agent's `json` with dumps() adding `extra` to run_start records only."""

    def __init__(self, real, extra):
        super().__init__(**{k: getattr(real, k) for k in dir(real) if not k.startswith("__") and k != "dumps"})
        self._real, self._extra = real, extra

    def dumps(self, obj, *a, **k):
        if isinstance(obj, dict) and obj.get("record") == "run_start":
            obj = {**obj, **self._extra}
        return self._real.dumps(obj, *a, **k)


def narrowed_plan(ag, tiers, seeds, turns=TURNS):
    """A replacement for the agent's strong_plan returning only the named tiers and seeds."""
    def plan(call2_tools="off", **_ignored):
        sfx = ag.MODE_SUFFIX[ag.as_mode(call2_tools)]
        return [(t + sfx, tuple(seeds), turns) for t in tiers]
    return plan


def argv_for(models, out: Path, results: Path = RESULTS) -> list[str]:
    return ["--mode", "real", "--plan", "strong", "--call2-tools", "off", "--rules-from", str(results / VALIDATION),
            "--require-validation-gates", "--per-model-refusal", "--order", "seed_major", "--models", ",".join(models),
            "--out", str(out)]


def session_texts(rows, ag, model, arm, seed) -> dict | None:
    """{turn_idx: final_text} of the last completed session of (model, arm, seed), None when there is none."""
    s = [x for x in ag.completed_sessions(rows) if x["model_id"] == model and x["arm_id"] == arm
         and x["seed"] == seed and x.get("mode") == "real"]
    if not s:
        return None
    att = s[-1]["attempt_id"]
    return {r["turn_idx"]: r.get("final_text") for r in rows if r.get("record") == "r2a_turn"
            and r.get("attempt_id") == att}


def compare_repro(ref: dict | None, new: dict | None, turns=TURNS) -> dict:
    """The pre-registered pass rule: both sessions complete with turns 1..turns and every final_text byte-identical."""
    if ref is None or new is None:
        return {"pass": False, "reason": "reference session missing" if ref is None else "rerun session missing",
                "mismatch_turns": [], "n_identical": 0, "n_turns": turns}
    want = list(range(1, turns + 1))
    missing = [t for t in want if t not in ref or t not in new]
    mism = [t for t in want if t in ref and t in new and ref[t] != new[t]]
    first = mism[0] if mism else None
    return {"pass": not missing and not mism, "reason": None if not missing and not mism else
            (f"turns missing {missing}" if missing else f"{len(mism)} turns differ"),
            "mismatch_turns": mism, "missing_turns": missing, "n_identical": len(want) - len(mism) - len(missing),
            "n_turns": turns, "first_mismatch": None if first is None else
            {"turn": first, "v1b": str(ref[first])[:400], "rerun": str(new[first])[:400]}}


def file_sha256(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def run_job(ag, results: Path = RESULTS, log=print, prov=None) -> str:
    """Stages 1 and 2 (the caller handles the queue). Returns the exit note."""
    prov = prov if prov is not None else provenance(ag)
    if prov["problems"]:
        return ("stopped: provenance check failed: " + "; ".join(prov["problems"]))[:400]
    real_json = ag.json
    orig_plan = ag.strong_plan
    try:
        v1b_path = results / V1B_FILE
        extra = {"harness_provenance": prov, "launcher_stage": "repro_check",
                 "v1b_reference": {"file": str(v1b_path), "sha256": file_sha256(v1b_path), **REPRO}}
        ag.json = _JsonShim(real_json, extra)
        ag.strong_plan = narrowed_plan(ag, (REPRO["arm"][:-len(SFX)],), (REPRO["seed"],))
        rep_out = results / REPRO_OUT
        note1 = ag.main(argv_for([REPRO["model"]], rep_out, results), advance=False)
        ref = session_texts(ag.read_rows(v1b_path), ag, REPRO["model"], REPRO["arm"], REPRO["seed"])
        new = session_texts(ag.read_rows(rep_out), ag, REPRO["model"], REPRO["arm"], REPRO["seed"])
        cmp = compare_repro(ref, new)
        if note1 != "completed":
            cmp = dict(cmp, **{"pass": False, "reason": f"rerun note {note1!r}"})
        with open(rep_out, "a", encoding="utf-8") as f:
            f.write(real_json.dumps({"record": "repro_check", **REPRO, "agent_note": note1, **cmp,
                                     "ts_utc": ag.utc_iso()}, default=str) + "\n")
        log(f"reproduction check: pass={cmp['pass']} identical {cmp['n_identical']}/{cmp['n_turns']} "
            f"reason={cmp['reason']}")
        if not cmp["pass"]:
            return (f"stopped: reproduction check failed ({cmp['reason']}; identical {cmp['n_identical']}/"
                    f"{cmp['n_turns']}); no new seed run")[:400]
        ag.json = _JsonShim(real_json, {"harness_provenance": prov, "launcher_stage": "new_seeds",
                                        "repro_check_passed": True})
        ag.strong_plan = narrowed_plan(ag, TIERS, NEW_SEEDS)
        return ag.main(argv_for(MODELS, results / OUT, results), advance=False)
    finally:
        ag.json = real_json
        ag.strong_plan = orig_plan


def check(ag, results: Path = RESULTS) -> dict:
    prov = provenance(ag)
    out = {"provenance": prov}
    for name, (tiers, seeds, models, o) in {
            "repro": ((REPRO["arm"][:-len(SFX)],), (REPRO["seed"],), [REPRO["model"]], results / REPRO_OUT),
            "new_seeds": (TIERS, NEW_SEEDS, list(MODELS), results / OUT)}.items():
        argv = argv_for(models, o, results)
        args = ag.build_arg_parser().parse_args(argv)
        out[name] = {"argv": argv, "parsed_models": args.models,
                     "plan": narrowed_plan(ag, tiers, seeds)(args.call2_tools),
                     "num_ctx": {t + SFX: ag.ARMS[t + SFX]["num_ctx"] for t in tiers},
                     "out_exists": o.exists()}
    ref = session_texts(ag.read_rows(results / V1B_FILE), ag, REPRO["model"], REPRO["arm"], REPRO["seed"])
    out["v1b_reference"] = {"turns": sorted(ref) if ref else None, "sha256": file_sha256(results / V1B_FILE)}
    vrows = ag.read_rows(results / VALIDATION)
    pf = ag.validation_preflight_per_model([(VALIDATION, vrows)], "off", list(MODELS))
    out["preflight"] = {k: pf.get(k) for k in ("ok", "models_ok", "refused", "rules_in_use")}
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    setup_path()
    import x2_r2_agent as ag
    if argv[:1] == ["--check"]:
        print(json.dumps(check(ag), indent=1, default=str))
        return 0
    note = "stopped: launcher did not finish"
    try:
        note = run_job(ag)
    except Exception as e:
        import traceback
        note = f"stopped: {e!r}"[:400]
        traceback.print_exc()
    finally:
        print(f"[{ag.utc_iso()}] x2_r2_v1b_seeds note: {note}", flush=True)
        try:
            import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
