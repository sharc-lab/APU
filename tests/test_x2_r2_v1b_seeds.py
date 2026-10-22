"""x2_r2_v1b_seeds (docs/FINDINGS.md 2026-10-22, R2 at 10 seeds on the v1b blob with a reproduction check first).
Fakes only: no server, no model call."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
import x2_r2_v1b_seeds as vs  # noqa: E402

FROZEN = REPO / "harness" / "frozen_v1b" / "x2_r2_agent.py"


def _frozen_agent():
    spec = importlib.util.spec_from_file_location("x2_r2_agent_v1b_frozen", FROZEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses look the module up while the class body runs
    spec.loader.exec_module(mod)
    return mod


def test_frozen_copy_is_the_v1b_blob_and_matches_the_2026_10_12_manifest():
    assert vs.git_blob_sha1(FROZEN) == vs.V1B_AGENT_BLOB
    man = json.loads((REPO / "results" / "t2s_k2_pause_resume_20261012T021512Z_manifest.json").read_text(
        encoding="utf-8"))
    blobs = {f["path"]: f["actual_git_blob_sha"] for f in man["script_provenance"]["files"]}
    assert blobs["x2_r2_agent.py"] == vs.V1B_AGENT_BLOB
    for name, want in vs.DEP_BLOBS.items():
        assert blobs[name + ".py"] == want, name


def test_narrowed_plan_and_argv_on_the_frozen_agent():
    ag = _frozen_agent()
    plan = vs.narrowed_plan(ag, vs.TIERS, vs.NEW_SEEDS)("off")
    assert plan == [(t + "_call2_notools", vs.NEW_SEEDS, 40) for t in vs.TIERS]
    assert [ag.ARMS[a]["num_ctx"] for a, _, _ in plan] == [None, 16384, 8192, 4096]
    rp = vs.narrowed_plan(ag, ("ollama_ctx_4096",), (20260904,))("off")
    assert rp == [("ollama_ctx_4096_call2_notools", (20260904,), 40)]
    args = ag.build_arg_parser().parse_args(vs.argv_for(vs.MODELS, Path("o.jsonl"), Path("r")))
    assert args.mode == "real" and args.plan == "strong" and args.call2_tools == "off"
    assert args.require_validation_gates and args.per_model_refusal and args.order == "seed_major"
    assert args.models == "llama3.1:8b,qwen3:14b" and args.rules_from.endswith("x2_r2_validation_v2.jsonl")
    assert not hasattr(args, "seeds") and not hasattr(args, "tiers")     # the blob has no such options


def test_compare_repro_rules():
    ref = {t: f"a{t}" for t in range(1, 41)}
    assert vs.compare_repro(ref, dict(ref))["pass"] is True
    bad = {**ref, 7: "x"}
    c = vs.compare_repro(ref, bad)
    assert not c["pass"] and c["mismatch_turns"] == [7] and c["first_mismatch"]["turn"] == 7
    short = {t: v for t, v in ref.items() if t < 40}
    assert not vs.compare_repro(ref, short)["pass"]
    assert not vs.compare_repro(None, ref)["pass"] and not vs.compare_repro(ref, None)["pass"]


def test_json_shim_touches_run_start_only():
    sh = vs._JsonShim(json, {"agent_blob": "abc"})
    assert json.loads(sh.dumps({"record": "run_start", "x": 1}))["agent_blob"] == "abc"
    assert "agent_blob" not in json.loads(sh.dumps({"record": "r2a_turn"}))
    assert sh.loads('{"a": 1}') == {"a": 1}


def test_provenance_flags_wrong_agent_blob(tmp_path):
    fake = types.SimpleNamespace(__file__=str(FROZEN))
    p = vs.provenance(fake, here=FROZEN.parent, ovn=REPO / "harness", dep_blobs={})
    assert p["agent_blob"] == vs.V1B_AGENT_BLOB and p["problems"] == []
    other = tmp_path / "x2_r2_agent.py"
    other.write_text("print(1)\n", encoding="utf-8")
    p2 = vs.provenance(types.SimpleNamespace(__file__=str(other)), here=tmp_path, ovn=REPO / "harness", dep_blobs={})
    assert any("blob" in x for x in p2["problems"])


class _FakeAg:
    """Minimal stand-in for the agent: main() writes a completed session whose final texts come from `texts`."""

    def __init__(self, v1b_texts, rerun_texts, note="completed"):
        self.json = json
        self.MODE_SUFFIX = {"off": "_call2_notools"}
        self.calls = []
        self.v1b_texts, self.rerun_texts, self.note = v1b_texts, rerun_texts, note

    def as_mode(self, c):
        return c

    def strong_plan(self, c2="off"):
        raise AssertionError("must be replaced")

    def utc_iso(self):
        return "2026-10-22T00:00:00+00:00"

    def read_rows(self, p):
        p = Path(p)
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []

    def completed_sessions(self, rows):
        return [r for r in rows if r.get("record") == "r2a_session"]

    def main(self, argv, advance=True):
        assert advance is False
        out = Path(argv[argv.index("--out") + 1])
        plan = self.strong_plan("off")
        self.calls.append((argv, plan))
        rows = [{"record": "run_start", "plan": plan}]
        if "repro" in out.name:
            for t, txt in self.rerun_texts.items():
                rows.append({"record": "r2a_turn", "attempt_id": "n", "turn_idx": t, "final_text": txt})
            rows.append({"record": "r2a_session", "attempt_id": "n", "mode": "real", **{
                "model_id": vs.REPRO["model"], "arm_id": vs.REPRO["arm"], "seed": vs.REPRO["seed"]}})
        with open(out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(self.json.dumps(r) + "\n")
        return self.note


def _v1b_file(results, texts):
    rows = [{"record": "r2a_turn", "attempt_id": "o", "turn_idx": t, "final_text": x} for t, x in texts.items()]
    rows.append({"record": "r2a_session", "attempt_id": "o", "mode": "real", "model_id": vs.REPRO["model"],
                 "arm_id": vs.REPRO["arm"], "seed": vs.REPRO["seed"]})
    (results / vs.V1B_FILE).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


PROV = {"problems": [], "agent_blob": vs.V1B_AGENT_BLOB}


def test_run_job_passes_repro_then_runs_new_seeds(tmp_path):
    texts = {t: f"answer {t}" for t in range(1, 41)}
    _v1b_file(tmp_path, texts)
    ag = _FakeAg(texts, dict(texts))
    note = vs.run_job(ag, results=tmp_path, log=lambda m: None, prov=PROV)
    assert note == "completed" and len(ag.calls) == 2
    assert ag.calls[0][1] == [(vs.REPRO["arm"], (vs.REPRO["seed"],), 40)]
    assert ag.calls[1][1] == [(t + "_call2_notools", vs.NEW_SEEDS, 40) for t in vs.TIERS]
    rep = ag.read_rows(tmp_path / vs.REPRO_OUT)
    assert rep[0]["harness_provenance"] == PROV and rep[0]["launcher_stage"] == "repro_check"
    assert rep[-1]["record"] == "repro_check" and rep[-1]["pass"] is True
    new = ag.read_rows(tmp_path / vs.OUT)
    assert new[0]["harness_provenance"] == PROV and new[0]["repro_check_passed"] is True
    assert ag.json is json                                     # shim removed afterwards


def test_run_job_stops_on_a_mismatch_and_runs_no_new_seed(tmp_path):
    texts = {t: f"answer {t}" for t in range(1, 41)}
    _v1b_file(tmp_path, texts)
    ag = _FakeAg(texts, {**texts, 12: "different"})
    note = vs.run_job(ag, results=tmp_path, log=lambda m: None, prov=PROV)
    assert note.startswith("stopped: reproduction check failed") and len(ag.calls) == 1
    assert not (tmp_path / vs.OUT).exists()
    rec = ag.read_rows(tmp_path / vs.REPRO_OUT)[-1]
    assert rec["record"] == "repro_check" and rec["pass"] is False and rec["mismatch_turns"] == [12]


def test_run_job_refuses_on_provenance_problems_without_any_call(tmp_path):
    ag = _FakeAg({}, {})
    note = vs.run_job(ag, results=tmp_path, prov={"problems": ["x2_r2_agent blob x != y"]})
    assert note.startswith("stopped: provenance check failed") and ag.calls == []


def test_run_job_fails_when_the_rerun_did_not_complete(tmp_path):
    texts = {t: f"answer {t}" for t in range(1, 41)}
    _v1b_file(tmp_path, texts)
    ag = _FakeAg(texts, dict(texts), note="stopped: RuntimeError('x')")
    note = vs.run_job(ag, results=tmp_path, log=lambda m: None, prov=PROV)
    assert note.startswith("stopped: reproduction check failed") and len(ag.calls) == 1
