"""
Byte-identical resume test for sdk_direct's JSONL + build_aggregate path.

Verifies that interrupting after seed K and resuming produces a
build_aggregate() output that is JSON-identical to a complete uninterrupted
run.

This tests two things:
  1. The json.loads(json.dumps(artifact)) round-trip preserves all values that
     build_aggregate touches (per_category, totals, per_task, provenance_totals).
  2. The _load_jsonl_seeds / JSONL-write bookkeeping correctly recovers the
     completed seed set, so the resumed run does not re-run completed seeds.

The fake artifact builder is deterministic per seed and returns a clean dict
(no underscore-prefixed internal fields) matching what main() writes to JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import harness.adapters.sdk_direct as m


# ── Fake artifact ─────────────────────────────────────────────────────────────

def _fake_artifact(seed: int) -> dict:
    """Minimal seed artifact with all fields that build_aggregate accesses.

    Values vary deterministically by seed so aggregates are non-trivial.
    This is a clean artifact (no _ fields) matching what is written to JSONL.
    """
    cpu_ns   = 1_000_000 + seed * 137_000     # distinct per seed, integer
    residual = cpu_ns // 8
    per_cat  = {
        "ORCH_SETUP":   {"cpu_ns": 40_000 + seed * 1_000, "wall_ns": 44_000, "count": 5,  "b_in": 0, "b_out": 0},
        "ORCH_DISPATCH": {"cpu_ns": 15_000,                "wall_ns": 16_000, "count": 5,  "b_in": 0, "b_out": 0},
        "HTTP_CLIENT":  {"cpu_ns": 0,                      "wall_ns": 700_000 + seed * 50_000, "count": 5, "b_in": 0, "b_out": 0},
        "TOOL_COMPUTE": {"cpu_ns": 80_000 + seed * 5_000,  "wall_ns": 80_000, "count": 10, "b_in": 0, "b_out": 0},
        "FRAMEWORK":    {"cpu_ns": 25_000,                 "wall_ns": 25_000, "count": 5,  "b_in": 0, "b_out": 0},
        "RESIDUAL_UNATTRIBUTED": {"cpu_ns": residual,      "wall_ns": residual, "count": 1, "b_in": 0, "b_out": 0},
    }
    return {
        "experiment":      "real_agent_breakdown",
        "result_validity": "valid",
        "generated_utc":   "2026-01-01T00:00:00+00:00",
        "setup_ref":       {},
        "config": {
            "profile":          "test",
            "payload_profile":  "test",
            "search_locality":  "local",
            "seed":             seed,
            "sessions":         1,
            "mode":             "sync/openai",
            "llm_median_scale": 1.0,
            "instr_version":    3,
        },
        "batch_wall_s": 1.0 + seed * 0.15,
        "run": {
            "env":    {},
            "config": {},
            "per_category":         per_cat,
            "per_session":          [],
            "per_session_category": {},
            "residual_cpu_ns":      residual,
            "residual_fraction":    residual / cpu_ns,
            "totals": {
                "thread_cpu_ns":       cpu_ns,
                "wall_ns":             cpu_ns * 2,
                "instrumented_cpu_ns": cpu_ns,
            },
            "os_times_user_sys": {},
            "timeline":          [],
            "provenance_totals": {
                "measured":      cpu_ns - residual,
                "step_inferred": 0,
                "residual":      residual,
            },
            "provenance_detail": {},
        },
        "per_task": {
            "CH-01": {
                "categories":          {},
                "sessions":            ["agent_0"],
                "instrumented_cpu_ns": cpu_ns,
                "amenable_strict_ns":  40_000,
                "amenable_broad_ns":   120_000,
                "amenable_strict_share": 40_000 / cpu_ns,
                "amenable_broad_share":  120_000 / cpu_ns,
            },
        },
        "per_task_wall_cpu": {},
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

_CFG_HASH = "test_cfg_abc1"   # stable sentinel; not validated in these tests


def _write_jsonl(path: Path, artifacts: list[dict]) -> None:
    """Write artifacts to JSONL as main() would (clean + config_hash)."""
    with open(path, "w", encoding="utf-8") as fout:
        for a in artifacts:
            row = {k: v for k, v in a.items() if not k.startswith("_")}
            row["config_hash"] = _CFG_HASH
            fout.write(json.dumps(row) + "\n")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_full_vs_resumed_aggregate_byte_identical(tmp_path: Path) -> None:
    """Resuming from a partial JSONL + collecting the rest produces the same
    build_aggregate() JSON as a complete uninterrupted run."""
    n_seeds = 4
    full_artifacts = [_fake_artifact(s) for s in range(n_seeds)]

    # Write all artifacts to JSONL (full run)
    jsonl = tmp_path / "full.jsonl"
    _write_jsonl(jsonl, full_artifacts)

    # Simulate interruption after seed 1: keep only first 2 lines
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    cut = 2
    assert 0 < cut < n_seeds

    jsonl_resume = tmp_path / "resume.jsonl"
    jsonl_resume.write_text("\n".join(lines[:cut]) + "\n", encoding="utf-8")

    completed_seeds, loaded = m._load_jsonl_seeds(jsonl_resume, _CFG_HASH)
    assert completed_seeds == {0, 1}
    assert len(loaded) == 2

    # "Fresh" collection of remaining seeds (deterministic, same values as full run)
    fresh = [_fake_artifact(s) for s in range(n_seeds) if s not in completed_seeds]

    # Resume raw_artifacts: loaded (json-round-tripped) + freshly collected
    resume_raw = list(loaded) + fresh
    full_raw   = full_artifacts

    agg_full   = m.build_aggregate(full_raw)
    agg_resume = m.build_aggregate(resume_raw)

    full_json   = json.dumps(agg_full)
    resume_json = json.dumps(agg_resume)

    assert full_json == resume_json, (
        "build_aggregate output differs after resume.\n"
        "This means either the json round-trip changed a value, or the "
        "aggregate computation is sensitive to internal vs. clean artifact shape.\n"
        f"Full   (first 400): {full_json[:400]}\n"
        f"Resume (first 400): {resume_json[:400]}"
    )


def test_load_jsonl_seeds_recovers_correct_set(tmp_path: Path) -> None:
    """_load_jsonl_seeds correctly identifies completed seed indices."""
    artifacts = [_fake_artifact(s) for s in range(5)]
    jsonl = tmp_path / "seeds.jsonl"
    _write_jsonl(jsonl, artifacts)

    completed, loaded = m._load_jsonl_seeds(jsonl, _CFG_HASH)
    assert completed == {0, 1, 2, 3, 4}
    assert len(loaded) == 5
    assert [a["config"]["seed"] for a in loaded] == list(range(5))


def test_load_jsonl_seeds_config_hash_mismatch(tmp_path: Path) -> None:
    """_load_jsonl_seeds raises ValueError on a config hash mismatch."""
    jsonl = tmp_path / "seeds.jsonl"
    _write_jsonl(jsonl, [_fake_artifact(0)])

    with pytest.raises(ValueError, match="Config hash mismatch"):
        m._load_jsonl_seeds(jsonl, "different_hash_xyz")


def test_load_jsonl_seeds_truncated_final_line(tmp_path: Path) -> None:
    """A truncated final line is discarded with a warning, not an exception."""
    jsonl = tmp_path / "seeds.jsonl"
    _write_jsonl(jsonl, [_fake_artifact(0), _fake_artifact(1)])

    # Append a partial line (simulates a power-loss mid-write)
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write('{"config": {"seed": 2}, "config_hash": "' + _CFG_HASH + '", "run": ')
        # deliberately unterminated — no closing brace

    completed, loaded = m._load_jsonl_seeds(jsonl, _CFG_HASH)
    assert completed == {0, 1}    # seed 2 row was discarded
    assert len(loaded) == 2


def test_json_roundtrip_preserves_aggregate(tmp_path: Path) -> None:
    """A single artifact round-tripped through JSON produces identical build_aggregate.

    This is the core invariant: loading from JSONL must be equivalent to having
    the original artifact in memory. Fails if any float loses precision or if
    build_aggregate silently reads a field only present in the full artifact.
    """
    a = _fake_artifact(seed=7)
    clean = {k: v for k, v in a.items() if not k.startswith("_")}
    clean["config_hash"] = _CFG_HASH

    # Simulate JSONL round-trip
    a_from_jsonl = json.loads(json.dumps(clean))

    agg_orig   = m.build_aggregate([a])
    agg_loaded = m.build_aggregate([a_from_jsonl])

    assert json.dumps(agg_orig) == json.dumps(agg_loaded), (
        "build_aggregate produced different output for the same artifact before "
        "and after JSON round-trip. Check for float precision loss or access to "
        "internal-only fields (prefixed with _)."
    )
