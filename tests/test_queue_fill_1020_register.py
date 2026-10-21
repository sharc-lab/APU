"""Register rows of the 2026-10-21 evo-x2 queue fill (analysis/numbers_register.py): the pre-registered qwen3:8b
reversal verdict on a synthetic seeds-06-10 file, the hours estimate, and the default-tier row. No network."""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import numbers_register as nr  # noqa: E402


@pytest.mark.parametrize("a8,a4,verdict", [(4, 1, "resolved as noise"), (7, 1, "reversal confirmed"),
                                            (8, 2, "reversal confirmed"), (1, 7, "ordering of the other models"),
                                            (5, 3, "resolved as noise")])
def test_reversal_verdict(a8, a4, verdict):
    assert nr.reversal_verdict(True, a8, 10, a4, 10) == verdict


def test_reversal_verdict_incomplete_and_fisher_matches_old_row():
    assert nr.reversal_verdict(False, 9, 10, 0, 10) == "incomplete"
    assert f"{nr.fisher_two_sided(4, 5, 1, 5):.3f}" == "0.206"   # R2-v1b-qwen3-8b-reversal's p


def _synthetic_new_file(dst: Path):
    """The v1b qwen3:8b 8192/4096 sessions renumbered to seeds 20260906-10, as if the new job had produced them."""
    src = REPO / nr.R2_REAL_V1B_FILE
    shift = {s: s + 5 for s in (20260901, 20260902, 20260903, 20260904, 20260905)}
    arms = {"ollama_ctx_8192_call2_notools", "ollama_ctx_4096_call2_notools"}
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("record") == "run_start":
            r["models"] = ["qwen3:8b"]
            out.append(r)
        elif r.get("model_id") == "qwen3:8b" and r.get("arm_id") in arms and r.get("seed") in shift:
            r["seed"] = shift[r["seed"]]
            if r.get("attempt_id"):
                r["attempt_id"] = r["attempt_id"] + "_s"
            out.append(r)
    out.append({"record": "run_end"})
    dst.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")


def test_reversal_10seed_on_synthetic_replication(tmp_path):
    for f in nr.R2_V1B_DATA:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / f, tmp_path / f)
    _synthetic_new_file(tmp_path / nr.R2_Q8_SEEDS_06_10_FILE)
    v = nr.compute_r2_qwen3_8b_reversal_10seed(tmp_path)["value"]
    assert "num_ctx_8192: 8/10 gated silent" in v
    assert "num_ctx_4096: 2/10 gated silent" in v
    assert "valid True" in v
    assert "verdict: reversal confirmed" in v


def test_reversal_10seed_incomplete_without_run_end(tmp_path):
    for f in nr.R2_V1B_DATA:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / f, tmp_path / f)
    p = tmp_path / nr.R2_Q8_SEEDS_06_10_FILE
    _synthetic_new_file(p)
    p.write_text("".join(l + "\n" for l in p.read_text(encoding="utf-8").splitlines() if "run_end" not in l),
                 encoding="utf-8")
    assert "verdict: incomplete" in nr.compute_r2_qwen3_8b_reversal_10seed(tmp_path)["value"]


def test_hours_row_is_labelled_estimate_and_keeps_the_reserve():
    v = nr.compute_x2_queue_fill_1020_hours(REPO)["value"]
    for j in nr.QUEUE_FILL_ORDER:
        assert f"{j} ESTIMATE" in v
    est = nr.queue_fill_estimates(REPO)
    assert sum(e["high_h"] for e in est.values()) < 24 * 19 - 24   # far inside the window with the 24 h reserve


def test_default_tier_row():
    v = nr.compute_x2_scale_real_default_tier(REPO)["value"]
    assert "qwen3:32b: 3 default-tier sessions completed" in v
