"""analysis/impact_model.py hybrid savings v1 (operator item 3): savings vs all-cloud at matched quality, matched on
shared seeds only, escalation cost INTERIM until the full-history redo is VERIFIED, PENDING without its register rows.
No network."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
import impact_model as im  # noqa: E402

# local {(seed, task): (flag, wrong)}, cloud {(seed, task): (ok, usd)}
LOCAL = {(1, "a"): (False, False), (1, "b"): (True, True), (1, "c"): (False, True), (1, "d"): (True, False),
         (9, "a"): (True, True)}   # seed 9 not shared with the cloud run
CLOUD = {(1, "a"): (True, 0.01), (1, "b"): (True, 0.02), (1, "c"): (False, 0.03), (1, "d"): (True, 0.04)}


def test_reference_is_all_cloud_and_unshared_seed_dropped():
    r = im.hybrid_policies(LOCAL, CLOUD, seeds=[1])
    assert r["n"] == 4
    ac = r["policies"]["all_cloud"]
    assert ac["usd_task"] == pytest.approx(0.025) and ac["savings_task"] == 0 and ac["correct"] == 3
    assert ac["matched"]


def test_naive_local_is_free_and_below_cloud():
    p = im.hybrid_policies(LOCAL, CLOUD, seeds=[1])["policies"]["naive_local"]
    assert p["usd_task"] == 0 and p["correct"] == 2 and p["silent_wrong"] == 2
    assert p["savings_task"] == pytest.approx(0.025)
    assert p["savings_month"][1000] == pytest.approx(0.025 * 1000 * 30)
    assert not p["matched"]


def test_flag_policy_escalates_flagged_and_misses_wrong_inside_window():
    r = im.hybrid_policies(LOCAL, CLOUD, seeds=[1])
    p = r["policies"]["hybrid_flag"]
    assert p["escalated"] == 2 and p["usd_task"] == pytest.approx((0.02 + 0.04) / 4)
    assert p["correct"] == 3 and p["silent_wrong"] == 1
    assert r["wrong_inside_window"] == 1 and r["flag_tp"] == 1 and r["flagged"] == 2


def test_perfect_detector_upper_bound_matches_quality():
    r = im.hybrid_policies(LOCAL, CLOUD, seeds=[1])
    p = r["policies"]["hybrid_perfect"]
    assert p["escalated"] == 2 and p["usd_task"] == pytest.approx((0.02 + 0.03) / 4)
    assert p["matched"] and p["correct"] == 3
    assert r["matched_local"] == 2   # local correct and cloud correct: (1,a) and (1,d)


def test_measured_redo_cost_replaces_interim_for_escalations_only():
    r = im.hybrid_policies(LOCAL, CLOUD, seeds=[1], esc_usd=0.1)
    assert r["policies"]["hybrid_perfect"]["usd_task"] == pytest.approx(0.2 / 4)
    assert r["policies"]["all_cloud"]["usd_task"] == pytest.approx(0.025)


def test_no_shared_pairs_raises():
    with pytest.raises(ValueError):
        im.hybrid_policies(LOCAL, CLOUD, seeds=[7])


def test_pending_without_register_rows(tmp_path):
    h = im.hybrid_savings(tmp_path)
    assert h["status"] == "PENDING" and set(h["pending"]) == set(im.HYBRID_REQUIRED)
    assert im.hybrid_savings_text(h).startswith("PENDING")
    assert "PENDING" in im.render_hybrid_block(h)


def test_full_context_redo_none_while_row_pending(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "NUMBERS_REGISTER.md").write_text("| cloud-redo-full-context | PENDING | x |\n",
                                                            encoding="utf-8")
    assert im.full_context_redo(tmp_path) is None


def test_real_data_block_if_available():
    if not all(im.register_has(REPO, c) for c in im.HYBRID_REQUIRED):
        pytest.skip("register rows not VERIFIED in this checkout")
    h = im.hybrid_savings(REPO)
    assert h["status"] == "VERIFIED" and len(h["seeds"]) == 3
    devs = {c["device"]: c["label"] for c in h["cells"]}
    assert devs["evo-t2s Intel CPU"] == "projected" and devs["evo-x2"] == "measured"
    if h["redo"] is None:
        assert all(c["esc_label"] == im.INTERIM_ESC for c in h["cells"])
    block = im.render_hybrid_block(h)
    assert "Blade" in block and "not measured" in block and "UPPER BOUND" in block
    assert "\u2014" not in block and "\u2014" not in im.hybrid_savings_text(h)
