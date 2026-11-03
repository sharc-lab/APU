"""Intake for Zach's XPS raw files (analysis/xps_intake.py) and their register rows: synthetic files only."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))

import numbers_register as nr  # noqa: E402
import xps_intake as XI  # noqa: E402

START = {"record": "run_start", "run_id": "xps-test-1", "ts_utc": "2026-11-05T10:00:00Z",
         "host": {"hostname": "XPS", "model": "Dell XPS (synthetic)", "ram_gb": 16, "cpu": "Intel", "gpu": "Intel"},
         "runtime_config": {"runtime": "openvino", "version": "2025.3", "default_window": 16384,
                            "model": "Qwen3-4B int4",
                            "device": "GPU", "kv_precision": "u8", "overrides": []},
         "tool": {"name": "synthetic", "commit": "0000000"}}
END = {"record": "run_end", "run_id": "xps-test-1", "ts_utc": "2026-11-05T11:00:00Z"}


def write(path: Path, kind: str, recs: list[dict], start=None, end=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(start or START, kind=kind)] + [{"run_id": "xps-test-1", **r} for r in recs] + ([END] if end else [])
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def ceiling_recs():
    out = []
    # 10 fit configs exact or 2% off; 4 held-out: errors 2%, 4%, 6%, 0%
    for i in range(10):
        m = 10000 + 1000 * i
        out.append({"record": "ceiling_config", "config_id": f"f{i}", "config": {"kv": "u8"}, "split": "fit",
                    "predicted_ceiling_tokens": m if i % 2 else round(m * 1.02), "measured_ceiling_tokens": m,
                    "measured_method": "bisection"})
    for i, e in enumerate((0.02, 0.04, -0.06, 0.0)):
        m = 20000
        out.append({"record": "ceiling_config", "config_id": f"h{i}", "config": {"kv": "u4"}, "split": "heldout",
                    "predicted_ceiling_tokens": round(m * (1 + e)), "measured_ceiling_tokens": m,
                    "measured_method": "bisection"})
    return out


def test_ceiling_rows(tmp_path):
    d = XI.load(write(tmp_path / "c.jsonl", "ceiling", ceiling_recs()), "ceiling")
    p = XI.ceiling_prediction(d)
    assert p["heldout_n"] == 4 and p["heldout_within"] == 3
    assert p["mean_heldout_pct"] == pytest.approx(3.0)
    assert "run_id xps-test-1" in p["value"] and "3/4" in p["value"] and "openvino 2025.3" in p["value"]
    w = XI.ceiling_within(d)
    assert (w["within"], w["configs"]) == (13, 14) and "13/14" in w["value"]


def test_tuned_window_gpu_share_and_loud_failure(tmp_path):
    tw = [{"record": "window_task", "config": "default", "window_tokens": 18000, "seed": 1, "task_id": f"t{i}",
           "ok": i < 20} for i in range(30)]
    tw += [{"record": "window_task", "config": "tuned", "window_tokens": 40448, "seed": 1, "task_id": f"t{i}",
            "ok": True} for i in range(30)]
    t = XI.tuned_window(XI.load(write(tmp_path / "t.jsonl", "tuned_window", tw), "tuned_window"))
    assert t["default"] == {"windows": [18000], "ok": 20, "n": 30} and t["tuned"]["ok"] == 30
    assert "40,448: 30/30" in t["value"]
    gs = [{"record": "gpu_share_trial", "gpu_share_pct": 50, "trial": i, "ok": True} for i in range(6)]
    gs += [{"record": "gpu_share_trial", "gpu_share_pct": 74.0, "trial": i, "ok": False} for i in range(6)]
    g = XI.gpu_share(XI.load(write(tmp_path / "g.jsonl", "gpu_share", gs), "gpu_share"))
    assert g["by_share"] == {50.0: [6, 6], 74.0: [0, 6]} and "GPU share 74%: 0/6" in g["value"]
    lf = [{"record": "overflow_call", "prompt_tokens": 20000, "window_tokens": 16384, "outcome": "error",
           "http_status": 500, "error_text": "exceeds", "latency_s": s} for s in (0.4, 0.5, 0.6)]
    lf += [{"record": "overflow_call", "prompt_tokens": 8000, "window_tokens": 16384, "outcome": "ok",
            "http_status": 200, "error_text": None, "latency_s": 12.0}]
    x = XI.loud_failure(XI.load(write(tmp_path / "l.jsonl", "loud_failure", lf), "loud_failure"))
    assert (x["over"], x["loud"]) == (3, 3) and "median 0.50 s" in x["value"] and "1/1 ok" in x["value"]


@pytest.mark.parametrize("mutate,msg", [
    (lambda rows: rows[:-1], "no run_end"),
    (lambda rows: [dict(rows[0], kind="gpu_share")] + rows[1:], "kind"),
    (lambda rows: [rows[0], {**rows[1], "split": "test"}] + rows[2:], "split"),
    (lambda rows: [rows[0], {**rows[1], "predicted_ceiling_tokens": "12000"}] + rows[2:], "predicted_ceiling_tokens"),
    (lambda rows: [rows[0], {**rows[1], "run_id": "other"}] + rows[2:], "outside an open run"),
    (lambda rows: [{k: v for k, v in rows[0].items() if k != "runtime_config"}] + rows[1:], "lacks"),
    (lambda rows: [rows[0], {"record": "surprise", "run_id": "xps-test-1"}] + rows[1:], "unexpected record"),
    (lambda rows: [rows[0], rows[-1]], "no ceiling_config"),
])
def test_validation_errors(tmp_path, mutate, msg):
    p = write(tmp_path / "c.jsonl", "ceiling", ceiling_recs())
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    p.write_text("\n".join(json.dumps(r) for r in mutate(rows)) + "\n", encoding="utf-8")
    with pytest.raises(XI.IntakeError, match=msg):
        XI.load(p, "ceiling")


def test_bool_is_not_an_int_field(tmp_path):
    recs = ceiling_recs()
    recs[0]["measured_ceiling_tokens"] = True
    with pytest.raises(XI.IntakeError, match="measured_ceiling_tokens"):
        XI.load(write(tmp_path / "c.jsonl", "ceiling", recs), "ceiling")


XPS_ROWS = ("xps-zach-ceiling-prediction", "xps-zach-ceiling-within-5pct", "xps-zach-tuned-window",
            "xps-zach-gpu-share", "xps-zach-intel-loud-failure-latency", "xps-demo-comparison")


def _entry(cid):
    return next(e for e in nr.NUMBER_ENTRIES if e["claim_id"] == cid)


def test_register_rows_are_pending_without_files_and_carry_no_reported_value(tmp_path):
    for cid in XPS_ROWS:
        e = _entry(cid)
        assert e["pending_ok"] is True and "reported_value" not in e
        assert nr._pending(e, tmp_path), cid          # no file in an empty repo dir -> PENDING
    # the description may say what Zach reported; the value is computed only from his file
    assert "REPORTED (not measured here) 5.1% mean error" in _entry("xps-zach-ceiling-prediction")["description"]


def test_register_rows_compute_from_synthetic_files(tmp_path):
    write(tmp_path / XI.DIR / "ceiling.jsonl", "ceiling", ceiling_recs())
    for cid in ("xps-zach-ceiling-prediction", "xps-zach-ceiling-within-5pct"):
        e = _entry(cid)
        assert nr._pending(e, tmp_path) == []
        assert "run_id xps-test-1" in e["compute"](tmp_path)["value"]
    write(tmp_path / XI.DIR / "gpu_share.jsonl", "gpu_share", [{"record": "gpu_share_trial", "gpu_share_pct": 50,
                                                                "trial": 0, "ok": True}], end=False)
    e = _entry("xps-zach-gpu-share")
    assert nr._pending(e, tmp_path) == [XI.DIR + "/gpu_share.jsonl (partial: no run_end)"]


def test_demo_comparison_row_computes_from_kit_files(tmp_path):
    import argparse
    sys.path[:0] = [str(REPO / "harness"), str(REPO / "scripts")]
    import xps_demo_kit as K
    d = tmp_path / "results" / "xps_demo"
    ns = argparse.Namespace(arms="a", seeds=[K.SEEDS[0]], max_turns=3, runtime="ollama", base_url="fake",
                            model="qwen3:4b", allow_nondefault=None, pipeline_window=None,
                            out=str(d / "xps_demo_local_v1.jsonl"), note=None, runtime_exe=None,
                            runtime_config_file=None, declared_default_window=None)
    K.run(ns, runtime=K.FakeRuntime(), cfg={"runtime": "ollama", "overrides": [], "unverified": [],
                                             "default_window": None}, log=lambda m: None)
    for a in "bcd":
        (d / f"xps_demo_cloud_{a}.jsonl").write_text(json.dumps({"record": "run_start"}) + "\n"
                                                     + json.dumps({"record": "run_end", "stopped": None}) + "\n",
                                                     encoding="utf-8")
    e = _entry("xps-demo-comparison")
    assert nr._pending(e, tmp_path) == []
    v = e["compute"](tmp_path)["value"]
    assert "local_default: 0/90 correct" in v and "complete False" in v


def test_cloud_estimate_row_computes_from_committed_files():
    v = _entry("xps-demo-cloud-estimate")["compute"](nr.REPO)["value"]
    assert v.startswith("ESTIMATE, not run") and "claude-sonnet-5-5" in v
