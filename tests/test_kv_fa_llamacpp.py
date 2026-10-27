"""KV precision x flash attention on llama.cpp server (harness/kv_fa_llamacpp.py): cells and commands, the 24-turn
prefix of the B5 session, the KV log parser and check, the refusal class, a session and the refusal probe on fake
servers, the pre-registered outcome on fake records, the hours estimate, the queue entry and the register rows.
No network, no server, no queue access."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "analysis"))

import kv_fa_llamacpp as K  # noqa: E402
import x2_queue_fill as qf  # noqa: E402

ag = K.ag


# ── cells, commands, plan ────────────────────────────────────────────────────────────────────────────

def test_cells_and_the_refusal_cell():
    assert K.REFUSAL_CELL not in K.CELL_ORDER and set(K.CELL_ORDER) | {K.REFUSAL_CELL} == set(K.CELLS)
    assert K.CELLS[K.REFUSAL_CELL] == {"ctk": "q8_0", "ctv": "q8_0", "fa": "off"}
    assert all(K.CELLS[c]["ctv"] == "f16" for c in K.SQUARE_K.values())          # the primary square: K only
    assert {K.CELLS[c]["fa"] for c in K.SQUARE_K.values()} == {"on", "off"}
    assert K.SQUARE_OPERATOR[("q8", "on")] == "q8_fa_on" and K.SQUARE_KV[("q8", "off")] == K.REFUSAL_CELL
    assert all(ag.ARMS[K.ARM[c]]["num_ctx"] == 40960 and ag.ARMS[K.ARM[c]]["call2_mode"] == "off" for c in K.CELLS)
    assert K.SEEDS == (20260901, 20260902, 20260903) and K.TURNS == 24 and K.PORT != K.XR.PORT


def test_server_commands_differ_only_in_kv_and_fa():
    base = K.server_cmd("f16_fa_off", "L")
    for c, cfg in K.CELLS.items():
        cmd = K.server_cmd(c, "L")
        assert cmd[:-6] == base[:-6]
        assert cmd[-6:] == ["-ctk", cfg["ctk"], "-ctv", cfg["ctv"], "-fa", cfg["fa"]]
        assert cmd[cmd.index("-c") + 1] == "40960" and cmd[cmd.index("--log-verbosity") + 1] == "4"
        assert K.guard_expected(c) == {"model_path": K.GGUF, "np": 1, "ngl": 99, "n_ctx": 40960, **cfg}
    assert not any("context-shift" in t for t in base) and base[base.index("--port") + 1] == str(K.PORT)


def test_run_order_is_seed_major_balanced_and_rotated():
    o = K.run_order()
    assert len(o) == 15 and len(set(o)) == 15
    for i, s in enumerate(K.SEEDS):
        block = o[5 * i:5 * i + 5]
        assert {c for c, _s in block} == set(K.CELL_ORDER) and all(sd == s for _c, sd in block)
    assert [b[0] for b in (o[0], o[5], o[10])] == list(K.CELL_ORDER[:3])


def test_24_turn_session_is_the_prefix_of_the_b5_session():
    for s in K.SEEDS:
        a, b = ag.build_agent_session(s, K.B5_TURNS), ag.build_agent_session(s, K.TURNS)
        assert a.system_prompt == b.system_prompt and len(b.spec.turns) == K.TURNS
        assert [ag.user_message(t, a) for t in a.spec.turns[:K.TURNS]] == [ag.user_message(t, b) for t in b.spec.turns]
        assert all(a.store.get(k) == v for k, v in b.store.items())
        assert [t.idx for t in b.spec.turns if t.canary_check] == [5, 10, 15, 20]


# ── log parser, KV check, refusal ────────────────────────────────────────────────────────────────────

LOAD_LOG = ("llama_context: n_ctx         = 40960\n"
            "llama_context: flash_attn    = disabled\n"
            "llama_kv_cache:    Vulkan0 KV buffer size =  4800.00 MiB\n"
            "llama_kv_cache: size = 4800.00 MiB ( 40960 cells,  40 layers,  1/1 seqs), K (q8_0): 1700.00 MiB, "
            "V (f16): 3200.00 MiB\n"
            "llama_context:    Vulkan0 compute buffer size =   820.50 MiB\n")


def test_parse_kv_log_and_kv_check():
    p = K.parse_kv_log("\x1b[32m" + LOAD_LOG)
    assert p["k_type"] == "q8_0" and p["v_type"] == "f16" and p["k_mib"] == 1700.0 and p["v_mib"] == 3200.0
    assert p["kv_buffer_mib"] == 4800.0 and p["compute_buffer_mib"] == 820.5 and p["flash_attn"] == "disabled"
    assert K.kv_check("q8k_fa_off", p)["status"] == "ok"
    assert K.kv_check("q8k_fa_on", p)["status"] == "mismatch"          # flash_attn disabled != on
    assert K.kv_check("f16_fa_off", p)["status"] == "mismatch"         # K q8_0 != f16
    assert K.kv_check("q8k_fa_off", K.parse_kv_log(""))["status"] == "absent"
    no_fa = K.parse_kv_log(LOAD_LOG.replace("flash_attn    = disabled", ""))
    assert K.kv_check("q8k_fa_on", no_fa) == {"status": "ok", "reason": None, "flash_attn_line_present": False}


def test_refusal_class():
    line = "llama_init_from_model: quantized V cache requires flash_attn to be enabled"
    assert K.refusal_class(False, "x\n" + line + "\n")["result"] == "refused"
    assert K.refusal_class(False, "segfault")["result"] == "start_failed"
    assert K.refusal_class(True, "")["result"] == "accepted"
    assert K.refusal_class(False, "llama_init_from_model: model does not support different K (q8_0) and V (f16) "
                                  "cache types")["result"] == "refused"


def test_help_lines_for_kv_and_fa():
    text = ("-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: "
            "'auto')\n                                        (env: LLAMA_ARG_FLASH_ATTN)\n"
            "-ctv,  --cache-type-v TYPE              KV cache data type for V\n")
    h = K.help_lines(text)
    assert "default: 'auto'" in h["--flash-attn"] and h["--cache-type-v"].startswith("-ctv")
    assert K.XR.HELP_KEYS != K.HELP_KEYS     # restored


# ── fake servers ─────────────────────────────────────────────────────────────────────────────────────

class FakeGuard:
    def check(self):
        return True, [111]


class FakeServer:
    refuse = False

    def __init__(self, cell, log_path):
        self.cell, self.log_path, self.pid, self.url = cell, str(log_path), 111, "http://fake"
        self.n_ctx, self.guard = 40960, FakeGuard()

    def _log_text(self):
        c = K.CELLS[self.cell]
        return (f"llama_context: flash_attn    = {'enabled' if c['fa'] == 'on' else 'disabled'}\n"
                f"Vulkan0 KV buffer size = {6400.0 if c['ctk'] == 'f16' else 4800.0} MiB\n"
                f"llama_kv_cache: size = 1 MiB ( 40960 cells, 40 layers, 1/1 seqs), K ({c['ctk']}): 1 MiB, "
                f"V ({c['ctv']}): 1 MiB\n")

    def start(self):
        Path(self.log_path).write_text("", encoding="utf-8")
        if self.refuse:
            return {"cell": self.cell, "started": False, "exit_rc": 1, "refusal": K.refusal_class(
                False, "quantized V cache requires flash_attn to be enabled")}
        p = K.parse_kv_log(self._log_text())
        return {"cell": self.cell, "started": True, "props_n_ctx": 40960, "kv_log": p,
                "kv_check": K.kv_check(self.cell, p)}

    def alive(self):
        return True

    def stop(self):
        return {"after_stop_listeners": []}


def fake_post(url, body):
    return 200, {"choices": [{"finish_reason": "stop", "message": {"content": "Done."}}],
                 "usage": {"prompt_tokens": 900, "completion_tokens": 3},
                 "timings": {"prompt_n": 900, "prompt_ms": 450.0, "predicted_n": 3, "predicted_ms": 150.0}}, None, 0.01


def test_session_rows_on_a_fake_server(tmp_path):
    rows = []
    res = K.run_cell_session("q8k_fa_off", 20260901, tmp_path / "o.jsonl", rows.append, lambda m: None,
                             server_factory=FakeServer, post=fake_post, settle_s=0)
    assert res == "ok"
    turns = [r for r in rows if r.get("record") == "r2a_turn"]
    assert len(turns) == 24 and all(t["arm_id"] == K.ARM["q8k_fa_off"] and t["mode"] == K.MODE for t in turns)
    assert all(t["cell"] == "q8k_fa_off" and t["context_event"] == "none" for t in turns)
    assert all(ev["timings"]["prompt_ms"] == 450.0 for t in turns for ev in t["llamacpp_calls"])
    v = K._session_values(turns, K.B5_RULES_QWEN)
    assert v["prefill_tok_s"] == pytest.approx(2000.0) and v["decode_tok_s"] == pytest.approx(20.0)
    assert v["canary_turns"] == 4 and v["bad_calls"] == 0
    stop = rows[-1]
    assert stop["record"] == "server_stop" and stop["kv_check_after"]["status"] == "ok"
    assert [r["record"] for r in rows if r["record"] in ("server_start", "r2a_session", "server_stop")] == [
        "server_start", "r2a_session", "server_stop"]


def test_refusal_probe_and_refused_cell_are_results(tmp_path):
    class Refusing(FakeServer):
        refuse = True
    rows = []
    r = K.refusal_probe(tmp_path, rows.append, lambda m: None, server_factory=Refusing)
    assert r["result"] == "refused" and "requires flash_attn" in r["lines"][0]
    assert [x["record"] for x in rows] == ["server_start", "server_stop", "refusal_probe"]
    res = K.run_cell_session("q8k_fa_off", 20260901, tmp_path / "o.jsonl", rows.append, lambda m: None,
                             server_factory=Refusing, post=fake_post, settle_s=0)
    assert res == "refused"
    r = K.refusal_probe(tmp_path, [].append, lambda m: None, server_factory=FakeServer)
    assert r["result"] == "accepted"


def test_kv_mismatch_stops_before_the_session(tmp_path):
    class Wrong(FakeServer):
        def _log_text(self):
            return super()._log_text().replace("K (q8_0)", "K (f16)")
    rows = []
    res = K.run_cell_session("q8k_fa_on", 20260901, tmp_path / "o.jsonl", rows.append, lambda m: None,
                             server_factory=Wrong, post=fake_post, settle_s=0)
    assert res == "kv_mismatch" and not any(r["record"] == "r2a_turn" for r in rows)


# ── outcome on fake records ──────────────────────────────────────────────────────────────────────────

KV_MIB = {"f16": 6400.0, "q8_0": 4800.0}


def fake_rows(minutes, kv=None, refusal="refused", run_end=True, bad_cell=None, drop=None):
    rows = [{"record": "run_start", "harness": K.HARNESS},
            {"record": "refusal_probe", "cell": K.REFUSAL_CELL, "result": refusal, "lines": ["x"]}]
    cells = list(K.CELL_ORDER) + ([K.REFUSAL_CELL] if refusal == "accepted" else [])
    for c in cells:
        cfg = K.CELLS[c]
        kvm = (kv or {}).get(c, KV_MIB[cfg["ctk"]] if cfg["ctv"] == "f16" else 3400.0)
        for i, seed in enumerate(K.SEEDS):
            if drop == (c, seed):
                continue
            att = f"{c}{seed}"
            log = {"k_type": cfg["ctk"], "v_type": cfg["ctv"], "kv_buffer_mib": kvm, "compute_buffer_mib": 800.0,
                   "flash_attn": "enabled" if cfg["fa"] == "on" else "disabled"}
            rows.append({"record": "server_start", "cell": c, "seed": seed, "started": True, "props_n_ctx": 40960,
                         "kv_log": log, "kv_check": {"status": "ok"}})
            per_turn = minutes[c][i] * 60.0 / K.TURNS
            for j in range(K.TURNS):
                ev = [{"class": "flagged_reject" if (c == bad_cell and j == 23) else "none",
                       "timings": {"prompt_n": 1000, "prompt_ms": 500.0, "predicted_n": 10, "predicted_ms": 500.0}}]
                rows.append({"record": "r2a_turn", "attempt_id": att, "mode": K.MODE, "model_id": K.MODEL_ID,
                             "arm_id": K.ARM[c], "seed": seed, "turn_idx": j + 1, "turn_wall_s": per_turn,
                             "llamacpp_calls": ev, "any_http_error": False, "canary_check": (j + 1) % 5 == 0,
                             "canary_hist_ok": True, **{r: True for r in K.B5_RULES_QWEN}})
            rows.append({"record": "r2a_session", "mode": K.MODE, "model_id": K.MODEL_ID, "arm_id": K.ARM[c],
                         "seed": seed, "attempt_id": att})
            rows.append({"record": "server_stop", "cell": c, "seed": seed, "kv_log_after": log,
                         "kv_check_after": {"status": "ok"}})
    if run_end:
        rows.append({"record": "run_end"})
    return rows


SAME = {"f16_fa_off": [50, 52, 48], "f16_fa_on": [40, 41, 39], "q8k_fa_off": [25, 26, 24], "q8k_fa_on": [20, 20.5, 19.5],
        "q8_fa_on": [10, 10, 10], "q8_fa_off": [30, 30, 30]}
DIFF = dict(SAME, q8k_fa_off=[50, 51, 49])       # q8_0 saves half with FA on, nothing with FA off


def test_outcome_no_interaction():
    o = K.outcome(fake_rows(SAME), n_boot=500)
    assert o["valid"] and o["refusal_probe"]["result"] == "refused" and o["window"] == 40960
    e = o["primary"]["speed"]["estimate"]
    assert e["I"] == pytest.approx(0.0, abs=1e-9) and e["ratio_fa_off"] == pytest.approx(0.5)
    assert o["primary"]["speed"]["reading"].startswith("no interaction shown")
    m = o["primary"]["memory"]
    assert m["d_fa_off_mib"] == m["d_fa_on_mib"] == -1600.0 and m["reading"].startswith("no memory interaction")
    c = o["cells"]["q8k_fa_off"]
    assert c["n_sessions"] == 3 and c["mean_min"] == pytest.approx(25.0) and c["kv_buffer_mib"] == 4800.0
    assert c["prefill_tok_s"] == pytest.approx(2000.0) and c["decode_tok_s"] == pytest.approx(20.0)
    assert c["rule_k"] == c["rule_n"] == 3 * 24 * 5 and c["hist_k"] == c["hist_n"] == 12
    assert "kv_square_minutes" not in o["secondary"] and "confounded" in o["secondary"]["operator_square_minutes"][
        "label"]


def test_outcome_interaction_and_memory_interaction():
    o = K.outcome(fake_rows(DIFF, kv={"q8k_fa_off": 6400.0}), n_boot=500)
    e = o["primary"]["speed"]["estimate"]
    assert e["I"] == pytest.approx(math.log(1.0) - math.log(0.5)) and e["ci95"][0] > 0
    assert o["primary"]["speed"]["reading"].startswith("interaction: the q8_0 effect differs")
    assert "helps less" in o["primary"]["speed"]["reading"]
    assert o["primary"]["memory"]["reading"].startswith("memory interaction")


def test_outcome_accepted_refusal_cell_adds_the_kv_square():
    o = K.outcome(fake_rows(SAME, refusal="accepted"), n_boot=200)
    assert o["valid"] and "q8_fa_off" in o["cells"] and o["secondary"]["kv_square_minutes"]["I"] is not None


def test_outcome_invalid_runs_are_not_read():
    o = K.outcome(fake_rows(SAME, run_end=False), n_boot=100)
    assert not o["valid"] and o["primary"]["speed"]["reading"].startswith("not read (invalid: no run_end")
    o = K.outcome(fake_rows(SAME, bad_cell="q8k_fa_on"), n_boot=100)
    assert not o["valid"] and any("did not fit cleanly" in p for p in o["problems"])
    o = K.outcome(fake_rows(SAME, drop=("f16_fa_on", 20260903)), n_boot=100)
    assert not o["valid"] and any(p.startswith("f16_fa_on: sessions completed") for p in o["problems"])
    rows = [r for r in fake_rows(SAME) if r.get("record") != "refusal_probe"]
    assert "no refusal probe record" in K.outcome(rows, n_boot=100)["problems"]


def test_interaction_reading_bands():
    assert K.interaction_reading(None).startswith("not read")
    assert K.interaction_reading({"I": 0.05, "ci95": [0.01, 0.09]}).startswith("no interaction shown")
    assert K.interaction_reading({"I": 0.3, "ci95": [-0.05, 0.6]}) == "inconclusive"
    assert "helps more" in K.interaction_reading({"I": -0.3, "ci95": [-0.5, -0.2]})


def test_b5_prefix_minutes():
    rows = [{"record": "r2a_turn", "model_id": K.MODEL_ID, "arm_id": K.B5_ARMS["g"], "seed": 20260901,
             "turn_idx": j, "turn_wall_s": 60.0} for j in range(1, 41)]
    m = K.b5_prefix_minutes(rows)
    assert m["g"]["per_seed_min"] == {20260901: 24.0} and m["a"]["mean_min"] is None


# ── hours, queue, register ───────────────────────────────────────────────────────────────────────────

def test_hours_estimate_from_fake_rows():
    def b5(arm, w):
        return [{"record": "r2a_turn", "model_id": K.MODEL_ID, "arm_id": arm, "seed": s, "turn_idx": j,
                 "turn_wall_s": w} for s in K.SEEDS for j in range(1, 41)]
    b = b5(K.B5_ARMS["a"], 100.0) + b5(K.B5_ARMS["g"], 10.0)
    bfa = b5(K.B5_ARMS["i"], 50.0)
    xrt = ([{"record": "r2a_turn", "arm_id": K.XR.ARM["fixed"], "turn_wall_s": 12.0}] * 2
           + [{"record": "server_stop", "ts_utc": "2026-10-27T00:00:00+00:00"},
              {"record": "server_start", "started": True, "ts_utc": "2026-10-27T00:00:08+00:00"}])
    b3 = [{"record": "r2a_turn", "model_id": K.MODEL_ID, "arm_id": K.XR.OLLAMA_DEFAULT_ARM, "turn_wall_s": 10.0}]
    h = K.hours_estimate(b, bfa, xrt, b3)
    assert h["latency_ratio"] == pytest.approx(1.2) and h["start_overhead_s"] == pytest.approx(8.0)
    assert h["b5_prefix_mean_min"]["a"] == pytest.approx(40.0) and h["b5_prefix_mean_min"]["i"] == pytest.approx(20.0)
    want = 3 * 1.2 * (40 + 20 + 40 + 20 + 4) / 60
    assert h["session_h"] == pytest.approx(want) and h["n_starts"] == 16
    assert h["point_h"] == pytest.approx(want + 16 * 8 / 3600) and h["high_h"] == pytest.approx(h["point_h"] * 1.5)


def test_queue_entry():
    e = qf.entry("x2_kv_fa_llamacpp")
    assert e["cmd"][1] == "kvfa\\kv_fa_llamacpp.py" and e["cmd"][-1].endswith("kv_fa_llamacpp_qwen3_14b.jsonl")
    assert e["projected_duration_h"] == qf.KVFA_HOURS[0]
    assert "tq.advance()" in e["note"] and "sets no pause" in e["note"] and "refused" in e["note"]


def test_register_rows_pending_and_hours_match_the_queue(tmp_path):
    import numbers_register as nr
    ent = {e["claim_id"]: e for e in nr.NUMBER_ENTRIES}
    e = ent["kv-fa-interaction-llamacpp"]
    assert e["pending_ok"] and e["require_run_end"] == [nr.KVFA_FILE] and nr._pending(e, tmp_path)
    v = nr.compute_kv_fa_interaction_llamacpp_hours(REPO)["value"]
    assert v.startswith(f"ESTIMATE {qf.KVFA_HOURS[0]:.2f} h (high {qf.KVFA_HOURS[1]:.2f} h)")


def test_register_result_row_computes_from_a_fake_file(tmp_path):
    import numbers_register as nr
    (tmp_path / "results").mkdir()
    with open(tmp_path / nr.KVFA_FILE, "w", encoding="utf-8") as f:
        for r in fake_rows(SAME):
            f.write(json.dumps(r) + "\n")
    for name in (nr.B5_X2_FILE, nr.B5_FA_FILE):
        (tmp_path / name).write_text("", encoding="utf-8")
    v = nr.compute_kv_fa_interaction_llamacpp(tmp_path)["value"]
    assert "valid True" in v and "refusal probe q8_fa_off (-ctk q8_0 -ctv q8_0 -fa off): refused" in v
    assert "reading: no interaction shown" in v and "reading: no memory interaction" in v
