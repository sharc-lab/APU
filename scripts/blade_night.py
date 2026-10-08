"""Blade night wrapper: `py -3.12 scripts/blade_night.py --night 1` (docs/BLADE_PLAN.md "Night protocol").

Order, every step logged:
  1. refuse unless the operator's start flag C:\\apu\\blade\\START_BLADE_NIGHT_<N>.flag exists (created only after the
     operator says "start Blade night N"); it is renamed .used when the night starts, so it never starts a second
     night by itself;
  2. refuse unless the machine is on AC power and the paused condition holds (blade_queue.paused_condition:
     C:\\apu\\blade\\CLAUDE_CODE_PAUSED.flag present, no other python / pytest / git / llama-server process);
  3. record the current power values (AC standby, AC hibernate, AC lid action, plus the active scheme) to
     C:\\apu\\blade\\logs\\night<N>_power_<ts>.json BEFORE changing anything, then set all three to 0 (never sleep,
     never hibernate, lid does nothing) for this night only. If an earlier night's power log says it never restored,
     its originals are used (the values read now would be this wrapper's own temporary zeros);
  4. run the night's jobs in order with resume (harness/blade_queue.run_night);
  5. in a finally: restore every recorded value, read them back, log both, then write the night summary
     results/blade_night<N>_summary_<ts>.json.

Never changes NVIDIA settings. Every child process is hidden (proc_util). --stub runs the whole night with fakes
(powercfg, nvidia-smi, AC, process list, job runner): it parses every job's argv with the job's own parser, checks the
output paths, and prints the order of power set/restore; nothing is loaded and no setting changes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))

import blade_common as bc  # noqa: E402
import blade_queue as bq  # noqa: E402

POWER_SETTINGS = (("SUB_SLEEP", "STANDBYIDLE", 0), ("SUB_SLEEP", "HIBERNATEIDLE", 0), ("SUB_BUTTONS", "LIDACTION", 0))
_AC_RE = re.compile(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)")
_DC_RE = re.compile(r"Current DC Power Setting Index:\s*0x([0-9a-fA-F]+)")
_SCHEME_RE = re.compile(r"GUID:\s*([0-9a-fA-F-]{36})")


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def start_flag(night: int, blade_dir=None) -> Path:
    return Path(blade_dir or bc.BLADE_DIR) / f"START_BLADE_NIGHT_{night}.flag"


class Power:
    """powercfg through an injectable runner: run(argv) -> stdout text. Only AC values are changed."""

    def __init__(self, run=None, log=print):
        self.run = run or self._real_run
        self.log = log
        self.events = []

    @staticmethod
    def _real_run(argv):
        from proc_util import run_hidden
        p = run_hidden(argv, capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            raise RuntimeError(f"{' '.join(argv)} -> rc {p.returncode}: {(p.stderr or p.stdout)[:200]}")
        return p.stdout or ""

    def scheme(self) -> str | None:
        m = _SCHEME_RE.search(self.run(["powercfg", "/getactivescheme"]))
        return m.group(1) if m else None

    def query(self, sub, setting) -> dict:
        out = self.run(["powercfg", "/q", "SCHEME_CURRENT", sub, setting])
        ac, dc = _AC_RE.search(out), _DC_RE.search(out)
        return {"ac": int(ac.group(1), 16) if ac else None, "dc": int(dc.group(1), 16) if dc else None}

    def set_ac(self, sub, setting, value: int):
        self.events.append(("set", sub, setting, value))
        self.run(["powercfg", "/setacvalueindex", "SCHEME_CURRENT", sub, setting, str(int(value))])

    def apply(self):
        self.events.append(("apply",))
        self.run(["powercfg", "/setactive", "SCHEME_CURRENT"])

    def read_all(self) -> dict:
        return {f"{s}/{k}": self.query(s, k) for s, k, _ in POWER_SETTINGS}


def unrestored_originals(log_dir: Path):
    """The originals of the newest earlier power log that never reached restored=true, else None."""
    logs = sorted(Path(log_dir).glob("night*_power_*.json"))
    for p in reversed(logs):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d.get("restored"):
            return d.get("original"), str(p)
        return None
    return None


def run_night(night: int, *, power: Power, ac_fn, paused_fn, runner, blade_dir: Path, results_dir: Path,
              state_path=None, log=print) -> dict:
    blade_dir, results_dir = Path(blade_dir), Path(results_dir)
    log_dir = blade_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_stamp()
    summary = {"night": night, "started_utc": bc.utc_iso(), "refused": None, "power": None, "queue": None}
    flag = start_flag(night, blade_dir)
    if not flag.exists():
        summary["refused"] = f"no start flag {flag} (create it only after the operator says 'start Blade night {night}')"
        log(summary["refused"])
        return summary
    if not ac_fn():
        summary["refused"] = "not on AC power"
        log(summary["refused"])
        return summary
    cond = paused_fn()
    if not cond["ok"]:
        summary["refused"] = "paused condition not met: " + "; ".join(cond["reasons"])
        log(summary["refused"])
        return summary
    used = flag.with_suffix(".flag.used")
    if used.exists():
        used.unlink()
    flag.rename(used)
    power_log = log_dir / f"night{night}_power_{ts}.json"
    prior = unrestored_originals(log_dir)
    original = power.read_all()
    rec = {"night": night, "scheme": power.scheme(), "original_read_now": original, "restored": False}
    if prior:
        rec["original"], rec["original_from_unrestored_log"] = prior[0], prior[1]
        log(f"earlier night never restored power ({prior[1]}); using its originals")
    else:
        rec["original"] = original
    power_log.write_text(json.dumps(rec, indent=1), encoding="utf-8")   # originals on disk before any change
    log(f"power originals recorded: {rec['original']}")
    try:
        for sub, setting, value in POWER_SETTINGS:
            power.set_ac(sub, setting, value)
        power.apply()
        rec["night_values"] = power.read_all()
        power_log.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        log(f"no-sleep set: {rec['night_values']}")
        summary["queue"] = bq.run_night(night, runner=runner, state_path=state_path, paused_fn=paused_fn, log=log)
    finally:
        errors = []
        for sub, setting, _ in POWER_SETTINGS:
            orig = (rec["original"].get(f"{sub}/{setting}") or {}).get("ac")
            if orig is None:
                errors.append(f"{sub}/{setting}: no original value recorded")
                continue
            try:
                power.set_ac(sub, setting, orig)
            except Exception as e:
                errors.append(f"{sub}/{setting}: {e!r}")
        try:
            power.apply()
            rec["restored_values"] = power.read_all()
        except Exception as e:
            errors.append(f"apply/readback: {e!r}")
        rec["restore_errors"] = errors
        rec["restored"] = not errors and all(
            (rec["restored_values"].get(k) or {}).get("ac") == (v or {}).get("ac") for k, v in rec["original"].items())
        power_log.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        log(f"power restored={rec['restored']}: {rec.get('restored_values')} errors={errors}")
        summary["power"] = {"log": str(power_log), "original": rec["original"],
                            "night_values": rec.get("night_values"), "restored_values": rec.get("restored_values"),
                            "restored": rec["restored"], "errors": errors}
        summary["ended_utc"] = bc.utc_iso()
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"blade_night{night}_summary_{ts}.json"
        out.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        summary["summary_file"] = str(out)
    return summary


# ── stub (fakes only) ───────────────────────────────────────────────────────────────────────────────

class FakePowercfg:
    """Holds AC/DC values in memory; answers powercfg's own output format."""

    def __init__(self, values=None):
        self.values = dict(values or {"SUB_SLEEP/STANDBYIDLE": 1800, "SUB_SLEEP/HIBERNATEIDLE": 10800,
                                      "SUB_BUTTONS/LIDACTION": 1})
        self.pending = dict(self.values)
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if argv[1] == "/getactivescheme":
            return "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\n"
        if argv[1] == "/q":
            v = self.values[f"{argv[3]}/{argv[4]}"]
            return f"    Current AC Power Setting Index: 0x{v:08x}\n    Current DC Power Setting Index: 0x{v:08x}\n"
        if argv[1] == "/setacvalueindex":
            self.pending[f"{argv[3]}/{argv[4]}"] = int(argv[5])
            return ""
        if argv[1] == "/setactive":
            self.values = dict(self.pending)
            return ""
        raise ValueError(argv)


def stub_runner_factory(events: list, tmp_results: Path):
    """Parses each job's argv with the job module's own build_arg_parser and checks its outputs; no model loads."""
    def runner(job, log_path):
        script = job["argv"][0]
        mod = importlib.import_module(Path(script).stem)
        args = mod.build_arg_parser().parse_args(job["argv"][1:])
        outs = [Path(o).name for o in job["outputs"]]
        bad = [o for o in outs if not o.startswith("blade_")]
        events.append({"event": "job", "id": job["id"], "module": mod.__name__, "parsed": vars(args),
                       "outputs": job["outputs"], "outputs_prefixed_blade": not bad})
        if bad:
            return 2
        return 0
    return runner


def stub_night(night: int, log=print) -> dict:
    events = []
    tmp = Path(tempfile.mkdtemp(prefix="blade_night_stub_"))
    start_flag(night, tmp).write_text("start Blade night (stub)\n", encoding="utf-8")
    fake = FakePowercfg()

    def run(argv):
        out = fake(argv)
        if argv[1] in ("/setacvalueindex", "/setactive"):
            events.append({"event": "powercfg", "argv": argv[1:]})
        return out

    def paused_fn():
        return {"ok": True, "reasons": [], "heavy": [], "flag": "(stub)"}

    def logf(msg):
        events.append({"event": "log", "msg": msg})
        log(msg)

    summ = run_night(night, power=Power(run=run, log=logf), ac_fn=lambda: True, paused_fn=paused_fn,
                     runner=stub_runner_factory(events, tmp), blade_dir=tmp, results_dir=tmp / "results",
                     state_path=tmp / "state.json", log=logf)
    order = [e["argv"][0] + " " + " ".join(e["argv"][2:5]) if e["argv"][0] != "/setactive" else "/setactive"
             for e in events if e["event"] == "powercfg"]
    jobs = [e for e in events if e["event"] == "job"]
    first_job = next(i for i, e in enumerate(events) if e["event"] == "job") if jobs else None
    sets = [i for i, e in enumerate(events) if e["event"] == "powercfg"]
    report = {"night": night, "summary": summ, "power_call_order": order, "jobs": jobs,
              "set_before_first_job": bool(sets) and first_job is not None and sets[0] < first_job,
              "restore_after_last_job": bool(sets) and bool(jobs) and sets[-1] > max(
                  i for i, e in enumerate(events) if e["event"] == "job"),
              "final_values_equal_original": fake.values == FakePowercfg().values,
              "all_outputs_prefixed_blade": all(j["outputs_prefixed_blade"] for j in jobs)}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night", type=int, choices=sorted(bq.NIGHTS), required=True)
    ap.add_argument("--stub", action="store_true", help="fakes only: argv parse, outputs, power set/restore order")
    args = ap.parse_args(argv)
    if args.stub:
        rep = stub_night(args.night)
        out = bc.DRYRUN_DIR / f"blade_dryrun_night{args.night}_stub.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
        ok = (rep["set_before_first_job"] and rep["restore_after_last_job"] and rep["final_values_equal_original"]
              and rep["all_outputs_prefixed_blade"] and not rep["summary"]["refused"])
        print(f"STUB NIGHT {args.night}: {'PASS' if ok else 'FAIL'} -> {out}")
        return 0 if ok else 1
    bc.require_blade()

    def ac():
        out = bc.ps("(Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | "
                    "Select-Object -First 1).PowerOnline").strip()
        return out.lower() == "true"

    def log(msg):
        print(f"[{bc.utc_iso()}] {msg}", flush=True)

    summ = run_night(args.night, power=Power(log=log), ac_fn=ac, paused_fn=bq.paused_condition,
                     runner=bq.run_job_subprocess, blade_dir=bc.BLADE_DIR, results_dir=bc.RESULTS, log=log)
    print(json.dumps(summ, indent=1, default=str))
    return 0 if not summ.get("refused") else 2


if __name__ == "__main__":
    sys.exit(main())
