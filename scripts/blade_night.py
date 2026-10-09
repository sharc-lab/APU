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

A setting the machine does not expose (powercfg prints no AC index for it, so its original is None; on the Blade
SUB_BUTTONS/LIDACTION, night 1 2026-10-09) is "not applicable": never set, never restored, not an error, listed under
"not_applicable" in the power log, and it does not make restored false. restored is false only when an exposed setting
failed to restore or read back different from its original.

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


def exposed(original: dict) -> tuple[list, list]:
    """(exposed POWER_SETTINGS entries, not-applicable keys): a setting whose recorded original AC value is None is
    not exposed by this machine."""
    ok, na = [], []
    for sub, setting, value in POWER_SETTINGS:
        if (original.get(f"{sub}/{setting}") or {}).get("ac") is None:
            na.append(f"{sub}/{setting}")
        else:
            ok.append((sub, setting, value))
    return ok, na


def run_night(night: int, *, power: Power, ac_fn, paused_fn, runner, blade_dir: Path, results_dir: Path,
              state_path=None, log=print, read_decision=None) -> dict:
    blade_dir, results_dir = Path(blade_dir), Path(results_dir)
    log_dir = blade_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = utc_stamp()
    night_start_epoch = time.time()
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
    settings, not_applicable = exposed(rec["original"])
    rec["not_applicable"] = not_applicable
    power_log.write_text(json.dumps(rec, indent=1), encoding="utf-8")   # originals on disk before any change
    log(f"power originals recorded: {rec['original']}"
        + (f" (not exposed on this machine, left alone: {not_applicable})" if not_applicable else ""))
    try:
        for sub, setting, value in settings:
            power.set_ac(sub, setting, value)
        power.apply()
        rec["night_values"] = power.read_all()
        power_log.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        log(f"no-sleep set: {rec['night_values']}")
        summary["queue"] = bq.run_night(night, runner=runner, state_path=state_path, paused_fn=paused_fn, log=log,
                                        **({"read_decision": read_decision} if read_decision else {}))
    finally:
        errors = []
        for sub, setting, _ in settings:
            try:
                power.set_ac(sub, setting, rec["original"][f"{sub}/{setting}"]["ac"])
            except Exception as e:
                errors.append(f"{sub}/{setting}: {e!r}")
        try:
            power.apply()
            rec["restored_values"] = power.read_all()
        except Exception as e:
            errors.append(f"apply/readback: {e!r}")
        mismatched = [f"{s}/{k}" for s, k, _ in settings if "restored_values" in rec
                      and (rec["restored_values"].get(f"{s}/{k}") or {}).get("ac")
                      != rec["original"][f"{s}/{k}"]["ac"]]
        rec["restore_errors"] = errors
        rec["restore_mismatched"] = mismatched
        rec["restored"] = not errors and not mismatched
        power_log.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        log(f"power restored={rec['restored']}: {rec.get('restored_values')} errors={errors} "
            f"mismatched={mismatched} not_applicable={not_applicable}")
        summary["power"] = {"log": str(power_log), "original": rec["original"],
                            "night_values": rec.get("night_values"), "restored_values": rec.get("restored_values"),
                            "restored": rec["restored"], "errors": errors, "mismatched": mismatched,
                            "not_applicable": not_applicable}
        if night == 2:
            import blade_c3
            summary["c3_setting_readback"] = rb = blade_c3.read_readback(blade_dir, night_start_epoch)
            log(rb["status"])
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
            if v is None:   # a setting this machine does not expose: powercfg prints no index lines
                return "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\n"
            return f"    Current AC Power Setting Index: 0x{v:08x}\n    Current DC Power Setting Index: 0x{v:08x}\n"
        if argv[1] == "/setacvalueindex":
            if self.values.get(f"{argv[3]}/{argv[4]}") is None:
                raise RuntimeError(f"set on a setting that is not exposed: {argv[3]}/{argv[4]}")
            self.pending[f"{argv[3]}/{argv[4]}"] = int(argv[5])
            return ""
        if argv[1] == "/setactive":
            self.values = dict(self.pending)
            return ""
        raise ValueError(argv)


def stub_runner_factory(events: list, tmp_results: Path, gate_misses: int = 0, decisions: dict | None = None):
    """Parses each job's argv with the job module's own build_arg_parser and checks its outputs; no model loads.
    A gate job runs its real decide() on synthetic validation rows with `gate_misses` negative-control canary misses
    and puts the decision in `decisions` (keyed by job id) instead of a results file."""
    decisions = {} if decisions is None else decisions

    def runner(job, log_path):
        script = job["argv"][0]
        mod = importlib.import_module(Path(script).stem)
        args = mod.build_arg_parser().parse_args(job["argv"][1:])
        outs = [Path(o).name for o in job["outputs"]]
        bad = [o for o in outs if not o.startswith("blade_")]
        ev = {"event": "job", "id": job["id"], "module": mod.__name__, "parsed": vars(args),
              "outputs": job["outputs"], "outputs_prefixed_blade": not bad}
        if mod.__name__ == "blade_r2" and args.models_if_fit:
            # the automatic fit decision against whatever K1 summary exists (none in a stub: "do not run")
            ev["fit_decision"] = mod.fit_decision(args.models_if_fit.split(","), args.k1_summary,
                                                  mod.fit_ctxs(args.mode, (args.tiers or "default").split(",")))
        if mod.__name__ == "blade_r2" and args.mode == "mitigation":
            # the equivalent x2_r2_agent argv, parsed by x2_r2_agent's own parser (raises SystemExit if it does not)
            ev["x2_r2_agent_argv"], ev["client_trim"] = mod.check_mitigation_args(args, args.models.split(","))
        if mod.__name__ == "blade_r2" and args.mode == "validation":
            ev["plan"] = mod.plan_for("validation", [], neg_sessions=args.neg_sessions)
        if mod.__name__ == "blade_r2_gate":
            rows = mod.synthetic_validation_rows(args.model, args.neg_sessions, gate_misses)
            decisions[job["id"]] = ev["decision"] = mod.decide(rows, args.model, args.neg_sessions,
                                                               validation=args.validation)
        events.append(ev)
        if bad:
            return 2
        return 0
    return runner


def stub_night(night: int, log=print, gate_misses: int = 0, power_values=None) -> dict:
    events = []
    decisions = {}
    tmp = Path(tempfile.mkdtemp(prefix="blade_night_stub_"))
    start_flag(night, tmp).write_text("start Blade night (stub)\n", encoding="utf-8")
    fake = FakePowercfg(power_values)
    start_values = dict(fake.values)

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
                     runner=stub_runner_factory(events, tmp, gate_misses, decisions), blade_dir=tmp,
                     results_dir=tmp / "results", state_path=tmp / "state.json", log=logf,
                     read_decision=lambda job: decisions.get(job["id"]))
    order = [e["argv"][0] + " " + " ".join(e["argv"][2:5]) if e["argv"][0] != "/setactive" else "/setactive"
             for e in events if e["event"] == "powercfg"]
    jobs = [e for e in events if e["event"] == "job"]
    first_job = next(i for i, e in enumerate(events) if e["event"] == "job") if jobs else None
    sets = [i for i, e in enumerate(events) if e["event"] == "powercfg"]
    report = {"night": night, "summary": summ, "power_call_order": order, "jobs": jobs,
              "set_before_first_job": bool(sets) and first_job is not None and sets[0] < first_job,
              "restore_after_last_job": bool(sets) and bool(jobs) and sets[-1] > max(
                  i for i, e in enumerate(events) if e["event"] == "job"),
              "final_values_equal_original": fake.values == start_values,
              "power_restored": (summ.get("power") or {}).get("restored"),
              "all_outputs_prefixed_blade": all(j["outputs_prefixed_blade"] for j in jobs),
              "gate_misses": gate_misses, "gate_decisions": (summ.get("queue") or {}).get("gate_decisions"),
              "ran": [j["id"] for j in jobs],
              "skipped": {j["id"]: j.get("reason") for j in (summ.get("queue") or {}).get("jobs", [])
                          if j["status"] == "skipped"}}
    report["branch_ok"] = branch_ok(night, report)
    return report


BLADE_POWER_LAYOUT = {"SUB_SLEEP/STANDBYIDLE": 3600, "SUB_SLEEP/HIBERNATEIDLE": 0, "SUB_BUTTONS/LIDACTION": None}


def stub_ok(rep: dict) -> bool:
    return bool(rep["set_before_first_job"] and rep["restore_after_last_job"] and rep["final_values_equal_original"]
                and rep["power_restored"] and rep["all_outputs_prefixed_blade"] and rep["branch_ok"]
                and not rep["summary"]["refused"])


def branch_ok(night: int, rep: dict) -> bool:
    """Every job either ran or was skipped by its gate's branch, and the ones that ran are exactly the jobs with no
    "when" plus those whose "when" branch matches the gate decision."""
    decs = rep.get("gate_decisions") or {}
    want = []
    for j in bq.NIGHTS[night]:
        w = j.get("when")
        if not w or (decs.get(w["gate"]) or {}).get("branch") == w["branch"]:
            want.append(j["id"])
    skipped_ok = all(r and r.startswith("skipped:") for r in rep["skipped"].values())
    return rep["ran"] == want and set(rep["ran"]) | set(rep["skipped"]) == {j["id"] for j in bq.NIGHTS[night]} \
        and skipped_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night", type=int, choices=sorted(bq.NIGHTS), required=True)
    ap.add_argument("--stub", action="store_true", help="fakes only: argv parse, outputs, power set/restore order")
    args = ap.parse_args(argv)
    if args.stub:
        # a night with a gate is stubbed once per branch (0 misses -> real; 2 misses, as night 1's first run ->
        # mechanism), with the Blade's own power layout (LIDACTION not exposed)
        has_gate = any(j.get("gate") for j in bq.NIGHTS[args.night])
        variants = [0, 2] if has_gate else [0]
        reps = [stub_night(args.night, gate_misses=m, power_values=BLADE_POWER_LAYOUT) for m in variants]
        out = bc.DRYRUN_DIR / f"blade_dryrun_night{args.night}_stub.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(reps if has_gate else reps[0], indent=1, default=str)
        home = json.dumps(str(Path.home()))[1:-1]          # the user's profile path as it appears inside JSON
        out.write_text(text.replace(home, "%USERPROFILE%"), encoding="utf-8")
        oks = [stub_ok(r) for r in reps]
        for r, ok in zip(reps, oks):
            br = ", ".join(f"{g}: {d.get('branch')}" for g, d in (r["gate_decisions"] or {}).items()) or "no gate"
            print(f"STUB NIGHT {args.night} (gate misses {r['gate_misses']}; {br}): {'PASS' if ok else 'FAIL'}; "
                  f"ran {r['ran']}; skipped {r['skipped']}")
        print(f"STUB NIGHT {args.night}: {'PASS' if all(oks) else 'FAIL'} -> {out}")
        return 0 if all(oks) else 1
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
