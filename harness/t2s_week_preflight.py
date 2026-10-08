"""Queue job (evo-t2s): the T2S week's own start check (docs/T2S_WEEK_PLAN.md step 1). Read-only: it installs,
pulls, deletes and changes nothing. Every later job of queues/t2s_week_queue.json is gated on this entry being
"done", so a failed hard check blocks the whole week's measurements instead of letting them run on the wrong
runtime version, a missing model, or an occupied machine.

Hard checks (any failure -> exit note "stopped: preflight failed: ...", the queue marks this entry error, every gated
entry stays pending and queue_empty.flag names them):
  host          the machine reports EVO-T2S (harness/host_config.py), and nobody is logged in (interactive_guard)
  handover      C:\\apu\\ovn\\HANDOVER_DONE.json exists (written by scripts/t2s_handover.py when the machine was
                handed back to Zach on 2026-10-06; its absence means the last handover never completed and the machine
                state is unknown)
  ollama        the Ollama this week will use (the pin file C:\\apu\\ovn\\ollama_pin.json if present, else the host's
                normal resolution) reports version 0.34.4, the evo-x2 version every R2 and outcome-table row was
                measured with
  llama_cpp     C:\\apu\\bin\\llama-b10970\\llama-server.exe and llama-tokenize.exe exist and report build 10970
                (b10970-bfdc32183 on every evo-t2s and evo-x2 row that records llama_build)
  models        the Ollama manifests for llama3.1:8b, qwen3:8b, qwen3:14b and the GGUFs the llama-server legs use
  files         the deployed harness files and the workload-pack weights file the queued jobs import or read
Soft checks (recorded, never fail the job): a pending Windows reboot, a persistent OLLAMA_IGPU_ENABLE in the HKCU/HKLM
environment, and which x2_r2_agent.py capabilities the R2 entries still lack (harness/t2s_r2_agent.py).

Output: results/t2s_week_preflight.jsonl (one row per check plus a summary row). Calls t2s_queue.advance() once.

The Ollama pin. If evo-t2s still has Ollama 0.33.2 (docs/T2S_CHANGELOG.md, 2026-09-29), 0.34.4 is installed SIDE BY
SIDE (never replacing the existing install) and C:\\apu\\ovn\\ollama_pin.json is written by the operator:
  {"ollama_bin": "C:\\\\apu\\\\bin\\\\ollama-0.34.4\\\\ollama.exe", "version": "0.34.4"}
apply_ollama_pin() sets OLLAMA_BIN from it, which host_config._resolve_ollama_exe_for_serve() checks first, so every
queued job that starts Ollama (t2s_r2_agent.py, t2s_outcome_table.py) uses the pinned binary.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
if (_here.parent / "harness").is_dir():
    sys.path.insert(0, str(_here.parent / "harness"))

HOST_KEY = "EVO-T2S"
DEPLOY = Path(r"C:\apu\ovn")
HANDOVER_FILE = DEPLOY / "HANDOVER_DONE.json"
OLLAMA_PIN_FILE = DEPLOY / "ollama_pin.json"
TARGET_OLLAMA_VERSION = "0.34.4"
LLAMA_BIN_DIR = Path(r"C:\apu\bin\llama-b10970")
TARGET_LLAMA_BUILD = "10970"
MODELS_DIR = Path(r"C:\apu\models")
OLLAMA_TAGS = ("llama3.1:8b", "qwen3:8b", "qwen3:14b")
GGUFS = ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf", "Qwen3-14B-Q4_K_M.gguf")
# Every file a queued job of this week imports or runs, flat in C:\apu\ovn (scripts/deploy_evo.py layout).
DEPLOYED_FILES = ("t2s_queue.py", "queue_watchdog.py", "stale_server_cleanup.py", "host_config.py", "proc_util.py",
                  "argparse_probe.py", "t2s_week_preflight.py", "t2s_r2_agent.py", "x2_r2_agent.py",
                  "t2s_r2_session_growth.py", "x2_r2_mechanism.py", "prompt_token_check.py", "t2s_k1_ollama.py",
                  "t2s_outcome_table.py", "chat_template_source.py", "t2s_night2.py", "t2s_lab.py", "t2s_overnight.py",
                  "t2s_amech.py", "level_zero_sysman.py", "win_cpu_topology.py", "bw_hog.py",
                  "spin_hog_affinity.py", "t2s_m3_power_coupling.py", "run_provenance.py", "server_guard.py")
# t2s_outcome_table.py's flat layout reads these relative to C:\apu\ovn (analysis/ next to it marks that layout).
DATA_FILES = (Path("results") / "workload_pack" / "item_weights_trace_weighted.json",
              Path("results") / "workload_pack" / "grade.py",
              Path("results") / "workload_pack" / "items",
              Path("analysis") / "trace_weighted_pack.py")
OUT_REL = Path("results") / "t2s_week_preflight.jsonl"


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def apply_ollama_pin(pin_file=OLLAMA_PIN_FILE, environ=None) -> dict:
    """Sets OLLAMA_BIN from the pin file when it names an existing exe. Returns what it did (never raises)."""
    environ = os.environ if environ is None else environ
    try:
        pin = json.loads(Path(pin_file).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {"pinned": False, "why": f"no pin file {pin_file}"}
    except Exception as e:
        return {"pinned": False, "why": f"unreadable pin file: {e!r}"}
    exe = pin.get("ollama_bin")
    if not exe or not Path(exe).exists():
        return {"pinned": False, "why": f"pinned exe missing: {exe!r}", "pin": pin}
    environ["OLLAMA_BIN"] = exe
    return {"pinned": True, "ollama_bin": exe, "pin": pin}


_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")


def parse_ollama_version(text) -> str | None:
    """'ollama version is 0.34.4' (and the client-only 'Warning: client version is 0.34.4') -> '0.34.4'."""
    m = _VERSION_RE.search(text or "")
    return m.group(1) if m else None


def parse_llama_build(text) -> str | None:
    """llama.cpp --version prints 'version: 10970 (bfdc3218)'."""
    m = re.search(r"version:\s*(\d+)", text or "")
    return m.group(1) if m else None


def _run(cmd, timeout=60):
    from proc_util import run_hidden
    try:
        p = run_hidden(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return f"error: {e!r}"


def _reg_value(root_name, key, value):
    try:
        import winreg
        root = getattr(winreg, root_name)
        with winreg.OpenKey(root, key) as k:
            return winreg.QueryValueEx(k, value)[0]
    except Exception:
        return None


def _reg_key_exists(root_name, key):
    try:
        import winreg
        with winreg.OpenKey(getattr(winreg, root_name), key):
            return True
    except Exception:
        return False


class Probe:
    """The machine-facing reads, one method each, so tests replace them with fakes."""

    def hostname(self):
        return socket.gethostname()

    def query_user(self):
        import host_config as hc
        return hc._query_user_raw()

    def exists(self, path):
        return Path(path).exists()

    def read_text(self, path):
        return Path(path).read_text(encoding="utf-8-sig")

    def ollama_exe(self):
        import host_config as hc
        return hc._resolve_ollama_exe_for_serve()

    def ollama_models_dir(self):
        import host_config as hc
        return os.environ.get("OLLAMA_MODELS") or hc._this_host_entry().get("ollama_models")

    def run(self, cmd):
        return _run(cmd)

    def reboot_pending(self):
        return (_reg_key_exists("HKEY_LOCAL_MACHINE", r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based "
                                r"Servicing\RebootPending")
                or _reg_key_exists("HKEY_LOCAL_MACHINE", r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto "
                                   r"Update\RebootRequired"))

    def persistent_igpu_env(self):
        return {"hkcu": _reg_value("HKEY_CURRENT_USER", "Environment", "OLLAMA_IGPU_ENABLE"),
                "hklm": _reg_value("HKEY_LOCAL_MACHINE",
                                   r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                                   "OLLAMA_IGPU_ENABLE")}

    def x2_missing(self):
        import t2s_r2_agent as w
        import x2_r2_agent as x2
        caps = w.x2_capabilities(x2)
        fwd = ["--mode", "mitigation", "--tiers", "x", "--server-env", "A=1", "--client-trim", "margin=0.05"]
        return w.missing_capabilities(caps, fwd)


def manifest_path(tag, models_dir):
    name, _, version = tag.partition(":")
    return Path(models_dir) / "manifests" / "registry.ollama.ai" / "library" / name / (version or "latest")


def run_checks(probe: Probe, deploy=DEPLOY, pin_file=OLLAMA_PIN_FILE) -> list[dict]:
    checks = []

    def add(name, ok, hard=True, **detail):
        checks.append({"record": "t2s_preflight_check", "check": name, "ok": bool(ok), "hard": hard,
                       **detail, "ts_utc": utc_iso()})

    host = (probe.hostname() or "").upper()
    add("host", host == HOST_KEY, hostname=host)
    qu = probe.query_user() or ""
    add("interactive_session", "No User exists" in qu, query_user=qu.strip()[:300])
    hf = Path(deploy) / HANDOVER_FILE.name
    if probe.exists(hf):
        try:
            content = json.loads(probe.read_text(hf))
        except Exception as e:
            content = {"unreadable": repr(e)}
        add("handover_file", True, path=str(hf), content=content)
    else:
        add("handover_file", False, path=str(hf), why="missing: the 2026-10-06 handover never wrote its record")
    pin = apply_ollama_pin(pin_file)
    exe = probe.ollama_exe()
    ver_text = probe.run([exe, "--version"])
    ver = parse_ollama_version(ver_text)
    add("ollama_version", ver == TARGET_OLLAMA_VERSION, exe=exe, version=ver, target=TARGET_OLLAMA_VERSION,
        pin=pin, raw=ver_text.strip()[:300])
    for tool in ("llama-server.exe", "llama-tokenize.exe"):
        p = LLAMA_BIN_DIR / tool
        if not probe.exists(p):
            add(f"llama_cpp_{tool}", False, path=str(p), why="missing")
            continue
        out = probe.run([str(p), "--version"])
        build = parse_llama_build(out)
        add(f"llama_cpp_{tool}", build == TARGET_LLAMA_BUILD, path=str(p), build=build,
            target=TARGET_LLAMA_BUILD, raw=out.strip()[:300])
    mdir = probe.ollama_models_dir()
    for tag in OLLAMA_TAGS:
        mp = manifest_path(tag, mdir) if mdir else None
        add(f"ollama_model_{tag}", bool(mp) and probe.exists(mp), manifest=str(mp))
    for g in GGUFS:
        add(f"gguf_{g}", probe.exists(MODELS_DIR / g), path=str(MODELS_DIR / g))
    missing = [f for f in DEPLOYED_FILES if not probe.exists(Path(deploy) / f)]
    missing += [str(f) for f in DATA_FILES if not probe.exists(Path(deploy) / f)]
    add("deployed_files", not missing, missing=missing)
    add("reboot_pending", not probe.reboot_pending(), hard=False)
    env = probe.persistent_igpu_env()
    add("persistent_igpu_env_absent", not any(env.values()), hard=False, values=env)
    try:
        x2m = probe.x2_missing()
    except Exception as e:
        x2m = [f"capability probe failed: {e!r}"]
    add("x2_r2_agent_capabilities", not x2m, hard=False, missing=x2m)
    return checks


def summarize(checks) -> tuple[bool, str]:
    failed = [c["check"] for c in checks if c["hard"] and not c["ok"]]
    warn = [c["check"] for c in checks if not c["hard"] and not c["ok"]]
    if failed:
        return False, ("stopped: preflight failed: " + ", ".join(failed)
                       + (f"; warnings: {', '.join(warn)}" if warn else ""))[:400]
    return True, "completed" + (f"; warnings: {', '.join(warn)}" if warn else "")


def main(argv=None, probe=None, tq=None, deploy=DEPLOY):
    note = "completed"
    try:
        checks = run_checks(probe or Probe(), deploy=deploy)
        ok, note = summarize(checks)
        out = Path(deploy) / OUT_REL
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            for c in checks:
                f.write(json.dumps(c, default=str) + "\n")
            f.write(json.dumps({"record": "t2s_preflight_summary", "ok": ok, "note": note, "ts_utc": utc_iso()})
                    + "\n")
        print(note, flush=True)
    except Exception as e:
        note = f"stopped: {e!r}"[:400]
    finally:
        try:
            if tq is None:
                import t2s_queue as tq
            tq.advance(note)
        except Exception as e:
            print(f"queue advance failed: {e!r}", flush=True)
    return note


if __name__ == "__main__":
    main()
