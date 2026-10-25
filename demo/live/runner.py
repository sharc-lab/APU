"""The live run behind the screen (controller side).

Modes:
  live     local lanes on evo-x2 through the agent (harness/demo_live_x2.py --mode agent) started over SSH; cloud
           parts through cloud_lane.CloudLane (LIVE within budget, else REPLAYED). Never while an evo-x2 queue job runs
           (docs/RUN.md has the check).
  laptop   development only: the same agent run on the controller with a small model; the screen says so.
  replay   a recorded run file (results/demo_live_run_<date>.jsonl) played back with its recorded timings; every
           number labelled REPLAYED. The fallback when evo-x2 is unreachable.

Events (one dict each, appended to LiveRun.events, polled by the page): run_start, task_start, lane_start, call,
decision, escalate, lane_end, task_end, tally, run_end, error.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "harness", REPO / "scripts", REPO / "src" / "dse", REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import demo_live as L  # noqa: E402
import demo_run as DR  # noqa: E402
from proc_util import popen_hidden  # noqa: E402

SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
X2_HOST = "evo-x2"
X2_DIR = r"C:\apu\ovn\demo"
X2_RESULTS = r"C:\apu\ovn\results"
X2_PY = "py -3.12"
PICKS = REPO / "demo" / "live" / "picks_v1.json"


ENV_SSH_TARGET = "DEMO_X2_SSH_TARGET"   # rehearsal only: point the agent at another (e.g. unroutable) address
CONNECT_TIMEOUT_S = 10                  # an unreachable evo-x2 fails in seconds, not after the OS default
FIRST_LINE_TIMEOUT_S = 90               # agent start (SSH + Python import) must answer within this
LINE_TIMEOUT_S = 300                    # a lane that writes nothing for this long counts as stalled
RECORDED_RUN = REPO / "results" / "demo_live_run_20261023.jsonl"


def x2_agent_argv(host: str = X2_HOST, py: str = X2_PY, env=None) -> list[str]:
    import os
    import host_config as hc
    env = os.environ if env is None else env
    target = env.get(ENV_SSH_TARGET) or hc.HOSTS[hc.ALIASES[host]]["ssh_host"]
    return [SSH, "-o", "BatchMode=yes", "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}", "-T", target,
            f'{py} {X2_DIR}\\demo_live_x2.py --mode agent --results-dir {X2_RESULTS}']


def laptop_agent_argv(model: str) -> list[str]:
    return [sys.executable, str(REPO / "harness" / "demo_live_x2.py"), "--mode", "agent", "--model", model,
            "--log-dir", str(Path.home() / "AppData" / "Local" / "Temp" / "demo_dev" / "agent")]


class Agent:
    """The local-lane agent as a child process (hidden; SSH for evo-x2), JSON lines both ways."""

    def __init__(self, argv: list[str], popen=popen_hidden, on_line=None, first_timeout=FIRST_LINE_TIMEOUT_S,
                 line_timeout=LINE_TIMEOUT_S):
        import queue
        import subprocess
        self.proc = popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, encoding="utf-8", bufsize=1)
        self.on_line = on_line or (lambda r: None)
        self.line_timeout = line_timeout
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        try:
            first = self._read(first_timeout)
        except Exception:
            self.kill()
            raise
        if first.get("record") != "agent_ready":
            self.kill()
            raise RuntimeError(f"agent did not start: {first}")
        self.ready = first

    def _pump(self):
        try:
            for line in self.proc.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def _read(self, timeout=None) -> dict:
        import queue
        while True:
            try:
                line = self.lines.get(timeout=timeout or self.line_timeout)
            except queue.Empty:
                raise TimeoutError(f"no answer from the agent in {timeout or self.line_timeout:.0f} s") from None
            if line is None:
                raise RuntimeError("agent closed its output (evo-x2 unreachable or the agent stopped)")
            try:
                return json.loads(line)
            except ValueError:
                continue

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass

    def local(self, lane, task, num_ctx) -> dict:
        self.proc.stdin.write(json.dumps({"cmd": "local", "lane": lane, "seed": task["seed"],
                                          "task_id": task["task_id"], "num_ctx": num_ctx}) + "\n")
        self.proc.stdin.flush()
        while True:
            r = self._read()
            if r.get("record") == "local_result":
                return r["result"]
            if r.get("record") == "local_error":
                raise RuntimeError(r.get("error"))
            self.on_line(r)

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.kill()


class LiveRun:
    def __init__(self, scenario: str, mode: str, picks_path: Path = PICKS, agent_factory=None, cloud=None,
                 replay_file: Path | None = None, clock=time.monotonic, sleep=time.sleep, idle_speedup: float = 1.0,
                 fallback_file: Path | None = RECORDED_RUN):
        if scenario not in DR.SCENARIOS:
            raise ValueError(scenario)
        self.scenario, self.mode, self.picks_path = scenario, mode, Path(picks_path)
        self.agent_factory, self.cloud, self.replay_file = agent_factory, cloud, replay_file
        self.clock, self.sleep = clock, sleep
        self.idle_speedup = idle_speedup
        self.fallback_file = fallback_file
        self.events, self.lock = [], threading.Lock()
        self.done = False
        self.results = []

    def push(self, ev: dict):
        with self.lock:
            self.events.append({"i": len(self.events), "t": round(self.clock() - self.t0, 2), **ev})

    def since(self, i: int) -> list[dict]:
        with self.lock:
            return self.events[i:]

    def start(self):
        self.t0 = self.clock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            if self.mode == "replay":
                self._replay()
            else:
                try:
                    self._live()
                except Exception as e:
                    self._fall_back(e)
        except Exception as e:
            self.push({"event": "error", "error": f"{type(e).__name__}: {e}"[:400]})
        finally:
            self.done = True
            self.push({"event": "run_end"})

    def _live(self):
        import route_at_start as R
        prof = R.load_profiles()
        picks = json.loads(self.picks_path.read_text(encoding="utf-8"))["picks"]
        b3 = L.read_jsonl(REPO / "results" / L.B3_FILE)
        self.push({"event": "run_start", "mode": self.mode, "scenario": self.scenario,
                   "label": "LIVE" if self.mode == "live" else "LIVE (development, controller laptop)",
                   "n_tasks": len(picks)})
        agent = self.agent_factory(lambda r: self._agent_line(r))
        host = str((agent.ready or {}).get("host") or "")
        self.push({"event": "agent", "host": host})
        if self.mode == "live" and host.upper() != "EVO-X2":
            agent.close()
            raise RuntimeError(f"the agent answered from {host!r}, not EVO-X2; refusing to run")
        try:
            for p in picks:
                task = L.build_task(b3, p["seed"], p["task_id"], with_cloud=True)
                self.push({"event": "task_start", "seed": task["seed"], "task_id": task["task_id"],
                           "turn_idx": task["turn_idx"], "history_turns": task["history_turns"],
                           "task_type": task["task_type"], "projected_prompt": task["projected_prompt"]})
                res = DR.run_task(self.scenario, prof, task, agent.local, self.cloud, self.cloud.budget_left,
                                  emit=self.push)
                self.results.append(res)
                self.push({"event": "task_end", "seed": task["seed"], "task_id": task["task_id"]})
                self.push({"event": "tally", "tally": DR.tally(self.results)})
        finally:
            agent.close()

    def _fall_back(self, err: Exception):
        """No manual step when evo-x2 is unreachable or a lane stalls: the tasks not yet finished are played from the
        recorded acceptance run, every number labelled REPLAYED, with the reason on screen."""
        reason = f"{type(err).__name__}: {err}"[:300]
        if self.fallback_file is None or not Path(self.fallback_file).exists():
            raise err
        done = {(r["seed"], r["task_id"]) for r in self.results}
        self.push({"event": "fallback", "reason": reason, "source": Path(self.fallback_file).name, "label": "REPLAYED"})
        self.replay_file = Path(self.fallback_file)
        self._replay(skip=done, announce=False)

    def _agent_line(self, r: dict):
        if r.get("record") == "call":
            self.push({"event": "call", **{k: r.get(k) for k in ("lane", "seed", "task_id", "num_ctx", "n", "seconds",
                                                                 "own_tokens", "used_tokens", "loaded_window", "cut",
                                                                 "evidence", "tool_calls")}})

    def _replay(self, skip=frozenset(), announce=True):
        rows = L.read_jsonl(self.replay_file)
        tasks = [r for r in rows if r.get("record") == "task_result" and r.get("scenario") == self.scenario
                 and (r["seed"], r["task_id"]) not in skip]
        if announce:
            self.push({"event": "run_start", "mode": "replay", "scenario": self.scenario, "label": "REPLAYED",
                       "source": str(self.replay_file.name), "n_tasks": len(tasks)})
        for r in tasks:
            self.push({"event": "task_start", "seed": r["seed"], "task_id": r["task_id"], "turn_idx": r["turn_idx"],
                       "history_turns": r["history_turns"], "projected_prompt": r["projected_prompt"],
                       "task_type": r.get("task_type")})
            self._play(replay_timeline(r))
            self.results.append(r)
            self.push({"event": "task_end", "seed": r["seed"], "task_id": r["task_id"]})
            self.push({"event": "tally", "tally": DR.tally(self.results)})

    def _play(self, timeline: list[tuple[float, dict]]):
        """Events at their recorded offsets. With idle_speedup (the backup clip only), a gap longer than IDLE_S in
        which nothing changes on screen is played at that speed, with a visible speed event before and after."""
        now = 0.0
        for at, ev in timeline:
            gap = max(0.0, at - now)
            if self.idle_speedup > 1 and gap > IDLE_S:
                self.push({"event": "speed", "x": self.idle_speedup})
                self.sleep(gap / self.idle_speedup)
                self.push({"event": "speed", "x": 1})
            else:
                self.sleep(gap)
            now = max(now, at)
            self.push(ev)


IDLE_S = 3.0


def replay_timeline(r: dict) -> list[tuple[float, dict]]:
    """(offset seconds from task start, event) for one recorded task: Default local then Ours one after another (each
    call at its recorded duration), the All cloud lane from the task start alongside them (as recorded)."""
    out, t = [], 0.0
    for lane in ("default", "ours"):
        res = r["lanes"][lane]
        out.append((t, {"event": "lane_start", "lane": lane}))
        if lane == "ours" and res.get("decision"):
            out.append((t, {"event": "decision", "lane": "ours", "decision": res["decision"]}))
        for c in res.get("calls") or []:
            t += float(c.get("seconds") or 0)
            out.append((t, {"event": "call", "lane": lane, **c}))
        if res.get("escalated"):
            out.append((t, {"event": "escalate", "lane": lane}))
        if res.get("route") == "cloud" or res.get("escalated"):
            t += float((res.get("cloud") or res).get("seconds") or 0)
        out.append((t, {"event": "lane_end", "lane": lane, "result": res}))
    c = r["lanes"]["cloud"]
    out.append((0.0, {"event": "lane_start", "lane": "cloud"}))
    out.append((float(c.get("seconds") or 0), {"event": "lane_end", "lane": "cloud", "result": c}))
    end = max(a for a, _ in out)
    return sorted(out, key=lambda x: x[0]) + [(end, {"event": "task_idle_end"})]
