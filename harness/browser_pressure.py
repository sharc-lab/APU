"""K2 pressure arm (c), everyday_apps: a headless browser holding local pages open, each allocating a chunk of JS
heap, standing in for ordinary background-app memory pressure during an agent session -- more realistic than the
two synthetic mechanisms K2 already has (t2s_lab.Balloon's AWE-locked balloon, harness/pageable_touch.py's ordinary
pageable touch). See docs/FINDINGS.md's "K2 arm (c), everyday_apps" section (2026-09-29) for the pre-registered
kill criterion this arm exists to test, written before the run-wiring code in t2s_k2_pressure.py existed.

Browser choice: this module launches a real browser as a plain subprocess (--headless --disable-gpu, local
file:// URLs only, no network) instead of adding Playwright as a project dependency. Reasoning (this module's own
call; not given numerically or by name in the task):
  - Playwright's bundled Chromium is a large one-time download this repo has never needed before, and both target
    machines (evo-t2s, evo-x2) are deployed-to hosts this repo reaches only by copying files (see
    scripts/deploy_evo.py) -- adding a step that needs `playwright install chromium` to run once per host, with
    network access, is a worse fit than reusing what is already there.
  - This harness already uses exactly the "spawn a subprocess, heartbeat/poll it, kill_tree() to clean it up"
    shape for every other pressure arm (t2s_lab.Balloon, harness/pageable_touch.py, this file's own
    EverydayAppsPressure below); a plain subprocess browser launch is not a new pattern, it is the same one.
  - Both target Windows hosts ship Microsoft Edge by default, so find_browser_executable() below finds a real,
    already-installed headless-capable browser with no install step at all on a real run.
If neither Edge nor Chrome is found on the host, this arm cannot run there (start() returns ok=False with a reason
rather than falling back to Playwright); a reviewer who wants the Playwright path can revisit this call once/if a
target host lacks any Chromium-family browser.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

N_PAGES = 20
TARGET_MB_PER_PAGE = 150

# Turn-scheduling constants for the arm (session "turn" == one quality-suite item in the K2 sense). Not given
# numerically in the task; flagged the same way t2s_k2_pressure.py flags SCORE_TOL_REL/RESP_TOL_FACTOR as this
# module's own choice for a reviewer to confirm or replace.
DEFAULT_START_TURN = 5
DEFAULT_HOLD_TURNS = 10

_PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>pressure page {idx}</title></head>
<body>
<script>
  // Deterministic ~{target_mb}MB allocation: one Float64Array (8 bytes/element) sized to hit the target byte
  // count, filled with a cheap deterministic pattern so the engine cannot elide the allocation as dead code, and
  // kept referenced on window so the GC cannot collect it while the page stays open.
  (function() {{
    var targetBytes = {target_bytes};
    var arr = new Float64Array(Math.floor(targetBytes / 8));
    for (var i = 0; i < arr.length; i += 4096) {{ arr[i] = i; }}
    window.__pressure_arr_{idx} = arr;
  }})();
</script>
<p>pressure page {idx}</p>
</body></html>
"""


def generate_pressure_pages(out_dir, n_pages=N_PAGES, target_mb=TARGET_MB_PER_PAGE):
    """Writes n_pages static, self-contained HTML files under out_dir, each allocating ~target_mb of JS heap on
    load via one inline <script> (no external resources referenced, no network). Deterministic: the same
    (idx, target_mb) always produces byte-identical content. Returns the list of file paths written, in order.
    Independently testable: takes no lab/server/subprocess dependency at all."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_bytes = int(target_mb * 1024 * 1024)
    paths = []
    for idx in range(n_pages):
        p = out_dir / f"pressure_page_{idx:02d}.html"
        p.write_text(_PAGE_TEMPLATE.format(idx=idx, target_mb=target_mb, target_bytes=target_bytes),
                     encoding="utf-8")
        paths.append(p)
    return paths


def find_browser_executable(candidates=None):
    """Best-effort local lookup for a headless-capable Chromium-family browser already on this machine: explicit
    candidates first, then well-known Edge/Chrome install paths, then whichever of msedge/chrome is on PATH.
    Returns None (never raises) if nothing is found -- callers must treat that as "cannot run this arm here"."""
    candidates = list(candidates or [])
    candidates += [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    for exe in ("msedge", "msedge.exe", "chrome", "chrome.exe", "google-chrome"):
        found = shutil.which(exe)
        if found:
            return found
    return None


def build_launch_cmd(executable, page_paths, user_data_dir):
    """The exact command line used to hold N local pages open headlessly with no network: --headless/--disable-gpu,
    a throwaway --user-data-dir (never touches a real browser profile), and one file:// URL per page."""
    cmd = [executable, "--headless", "--disable-gpu", "--no-sandbox", "--disable-extensions",
           f"--user-data-dir={user_data_dir}", "--disable-background-networking", "--no-first-run"]
    cmd += [Path(p).resolve().as_uri() for p in page_paths]
    return cmd


class EverydayAppsPressure:
    """Balloon/PageableToucher-shaped lifecycle (start(...)/alive()/stop()) for K2 pressure arm (c): generate the
    local pages, launch a headless browser holding all of them open, then later kill the whole process tree
    cleanly. Same start/alive/stop shape t2s_lab.Balloon and t2s_k2_pressure.PageableToucher already expose, so
    call sites that drive either of those two can drive this one identically.

    Dependency-injected so no test here ever launches a real browser or touches a real process tree:
      launch_fn(cmd) -> object with .pid and .poll()   (default: subprocess.Popen)
      kill_tree_fn(pid)                                (default: t2s_m3_power_coupling.kill_tree, the same
                                                         taskkill /F /T convention Balloon/PageableToucher/
                                                         Server.stop() all already use)
    """

    def __init__(self, lab, tag, page_dir=None, n_pages=N_PAGES, target_mb=TARGET_MB_PER_PAGE, executable=None,
                 launch_fn=None, kill_tree_fn=None, page_generator=generate_pressure_pages,
                 executable_finder=find_browser_executable):
        self.lab, self.tag = lab, tag
        self.n_pages, self.target_mb = n_pages, target_mb
        self.page_dir = Path(page_dir) if page_dir else Path(lab.prefix + f"_everyday_apps_{tag}_pages")
        self.user_data_dir = Path(lab.prefix + f"_everyday_apps_{tag}_profile")
        self.executable = executable
        self._executable_finder = executable_finder
        self._launch_fn = launch_fn or subprocess.Popen
        self._kill_tree_fn = kill_tree_fn or _default_kill_tree
        self._page_generator = page_generator
        self.proc = None
        self.pages = []

    def start(self, target_available_mb=None):
        """target_available_mb is accepted (and only recorded, never targeted) so this class matches the call
        signature run_k2_run already uses for the other two pressure arms; real memory pressure here comes from
        the browser's own footprint (n_pages * target_mb), not a controlled available-memory setpoint."""
        self.pages = self._page_generator(self.page_dir, self.n_pages, self.target_mb)
        executable = self.executable or self._executable_finder()
        if executable is None:
            return {"ok": False, "why": "no headless-capable browser executable found on this host", "pid": None,
                    "n_pages": len(self.pages)}
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_launch_cmd(executable, self.pages, self.user_data_dir)
        self.proc = self._launch_fn(cmd)
        return {"ok": True, "why": None, "pid": getattr(self.proc, "pid", None), "n_pages": len(self.pages),
                "target_mb_per_page": self.target_mb, "requested_available_mb": target_available_mb,
                "executable": executable}

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        stopped = False
        if self.proc is not None:
            self._kill_tree_fn(self.proc.pid)
            stopped = True
        return {"stopped": stopped}


def _default_kill_tree(pid):
    # Imported lazily so a plain `import browser_pressure` (as tests do, with everything faked) never pulls in
    # t2s_m3_power_coupling's own Windows-only imports unless the real kill path is actually exercised.
    import t2s_m3_power_coupling as m3
    m3.kill_tree(pid)
