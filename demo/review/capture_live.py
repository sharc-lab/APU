"""Recordings of the live run screen replaying the acceptance run (Playwright, headless Chromium, 1600x1000, no audio).

    py -3.8 demo/review/capture_live.py amd8192 results/demo_live_run_20261023.jsonl            # main clip, real speed
    py -3.8 demo/review/capture_live.py intel results/demo_live_run_20261023.jsonl --idle-speedup 2   # backup clip

The screen is started here as a hidden child process (Python 3.12, `-m demo.live --mode replay`) on 127.0.0.1 and
always stopped in a finally block. Replay mode labels every number REPLAYED; with --idle-speedup 2 only stretches where
nothing changes on screen are played faster, and the page shows "2x speed" while they play. Output:
demo/review/demo_live_<scenario>.webm.
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "harness"))
from proc_util import check_output_hidden, popen_hidden  # noqa: E402

PORT = 8802
BASE = "http://127.0.0.1:%d" % PORT
SIZE = {"width": 1600, "height": 1000}


def start(run_file, speedup):
    py312 = check_output_hidden(["py", "-3.12", "-c", "import sys; print(sys.executable)"], text=True).strip()
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "DEMO_CLOUD_LIVE", "OPENAI_API_KEY")}
    proc = popen_hidden([py312, "-m", "demo.live", "--mode", "replay", "--replay", str(run_file), "--port", str(PORT),
                         "--idle-speedup", str(speedup)], cwd=str(REPO), env=env,
                        stdout=open(os.devnull, "w"), stderr=open(os.devnull, "w"))
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + "/api/meta", timeout=2).read()
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("screen exited early")
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("screen did not start")


def record(pw, scenario):
    vdir = HERE / "_video_live"
    vdir.mkdir(exist_ok=True)
    b = pw.chromium.launch()
    ctx = b.new_context(viewport=SIZE, record_video_dir=str(vdir), record_video_size=SIZE)
    page = ctx.new_page()
    t0 = time.time()
    page.goto(BASE + "/?scenario=%s&autorun=1" % scenario)
    page.wait_for_timeout(2000)
    page.wait_for_function('document.querySelector("#go").disabled === false', timeout=900000)
    page.wait_for_timeout(4000)
    path = page.video.path()
    ctx.close()
    b.close()
    out = HERE / ("demo_live_%s.webm" % scenario)
    if out.exists():
        out.unlink()
    Path(path).replace(out)
    try:
        vdir.rmdir()
    except OSError:
        pass
    print("video %s, wall %.1f s" % (out.name, time.time() - t0))


def main():
    scenario, run_file = sys.argv[1], sys.argv[2]
    speedup = float(sys.argv[sys.argv.index("--idle-speedup") + 1]) if "--idle-speedup" in sys.argv else 1.0
    from playwright.sync_api import sync_playwright
    proc = start(run_file, speedup)
    try:
        with sync_playwright() as pw:
            record(pw, scenario)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    print("screen stopped")


if __name__ == "__main__":
    main()
