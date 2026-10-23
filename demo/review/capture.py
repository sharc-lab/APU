"""Screenshots and the backup recording of the demo dashboard, for demo/review/ (Playwright, headless Chromium).

    py -3.8 demo/review/capture.py shots     # 1600x1000 screenshots into demo/review/shots/
    py -3.8 demo/review/capture.py video     # about 3 minutes of story mode into demo/review/demo_story.webm

Python 3.8 because that is the interpreter with Playwright 1.48 (chromium-1140) on the controller laptop; the
dashboard itself runs under Python 3.12, started here as a hidden child process on 127.0.0.1 and always stopped in a
finally block (no process is left running). The child gets no ANTHROPIC_API_KEY and no DEMO_CLOUD_LIVE, so the
escalation step and the step 4 live button show the recorded redo and no cloud call can happen. No audio.
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

PORT = 8799
BASE = "http://127.0.0.1:%d" % PORT
SIZE = {"width": 1600, "height": 1000}
SHOTS = HERE / "shots"


def start_server():
    py312 = check_output_hidden(["py", "-3.12", "-c", "import sys; print(sys.executable)"], text=True).strip()
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "DEMO_CLOUD_LIVE", "OPENAI_API_KEY")}
    proc = popen_hidden([py312, "-m", "demo.dashboard", "--port", str(PORT)], cwd=str(REPO), env=env,
                        stdout=open(os.devnull, "w"), stderr=open(os.devnull, "w"))
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + "/api/meta", timeout=2).read()
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("dashboard exited early")
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("dashboard did not start")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def settle(page, ms=900):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def shots(pw):
    SHOTS.mkdir(exist_ok=True)
    b = pw.chromium.launch()
    page = b.new_page(viewport=SIZE)
    # the landing page is story step 1; each step at 1600x1000, plus a crop of its main screen for the leave-behind
    for n in range(1, 6):
        page.goto(BASE + ("/" if n == 1 else "/?step=%d" % n))
        settle(page)
        page.screenshot(path=str(SHOTS / ("story_step%d.png" % n)))
        page.locator("#story-panel").first.screenshot(path=str(SHOTS / ("leave_step%d.png" % n)))
    # the accuracy-tolerance slider (step 4) at 12 points, and the tool card (step 5) on the Intel laptop
    page.goto(BASE + "/?step=4&tol=12")
    settle(page)
    page.screenshot(path=str(SHOTS / "story_step4_t12.png"))
    page.goto(BASE + "/?step=4")
    settle(page)
    page.click("#s-live")  # no key in this server: the recorded fallback
    page.wait_for_selector("#s-live-out .s-line b")
    settle(page, 300)
    page.screenshot(path=str(SHOTS / "story_step4_live.png"))
    page.goto(BASE + "/?step=5&machine=Intel%20laptop")
    settle(page)
    page.screenshot(path=str(SHOTS / "story_step5_intel.png"))
    page.goto(BASE + "/?step=2")
    settle(page)
    page.click("#story-panel details summary")
    page.wait_for_timeout(400)
    page.screenshot(path=str(SHOTS / "story_step2_details.png"))
    # the engineering view: header (hardware, cloud banner), then each section in its default state and one
    # interaction each
    page.goto(BASE + "/?explore=1")
    settle(page, 1500)
    page.screenshot(path=str(SHOTS / "eng_top.png"))
    page.locator("#inputs").screenshot(path=str(SHOTS / "eng_inputs.png"))
    page.locator("#frontier").screenshot(path=str(SHOTS / "eng_frontier.png"))
    page.locator("#live").screenshot(path=str(SHOTS / "eng_replay.png"))
    page.locator("#silent").screenshot(path=str(SHOTS / "A_default.png"))
    page.click("#silent details summary")
    page.wait_for_timeout(400)
    page.locator("#silent").screenshot(path=str(SHOTS / "A_details.png"))
    page.locator("#device").screenshot(path=str(SHOTS / "B_default.png"))
    page.select_option("#devform select[name=device]", "evo-t2s-cpu")
    settle(page)
    page.locator("#device").screenshot(path=str(SHOTS / "B_cpu.png"))
    page.select_option("#devform select[name=device]", "blade")
    settle(page)
    page.locator("#device").screenshot(path=str(SHOTS / "B_blade.png"))
    page.locator("#cost").screenshot(path=str(SHOTS / "C_default.png"))
    page.click("#cost .lead a.num")
    page.wait_for_selector("#src-dialog[open]")
    page.wait_for_timeout(500)
    page.screenshot(path=str(SHOTS / "C_source.png"))
    page.keyboard.press("Escape")
    page.locator("#escalate").screenshot(path=str(SHOTS / "D_default.png"))
    page.click("#esc-run")
    page.wait_for_selector("#esc-result .devcard")
    settle(page, 500)
    page.locator("#escalate").screenshot(path=str(SHOTS / "D_sent.png"))
    b.close()


def video(pw, seconds_per_step=36):
    vdir = HERE / "_video"
    vdir.mkdir(exist_ok=True)
    b = pw.chromium.launch()
    ctx = b.new_context(viewport=SIZE, record_video_dir=str(vdir), record_video_size=SIZE)
    page = ctx.new_page()
    t0 = time.time()
    page.goto(BASE + "/?step=1")
    settle(page, 300)
    for n in range(1, 6):
        start = time.time()
        if n == 4:  # the live button (recorded fallback: no key here), then the slider: 5 -> 12 -> 20 -> 5
            page.wait_for_timeout(5000)
            page.click("#s-live")
            page.wait_for_selector("#s-live-out .s-line b")
            for v in (12, 20, 5):
                page.wait_for_timeout(7000)
                page.evaluate("(v) => { const s = document.querySelector('#s-tol'); s.value = v; "
                              "s.dispatchEvent(new Event('input')); }", v)
        if n == 5:  # pick each machine
            for m in ("Intel laptop", "NVIDIA laptop", "AMD workstation"):
                page.wait_for_timeout(6000)
                page.select_option("#s-machine", m)
        # slow scroll through the step, then back to the top
        for y in (0, 300, 600, 900, 300, 0):
            page.mouse.wheel(0, y - page.evaluate("window.scrollY"))
            page.wait_for_timeout(1500)
        left = seconds_per_step - (time.time() - start)
        if left > 0:
            page.wait_for_timeout(int(left * 1000))
        if n < 5:
            page.click("#story-next")
            settle(page, 200)
    page.wait_for_timeout(3000)
    path = page.video.path()
    ctx.close()
    b.close()
    out = HERE / "demo_story.webm"
    if out.exists():
        out.unlink()
    Path(path).replace(out)
    try:
        vdir.rmdir()
    except OSError:
        pass
    print("video %s, wall %.1f s" % (out.name, time.time() - t0))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "shots"
    from playwright.sync_api import sync_playwright
    proc = start_server()
    try:
        with sync_playwright() as pw:
            shots(pw) if mode == "shots" else video(pw)
    finally:
        stop_server(proc)
    print("server stopped")


if __name__ == "__main__":
    main()
