"""Every OpenAI call must go through the capped client in src/cloud/client.py.

Operator decision 2026-10-08: src/cloud/client.py enforces a hard USD 50 total cap (HARD_SPEND_CAP_USD),
writes results/cloud_ledger.jsonl and fires alerts at 50/75/90%. Before this test existed, three scripts
(harness/adapters/sdk_direct.py, harness/tail_latency_instrument.py, harness/backends/cloud_openai.py) read
OPENAI_API_KEY and built their own OpenAI SDK client, so none of their spend was capped or recorded.

This test scans every tracked .py file (git ls-files) for a direct OpenAI client path and fails if one is
found outside the allowlist, which holds only the capped client module and this test file (which has to
spell the patterns out). Flagged:
  - `import openai` / `from openai ...` / importlib.import_module("openai") / __import__("openai")
  - `OpenAI(` / `AsyncOpenAI(` (constructing an SDK client)
  - `api.openai.com` (raw HTTP to OpenAI via requests/httpx/urllib)
  - os.environ[...] / os.environ.get(...) / os.getenv(...) / .pop / .setdefault of "OPENAI_API_KEY"
A local OpenAI-compatible server (Ollama, llama-server) is reached through
src.cloud.client.local_openai_compatible_client, which refuses openai.com hosts.

Modelled on tests/test_no_visible_console_windows.py and tests/test_no_absolute_paths.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
from proc_util import run_hidden  # noqa: E402

ALLOWLIST = {
    "src/cloud/client.py",                    # the capped client: the only module allowed to talk to OpenAI
    "tests/test_no_direct_openai_clients.py",  # this file (spells the patterns out)
}

PATTERNS = {
    "import openai": re.compile(r"^\s*import\s+openai\b", re.MULTILINE),
    "from openai": re.compile(r"^\s*from\s+openai\b", re.MULTILINE),
    "dynamic import of openai": re.compile(r"(?:import_module|__import__)\s*\(\s*[\"']openai[\"']"),
    "OpenAI( / AsyncOpenAI(": re.compile(r"(?<![A-Za-z0-9_])(?:Async)?OpenAI\s*\("),
    "api.openai.com": re.compile(r"api\.openai\.com", re.IGNORECASE),
    "OPENAI_API_KEY env read": re.compile(
        r"os\.(?:environ\s*(?:\[|\.get\s*\(|\.pop\s*\(|\.setdefault\s*\()|getenv\s*\()\s*[\"']OPENAI_API_KEY[\"']"
    ),
}


def _tracked_py_files() -> list[str]:
    out = run_hidden(["git", "ls-files"], capture_output=True, text=True, cwd=REPO, check=True)
    files = (line.strip().replace("\\", "/") for line in out.stdout.splitlines())
    return [f for f in files if f.endswith(".py")]


def _scan(text: str) -> list[tuple[str, int]]:
    hits = []
    for name, pat in PATTERNS.items():
        for m in pat.finditer(text):
            hits.append((name, text.count("\n", 0, m.start()) + 1))
    return hits


def test_allowlisted_files_exist():
    """A stale allowlist entry (renamed module) would otherwise go unnoticed."""
    for rel in ALLOWLIST:
        assert (REPO / rel).exists(), f"allowlisted file missing: {rel}"


def test_patterns_catch_known_bypasses():
    """The scan itself must flag every form it claims to; guards against a regex that matches nothing."""
    samples = [
        "import openai\n",
        "from openai import OpenAI\n",
        "client = OpenAI(api_key=k)\n",
        "client = openai.AsyncOpenAI()\n",
        "requests.post('https://api.openai.com/v1/chat/completions')\n",
        "k = os.environ['OPENAI_API_KEY']\n",
        'k = os.environ.get("OPENAI_API_KEY")\n',
        'k = os.getenv("OPENAI_API_KEY")\n',
        'm = importlib.import_module("openai")\n',
    ]
    for s in samples:
        assert _scan(s), f"pattern scan missed: {s!r}"
    for s in ["backend = CloudOpenAIBackend(model=m)\n", "def to_openai_messages(x):\n",
              "# talks to /v1/chat/completions on the local llama-server\n"]:
        assert not _scan(s), f"false positive on: {s!r}"


def test_no_direct_openai_client_use_outside_capped_client():
    files = _tracked_py_files()
    assert "src/cloud/client.py" in files, "git ls-files returned an unexpected list; scan would be vacuous"
    violations = []
    for rel in files:
        if rel in ALLOWLIST:
            continue
        path = REPO / rel
        if not path.exists():
            continue  # deleted in the working tree but still tracked
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, line in _scan(text):
            violations.append(f"{rel}:{line}: {name}")
    assert not violations, (
        "Direct OpenAI client use outside src/cloud/client.py (route it through CloudClient(mode='real') so "
        "the USD 50 cap, ledger and alerts apply):\n  " + "\n  ".join(violations)
    )
