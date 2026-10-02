"""2026-10-02: terminal windows were flashing open and closing on the controller laptop while scheduled/
background work ran. Root cause: the scheduled task's own parent (pythonw.exe) is console-less by design
(confirmed live against the real APU-SyncResults task registration), but a bare subprocess.run/Popen/
check_output call for a console-subsystem child (ssh.exe, scp.exe, git.exe, powershell.exe) has no console
to inherit from a console-less parent, so Windows allocates a brand-new, visible console for EACH such call.

Fix: harness/proc_util.py's run_hidden/popen_hidden/check_output_hidden wrap the real subprocess functions
with CREATE_NO_WINDOW + a hidden STARTUPINFO. This test greps every controller-side script (one that can run
from a Windows scheduled task or in the background on this laptop -- not a script deployed to and run ON
evo-t2s/evo-x2, and not a Linux-only script that never runs on this laptop at all) for a bare
subprocess.run/Popen/check_output/call outside of proc_util.py itself, and fails if one is found, so a future
edit cannot silently reintroduce a flashing console.

Scripts deliberately NOT covered here:
  - scripts/t2s_handover.py: deployed to and runs ON evo-t2s itself, not this laptop.
  - scripts/verify_platform.py: Linux-only checks (/proc/meminfo, `free`), never runs on this laptop.
  - anything under harness/: deployed to and runs on the remote machines, not this laptop.
"""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CONTROLLER_SCRIPTS = [
    REPO / "scripts" / "sync_results.py",
    REPO / "scripts" / "deploy_evo.py",
    REPO / "scripts" / "t2s_responsiveness_probe.py",
]

BARE_FUNCS = {"run", "Popen", "check_output", "call"}  # the subprocess functions that can spawn a visible console


def _find_bare_subprocess_calls(path):
    """AST-based (not regex/grep over text) so a docstring or comment that merely MENTIONS subprocess.run(...)
    -- as this very file's own module docstring does -- is never mistaken for a real call."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr in BARE_FUNCS
                and isinstance(func.value, ast.Name) and func.value.id == "subprocess"):
            try:
                label = path.relative_to(REPO)
            except ValueError:
                label = path
            violations.append(f"{label}:{node.lineno}: subprocess.{func.attr}(...)")
    return violations


def test_controller_scripts_exist():
    """A guard against the list above silently going stale (a renamed/moved file would make every other
    assertion in this file vacuously pass)."""
    for path in CONTROLLER_SCRIPTS:
        assert path.exists(), f"expected controller script not found: {path}"


def test_no_bare_subprocess_calls_in_controller_scripts():
    violations = []
    for path in CONTROLLER_SCRIPTS:
        violations.extend(_find_bare_subprocess_calls(path))
    assert not violations, (
        "bare subprocess call(s) found in controller-side script(s) -- each one spawns a visible console "
        "window on the controller when run under pythonw.exe (e.g. the APU-SyncResults scheduled task). "
        "Use harness/proc_util.run_hidden/popen_hidden/check_output_hidden instead:\n" + "\n".join(violations)
    )


def test_detector_actually_catches_a_real_bare_call(tmp_path):
    """Proves the AST-based detector is not vacuously passing (e.g. from a broken attribute match) by feeding
    it source that unambiguously contains a real bare call, outside of any comment or docstring."""
    bad = tmp_path / "bad.py"
    bad.write_text("import subprocess\nsubprocess.run(['echo', 'hi'])\n", encoding="utf-8")
    violations = _find_bare_subprocess_calls(bad)
    assert len(violations) == 1
    assert "subprocess.run" in violations[0]


def test_detector_ignores_a_docstring_mention_of_subprocess_run():
    """The exact false-positive this test file's own regex-based first draft hit: a docstring that merely
    MENTIONS subprocess.run(...) as prose must not be flagged."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mentions_only.py"
        path.write_text('"""Found live: subprocess.run(..., timeout=...) raises on timeout."""\n', encoding="utf-8")
        assert _find_bare_subprocess_calls(path) == []


def test_controller_scripts_actually_import_proc_util():
    """A script with zero subprocess calls at all would vacuously pass the test above without ever having
    adopted the fix -- confirm each listed script really does import the hidden-window helper."""
    for path in CONTROLLER_SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert "proc_util" in text, f"{path.relative_to(REPO)} does not import proc_util at all"
