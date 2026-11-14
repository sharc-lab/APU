"""Each Paper 1 figure script (analysis/figures/fig_*.py) runs and writes its vector PDF, and the shared helper refuses
a register row that is not VERIFIED. Scripts are imported and called in-process (no subprocess)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "analysis" / "figures"
SCRIPTS = ["fig_wrong_by_tier", "fig_fh1", "fig_f1", "fig_f2", "fig_f5", "fig_f6", "fig_f7"]


def _load(name):
    if str(FIG_DIR) not in sys.path:
        sys.path.insert(0, str(FIG_DIR))
    spec = importlib.util.spec_from_file_location(f"paper_{name}", FIG_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", SCRIPTS)
def test_figure_script_writes_pdf(name, tmp_path):
    mod = _load(name)
    out = Path(mod.main(tmp_path))
    assert out == tmp_path / f"{name}.pdf"
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 1000


@pytest.mark.parametrize("name", SCRIPTS)
def test_figure_script_reads_only_verified_rows(name):
    common = _load("_common")
    mod = _load(name)
    assert mod.ROWS, f"{name} names no register rows"
    for cid in mod.ROWS:
        assert common.register_status(cid) == "VERIFIED", cid
        assert cid in (mod.__doc__ or ""), f"{name} docstring does not name {cid}"


def test_unverified_row_is_refused():
    common = _load("_common")
    with pytest.raises(RuntimeError):
        common.entry("no-such-register-row")
    pending = next((line.split("|")[1].strip() for line in common.REGISTER_MD.read_text(encoding="utf-8").splitlines()
                    if line.startswith("| ") and "| PENDING |" in line), None)
    if pending:
        with pytest.raises(RuntimeError):
            common.entry(pending)
