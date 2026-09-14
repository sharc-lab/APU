"""Session-wide pytest configuration.

Fails immediately when the interpreter is below the declared Python floor
(requires-python = ">=3.12" in pyproject.toml).  A clear error here is
preferable to cryptic collection failures on unsupported syntax.
"""

from __future__ import annotations

import sys


def pytest_configure(config):
    if sys.version_info < (3, 12):
        import pytest
        pytest.exit(
            reason=(
                f"Python {sys.version_info.major}.{sys.version_info.minor} is below the "
                f"required floor (>=3.12).\n"
                "Install Python 3.12+ and re-run: py -3.12 -m pytest\n"
                f"Detected: {sys.version}"
            ),
            returncode=2,
        )
