"""JSON load for files PowerShell may have written with a UTF-8 BOM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Read JSON. ``utf-8-sig`` accepts a leading BOM and a BOM-less file."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
