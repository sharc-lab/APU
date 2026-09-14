"""Pre-commit hook: enforce the AM-014 raw/ payload ceiling.

AM-014 declares that ``raw/`` payloads are committed to git while total payload
size stays under a ceiling declared in ``configs/repo.yaml``. Above the ceiling,
payloads move to an externally archived bundle and ``raw/MANIFEST.sha256``
becomes the in-repo audit index.

The ceiling is enforced here so the transition is a pre-registered rule that
fires on its own, rather than an ad hoc reaction under pressure once M2
telemetry starts producing samples.ndjson at 1-10 Hz.

Delegates to ``seam.raw_retention.assert_raw_under_ceiling`` so there is exactly
one implementation of the policy.

Exit 0 = under ceiling, 1 = over.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    raw_dir = REPO_ROOT / "raw"
    if not raw_dir.is_dir():
        return 0

    try:
        from seam.raw_retention import assert_raw_under_ceiling
    except ImportError as exc:  # pragma: no cover - environment issue, not policy
        sys.stderr.write(
            f"note: could not import seam.raw_retention ({exc}); "
            "raw/ ceiling not verified this commit.\n"
        )
        return 0

    try:
        total = assert_raw_under_ceiling(REPO_ROOT)
    except Exception as exc:
        sys.stderr.write("\nBLOCKED — AM-014 raw/ ceiling exceeded:\n")
        sys.stderr.write(f"  {exc}\n")
        sys.stderr.write(
            "\nThis is the pre-registered transition point, not an error to work around.\n"
            "Move raw/ payloads to the external archive, regenerate raw/MANIFEST.sha256\n"
            "as the in-repo audit index, and record the transition in AUDIT_LOG.md.\n"
            "Do NOT raise ceiling_mb in configs/repo.yaml to make this pass.\n\n"
        )
        return 1

    sys.stderr.write(f"raw/ payload {total / 1_000_000:.1f} MB — under AM-014 ceiling\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
