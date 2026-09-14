"""SEAM - Silicon-aware Exploration of Agentic Model partitioning.

Sharc Lab @ Georgia Tech.

Governing documents:
    ``docs/SEAM_research_blueprint.md``           - governing research protocol (§5 audit standard)
    ``docs/PHASE_MINUS1_IMPLEMENTATION_SPEC.md``  - what this package implements

This package is scientific instrumentation, not a demo. Three invariants hold everywhere:

1. **No number without a manifest.** Any emitted value traces to a ``run_id``. If it cannot, it
   does not exist (spec §9.2).
2. **``raw/`` is write-once.** Never modified after a run completes (spec §9.1).
3. **No silent anything.** No swallowed exceptions, no silent retries, no silent fallbacks. A
   fallback is an event that gets logged (spec §9.6).
"""

from __future__ import annotations

__all__ = ["SPEC_VERSION", "__version__"]

__version__ = "0.1.0"

#: Manifest schema version. Bump only alongside ``seam/schemas/run_manifest.schema.json``
#: and an entry in ``AMENDMENTS.md``.
SPEC_VERSION = "1.0"
