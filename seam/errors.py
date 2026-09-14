"""Exception hierarchy for SEAM.

Every failure mode the harness can hit has a named type. This exists so that nothing is ever
caught as a bare ``Exception`` and quietly discarded (spec §9.6): a caller that wants to tolerate
a specific failure must name it, and naming it makes the tolerance reviewable.
"""

from __future__ import annotations

__all__ = [
    "BackendError",
    "BudgetExceededError",
    "ConfigError",
    "DirtyTreeError",
    "EscalationRefusedError",
    "GitError",
    "IsolationViolationError",
    "ManifestValidationError",
    "PinnedConditionError",
    "ProfileMismatchError",
    "ProvenanceError",
    "RawStoreError",
    "RunSealedError",
    "SeamError",
    "TopologyNotVerifiedError",
    "TopologyVerificationError",
]


class SeamError(Exception):
    """Base class for every SEAM error."""


class ConfigError(SeamError):
    """A configuration file is missing, malformed, or internally inconsistent."""


class GitError(SeamError):
    """Git state could not be determined.

    Raised rather than defaulted. A manifest with an unknown ``git_sha`` is not auditable, so an
    unavailable git is a hard failure and never a ``null`` field.
    """


class DirtyTreeError(SeamError):
    """The working tree has uncommitted changes and ``--allow-dirty`` was not passed.

    Spec §7/M1: a run on a dirty tree is not reproducible, because ``git_sha`` does not describe
    the code that actually executed.
    """


class ManifestValidationError(SeamError):
    """A manifest failed JSON-Schema validation."""


class ProvenanceError(SeamError):
    """A declared provenance artifact is missing or unreadable.

    Blueprint §5.2 requires every run to pin the provenance snapshot it relied on, so a missing
    artifact invalidates the run rather than degrading it.
    """


class PinnedConditionError(SeamError):
    """A session outside MACHINE.md's pinned run conditions tried to commit a platform result.

    MACHINE.md § "Pinned run conditions (MANDATORY)" classifies such a session as **INVALID, not
    noisy**. The session is still allowed to run and to emit its manifest - the measurement is real
    and stays citable - but it may not write a result back into platform config.
    """


class ProfileMismatchError(PinnedConditionError):
    """The host state does not match the pinned profile required by the measurement class.

    Spec §3.7: emit a refusal manifest recording the mismatch, then stop. Refusal is a data point.
    """


class TopologyVerificationError(SeamError):
    """Empirical P/LP-E verification ran but did not produce a clean, expected result."""


class TopologyNotVerifiedError(SeamError):
    """An affinity list was requested but the platform config carries no verified mapping.

    Spec §4 requires the verified mapping to be asserted on every run. Guessing the mapping would
    silently answer open question 4 with an assumption.
    """


class RawStoreError(SeamError):
    """A ``raw/`` invariant was violated."""


class RunSealedError(RawStoreError):
    """An attempt was made to write into a run directory that is already sealed.

    This is the write-once guard (spec §9.1, blueprint §5.3). Raised on *attempt*, before any
    bytes are written.
    """


class BudgetExceededError(SeamError):
    """A paid call or a run was refused because it would exceed a spend ceiling.

    Raised on *attempt*, before the request leaves the process, so the ceiling is a structural
    property rather than a post-hoc observation. The caller emits a refusal manifest first -
    a refused run is a data point, not an error to be swallowed.
    """


class IsolationViolationError(SeamError):
    """Two arms of a comparison differ in something other than the variable under test.

    AMENDMENTS.md AM-022: between ``cpu-p`` and ``cpu-lpe`` the only permitted difference is
    measured throughput. Anything else silently converts the experiment into a comparison of
    something nobody chose to study.
    """


class EscalationRefusedError(SeamError):
    """A step tried to escalate in an arm where escalation is disabled.

    Deliberately **not** a :class:`BackendError`: the harness treats a cloud ``BackendError`` as a
    transient failure and falls back to local with the step marked. That is correct for a hybrid run
    whose network broke, and wrong here - it would turn a violated arm invariant into a trajectory
    that looks fine.
    """


class BackendError(SeamError):
    """A local or cloud execution backend failed in a way the harness will not paper over.

    Spec §9.6 forbids silent retries and swallowed exceptions. Every occurrence is logged as an
    event before this is raised.
    """
