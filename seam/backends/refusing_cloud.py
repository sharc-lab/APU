"""Cloud backend that refuses to generate. Used by escalation-disabled arms.

``run_task`` takes ``cloud_backend`` unconditionally, so an arm that must never escalate still has
to pass one. A no-op stub is the wrong choice: it would return an empty result, the step would be
recorded as executed on ``cloud``, and the run would produce a hybrid trajectory carrying a
local-only label.

This backend raises instead - and raises :class:`~seam.errors.EscalationRefusedError`, **not**
:class:`~seam.errors.BackendError`, because ``run_task`` catches ``BackendError`` on a cloud step
and falls back to local. Falling back is right for a hybrid run whose network failed; here it would
silently convert a protocol violation into a successful-looking trajectory.
"""

from __future__ import annotations

from typing import Final

from seam.backends.base import GenerationRequest, GenerationResult, PreflightVerdict
from seam.errors import EscalationRefusedError
from seam.jsonlog import log_event

__all__ = ["RefusingCloudBackend"]

_REASON: Final = (
    "escalation is disabled in this arm (deadline = DEADLINE_DISABLED_S); no cloud call may be "
    "issued, so reaching this backend means the routing rule escalated a step that could not have "
    "exceeded the sentinel deadline"
)


class RefusingCloudBackend:
    """A ``Backend`` whose ``generate`` always raises."""

    def __init__(self, *, run_id: str = "", arm: str = "arm_l_escalation_disabled") -> None:
        self._run_id = run_id
        self._arm = arm
        self.call_attempts = 0

    @property
    def name(self) -> str:
        return "refusing_cloud"

    @property
    def model_ref(self) -> str:
        return "none/escalation-disabled"

    def preflight(self) -> PreflightVerdict:
        return PreflightVerdict("UNSUPPORTED", reason=_REASON, detail={"arm": self._arm})

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Always raise.

        Raises:
            EscalationRefusedError: Unconditionally.
        """
        self.call_attempts += 1
        log_event(
            "backend.cloud_refused",
            severity="critical",
            message=f"escalation attempted in {self._arm}: {_REASON}",
            run_id=self._run_id,
            arm=self._arm,
            attempts=self.call_attempts,
            n_messages=len(request.messages),
        )
        raise EscalationRefusedError(_REASON)
