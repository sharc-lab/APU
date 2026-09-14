"""Isolation mode: which machine state a run was measured under, and the rule against pooling.

Two prior runs differed by a factor of 1.98 with identical configuration. The difference was
machine state -- one measured with the desktop in use, one with it closed down. Nothing in the
manifest recorded that, so the difference was invisible to analysis and the ratio was
unexplainable rather than merely large.

This module makes that state a first-class recorded property:

``local``
    Measured with the machine in normal use. Development, smoke runs, anything untimed.

``remote``
    Measured in the mode of ``docs/SETUP_remote_measurement.md``: editor and browsers closed,
    driven over SSH, nobody touching the machine while the run is in flight.

Two protections follow from recording it.

**Declaration is mandatory and fails closed.** There is no default. A run that does not declare
its mode does not emit a manifest, because a defaulted mode is exactly the silent mislabel this
exists to prevent -- and it would be wrong in whichever direction the default was chosen.

**A declaration that contradicts the machine is refused.** Declaring ``remote`` while tier-1
operator-controlled software is resident is not a labelling slip, it is a claim about the
measurement that is false. Tier-1 processes are enumerated in the refusal so the operator knows
what to close. Tier-2 auto-respawning shell / vendor agents are recorded (in isolation evidence
and in every cell's ``environment_start`` / ``environment_peak``) but do not refuse: they return
within ~5 s of Stop-Process (verified 2026-08-09; hosts MicrosoftWindows.Client.CBS and
MicrosoftWindows.Client.WebExperience). The Available-MBytes floor remains the substantive
memory gate and is unchanged.

:func:`assert_poolable` is the consumption-side half: analysis that spans modes is invalid
regardless of how clean each half looks on its own.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from seam.errors import SeamError
from seam.jsonlog import log_event

__all__ = [
    "CONTENDING_PROCESS_NAMES",
    "ISOLATION_MODES",
    "TIER1_CONTENDING_PROCESS_NAMES",
    "TIER2_CONTENDING_PROCESS_NAMES",
    "IsolationModeError",
    "assert_poolable",
    "gather_isolation_evidence",
    "harness_termination_record",
    "resolve_isolation_mode",
]

ISOLATION_MODES: Final = ("remote", "local")

ENV_VAR: Final = "SEAM_ISOLATION_MODE"

# Tier 1 - REFUSE when resident under isolation_mode=remote.
# Operator-controlled interactive / background software. Cursor alone was roughly 20% of
# system RAM. Named rather than inferred from load because a process can be resident and
# momentarily idle during the quiescence window and still hold the memory that changes where
# the context ceiling falls.
#
# 2026-08-09: a 3.1 GB Available gap (loaded 4401 vs clean 7477 MB) moved gpu_only n=12000
# prefill ~2.5x. Names are matched case-insensitively; psutil reports the ``.exe`` suffix on
# Windows.
TIER1_CONTENDING_PROCESS_NAMES: Final = frozenset(
    {
        "cursor.exe",
        "code.exe",
        "chrome.exe",
        "msedge.exe",
        "firefox.exe",
        "brave.exe",
        "slack.exe",
        "discord.exe",
        "teams.exe",
        "ms-teams.exe",
        "spotify.exe",
        "outlook.exe",
        "obsidian.exe",
        "docker desktop.exe",
        "vmmem.exe",
        "claude.exe",
    }
)

# Tier 2 - RECORD only; do NOT refuse.
# Auto-respawning Windows shell packages and OEM vendor agents. Verified 2026-08-09:
# msedgewebview2 hosts are MicrosoftWindows.Client.CBS (Search / Copilot shell) and
# MicrosoftWindows.Client.WebExperience (Widgets); killing them alongside Widgets and
# SearchHost, six msedgewebview2 processes returned with new PIDs inside a 5 s sleep before
# the gate ran. Their private working set (~194 MiB) is below the noise of the Available
# floor (7000 MB; clean session had ~7891 MB Available). Presence and private WS are written
# into environment_start / environment_peak of every cell so contamination questions stay
# answerable from the artifact. The Available floor itself is unchanged.
TIER2_CONTENDING_PROCESS_NAMES: Final = frozenset(
    {
        "msedgewebview2.exe",
        "searchhost.exe",
        "widgets.exe",
        "workloadssessionhost.exe",
        "delloptimizer.systray.exe",
        "supportassistagent.exe",
        "icps.exe",
    }
)

# Union of both tiers (discovery / documentation). Refuse gates use TIER1 only.
CONTENDING_PROCESS_NAMES: Final = TIER1_CONTENDING_PROCESS_NAMES | TIER2_CONTENDING_PROCESS_NAMES


def _normalize_process_name(name: str) -> str:
    """Lowercase; ensure a trailing ``.exe`` so bare ProcessName matches the frozenset."""
    lowered = str(name or "").strip().lower()
    if lowered and not lowered.endswith(".exe"):
        lowered = f"{lowered}.exe"
    return lowered


def _process_row(name: str, pid: int, memory_info: Any) -> dict[str, Any]:
    private = None
    rss = None
    if memory_info is not None:
        rss = int(memory_info.rss)
        private_attr = getattr(memory_info, "private", None)
        private = int(private_attr) if private_attr is not None else rss
    return {
        "name": name,
        "pid": pid,
        "working_set_bytes": rss,
        "private_working_set_bytes": private,
    }


class IsolationModeError(SeamError):
    """The isolation mode was undeclared, invalid, or contradicted by the machine."""


def harness_termination_record() -> dict[str, Any]:
    """Record whether this harness terminated interactive software, and if so when.

    Remote mode refuses while tier-1 contending processes are resident; it does not kill them.
    Operator kills outside the harness are therefore unobservable. Never invent a kill
    timestamp from process absence or from launch timing.
    """
    return {
        "harness_did_not_terminate": True,
        "operator_kill_unobservable": True,
        "seconds_since_harness_interactive_termination": None,
        "note": (
            "resolve_isolation_mode / acceptance.ps1 refuse remote mode while tier-1 "
            "operator-controlled software is resident; the harness does not terminate those "
            "processes. Tier-2 shell/vendor agents are recorded only. If the operator killed "
            "processes outside the harness, that kill time is unobservable here - do not infer it."
        ),
    }


def gather_isolation_evidence() -> dict[str, Any]:
    """Enumerate the machine state that corroborates or contradicts a mode declaration.

    Never raises on inspection failure: evidence that could not be gathered is recorded as such,
    because a probe that fails is not the same as a machine that is quiet, and collapsing the two
    would let an unverifiable declaration pass as a verified one.

    ``contending_processes`` is tier 1 only (refuse under remote). ``tier2_recorded_processes``
    is tier 2 (record only; never refuses).
    """
    evidence: dict[str, Any] = {
        "contending_processes": [],
        "tier2_recorded_processes": [],
        "sshd_session_count": None,
        "probe_error": None,
    }
    try:
        import psutil
    except ImportError as exc:
        evidence["probe_error"] = f"psutil unavailable: {exc}"
        return evidence

    tier1: list[dict[str, Any]] = []
    tier2: list[dict[str, Any]] = []
    sshd_sessions = 0
    try:
        for process in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                name = str(process.info["name"] or "")
                lowered = _normalize_process_name(name)
                if lowered == "sshd.exe":
                    sshd_sessions += 1
                    continue
                pid = int(process.info["pid"])
                memory_info = process.info.get("memory_info")
                if lowered in TIER1_CONTENDING_PROCESS_NAMES:
                    tier1.append(_process_row(name, pid, memory_info))
                elif lowered in TIER2_CONTENDING_PROCESS_NAMES:
                    tier2.append(_process_row(name, pid, memory_info))
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
    except Exception as exc:  # pragma: no cover - psutil enumeration is platform-dependent
        evidence["probe_error"] = f"process enumeration failed: {exc}"
        return evidence

    tier1.sort(key=lambda row: (row["name"], row["pid"]))
    tier2.sort(key=lambda row: (row["name"], row["pid"]))
    evidence["contending_processes"] = tier1
    evidence["tier2_recorded_processes"] = tier2
    # The listener itself is one sshd.exe; anything beyond that is a live session.
    evidence["sshd_session_count"] = max(0, sshd_sessions - 1)
    return evidence


def resolve_isolation_mode(
    declared: str | None = None,
    *,
    evidence: Mapping[str, Any] | None = None,
    enforce: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Resolve and verify the isolation mode for a run.

    Args:
        declared: Explicit declaration. Falls back to the ``SEAM_ISOLATION_MODE`` environment
            variable, which is how a detached launcher declares the mode for a process it does
            not otherwise talk to.
        evidence: Pre-gathered evidence. Gathered here when omitted.
        enforce: Refuse a declaration the evidence contradicts. Only a caller that is deliberately
            recording a contradiction should pass False.

    Returns:
        The mode and the evidence recorded with it.

    Raises:
        IsolationModeError: Undeclared, not one of :data:`ISOLATION_MODES`, or -- when enforcing --
            declared ``remote`` while tier-1 (operator-controlled) contending software is resident.
            Tier-2 residents do not refuse.
    """
    value = declared if declared is not None else os.environ.get(ENV_VAR)
    if value is None or not str(value).strip():
        raise IsolationModeError(
            "isolation_mode is not declared. Every manifest must record whether the run was "
            f"measured in `remote` or `local` mode. Set {ENV_VAR}, or pass the mode explicitly. "
            "There is no default: a defaulted mode would mislabel exactly the runs this field "
            "exists to keep apart (docs/SETUP_remote_measurement.md)."
        )

    mode = str(value).strip().lower()
    if mode not in ISOLATION_MODES:
        raise IsolationModeError(
            f"isolation_mode must be one of {list(ISOLATION_MODES)}, got {value!r}"
        )

    resolved_evidence = dict(evidence) if evidence is not None else gather_isolation_evidence()
    if "tier2_recorded_processes" not in resolved_evidence:
        resolved_evidence["tier2_recorded_processes"] = []
    # contending_processes = tier 1 only (refuse under remote).
    contending = list(resolved_evidence.get("contending_processes") or [])
    tier2 = list(resolved_evidence.get("tier2_recorded_processes") or [])

    if mode == "remote" and contending and enforce:
        rendered_rows: list[str] = []
        for row in contending:
            private_bytes = (
                row.get("private_working_set_bytes") or row.get("working_set_bytes") or 0
            )
            private_mib = private_bytes / 1024 / 1024
            rendered_rows.append(
                f"  - {row['name']} (pid {row['pid']}, private {private_mib:.0f} MiB)"
            )
        rendered = "\n".join(rendered_rows)
        raise IsolationModeError(
            "isolation_mode was declared `remote` but tier-1 operator-controlled software is "
            f"resident on this machine:\n{rendered}\n"
            "Remote mode means the machine is closed down and driven over SSH. Measuring with "
            "these running and labelling it `remote` would pool a contended run with quiet ones, "
            "which is the failure behind the unexplained 1.98x. Close them, or declare `local` "
            "and accept that the result may not be compared against remote-mode runs. "
            "(Tier-2 shell/vendor agents such as msedgewebview2 are recorded only and do not "
            "refuse; the Available-MBytes floor is the memory gate.)"
        )

    resolved_evidence["consistent_with_declaration"] = not (mode == "remote" and contending)
    log_event(
        "isolation.resolved",
        message=f"isolation_mode={mode}",
        isolation_mode=mode,
        contending_process_count=len(contending),
        tier2_recorded_process_count=len(tier2),
        sshd_session_count=resolved_evidence.get("sshd_session_count"),
    )
    return mode, resolved_evidence


def assert_poolable(manifests: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]) -> str:
    """Refuse to pool or compare runs measured under different isolation modes.

    Returns the single shared mode, so a caller can record which one the comparison belongs to.

    Raises:
        IsolationModeError: If the runs span modes, or if any run predates the field. A run
            without the field is not assumed to match: its machine state is unknown, and an
            unknown state is exactly what cannot be pooled.
    """
    seen: dict[str, list[str]] = {}
    for manifest in manifests:
        run_id = str(manifest.get("run_id", "<unknown>"))
        mode = manifest.get("isolation_mode")
        if mode is None:
            raise IsolationModeError(
                f"run {run_id} records no isolation_mode, so the machine state it was measured "
                "under is unknown. It cannot be pooled with or compared against anything."
            )
        seen.setdefault(str(mode), []).append(run_id)

    if not seen:
        raise IsolationModeError("no manifests supplied")

    if len(seen) > 1:
        rendered = "; ".join(f"{mode}: {', '.join(runs)}" for mode, runs in sorted(seen.items()))
        raise IsolationModeError(
            "refusing to compare runs measured under different isolation modes "
            f"({rendered}). A comparison spanning modes is invalid regardless of how clean each "
            "half looks (docs/SETUP_remote_measurement.md)."
        )

    return next(iter(seen))
