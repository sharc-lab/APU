"""M0 provenance tests - the AF-001 correction (blueprint Appendix B, spec §7/M0).

``analysis/aipc-c1/MACHINE.md`` is the provenance source for the ``platform`` block of every run
manifest (blueprint §5.2), so a regression here silently corrupts every number the project ever
reports. These tests pin the correction so it cannot be undone by a later merge.

They also pin the *evidence classification*, which is the part most likely to erode: the
distinction between probe-attested and inferred evidence is what makes the identification auditable
rather than asserted, and it is invisible to anyone who has not read AF-001.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = REPO_ROOT / "analysis"
MACHINE_MD = ANALYSIS / "aipc-c1" / "MACHINE.md"

PROBE_ARTIFACTS = [
    ANALYSIS / "_c1_machine_probe.txt",
    ANALYSIS / "_c1_drivers_probe.txt",
    ANALYSIS / "_c1_probe.txt",
    ANALYSIS / "_c1_mem_probe.txt",
    ANALYSIS / "_c1_tools_probe.txt",
]

#: Microarchitecture code names that Windows does not report through any probed interface.
INFERRED_CORE_NAMES = ["Cougar", "Darkmont", "Lion Cove", "Skymont"]

#: Byte-order marks mapped to the encoding they announce. The committed probe artifacts were
#: produced by PowerShell output redirection, which on Windows PowerShell 5.1 writes UTF-16LE with a
#: BOM. Decoding them as UTF-8 yields mojibake in which *every* substring assertion below is
#: vacuously satisfiable, so the encoding must be honoured rather than forced.
_BOM_ENCODINGS = [
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
]


def read_probe_text(path: Path) -> str:
    """Decode a committed probe artifact using its byte-order mark.

    The artifacts are read-only provenance (spec §9.1): they must never be re-encoded on disk to
    suit a reader, so the reader adapts instead. Falls back to UTF-8 only when no BOM is present,
    and does not pass ``errors="replace"`` - a probe artifact this project cannot decode is a
    finding to report, not damage to paper over.
    """
    raw = path.read_bytes()
    for bom, encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(encoding)
    return raw.decode("utf-8")


@pytest.fixture(scope="module")
def machine_md() -> str:
    assert MACHINE_MD.is_file(), f"provenance document missing at {MACHINE_MD}"
    return MACHINE_MD.read_text(encoding="utf-8")


# ==================================================================================================
# AF-001 - the correction itself
# ==================================================================================================


def test_machine_md_identifies_panther_lake(machine_md: str) -> None:
    assert "Panther Lake" in machine_md
    assert "Intel(R) Core(TM) Ultra 5 325" in machine_md


def test_machine_md_no_longer_asserts_lunar_lake(machine_md: str) -> None:
    """Every remaining mention of Lunar Lake must be contrastive evidence, never an assertion.

    AF-001 named only line 12, but the mislabel occurred twice - see AMENDMENTS.md AM-001. Lunar
    Lake still appears legitimately in the evidence tables ("Lunar Lake *would be* 64A0"), so this
    checks the two original assertion sites are gone rather than banning the string.
    """
    assert "Ultra 5 325 (Lunar Lake)" not in machine_md
    assert "Lunar Lake memory-side cache" not in machine_md


def test_machine_md_records_the_correction(machine_md: str) -> None:
    assert "AF-001" in machine_md
    assert "Corrected 2026-07-29" in machine_md


# ==================================================================================================
# Discriminating evidence
# ==================================================================================================


def test_machine_md_states_the_sku_discriminator(machine_md: str) -> None:
    """E1: a 3xx part is Core Ultra Series 3 = Panther Lake; Lunar Lake is 2xxV."""
    assert "3xx" in machine_md
    assert "2xxV" in machine_md


def test_machine_md_states_the_graphics_did_discriminator(machine_md: str) -> None:
    """E2: DID B090 is Xe3; Lunar Lake is 64A0 (Xe2)."""
    assert "B090" in machine_md
    assert "64A0" in machine_md


def test_machine_md_cites_the_source_artifact_for_each_load_bearing_claim(
    machine_md: str,
) -> None:
    """Load-bearing evidence must name the artifact and line it came from."""
    assert "analysis/_c1_machine_probe.txt" in machine_md
    assert "analysis/_c1_drivers_probe.txt" in machine_md


def test_machine_md_marks_core_names_as_non_independent(machine_md: str) -> None:
    """Spec §7/M0 item 4: if inferred, it must be marked as non-independent evidence."""
    assert "NON-INDEPENDENT" in machine_md
    assert "NOT probe-reported" in machine_md


def test_machine_md_states_the_topology_does_not_discriminate(machine_md: str) -> None:
    """Spec §7/M0 item 3: Lunar Lake has the identical 4+4/8T signature."""
    lowered = machine_md.lower()
    assert "does not discriminate" in lowered or "non-discriminating" in lowered
    assert "8c/8t" in lowered
    assert "identical" in lowered


def test_machine_md_distinguishes_load_bearing_from_non_load_bearing(machine_md: str) -> None:
    """Spec §7/M0 item 3: "State which evidence is load-bearing.\" """
    lowered = machine_md.lower()
    assert "load-bearing" in lowered
    assert "inferred" in lowered


# ==================================================================================================
# The finding behind the classification
# ==================================================================================================


@pytest.mark.parametrize("core_name", INFERRED_CORE_NAMES)
def test_core_names_appear_in_no_probe_artifact(core_name: str) -> None:
    """The empirical basis for classifying E3 as inferred.

    Windows exposes no microarchitecture code-name field, so these strings cannot be probe-reported.
    If a future probe *does* capture them, this test fails and the classification in MACHINE.md must
    be revisited - which is the correct outcome, not a nuisance.
    """
    for artifact in PROBE_ARTIFACTS:
        if not artifact.is_file():
            continue
        content = read_probe_text(artifact).lower()
        assert core_name.lower() not in content, (
            f"{core_name!r} was found in {artifact.name}; it is currently documented as a human "
            f"inference in MACHINE.md, so that classification is now wrong"
        )


def test_sku_string_is_probe_attested() -> None:
    """E1's independence rests on this string being in the artifact, not in a human's notes."""
    content = read_probe_text(PROBE_ARTIFACTS[0])
    assert "Intel(R) Core(TM) Ultra 5 325" in content


def test_graphics_device_id_is_probe_attested() -> None:
    """E2's independence rests on the raw PCI enumeration."""
    content = read_probe_text(PROBE_ARTIFACTS[1])
    assert "DEV_B090" in content


def test_probe_artifacts_do_not_attest_the_core_topology() -> None:
    """The 4P/4LP-E split is unverified by artifact, which is why M1 measures it.

    Recorded as a test because "no artifact says this" is an easy claim to lose track of, and it is
    the justification for AMENDMENTS.md AM-008.
    """
    for artifact in PROBE_ARTIFACTS:
        if not artifact.is_file():
            continue
        content = read_probe_text(artifact).lower()
        assert "efficiencyclass" not in content


# ==================================================================================================
# Audit trail
# ==================================================================================================


def test_audit_log_records_both_findings() -> None:
    audit = (REPO_ROOT / "AUDIT_LOG.md").read_text(encoding="utf-8")
    assert "AF-001" in audit
    assert "AF-002" in audit
    assert "Panther Lake" in audit


def test_audit_log_states_whether_core_names_were_probe_reported() -> None:
    """The specific question spec §7/M0 item 4 asks must be answered in the log, not just the doc."""
    audit = (REPO_ROOT / "AUDIT_LOG.md").read_text(encoding="utf-8")
    assert "NOT probe-reported" in audit or "not a probe-reported" in audit


def test_amendments_log_exists_and_records_divergences() -> None:
    amendments = (REPO_ROOT / "AMENDMENTS.md").read_text(encoding="utf-8")
    assert "AM-001" in amendments
    assert "blueprint governs" in amendments.lower()
