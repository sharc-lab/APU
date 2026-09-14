"""Enforcement tests for blueprint v2.0 reconciliation - schema, locks, invariants, configs.

A protocol that exists only in prose is not landed. Every assertion here fails when the
corresponding enforcement is removed.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
import yaml

from seam.agent.policy import assert_identical_deadline_grid
from seam.analysis.slice_stats import compare_against_noise
from seam.backends.base import GenerationResult
from seam.backends.cloud_anthropic import is_retryable_protocol_error
from seam.config import ResolvedConfig
from seam.errors import ManifestValidationError, SeamError
from seam.gitinfo import GitState
from seam.locks import ExclusiveLock, exclusive
from seam.manifest import build_manifest, load_schema, validate_manifest
from seam.tools.fetch_model import FetchedModelSpec

VERIFIED_TOPOLOGY = {"p_cpus": [0, 1, 2, 3], "lpe_cpus": [4, 5, 6, 7], "verified": True}


def _git() -> GitState:
    return GitState(sha="a" * 40, dirty=False, branch="test", dirty_files=())


# --------------------------------------------------------------------------------------------------
# 2A - schema fields
# --------------------------------------------------------------------------------------------------


class TestManifestSchemaV2:
    def test_confinement_mechanism_enum(self) -> None:
        prop = load_schema()["properties"]["confinement_mechanism"]
        assert set(prop["enum"]) == {"process-affinity", "scheduling-core-type", None}

    def test_thermal_regime_enum(self) -> None:
        prop = load_schema()["properties"]["thermal"]["properties"]["regime"]
        assert set(prop["enum"]) == {"confound", "axis", None}

    def test_file_verification_methods(self) -> None:
        prop = load_schema()["properties"]["model"]["properties"]["provenance"]["properties"][
            "file_verification"
        ]
        assert set(prop["additionalProperties"]["enum"]) == {
            "sha256",
            "git-blob-sha1",
            "size-only",
        }

    def test_pinned_profile_enum(self) -> None:
        prop = load_schema()["properties"]["power_state"]["properties"]["pinned_profile"]
        assert set(prop["enum"]) == {"ac-pinned", "battery-pinned", None}

    def test_power_source_enum(self) -> None:
        prop = load_schema()["properties"]["power_state"]["properties"]["power_source"]
        assert set(prop["enum"]) == {"battery", "mains", None}

    def test_concurrency_field(self) -> None:
        prop = load_schema()["properties"]["workload"]["properties"]["concurrency"]
        assert prop["type"] == ["integer", "null"]
        assert prop["minimum"] == 1

    def test_escalation_semantics_and_kv_residency(self) -> None:
        pol = load_schema()["properties"]["policy"]["properties"]
        assert set(pol["escalation_semantics"]["enum"]) == {"predictive", "preemptive", None}
        assert set(pol["kv_residency"]["enum"]) == {"discard", "retain", "transfer", None}

    def test_invalid_thermal_regime_is_rejected(
        self, fake_config: ResolvedConfig, fake_repo: Path
    ) -> None:
        m = build_manifest(
            run_id=str(uuid.uuid4()),
            config=fake_config,
            git_state=_git(),
            allow_dirty=False,
            target="npu",
            workload={
                "kind": "microbench",
                "benchmark": "t",
                "task_ids": [],
                "seed": 1,
                "n_repeats": 1,
            },
            condition_label="A",
            blinded_label="cond_0123456789ab",
            repo_root=fake_repo,
            topology_override=VERIFIED_TOPOLOGY,
            thermal={"regime": "pooled"},  # illegal - confound|axis only
        )
        with pytest.raises(ManifestValidationError):
            validate_manifest(m)

    def test_build_manifest_emits_v2_null_defaults(
        self, fake_config: ResolvedConfig, fake_repo: Path
    ) -> None:
        m = build_manifest(
            run_id=str(uuid.uuid4()),
            config=fake_config,
            git_state=_git(),
            allow_dirty=False,
            target="npu",
            workload={
                "kind": "microbench",
                "benchmark": "t",
                "task_ids": [],
                "seed": 1,
                "n_repeats": 1,
            },
            condition_label="A",
            blinded_label="cond_0123456789ab",
            repo_root=fake_repo,
            topology_override=VERIFIED_TOPOLOGY,
        )
        validate_manifest(m)
        assert m["confinement_mechanism"] is None
        assert m["thermal"]["regime"] is None
        assert m["power_state"]["power_source"] is None
        assert m["policy"]["escalation_semantics"] is None
        assert m["policy"]["kv_residency"] is None
        assert m["model"]["provenance"]["file_verification"] is None


# --------------------------------------------------------------------------------------------------
# 2B - code invariants
# --------------------------------------------------------------------------------------------------


class TestInvariants:
    def test_per_target_deadline_grids_are_refused(self) -> None:
        with pytest.raises(SeamError, match="per-target deadline grids are refused"):
            assert_identical_deadline_grid(
                {"cpu-p": [1.0, 2.0, 4.0, 8.0], "cpu-lpe": [1.5, 3.0, 6.0, 12.0]}
            )

    def test_identical_deadline_grids_pass(self) -> None:
        grid = [1.0, 2.0, 4.0, 8.0]
        assert_identical_deadline_grid({"cpu-p": grid, "cpu-lpe": list(grid)})

    def test_effect_below_2x_cv_is_null(self) -> None:
        # Relative difference ~0.05; 2x CV with CV=0.10 → threshold 0.20 → NULL.
        verdict = compare_against_noise(
            metric="escalation_rate",
            a_values=[0.40] * 20,
            b_values=[0.42] * 20,
            cv=0.10,
            null_threshold_cv_multiple=2.0,
            resamples=200,
            seed=1,
        )
        assert verdict.verdict == "NULL"

    def test_tokenizer_primary_metrics_are_chars_bytes_not_tokens(self) -> None:
        cfg = yaml.safe_load((Path("configs/mslice.yaml")).read_text(encoding="utf-8"))
        primary = cfg["design"]["primary_behavioral_metrics"]
        cost = cfg["design"]["cost_metrics"]
        assert "completion_chars" in primary and "completion_bytes" in primary
        assert "completion_tokens" not in primary
        assert "completion_tokens" in cost
        # GenerationResult carries both; the behavioral fields must be present.
        result = GenerationResult(
            text="hi",
            tool_calls=(),
            prompt_tokens=1,
            completion_tokens=1,
            completion_chars=2,
            completion_bytes=2,
            wall_ns=1,
            backend="t",
            model_ref="t",
        )
        assert result.completion_chars == 2
        assert result.completion_bytes == 2

    def test_fetched_model_spec_documents_aggregate_as_self_constructed(self) -> None:
        doc = FetchedModelSpec.__doc__ or ""
        assert "self-constructed" in doc
        assert "not" in doc.lower() and "publisher-verifiable" in doc


# --------------------------------------------------------------------------------------------------
# 2C - mutual exclusion
# --------------------------------------------------------------------------------------------------


class TestMutualExclusion:
    def test_second_writer_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "shared.bin"
        with exclusive(target), pytest.raises(SeamError, match="refusing second writer"):
            ExclusiveLock(target).acquire()

    def test_lock_releases_so_a_later_writer_can_proceed(self, tmp_path: Path) -> None:
        target = tmp_path / "shared.bin"
        with exclusive(target):
            target.write_bytes(b"a")
        with exclusive(target):
            target.write_bytes(b"b")
        assert target.read_bytes() == b"b"


# --------------------------------------------------------------------------------------------------
# 2D - config values
# --------------------------------------------------------------------------------------------------


class TestConfigV2:
    def test_energy_design_is_full_no_stage_a(self) -> None:
        energy = yaml.safe_load(Path("configs/energy.yaml").read_text(encoding="utf-8"))
        assert energy["design"] == "full_per_target_multi_cycle"
        assert energy["load_levels_min"] >= 8
        assert set(energy["targets"]) == {"cpu-p", "cpu-lpe", "igpu", "npu"}
        assert energy["multi_cycle"] is True
        # Comments may mention the withdrawn staging; the parsed design must not encode it.
        assert energy["design"] != "stage_a"
        assert "stage_a" not in json.dumps(energy).lower()

        state = Path("configs/project_state.yaml").read_text(encoding="utf-8")
        assert "Stage A" not in state
        assert "contribution_ledger" in state
        assert "yield_queue" in state
        assert 'blueprint_version: "2.0"' in state

    def test_adopted_confinement_is_null_until_a1a6(self) -> None:
        state = yaml.safe_load(Path("configs/project_state.yaml").read_text(encoding="utf-8"))
        conf = state["milestones"]["M_SLICE"]["confinement"]
        assert conf["adopted_mechanism"] is None
        assert conf["citing_run_id"] is None
        mslice = yaml.safe_load(Path("configs/mslice.yaml").read_text(encoding="utf-8"))
        assert mslice["openvino"]["adopted_mechanism"] is None

    def test_no_stage_a_in_seam_python(self) -> None:
        """Withdrawn staging must not live as an active design constant in seam/."""
        root = Path("seam")
        offenders: list[str] = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "Stage A" in text or "stage_a" in text:
                offenders.append(str(path))
        assert offenders == []


# --------------------------------------------------------------------------------------------------
# 2E - cursor rules
# --------------------------------------------------------------------------------------------------


class TestCursorRules:
    def test_exactly_one_always_applied_rule(self) -> None:
        rules = list(Path(".cursor/rules").glob("*.mdc"))
        always = []
        for path in rules:
            text = path.read_text(encoding="utf-8")
            # front matter
            if "alwaysApply: true" in text.split("---", 2)[1]:
                always.append(path.name)
        assert always == ["seam-core.mdc"], always

    def test_seam_detail_has_no_withdrawn_staging_language(self) -> None:
        text = Path(".cursor/rules/seam-detail.mdc").read_text(encoding="utf-8")
        for banned in ("Stage A", "Stage B", "Paper 2", "Paper-2"):
            assert banned not in text, banned

    def test_always_on_byte_budget(self) -> None:
        core = Path(".cursor/rules/seam-core.mdc")
        assert "alwaysApply: true" in core.read_text(encoding="utf-8")
        budget = core.stat().st_size
        # Single always-on rule; keep the budget visible in the report.
        assert 1000 < budget < 8000


# --------------------------------------------------------------------------------------------------
# 2F - session findings
# --------------------------------------------------------------------------------------------------


class TestSessionFindings:
    def test_watcher_is_on_quiesce_forbidden_list(self) -> None:
        cfg = yaml.safe_load(Path("configs/mslice.yaml").read_text(encoding="utf-8"))
        forbidden = cfg["quiesce"]["forbidden_processes"]
        assert "seam.tools.fetch_progress" in forbidden

    def test_canary_requires_connectivity_and_sustained_burst(self) -> None:
        cfg = yaml.safe_load(Path("configs/canary.yaml").read_text(encoding="utf-8"))
        assert cfg["connectivity"]["enabled"] is True
        assert "api.anthropic.com" in cfg["connectivity"]["require_healthy"]
        burst = cfg["sustained_burst"]
        assert burst["required_before_collection"] is True
        assert burst["n_calls"] >= 10
        assert burst["prompt_chars_min"] >= 20_000

    def test_protocol_disconnect_is_retryable_auth_is_not(self) -> None:
        assert is_retryable_protocol_error(ConnectionError("TLS handshake failed"))
        assert is_retryable_protocol_error(TimeoutError("api connection reset"))
        assert not is_retryable_protocol_error(PermissionError("Unauthorized: invalid API key"))
