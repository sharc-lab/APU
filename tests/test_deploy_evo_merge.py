"""deploy_evo.merge_expected: expected_blobs.json accumulates across deploys (2026-10-07)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import deploy_evo as de  # noqa: E402


def test_first_deploy_with_no_prior_file():
    m = de.merge_expected("", "h1", {"a.py": "s1"}, {"a.py": "harness/a.py"})
    assert m == {"git_head": "h1", "blobs": {"a.py": "s1"}, "heads": {"a.py": "h1"},
                 "paths": {"a.py": "harness/a.py"}}


def test_second_deploy_keeps_earlier_files_and_replaces_redeployed_ones():
    prior = json.dumps({"git_head": "h0", "blobs": {"a.py": "s0", "b.py": "t0"}})  # old format: no heads/paths
    m = de.merge_expected(prior, "h1", {"a.py": "s1"}, {"a.py": "harness/a.py"})
    assert m["blobs"] == {"a.py": "s1", "b.py": "t0"}
    assert m["heads"] == {"a.py": "h1", "b.py": "h0"}
    assert m["git_head"] == "h1"
    assert m["paths"] == {"a.py": "harness/a.py"}


def test_unreadable_prior_file_starts_fresh():
    m = de.merge_expected("{not json", "h1", {"a.py": "s1"}, {})
    assert m["blobs"] == {"a.py": "s1"} and m["paths"] == {"a.py": "a.py"}
