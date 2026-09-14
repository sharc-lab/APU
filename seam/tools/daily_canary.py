"""Daily canary: model pin + connectivity, before collection variance hides incidents.

Connectivity is folded in deliberately. A contemporaneous TLS probe of ``api.anthropic.com``
that returns 8/8 establishes "not badly degraded", not "clean" - with n=8 and zero failures the
rule of three puts the 95% upper bound near 31%. This canary re-runs the probe so a mid-
collection network change surfaces as an incident rather than as unexplained variance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from seam.gitinfo import repo_root
from seam.tools.probe_tls import probe_session

__all__ = ["run_canary"]


def run_canary(*, config_path: Path | None = None) -> dict[str, Any]:
    root = repo_root(Path(__file__).parent)
    cfg_path = config_path or (root / "configs" / "canary.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "config": str(cfg_path),
        "model_pin": cfg.get("model_pin"),
        "sustained_burst_required": bool(
            (cfg.get("sustained_burst") or {}).get("required_before_collection")
        ),
        "connectivity": None,
        "verdict": "pass",
        "notes": [],
    }

    conn = cfg.get("connectivity") or {}
    if conn.get("enabled"):
        result = probe_session(attempts_per_target=int(conn.get("attempts_per_target", 8)))
        summary = result.to_dict()["summary"]
        report["connectivity"] = summary
        for host in conn.get("require_healthy") or []:
            label = summary.get("verdict", {}).get("label")
            if host == "api.anthropic.com" and label != "healthy":
                report["verdict"] = "fail"
                report["notes"].append(
                    f"{host} connectivity not healthy (label={label!r}); treat as an incident, "
                    f"not as run variance"
                )

    if not report["sustained_burst_required"]:
        report["verdict"] = "fail"
        report["notes"].append(
            "configs/canary.yaml must keep sustained_burst.required_before_collection=true; "
            "short unauthenticated probes do not exercise H1's traffic shape"
        )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_canary(config_path=args.config)
    out = args.out or (
        repo_root(Path(__file__).parent) / "derived" / "mslice" / "daily_canary_latest.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
