"""Contemporaneous TLS handshake probe across hosts this project depends on.

The measured fault on this host is selective: some HTTPS endpoints fail at the TLS handshake
(curl exit 35, TCP connected) while others succeed 8/8. Comparing today's ``api.anthropic.com``
against yesterday's ``huggingface.co`` numbers is not a valid comparison - network conditions
drift - so every target in a session is probed interleaved, not blocked by host.

No authentication is sent. This probe costs USD 0.00.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from seam.gitinfo import repo_root

__all__ = ["TARGETS", "Attempt", "ProbeResult", "probe_session", "summarize"]

#: Cheap unauthenticated paths. HEAD where the host accepts it; a short GET otherwise.
#: Anthropic's API rejects unauthenticated requests with 401/403 after a successful handshake -
#: that is a *success* for this probe. Exit 35 (SSL connect error) is the failure mode of interest.
TARGETS: tuple[tuple[str, str], ...] = (
    ("api.anthropic.com", "https://api.anthropic.com/v1/models"),
    ("huggingface.co", "https://huggingface.co/api/models/OpenVINO/Qwen3-0.6B-int4-ov"),
    ("cdn-lfs.hf.co", "https://cdn-lfs.hf.co/"),
    ("github.com", "https://github.com/"),
    ("google.com", "https://www.google.com/"),
)

_CURL_BASE = ["curl.exe", "-sS", "-o", "NUL", "-w", "%{http_code}", "--max-time", "20"]


@dataclass(slots=True)
class Attempt:
    host: str
    url: str
    mode: str
    attempt: int
    exit_code: int
    http_code: str
    elapsed_s: float
    #: Exit 35 = SSL connect error after TCP connected. The measured fault class.
    tls_handshake_failure: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(slots=True)
class ProbeResult:
    started_utc: str
    finished_utc: str
    attempts_per_target: int
    modes: list[str]
    attempts: list[Attempt]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "attempts_per_target": self.attempts_per_target,
            "modes": self.modes,
            "attempts": [asdict(a) for a in self.attempts],
            "summary": summarize(self),
        }


def _one(host: str, url: str, *, mode: str, attempt: int) -> Attempt:
    cmd = list(_CURL_BASE)
    if mode == "retry":
        cmd += ["--retry", "5", "--retry-all-errors", "--retry-delay", "1"]
    cmd.append(url)
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0
    return Attempt(
        host=host,
        url=url,
        mode=mode,
        attempt=attempt,
        exit_code=completed.returncode,
        http_code=(completed.stdout or "").strip(),
        elapsed_s=round(elapsed, 3),
        tls_handshake_failure=completed.returncode == 35,
    )


def probe_session(
    *,
    attempts_per_target: int = 8,
    modes: tuple[str, ...] = ("plain", "retry"),
    targets: tuple[tuple[str, str], ...] = TARGETS,
) -> ProbeResult:
    """Interleave hosts so a mid-session network shift hits every target equally."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    attempts: list[Attempt] = []
    for mode in modes:
        for i in range(1, attempts_per_target + 1):
            for host, url in targets:
                a = _one(host, url, mode=mode, attempt=i)
                attempts.append(a)
                flag = "TLS35" if a.tls_handshake_failure else ("ok" if a.ok else f"e{a.exit_code}")
                print(
                    f"{mode:5} {host:20} #{i}: exit={a.exit_code} http={a.http_code or '-'} "
                    f"{a.elapsed_s:.2f}s [{flag}]",
                    flush=True,
                )
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return ProbeResult(
        started_utc=started,
        finished_utc=finished,
        attempts_per_target=attempts_per_target,
        modes=list(modes),
        attempts=attempts,
    )


def summarize(result: ProbeResult) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in result.modes:
        mode_rows = [a for a in result.attempts if a.mode == mode]
        hosts: dict[str, Any] = {}
        for host, _ in TARGETS:
            rows = [a for a in mode_rows if a.host == host]
            ok = sum(1 for a in rows if a.ok)
            tls35 = sum(1 for a in rows if a.tls_handshake_failure)
            other = sum(1 for a in rows if not a.ok and not a.tls_handshake_failure)
            hosts[host] = {
                "ok": ok,
                "n": len(rows),
                "tls35": tls35,
                "other_fail": other,
                "exit_codes": [a.exit_code for a in rows],
                "http_codes": [a.http_code for a in rows],
            }
        out[mode] = hosts
    # Verdict keyed on the cloud host under the plain (no-retry) mode - that is the rate the
    # cloud backend would see without retry discipline.
    plain = out.get("plain", {}).get("api.anthropic.com", {})
    ok = int(plain.get("ok", 0))
    n = int(plain.get("n", 0)) or 1
    rate = ok / n
    if rate >= 0.875:
        verdict = "healthy"
    elif rate >= 0.5:
        verdict = "degraded"
    else:
        verdict = "severely_degraded"
    out["verdict"] = {
        "api.anthropic.com_plain_ok_rate": round(rate, 3),
        "label": verdict,
        "retry_required_before_spend": verdict != "healthy",
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON destination (default: derived/mslice/tls_probe_<utc>.json)",
    )
    args = parser.parse_args(argv)

    result = probe_session(attempts_per_target=args.attempts)
    root = repo_root(Path(__file__).parent)
    out = args.out or (
        root / "derived" / "mslice" / f"tls_probe_{result.started_utc.replace(':', '')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = result.to_dict()["summary"]
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
