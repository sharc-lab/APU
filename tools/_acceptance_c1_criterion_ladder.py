"""Acceptance: --criterion completion probe ladder vs session 83127e1b.

Same discipline as -ModelSpecs / -PipelineTypes: default path must reproduce
the pre-criterion C-1 bisection ladder when driven by the session oracle.

Oracle (completion): n passes iff the session has `repeats` completed results
at that n. Unknown n (never probed in session) is treated as pass, matching
the session's no-ceiling finding (high would have passed).

Reference ladder is the high-first algorithm already in run_c1_ceiling.py
(post salvage). Session 83127e1b historically used mid-only then final-hi;
that divergence is noted, not papered over.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SESSION = ROOT / "derived/c1_ceiling/83127e1b-9d6e-4103-bee6-2a63c00f479f"
LOW, HIGH, RES, REPEATS = 12_000, 45_000, 250, 2


def _session_oracle() -> dict[int, list[bool]]:
    by_n: dict[int, list[bool]] = {}
    work = SESSION / "work"
    for p in sorted(work.glob("*.result.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m = re.search(r"\.n(\d+)\.r(\d+)", p.name)
        if not m:
            continue
        n, r = int(m.group(1)), int(m.group(2))
        # Prefer a0 over later attempts when ranking by r; store in order seen.
        by_n.setdefault(n, [])
        # Keep best attempt per (n,r): completed wins.
        while len(by_n[n]) <= r:
            by_n[n].append(False)
        by_n[n][r] = by_n[n][r] or bool(d.get("completed"))
    return by_n


def _n_pass(by_n: dict[int, list[bool]], n: int, repeats: int) -> bool:
    xs = by_n.get(n)
    if xs is None:
        return True  # unknown: no wall observed in session
    return len(xs) >= repeats and all(bool(x) for x in xs[:repeats])


def _reference_ladder(by_n: dict[int, list[bool]]) -> list[int]:
    """High-first bisection (current run_c1_ceiling completion control flow)."""
    seq: list[int] = []

    def probe(n: int) -> bool:
        seq.append(n)
        return _n_pass(by_n, n, REPEATS)

    lo, hi = LOW, HIGH
    if not probe(lo):
        return seq
    if probe(hi):
        return seq
    while (hi - lo) > RES:
        mid = max(lo + 1, (lo + hi) // 2)
        if probe(mid):
            lo = mid
        else:
            hi = mid
    return seq


def _worker_ladder(by_n: dict[int, list[bool]]) -> list[int]:
    """Drive tools.run_c1_ceiling.bisect_arm with --criterion completion mocked."""
    import tools.run_c1_ceiling as m

    seq: list[int] = []

    def fake_probe_once(**kwargs: Any) -> dict[str, Any]:
        n = int(kwargs["n_tokens"])
        r = int(kwargs["repeat_i"])
        if r == 0:
            seq.append(n)
        ok = _n_pass(by_n, n, REPEATS)
        return {
            "arm_id": "gpu_only_f16",
            "n_tokens": n,
            "repeat_index": r,
            "outcome": "pass" if ok else "fail",
            "completed": ok,
            "admissible": True,
            "failure_classification": {
                "class": "pass" if ok else "fail",
                "failure_kind": None if ok else "other",
                "verbatim": None,
            },
            "failure_kind": None if ok else "other",
            "verbatim_error": None,
            "prefill_s": 1.0 if ok else None,
            "decode_tok_s": 20.0 if ok else None,
            "r_prefill_tok_s": None,
            "wall_s": 1.0 if ok else None,
            "started_utc": None,
            "ended_utc": None,
        }

    real_probe = m._probe_once
    real_tok = None

    class _Tok:
        pass

    try:
        m._probe_once = fake_probe_once  # type: ignore[assignment]

        # Avoid loading the real tokenizer / model dir.
        import transformers

        real_from = transformers.AutoTokenizer.from_pretrained

        def fake_from_pretrained(*_a, **_k):
            return _Tok()

        transformers.AutoTokenizer.from_pretrained = fake_from_pretrained  # type: ignore

        from seam.tools import delta_n as dn

        real_prompt_for = dn.prompt_for

        def fake_prompt_for(**_k):
            return {"text": "x", "n_tokens": _k.get("n_tokens")}

        dn.prompt_for = fake_prompt_for  # type: ignore

        probes_log: list[dict[str, Any]] = []
        m.bisect_arm(
            root=ROOT,
            cfg={},
            arm={"id": "gpu_only_f16"},
            model_dir=str(ROOT / "models" / "Qwen3-4B-int4-ov"),
            p_cpus=[0, 1, 2, 3],
            work_dir=SESSION / "work",
            prompt_cache={},
            unit="the quick brown fox",
            low=LOW,
            high=HIGH,
            resolution=RES,
            repeats=REPEATS,
            probes_log=probes_log,
            criterion=m.CRITERION_COMPLETION,
            slo_s=10.0,
            label_prefix="c1",
        )
    finally:
        m._probe_once = real_probe  # type: ignore[assignment]
        try:
            transformers.AutoTokenizer.from_pretrained = real_from  # type: ignore
            dn.prompt_for = real_prompt_for  # type: ignore
        except Exception:
            pass
    return seq


def _session_n_order() -> list[int]:
    """Unique n in first-seen mtime order (historical mid-only ladder)."""
    work = SESSION / "work"
    rows = []
    for p in work.glob("*.result.json"):
        m = re.search(r"\.n(\d+)\.", p.name)
        if not m:
            continue
        rows.append((p.stat().st_mtime, int(m.group(1))))
    rows.sort()
    out: list[int] = []
    for _, n in rows:
        if not out or out[-1] != n:
            out.append(n)
    return out


def main() -> int:
    by_n = _session_oracle()
    ref = _reference_ladder(by_n)
    got = _worker_ladder(by_n)
    hist = _session_n_order()
    identical = ref == got
    print("session_id", SESSION.name)
    print("oracle_ns", sorted(by_n))
    print("reference_ladder_high_first", ref)
    print("worker_ladder_criterion_completion", got)
    print("LADDER_IDENTICAL", identical)
    print("session_historical_n_order", hist)
    print(
        "NOTE_historical_vs_high_first",
        hist != ref,
        "(83127e1b ran mid-only; current worker probes high first)",
    )
    if not identical:
        print("DIFF:")
        for i, (a, b) in enumerate(zip(ref, got, strict=False)):
            if a != b:
                print(f"  first mismatch i={i} ref={a} got={b}")
                break
        if len(ref) != len(got):
            print(f"  length ref={len(ref)} got={len(got)}")
        return 1
    print("ACCEPTANCE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
