"""Phase E - first paid cloud path validation (E0 guard fire, E1 one call, E2 sustained burst).

E0 is unpaid. E1+E2 share a hard USD 1.00 ceiling for this dispatch. Slice ceiling remains USD 10.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

from seam.backends.base import GenerationRequest, ToolSpec
from seam.backends.cloud_anthropic import AnthropicBackend, is_retryable_protocol_error
from seam.budget import BudgetGuard, BudgetLedger, CallUsage, PricingTable
from seam.config import resolve_config
from seam.credentials import load_dotenv_once, require_api_key
from seam.errors import BudgetExceededError
from seam.gitinfo import repo_root
from seam.manifest import emit

__all__ = ["main"]

_TOOL = ToolSpec(
    name="lookup_status",
    description="Look up a named status code for a service.",
    input_schema={
        "type": "object",
        "properties": {"service": {"type": "string"}, "code": {"type": "string"}},
        "required": ["service", "code"],
    },
)

_PLATFORM = Path("configs/platforms/aipc-c1.yaml")


def _guard(*, ledger_dir: Path, slice_ceiling: float, phase: str) -> BudgetGuard:
    root = repo_root(Path(__file__).parent)
    return BudgetGuard(
        ledger=BudgetLedger(ledger_dir),
        pricing=PricingTable.load(root / "configs" / "pricing" / "anthropic.yaml"),
        slice_ceiling_usd=slice_ceiling,
        project_ceiling_usd=50.0,
        phase=phase,
    )


def _emit_refusal(*, reason: str, summary: dict[str, Any]) -> str:
    root = repo_root(Path(__file__).parent)
    config = resolve_config([root / _PLATFORM], repo_root=root)
    handle = emit(
        config=config,
        target="cloud",
        workload={
            "kind": "mslice_key_validation",
            "benchmark": "phase-e-budget-guard-fire",
            "task_ids": ["e0-guard-fire"],
            "seed": 20260802,
            "n_repeats": 1,
        },
        condition_label="phase_e_e0_refusal",
        repo_root=root,
        allow_dirty=True,
        self_check="fail",
        summary={"refusal_reason": reason, **summary},
        model={
            "name": "claude-sonnet-5",
            "revision": None,
            "quantization": None,
            "ir_sha256": None,
            "reasoning_mode": None,
            "provenance": {
                "kind": "cloud",
                "self_converted": None,
                "source_repo": None,
                "download_method": None,
                "export_command": None,
                "quantization_config": None,
                "ladder_position": None,
                "spec_path": None,
                "spec_sha256": None,
                "file_verification": None,
            },
            "cloud": {
                "provider": "anthropic",
                "model_snapshot_id": "claude-sonnet-5",
                "access_date": "2026-08-02",
                "pricing_table_version": "2026-08-02.1",
                "pin_convention": "anthropic_dateless_generation_id",
                "prompt_caching": True,
                "reasoning_mode": None,
            },
        },
        thermal={"regime": "confound", "excluded": False},
    )
    return handle.run_id


def run_e0(*, ledger_dir: Path, out: Path) -> dict[str, Any]:
    """Temporarily set ceiling below one call's worst case; confirm refuse-before-network."""
    load_dotenv_once()
    # Presence only - never log the value.
    key = require_api_key("ANTHROPIC_API_KEY")
    key_len = len(key)

    spent_before = BudgetLedger(ledger_dir).total_usd()
    # Tiny ceiling: any real worst-case projection exceeds this.
    guard = _guard(ledger_dir=ledger_dir, slice_ceiling=0.00001, phase="phase_e_e0")
    backend = AnthropicBackend(
        model_id="claude-sonnet-5",
        guard=guard,
        max_tokens=64,
        prompt_caching=True,
        max_retries=0,
        run_id="e0-guard-fire",
    )

    create_calls = {"n": 0}

    def _spy_create(*_a: Any, **_k: Any) -> Any:
        create_calls["n"] += 1
        raise RuntimeError("network create must not be reached after budget refusal")

    request = GenerationRequest(
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        system="You are a validation probe.",
        tools=(_TOOL,),
        max_tokens=64,
    )

    refused = False
    reason = ""
    try:
        client = backend._ensure_client()
        # Spy after client exists so authorization path is what we test.
        client.messages.create = _spy_create  # type: ignore[method-assign]
        backend.generate(request)
    except BudgetExceededError as exc:
        refused = True
        reason = str(exc)
    except Exception as exc:  # unexpected
        reason = f"UNEXPECTED {type(exc).__name__}: {exc}"

    spent_after = BudgetLedger(ledger_dir).total_usd()
    run_id = None
    if refused:
        run_id = _emit_refusal(
            reason=reason,
            summary={
                "key_present": True,
                "key_char_count": key_len,
                "messages_create_calls": create_calls["n"],
                "ledger_spend_before": spent_before,
                "ledger_spend_after": spent_after,
            },
        )

    report = {
        "refused": refused,
        "refusal_reason": reason,
        "refusal_manifest_run_id": run_id,
        "messages_create_calls": create_calls["n"],
        "ledger_spend_before_usd": spent_before,
        "ledger_spend_after_usd": spent_after,
        "evidence_no_request": create_calls["n"] == 0 and spent_after == spent_before,
        "key_present": True,
        "key_char_count": key_len,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _usage_fields(usage_obj: Any) -> dict[str, Any]:
    """Transcribe usage attribute names and values without assuming the schema."""
    names: list[str] = []
    values: dict[str, Any] = {}
    # Prefer model_fields / __dict__ / dir - record what is actually present.
    if hasattr(usage_obj, "model_dump"):
        dumped = usage_obj.model_dump()
        for k, v in dumped.items():
            names.append(k)
            values[k] = v
    else:
        for name in dir(usage_obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(usage_obj, name)
            except Exception:
                continue
            if callable(val):
                continue
            names.append(name)
            values[name] = val
    return {"field_names": names, "values": values}


def run_e1(*, ledger_dir: Path, ceiling: float, out: Path) -> dict[str, Any]:
    load_dotenv_once()
    guard = _guard(ledger_dir=ledger_dir, slice_ceiling=ceiling, phase="phase_e_e1")
    spent_before = guard.spent_usd()

    import anthropic

    api_key = require_api_key("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key, max_retries=0)

    tools = [
        {
            "name": _TOOL.name,
            "description": _TOOL.description,
            "input_schema": _TOOL.input_schema,
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "Using the lookup_status tool schema only for overhead, reply with exactly: pong",
        }
    ]
    system = "You are a validation probe. Prefer a short text reply; tool use is optional."

    # Pre-flight projection via the guard (same path as the backend).
    est_prompt = 200  # crude; ledger records projected-vs-actual
    projected = guard.worst_case_call_usd(
        model="claude-sonnet-5", prompt_tokens=est_prompt, max_tokens=64
    )
    guard.authorize_call(model="claude-sonnet-5", prompt_tokens=est_prompt, max_tokens=64)

    t0 = time.perf_counter()
    # claude-sonnet-5 rejects `temperature` (400 invalid_request_error: deprecated for this model).
    raw = client.messages.with_raw_response.create(
        model="claude-sonnet-5",
        max_tokens=64,
        system=[{"type": "text", "text": system}],
        tools=tools,
        messages=messages,
    )
    latency_s = time.perf_counter() - t0
    headers = dict(raw.headers.items())
    response = raw.parse()

    usage_info = _usage_fields(response.usage)
    vals = usage_info["values"]

    # Map known names if present; keep zeros explicit.
    def _i(name: str) -> int:
        v = vals.get(name)
        return int(v) if v is not None else 0

    usage = CallUsage(
        input_tokens=_i("input_tokens"),
        output_tokens=_i("output_tokens"),
        cache_read_input_tokens=_i("cache_read_input_tokens"),
        cache_creation_input_tokens=_i("cache_creation_input_tokens"),
    )
    actual = guard.record(
        model="claude-sonnet-5",
        usage=usage,
        projected_usd=projected,
        run_id="phase-e-e1",
    )
    rel_err = ((actual - projected) / projected) if projected > 0 else None

    cache_names = [n for n in usage_info["field_names"] if "cache" in n.lower()]
    cache_behavior = {
        name: (
            "PRESENT_AND_ZERO" if vals.get(name) == 0 else ("PRESENT" if name in vals else "ABSENT")
        )
        for name in (
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cache_creation_input_tokens",
        )
    }
    # Clarify absent vs zero for the two expected cache fields.
    for name in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        if name not in usage_info["field_names"]:
            cache_behavior[name] = "ABSENT"
        elif vals.get(name) == 0:
            cache_behavior[name] = "PRESENT_AND_ZERO"
        else:
            cache_behavior[name] = f"PRESENT_AND_NONZERO({vals.get(name)})"

    report = {
        "usage_field_names_verbatim": usage_info["field_names"],
        "usage_values": vals,
        "cache_field_behavior": cache_behavior,
        "cache_field_names_present": cache_names,
        "serving_model_id": getattr(response, "model", None),
        "pinned_model_id": "claude-sonnet-5",
        "model_id_matches_pin": getattr(response, "model", None) == "claude-sonnet-5",
        "rate_limit_headers": {
            k: v
            for k, v in headers.items()
            if "rate" in k.lower()
            or "limit" in k.lower()
            or "remaining" in k.lower()
            or "reset" in k.lower()
            or k.lower().startswith("anthropic-")
            or k.lower().startswith("x-")
        },
        "all_response_header_names": sorted(headers.keys()),
        "projected_usd": projected,
        "actual_usd": actual,
        "projected_vs_actual_rel_error": rel_err,
        "latency_s": latency_s,
        "retries": 0,
        "spent_before_usd": spent_before,
        "spent_after_usd": guard.spent_usd(),
        "remaining_usd": guard.remaining_usd(),
        "stop_reason": getattr(response, "stop_reason", None),
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_e2(*, ledger_dir: Path, ceiling: float, out: Path) -> dict[str, Any]:
    """10 sequential calls with spanning sizes; raw attempt outcomes; cache exercise."""
    load_dotenv_once()
    guard = _guard(ledger_dir=ledger_dir, slice_ceiling=ceiling, phase="phase_e_e2")
    import anthropic

    api_key = require_api_key("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key, max_retries=0)

    def _exact_bytes(n: int, seed: str) -> str:
        """Build an ASCII payload of exactly ``n`` UTF-8 bytes."""
        base = f"{seed}:"
        if n <= len(base):
            return "x" * n
        pad = "x" * (n - len(base))
        return base + pad

    # Size plan: 2x~2KB, 3x~10KB, 5x~30KB. Calls 4-10 share one identical ~30KB
    # cacheable system prefix (weighted to the large end); size schedule for 4-5
    # is met by that same prefix plus a short unique user turn - the cacheable
    # block must be byte-identical for cache_read to fire.
    sizes = [2_000, 2_000, 10_000, 10_000, 10_000, 30_000, 30_000, 30_000, 30_000, 30_000]
    shared_cache_block = _exact_bytes(30_000, "SEAM-CACHE-PREFIX")

    tools_base = [
        {
            "name": _TOOL.name,
            "description": _TOOL.description,
            "input_schema": _TOOL.input_schema,
        }
    ]
    tools_cached = [
        {
            **tools_base[0],
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        }
    ]

    calls: list[dict[str, Any]] = []
    inter_call_delay_s = 0.5
    total_attempts = 0
    failures: list[dict[str, Any]] = []

    for i, nbytes in enumerate(sizes, start=1):
        use_shared = i >= 4
        if use_shared:
            system_text = shared_cache_block
            system_block: Any = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            ]
            tools = tools_cached
            # Uncached suffix carries the call index; size schedule for calls 4-5
            # is the shared 30KB prefix (large-end weight) + short user turn.
            user_content = f"Call {i}: reply with ack. nonce={uuid.uuid4().hex}"
            actual_bytes = len(system_text.encode("utf-8")) + len(user_content.encode("utf-8"))
            est_tokens = (len(system_text) // 4) + 80 + 354  # + tool-use overhead
        else:
            system_text = "You are a validation probe. Reply with the single word: ack"
            system_block = system_text
            tools = tools_base
            user_content = _exact_bytes(nbytes, f"probe-{i}") + f"\n\nCall {i}: reply with ack."
            actual_bytes = len(user_content.encode("utf-8"))
            est_tokens = max(100, actual_bytes // 4) + 50 + 354
        projected = guard.worst_case_call_usd(
            model="claude-sonnet-5", prompt_tokens=est_tokens, max_tokens=32
        )
        if projected > guard.remaining_usd():
            calls.append(
                {
                    "call": i,
                    "skipped": True,
                    "reason": "projected exceeds remaining dispatch ceiling",
                    "projected_usd": projected,
                    "remaining_usd": guard.remaining_usd(),
                }
            )
            break

        guard.authorize_call(model="claude-sonnet-5", prompt_tokens=est_tokens, max_tokens=32)

        attempt_outcomes: list[dict[str, Any]] = []
        max_attempts = 2  # one retry
        final_success = None
        for attempt in range(1, max_attempts + 1):
            total_attempts += 1
            t0 = time.perf_counter()
            try:
                # Omit temperature: claude-sonnet-5 returns 400 if it is set.
                raw = client.messages.with_raw_response.create(
                    model="claude-sonnet-5",
                    max_tokens=32,
                    system=system_block,
                    tools=tools,
                    messages=[{"role": "user", "content": user_content}],
                )
                latency = time.perf_counter() - t0
                response = raw.parse()
                usage_info = _usage_fields(response.usage)
                vals = usage_info["values"]

                def _i(name: str, _vals: dict[str, Any] = vals) -> int:
                    v = _vals.get(name)
                    return int(v) if v is not None else 0

                usage = CallUsage(
                    input_tokens=_i("input_tokens"),
                    output_tokens=_i("output_tokens"),
                    cache_read_input_tokens=_i("cache_read_input_tokens"),
                    cache_creation_input_tokens=_i("cache_creation_input_tokens"),
                )
                actual = guard.record(
                    model="claude-sonnet-5",
                    usage=usage,
                    projected_usd=projected,
                    run_id=f"phase-e-e2-{i}",
                )
                outcome = {
                    "attempt": attempt,
                    "ok": True,
                    "latency_s": latency,
                    "failure_class": None,
                    "usage_field_names": usage_info["field_names"],
                    "usage": vals,
                    "actual_usd": actual,
                    "serving_model_id": getattr(response, "model", None),
                }
                attempt_outcomes.append(outcome)
                final_success = outcome
                break
            except Exception as exc:
                latency = time.perf_counter() - t0
                text = f"{type(exc).__name__}: {exc}"
                if is_retryable_protocol_error(exc):
                    fclass = "tls_protocol_disconnect"
                elif "timeout" in text.lower():
                    fclass = "timeout"
                elif "400" in text or "401" in text or "403" in text or "429" in text:
                    fclass = "http_error"
                else:
                    fclass = "other"
                failures.append({"call": i, "attempt": attempt, "class": fclass, "error": text})
                attempt_outcomes.append(
                    {
                        "attempt": attempt,
                        "ok": False,
                        "latency_s": latency,
                        "failure_class": fclass,
                        "error": text,
                    }
                )
                if not is_retryable_protocol_error(exc) or attempt >= max_attempts:
                    break

        n_fail = sum(1 for a in attempt_outcomes if not a["ok"])
        n_ok = sum(1 for a in attempt_outcomes if a["ok"])
        calls.append(
            {
                "call": i,
                "prompt_bytes": actual_bytes,
                "shared_prefix": use_shared,
                "projected_usd": projected,
                "attempt_outcomes": attempt_outcomes,
                "raw_successes": n_ok,
                "raw_failures": n_fail,
                # Post-retry convenience - NOT the primary accounting.
                "final_ok": final_success is not None,
                "final_usage": (final_success or {}).get("usage"),
                "final_cost_usd": (final_success or {}).get("actual_usd"),
                "final_latency_s": (final_success or {}).get("latency_s"),
            }
        )
        time.sleep(inter_call_delay_s)

    n_fail_total = sum(1 for f in failures)
    n_attempts = total_attempts
    from seam.analysis.proportions import PROPORTION_CI_METHOD, wilson_ci

    if n_attempts:
        w = wilson_ci(n_fail_total, n_attempts)
        rate = w.proportion
        ci = [w.lo, w.hi]
        ci_method = w.method
    else:
        rate = 0.0
        ci = [None, None]
        ci_method = PROPORTION_CI_METHOD

    by_class: dict[str, int] = {}
    for f in failures:
        by_class[f["class"]] = by_class.get(f["class"], 0) + 1

    report = {
        "rationale": (
            "TLS probe 8/8 short 401s only bounds failure rate ~31% by rule of three; "
            "this burst exercises long POSTs with bodies H1 will use."
        ),
        "inter_call_delay_s": inter_call_delay_s,
        "calls": calls,
        "total_attempts": n_attempts,
        "total_underlying_failures": n_fail_total,
        "failure_rate": rate,
        "failure_rate_95ci": ci,
        "failure_rate_ci_method": ci_method,
        "failures_by_class": by_class,
        "spent_usd": guard.spent_usd(),
        "remaining_usd": guard.remaining_usd(),
        "ledger_calibration": guard.ledger.calibration(),
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=["e0", "e1", "e2", "all"], default="all")
    parser.add_argument("--ceiling", type=float, default=1.0)
    parser.add_argument("--ledger-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = repo_root(Path(__file__).parent)
    ledger_dir = args.ledger_dir or (root / "derived" / "budget")
    out_dir = root / "derived" / "mslice"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.step in {"e0", "all"}:
        e0 = run_e0(ledger_dir=ledger_dir, out=out_dir / "phase_e_e0.json")
        print(json.dumps({"E0": e0}, indent=2))
        if not e0["refused"] or not e0["evidence_no_request"]:
            print("E0 FAILED - refusing to proceed to E1")
            return 1

    if args.step in {"e1", "all"}:
        e1 = run_e1(ledger_dir=ledger_dir, ceiling=args.ceiling, out=out_dir / "phase_e_e1.json")
        print(json.dumps({"E1": e1}, indent=2))
        print("=== E1 COMPLETE - review before E2 ===")

    if args.step == "e2" or args.step == "all":
        # For --step all, continue into E2 after E1 report printed above.
        if args.step == "all":
            e2 = run_e2(
                ledger_dir=ledger_dir, ceiling=args.ceiling, out=out_dir / "phase_e_e2.json"
            )
            print(
                json.dumps(
                    {
                        "E2_summary": {
                            "total_attempts": e2["total_attempts"],
                            "total_underlying_failures": e2["total_underlying_failures"],
                            "failure_rate": e2["failure_rate"],
                            "failure_rate_95ci": e2["failure_rate_95ci"],
                            "failures_by_class": e2["failures_by_class"],
                            "spent_usd": e2["spent_usd"],
                            "remaining_usd": e2["remaining_usd"],
                        }
                    },
                    indent=2,
                )
            )
        elif args.step == "e2":
            e2 = run_e2(
                ledger_dir=ledger_dir, ceiling=args.ceiling, out=out_dir / "phase_e_e2.json"
            )
            print(json.dumps(e2, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
