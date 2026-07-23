"""Basic integration tests for adapters."""

import pytest


def test_adapter_imports():
    """Verify adapter module can be imported."""
    try:
        from harness.adapters import sdk_direct
        assert sdk_direct.MODEL == "gpt-4o-mini"
        assert sdk_direct.BACKEND == "openai"
        assert len(sdk_direct.TASKS) == 14
        assert "CH-01" in sdk_direct.TASKS
    except ImportError as e:
        pytest.skip(f"Adapter import failed (expected if dependencies not installed): {e}")


def test_instrumentation_imports():
    """Verify instrumentation modules can be imported."""
    from harness.instrumentation import (
        Category, CPU_BOUND_CATS, Span,
        wall_ns, process_cpu_ns
    )
    assert "ORCH_SETUP" in CPU_BOUND_CATS
    assert "HTTP_CLIENT" not in CPU_BOUND_CATS
    assert callable(wall_ns)
    assert callable(process_cpu_ns)
