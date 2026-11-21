"""Tests for the worker process resource-limit backstop (DoS mitigation)."""

import pytest

from rlm_tools.worker_runtime import apply_resource_limits

resource = pytest.importorskip("resource")

def _soft(limit_name: str) -> int:
    soft, _hard = resource.getrlimit(getattr(resource, limit_name))
    return soft


def test_applies_cpu_limit():
    # RLIMIT_CPU is portable (incl. macOS) and reported back via getrlimit.
    original = resource.getrlimit(resource.RLIMIT_CPU)
    try:
        applied = apply_resource_limits(max_address_space_bytes=None, cpu_seconds=3600)
        expected = 3600 if original[1] == resource.RLIM_INFINITY else min(3600, original[1])
        assert applied.get("RLIMIT_CPU") == expected
        assert _soft("RLIMIT_CPU") == expected
    finally:
        resource.setrlimit(resource.RLIMIT_CPU, original)


def test_none_values_are_noops():
    original_cpu = resource.getrlimit(resource.RLIMIT_CPU)
    applied = apply_resource_limits(max_address_space_bytes=None, cpu_seconds=None)
    assert applied == {} or "RLIMIT_CPU" not in applied
    assert resource.getrlimit(resource.RLIMIT_CPU) == original_cpu


def test_does_not_exceed_hard_limit():
    # Never raise the soft limit above the inherited hard limit.
    _soft_cpu, hard_cpu = resource.getrlimit(resource.RLIMIT_CPU)
    if hard_cpu == resource.RLIM_INFINITY:
        pytest.skip("no finite hard CPU limit to test against")
    original = resource.getrlimit(resource.RLIMIT_CPU)
    try:
        applied = apply_resource_limits(
            max_address_space_bytes=None, cpu_seconds=hard_cpu + 10_000
        )
        assert _soft("RLIMIT_CPU") <= hard_cpu
        assert applied.get("RLIMIT_CPU", hard_cpu) <= hard_cpu
    finally:
        resource.setrlimit(resource.RLIMIT_CPU, original)
