"""Tests for the worker process resource-limit backstop (DoS mitigation)."""

import subprocess
import sys

import pytest

from rlm_tools.worker_runtime import apply_resource_limits

resource = pytest.importorskip("resource")

def _soft(limit_name: str) -> int:
    soft, _hard = resource.getrlimit(getattr(resource, limit_name))
    return soft


def test_applies_cpu_limit():
    # A reduced hard limit cannot be restored in the pytest process.
    subprocess.run(
        [sys.executable, "-c", """
import resource
from rlm_tools.worker_runtime import apply_resource_limits

_, hard = resource.getrlimit(resource.RLIMIT_CPU)
expected = 3600 if hard == resource.RLIM_INFINITY else min(3600, hard)
assert apply_resource_limits(None, 3600)["RLIMIT_CPU"] == expected
assert resource.getrlimit(resource.RLIMIT_CPU) == (expected, expected)
"""],
        check=True,
        timeout=10,
    )


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
