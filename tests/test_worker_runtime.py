import signal
import time

import pytest

from rlm_tools.sandbox import Sandbox
from rlm_tools.worker_runtime import WorkerRuntime


def _runtime(max_memory_bytes=10**9):
    return WorkerRuntime(
        conn=None,
        max_output_chars=1000,
        execution_timeout_seconds=0,
        max_memory_bytes=max_memory_bytes,
    )


def test_measure_user_memory_full_when_under_limit():
    rt = _runtime()
    rt._namespace["a"] = "x" * 100_000
    rt._namespace["b"] = "y" * 100_000
    measured = rt._measure_user_memory()
    assert measured >= 200_000


def test_measure_user_memory_short_circuits_over_limit():
    rt = _runtime()
    rt._namespace["a"] = "x" * 100_000
    rt._namespace["b"] = "y" * 100_000
    full = rt._measure_user_memory()
    capped = rt._measure_user_memory(limit=1_000)
    # Early-exit stops after the first variable already blows the cap, so the
    # capped measurement is over the limit but cheaper than the full traversal.
    assert capped >= 1_000
    assert capped < full


def test_worker_backed_sandbox_preserves_state(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    sandbox.execute("value = 41")
    result = sandbox.execute("value += 1\nprint(value)")
    assert result.stdout.strip() == "42"
    assert result.error is None


def test_worker_backed_sandbox_can_use_helpers(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello from worker")

    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    result = sandbox.execute("print(read_file('sample.txt'))")
    assert result.stdout.strip() == "hello from worker"
    assert result.error is None


def test_accounting_accepts_deeply_nested_values():
    runtime = _runtime()
    result = runtime.execute("value = []\nfor i in range(1500):\n    value = [value]")

    assert result.error is None
    assert result.memory_bytes_used >= 1500


def test_bad_final_value_does_not_stop_worker(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path))
    try:
        assert sandbox.execute("rlm_finalize({1, 2})").error.startswith("TypeError:")
        assert sandbox.execute("rlm_finalize({'result': [1, 2]})").final == {
            "final": {"result": [1, 2]}
        }
    finally:
        sandbox.close()


def test_closed_worker_reports_unavailability(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path))
    sandbox.close()
    sandbox.close()

    assert "worker is unavailable" in sandbox.execute("print('hello')").error


def test_stopped_worker_reports_unavailability(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path))
    try:
        sandbox._process.terminate()
        sandbox._process.join(timeout=5)
        assert not sandbox._process.is_alive()
        assert "worker is unavailable" in sandbox.execute("print('hello')").error
    finally:
        sandbox.close()


def test_reported_memory_limit_includes_helper_cache(tmp_path):
    (tmp_path / "sample.txt").write_text("a" * 500)
    sandbox = Sandbox(base_path=str(tmp_path), max_memory_bytes=1000)
    try:
        result = sandbox.execute("read_file('sample.txt')\nvalue = 'b' * 600")
        assert result.error.startswith("MemoryError:")
        assert result.memory_bytes_used > 1000
        assert sandbox.memory_bytes_used == 500
        assert sandbox.list_variables() == []
        assert "closed after exceeding" in sandbox.execute("print(1)").error
    finally:
        sandbox.close()


def test_over_limit_final_value_is_discarded(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_memory_bytes=1000)
    try:
        result = sandbox.execute("rlm_finalize('a' * 2000)")

        assert "worker state was discarded" in result.error
        assert result.final is None
        assert not sandbox._process.is_alive()
    finally:
        sandbox.close()


def test_memory_budget_counts_private_user_variables(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_memory_bytes=1000)
    try:
        result = sandbox.execute("_scratch = 'a' * 2000")

        assert "worker state was discarded" in result.error
        assert sandbox.memory_bytes_used == 0
    finally:
        sandbox.close()


def test_missing_worker_response_closes_worker(tmp_path, monkeypatch):
    sandbox = Sandbox(base_path=str(tmp_path), execution_timeout_seconds=1)
    try:
        monkeypatch.setattr(sandbox._parent_conn, "poll", lambda timeout: False)

        result = sandbox.execute("print('hello')")

        assert result.error.startswith("TimeoutError:")
        assert not sandbox._process.is_alive()
        assert "closed after an execution timeout" in sandbox.execute("print(1)").error
    finally:
        sandbox.close()


@pytest.mark.skipif(not hasattr(signal, "pthread_sigmask"), reason="requires POSIX signals")
def test_helper_timeout_keeps_worker_protocol_in_sync(tmp_path):
    def slow_helper():
        time.sleep(1.2)
        return "reply"

    sandbox = Sandbox(base_path=str(tmp_path), execution_timeout_seconds=1)
    try:
        sandbox.register_tool("slow_helper", slow_helper)
        assert sandbox.execute("print(slow_helper())").error.startswith("TimeoutError:")
        assert sandbox.execute("print('next request')").stdout == "next request\n"
    finally:
        sandbox.close()
