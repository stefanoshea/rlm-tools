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
        sandbox._process.wait(timeout=5)
        assert sandbox._process.poll() is not None
        assert "worker is unavailable" in sandbox.execute("print('hello')").error
    finally:
        sandbox.close()


def test_reported_memory_limit_evicts_helper_cache_first(tmp_path):
    (tmp_path / "sample.txt").write_text("a" * 500)
    sandbox = Sandbox(base_path=str(tmp_path), max_memory_bytes=1000)
    try:
        result = sandbox.execute("read_file('sample.txt')\nvalue = 'b' * 600")
        assert result.error is None
        assert sandbox.memory_bytes_used <= 1000
        assert "value" in sandbox.list_variables()
    finally:
        sandbox.close()


def test_over_limit_final_value_is_dropped_without_killing_worker(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_memory_bytes=1000)
    try:
        sandbox.execute("small = 42")
        result = sandbox.execute("rlm_finalize('a' * 2000)")

        assert result.error is not None
        assert "MemoryError:" in result.error
        assert "dropped <final>" in result.error
        assert result.final is None
        assert sandbox._process.poll() is None
        follow_up = sandbox.execute("print(small)")
        assert follow_up.error is None
        assert follow_up.stdout == "42\n"
    finally:
        sandbox.close()


def test_memory_budget_evicts_largest_vars_including_private_and_keeps_imports(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_memory_bytes=1000)
    try:
        sandbox.execute("import re\nsmall = 'ok'")
        result = sandbox.execute("_scratch = 'a' * 2000")

        assert result.error is not None
        assert "MemoryError:" in result.error
        assert "dropped _scratch" in result.error
        assert sandbox.memory_bytes_used <= 1000
        alive = sandbox.execute("print(small, bool(re.match('a', 'a')))")
        assert alive.error is None
        assert alive.stdout == "ok True\n"
    finally:
        sandbox.close()


def test_parent_hard_kill_backstop_triggers_if_worker_still_exceeds_limit(tmp_path, monkeypatch):
    sandbox = Sandbox(base_path=str(tmp_path), max_memory_bytes=1000)
    try:
        monkeypatch.setattr(
            sandbox,
            "_parent_conn",
            sandbox._parent_conn,
        )
        orig_receive = __import__("rlm_tools.sandbox", fromlist=["receive_message"]).receive_message

        def fake_receive(conn, deadline=None):
            msg = orig_receive(conn, deadline)
            if msg.get("type") == "execute_result":
                msg["memory_bytes_used"] = 5000
            return msg

        monkeypatch.setattr("rlm_tools.sandbox.receive_message", fake_receive)
        result = sandbox.execute("x = 1")
        assert "worker state was discarded" in result.error
        assert sandbox._process.poll() is not None
    finally:
        sandbox.close()


def test_missing_worker_response_closes_worker(tmp_path, monkeypatch):
    sandbox = Sandbox(base_path=str(tmp_path), execution_timeout_seconds=1)
    try:
        monkeypatch.setattr(sandbox._parent_conn, "poll", lambda timeout: False)

        result = sandbox.execute("print('hello')")

        assert result.error.startswith("TimeoutError:")
        assert sandbox._process.poll() is not None
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


def test_deque_and_partial_memory_counted_against_limit(tmp_path):
    with Sandbox(base_path=str(tmp_path), max_memory_bytes=2_000) as sandbox:
        deque_result = sandbox.execute("import collections\nd = collections.deque(['a' * 3000])")
        assert deque_result.error is not None
        assert "MemoryError:" in deque_result.error

    with Sandbox(base_path=str(tmp_path), max_memory_bytes=2_000) as sandbox:
        partial_result = sandbox.execute(
            "import functools\np = functools.partial(len, 'a' * 3000)"
        )
        assert partial_result.error is not None
        assert "MemoryError:" in partial_result.error


def test_accounting_does_not_invoke_list_subclass_memory_hooks(tmp_path):
    with Sandbox(base_path=str(tmp_path), max_memory_bytes=50_000) as sandbox:
        code = (
            "memory_hooks = []\n"
            "class CustomList(list):\n"
            "    def __getattribute__(self, name):\n"
            "        memory_hooks.append(name)\n"
            "        raise AttributeError(name)\n"
            "    @property\n"
            "    def __dict__(self):\n"
            "        memory_hooks.append('dict')\n"
            "        return {}\n"
            "    def __getattr__(self, name):\n"
            "        memory_hooks.append('getattr')\n"
            "        raise AttributeError(name)\n"
            "    def __sizeof__(self):\n"
            "        memory_hooks.append('sizeof')\n"
            "        return 0\n"
            "values = CustomList(['first', 'second'])\n"
            "print('created')\n"
        )
        first = sandbox.execute(code)
        assert first.error is None
        assert first.stdout.strip() == "created"
        second = sandbox.execute("print(memory_hooks)")
        assert second.error is None
        assert second.stdout.strip() == "[]"


def test_accounting_uses_partial_storage_and_inherited_slots():
    runtime = _runtime()
    result = runtime.execute(
        "import functools\n"
        "memory_hooks = []\n"
        "class CustomPartial(functools.partial):\n"
        "    @property\n"
        "    def args(self):\n"
        "        memory_hooks.append('args')\n"
        "        return ()\n"
        "class Base:\n"
        "    __slots__ = ('contents',)\n"
        "    def __init__(self, value):\n"
        "        self.contents = value\n"
        "class Child(Base):\n"
        "    __slots__ = ()\n"
        "partial_value = CustomPartial(len, 'x' * 3000)\n"
        "slotted_value = Child('y' * 3000)\n"
    )

    assert result.error is None
    assert result.memory_bytes_used >= 6000
    assert runtime.execute("print(memory_hooks)").stdout.strip() == "[]"


def test_sandbox_context_manager_closes_worker(tmp_path):
    with Sandbox(base_path=str(tmp_path)) as sandbox:
        process = sandbox._process
        assert process.poll() is None
    assert process.poll() is not None


def test_rebinding_protected_helper_name_is_counted_and_evicted():
    runtime = _runtime(max_memory_bytes=4_000)
    runtime.register_tools(["tree"])

    # Unmodified helper stays hidden and uncounted.
    assert runtime.list_variables() == []
    assert runtime._measure_user_memory() == 0

    # Rebinding a helper name to a large user payload must be measured and evicted.
    result = runtime.execute("tree = 'x' * 10_000")
    assert result.error is not None
    assert "MemoryError:" in result.error
    assert "tree" in result.error
    assert result.memory_bytes_used <= 4_000
    assert "tree" not in runtime._namespace


def test_evict_to_fit_shared_graph_aliases_is_bounded_and_preserves_modules():
    runtime = _runtime(max_memory_bytes=16_000)
    t0 = time.monotonic()
    result = runtime.execute(
        "import re\n"
        "shared = [f'{i}-' + 'x' * 200 for i in range(200)]\n"
        "aliases = {f'v{i}': shared for i in range(60)}\n"
        "a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = a9 = shared\n"
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0
    assert result.error is not None
    assert "MemoryError:" in result.error
    assert result.memory_bytes_used <= 16_000
    # Imported stdlib module remains intact after eviction.
    assert "re" in runtime._namespace
