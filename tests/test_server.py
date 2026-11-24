import asyncio
import json
import os
import tempfile
import threading
from unittest.mock import MagicMock, patch

import pytest

from rlm_tools.server import _rlm_start, _rlm_execute, _rlm_end


def test_full_rlm_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "example.py"), "w") as f:
            f.write("def hello():\n    return 'world'\n\ndef foo():\n    return 'bar'\n")

        start_result = _rlm_start(path=tmpdir, query="find all functions")
        result_data = json.loads(start_result)
        session_id = result_data["session_id"]
        assert session_id is not None
        assert "metadata" in result_data

        exec_result = _rlm_execute(
            session_id=session_id,
            code="files = glob_files('**/*.py')\nprint(f'Found {len(files)} Python files')"
        )
        exec_data = json.loads(exec_result)
        assert "Found 1 Python files" in exec_data["stdout"]

        exec_result2 = _rlm_execute(
            session_id=session_id,
            code="print(files)"
        )
        exec_data2 = json.loads(exec_result2)
        assert "example.py" in exec_data2["stdout"]

        end_result = _rlm_end(session_id=session_id)
        end_data = json.loads(end_result)
        assert end_data["success"] is True


def test_invalid_session():
    result = _rlm_execute(session_id="nonexistent", code="print('hi')")
    data = json.loads(result)
    assert "error" in data


def test_invalid_directory():
    result = _rlm_start(path="/nonexistent/path", query="test")
    data = json.loads(result)
    assert "error" in data


def test_server_shutdown_releases_open_sessions(tmp_path, monkeypatch):
    from rlm_tools import server
    from rlm_tools.session import SessionManager

    monkeypatch.setattr(server, "session_manager", SessionManager())
    monkeypatch.setattr(server, "_sandboxes", {})
    monkeypatch.setattr(server.mcp, "run", lambda transport: None)
    started = json.loads(server._rlm_start(str(tmp_path), "open session"))
    session_id = started["session_id"]
    worker = server._sandboxes[session_id]._process
    server.main()
    assert worker.poll() is not None
    assert server.session_manager.get(session_id) is None
    assert not server._sandboxes


@pytest.mark.asyncio
async def test_mcp_sessions_progress_independently_and_serialize_calls(tmp_path, monkeypatch):
    from rlm_tools import server

    limiter = server.to_thread.current_default_thread_limiter()
    monkeypatch.setattr(limiter, "total_tokens", 2)
    first_id = json.loads(await server.rlm_start(str(tmp_path), "first", max_execute_calls=2))["session_id"]
    other_id = json.loads(await server.rlm_start(str(tmp_path), "other"))["session_id"]
    entered = threading.Event()
    resume = threading.Event()

    def pause():
        entered.set()
        return resume.wait(timeout=5)

    server._sandboxes[first_id].register_tool("pause", pause)
    first = asyncio.create_task(server.rlm_execute(first_id, "pause()\nx = 41", "usage"))
    queued = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        queued = asyncio.create_task(server.rlm_execute(first_id, "print(x + 1)", "usage"))
        other = await asyncio.wait_for(server.rlm_execute(other_id, "print('other session')"), timeout=3)
        assert json.loads(other)["stdout"] == "other session\n"
        assert not first.done()
        assert not queued.done()

        resume.set()
        first_result, queued_result = map(json.loads, await asyncio.gather(first, queued))
        assert first_result["usage"]["execute_calls_used"] == 1
        assert queued_result["usage"]["execute_calls_used"] == 2
        assert queued_result["stdout"] == "42\n"
        over_budget = json.loads(await server.rlm_execute(first_id, "print(x)"))
        assert "Execution call limit exceeded" in over_budget["error"]
    finally:
        resume.set()
        tasks = [first] if queued is None else [first, queued]
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(server.rlm_end(first_id), server.rlm_end(other_id))


@pytest.mark.asyncio
async def test_mcp_start_preserves_positional_timeout_and_metadata(tmp_path):
    from rlm_tools import server

    result = json.loads(await server.rlm_start(str(tmp_path), "test", 15_000, 50, 50, 1, True, True))
    try:
        assert result["limits"]["execution_timeout_seconds"] == 1
        assert "strategy" in result
        assert result["metadata"]["total_files"] == 0
    finally:
        await server.rlm_end(result["session_id"])


def test_worker_start_failure_releases_session_slot(tmp_path, monkeypatch):
    from rlm_tools import server
    from rlm_tools.session import SessionManager

    monkeypatch.setattr(server, "session_manager", SessionManager(max_sessions=1))
    with patch("rlm_tools.server.Sandbox", side_effect=RuntimeError("worker failed")):
        failed = json.loads(server._rlm_start(str(tmp_path), "test"))
    assert "worker failed" in failed["error"]

    started = json.loads(server._rlm_start(str(tmp_path), "try again"))
    server._rlm_end(started["session_id"])


def test_flat_query_retries_respect_session_budget(external_test_roots, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = MagicMock()
    client.messages.create.side_effect = ConnectionError("provider unavailable")

    with patch("rlm_tools.llm_bridge.get_client", return_value=client), \
            patch("rlm_tools.llm_bridge.time.sleep"):
        started = json.loads(_rlm_start(str(tmp_path), "test", max_llm_calls=1))
        session_id = started["session_id"]
        try:
            result = json.loads(_rlm_execute(
                session_id, "print(llm_query('question'))", detail_level="usage"
            ))
        finally:
            _rlm_end(session_id)

    assert "LLM call limit exceeded" in result["error"]
    assert result["usage"]["llm_calls_used"] == 1
    client.messages.create.assert_called_once()


def test_tool_install_failure_closes_the_worker(external_test_roots, tmp_path):
    from rlm_tools import server

    with patch.object(server, "Sandbox") as make_sandbox, \
            patch.object(server, "_install_session_llm_tools", side_effect=RuntimeError("tool registration failed")):
        failed = json.loads(server._rlm_start(str(tmp_path), "test"))

    assert "tool registration failed" in failed["error"]
    make_sandbox.return_value.close.assert_called_once()


def test_metadata_includes_file_types():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.py"), "w").close()
        open(os.path.join(tmpdir, "b.py"), "w").close()
        open(os.path.join(tmpdir, "c.txt"), "w").close()

        result = _rlm_start(path=tmpdir, query="test")
        data = json.loads(result)
        assert data["metadata"] == {}

        _rlm_end(data["session_id"])


def test_metadata_includes_file_types_when_enabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.py"), "w").close()
        open(os.path.join(tmpdir, "b.py"), "w").close()
        open(os.path.join(tmpdir, "c.txt"), "w").close()

        result = _rlm_start(path=tmpdir, query="test", include_metadata=True)
        data = json.loads(result)
        assert data["metadata"]["total_files"] == 3
        assert ".py" in data["metadata"]["file_types"]

        _rlm_end(data["session_id"])


def test_metadata_counts_regular_files_and_skips_binary_lines(tmp_path):
    from rlm_tools.server import _scan_metadata

    (tmp_path / "notes.txt").write_text("first\nsecond\n")
    (tmp_path / "image.PNG").write_bytes(b"image\ndata\n")
    (tmp_path / "missing.txt").symlink_to(tmp_path / "removed.txt")

    metadata = _scan_metadata(str(tmp_path))

    assert metadata["total_files"] == 2
    assert metadata["total_lines"] == 2
    assert metadata["file_types"] == {".txt": 1, ".PNG": 1}


def test_read_file_in_sandbox():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "data.txt"), "w") as f:
            f.write("important data")

        result = _rlm_start(path=tmpdir, query="read file")
        data = json.loads(result)
        session_id = data["session_id"]

        exec_result = _rlm_execute(
            session_id=session_id,
            code="content = read_file('data.txt')\nprint(content)"
        )
        exec_data = json.loads(exec_result)
        assert "important data" in exec_data["stdout"]
        assert exec_data["error"] is None

        _rlm_end(session_id)


def test_grep_in_sandbox():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "code.py"), "w") as f:
            f.write("class MyController:\n    def handle_error(self):\n        pass\n")

        result = _rlm_start(path=tmpdir, query="find controllers")
        data = json.loads(result)
        session_id = data["session_id"]

        exec_result = _rlm_execute(
            session_id=session_id,
            code="results = grep('class.*Controller')\nprint(len(results))"
        )
        exec_data = json.loads(exec_result)
        assert "1" in exec_data["stdout"]

        _rlm_end(session_id)


def test_skip_metadata_scan():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.py"), "w").close()

        result = _rlm_start(path=tmpdir, query="test", include_metadata=False)
        data = json.loads(result)
        assert data["metadata"] == {}
        assert "session_id" in data

        _rlm_end(data["session_id"])


def test_new_helpers_in_sandbox():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "a.txt"), "w") as f:
            f.write("hello from a")
        with open(os.path.join(tmpdir, "b.txt"), "w") as f:
            f.write("hello from b")

        result = _rlm_start(path=tmpdir, query="test helpers")
        data = json.loads(result)
        session_id = data["session_id"]

        exec_result = _rlm_execute(
            session_id=session_id,
            code="result = read_files(['a.txt', 'b.txt'])\nfor k, v in sorted(result.items()):\n    print(f'{k}: {v}')"
        )
        exec_data = json.loads(exec_result)
        assert "a.txt: hello from a" in exec_data["stdout"]
        assert "b.txt: hello from b" in exec_data["stdout"]

        exec_result2 = _rlm_execute(
            session_id=session_id,
            code="print(grep_summary('hello'))"
        )
        exec_data2 = json.loads(exec_result2)
        assert "2 matches" in exec_data2["stdout"]

        exec_result3 = _rlm_execute(
            session_id=session_id,
            code="result = grep_read('hello')\nprint(result['summary'])"
        )
        exec_data3 = json.loads(exec_result3)
        assert "2 matches" in exec_data3["stdout"]

        _rlm_end(session_id)


def test_new_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.py"), "w").close()

        result = _rlm_start(path=tmpdir, query="test defaults")
        data = json.loads(result)
        assert data["limits"]["max_execute_calls"] == 50
        assert data["limits"]["execution_timeout_seconds"] == 30
        assert data["limits"]["max_memory_bytes"] == 10_000_000

        _rlm_end(data["session_id"])


def test_full_detail_excludes_helper_functions_from_variables():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "a.txt"), "w") as f:
            f.write("hello")

        start = json.loads(_rlm_start(path=tmpdir, query="detail vars"))
        session_id = start["session_id"]

        result = json.loads(
            _rlm_execute(
                session_id=session_id,
                code="x = 123",
                detail_level="full",
            )
        )

        assert "x" in result["variables"]
        assert "read_files" not in result["variables"]
        assert "grep_summary" not in result["variables"]
        assert "grep_read" not in result["variables"]

        _rlm_end(session_id)


def test_usage_includes_memory_bytes():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "a.txt"), "w") as f:
            f.write("hello")

        start = json.loads(_rlm_start(path=tmpdir, query="memory usage"))
        session_id = start["session_id"]
        result = json.loads(
            _rlm_execute(
                session_id=session_id,
                code="print(read_file('a.txt'))",
                detail_level="usage",
            )
        )

        assert result["usage"]["memory_bytes_used"] > 0
        _rlm_end(session_id)


def test_memory_limit_counts_worker_namespace_allocations():
    with tempfile.TemporaryDirectory() as tmpdir:
        start = json.loads(
            _rlm_start(
                path=tmpdir,
                query="memory limit",
                max_memory_bytes=1024,
            )
        )
        session_id = start["session_id"]
        try:
            result = json.loads(
                _rlm_execute(
                    session_id=session_id,
                    code="blob = 'x' * 50_000\nprint(len(blob))",
                    detail_level="usage",
                )
            )
            assert result["error"] is not None
            assert "memory" in result["error"].lower()
            assert result["usage"]["memory_bytes_used"] > 1024
        finally:
            _rlm_end(session_id)


def test_recursive_helpers_and_finalize(external_test_roots):
    # rlm_recurse wraps the recursive explorer; rlm_recurse_batched runs that
    # explorer once per prompt via the real batched wrapper (left unpatched).
    with tempfile.TemporaryDirectory() as tmpdir, \
            patch("rlm_tools.server.make_llm_query") as make_query, \
            patch("rlm_tools.server.make_recursive_query") as make_recursive:
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        make_query.return_value = lambda prompt, context="": f"flat:{prompt}:{context}"
        make_recursive.return_value = lambda prompt, context="": f"answer:{prompt}:{context}"

        start = json.loads(_rlm_start(path=tmpdir, query="recursive helpers"))
        session_id = start["session_id"]
        try:
            result = json.loads(
                _rlm_execute(
                    session_id=session_id,
                    code=(
                        "snippet = 'class DashboardReducer: pass'\n"
                        "response = rlm_recurse('What type is this?', snippet)\n"
                        "batched = rlm_recurse_batched(['one', 'two'], snippet)\n"
                        "rlm_finalize(response)\n"
                        "print(response)\n"
                        "print(','.join(batched))"
                    ),
                )
            )
            assert result["error"] is None
            assert result["final"] == {"final": "answer:What type is this?:class DashboardReducer: pass"}
            assert "answer:one:class DashboardReducer: pass" in result["stdout"]
            assert "answer:two:class DashboardReducer: pass" in result["stdout"]
        finally:
            _rlm_end(session_id)
            os.environ.pop("ANTHROPIC_API_KEY", None)


def test_the_gate_can_be_enabled_with_a_typesafe_key(tmp_path, monkeypatch):
    """The gate is server-side only: it must never appear in the sandbox surface."""
    import rlm_tools.server as server_module

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("RLM_RECURSE_GATE", "1")
    monkeypatch.setattr(server_module, "make_scorer", lambda **kwargs: (lambda items, q: []))

    gate = server_module._make_recurse_gate(session=None, reserve=lambda count: None)
    assert gate is not None

    response = json.loads(server_module._rlm_start(str(tmp_path), "query"))
    functions = " ".join(response["available_functions"])
    for removed in ("rank(", "keep(", "judge(", "classify("):
        assert removed not in functions
    server_module._rlm_end(response["session_id"])


def test_no_gate_without_a_typesafe_key(tmp_path, monkeypatch):
    import rlm_tools.server as server_module

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    assert server_module._make_recurse_gate(session=None) is None


def test_the_gate_can_be_switched_off(tmp_path, monkeypatch):
    import rlm_tools.server as server_module

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("RLM_RECURSE_GATE", "0")

    assert server_module._make_recurse_gate(session=None) is None


def test_start_scans_metadata_for_authorized_symlinked_root(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "main.py").write_text("print('hi')\n")
    symlink_root = tmp_path / "link_root"
    symlink_root.symlink_to(real_root)

    started = json.loads(_rlm_start(str(symlink_root), "query", include_metadata=True))
    session_id = started["session_id"]
    try:
        metadata = started["metadata"]
        assert metadata["total_files"] == 1
        assert metadata["total_lines"] == 1
        assert metadata["file_types"] == {".py": 1}
    finally:
        _rlm_end(session_id)


def test_start_guidance_omits_unavailable_model_helpers(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    data = json.loads(_rlm_start(str(tmp_path), "query", include_guidance=True))
    try:
        assert "rlm_finalize" in data["strategy"]
        assert "rlm_recurse" not in data["strategy"]
        assert "llm_query" not in data["strategy"]
    finally:
        _rlm_end(data["session_id"])


def test_start_includes_allowed_imports_and_supports_code_and_end_session(tmp_path):
    (tmp_path / "data.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    # Initial code on rlm_start keeps the session open and returns stdout/allowed_imports.
    started_with_error = json.loads(_rlm_start(
        path=str(tmp_path),
        query="bad import",
        code="import os\nprint(1)",
    ))
    sid_err = started_with_error["session_id"]
    try:
        assert "not allowed" in started_with_error["error"]
        assert "json" in started_with_error["allowed_imports"]
        assert "os" not in started_with_error["allowed_imports"]

        recovered = json.loads(_rlm_execute(
            sid_err,
            "print(read_file('data.txt').splitlines()[0])",
            end_session=True,
        ))
        assert recovered["error"] is None
        assert recovered["stdout"] == "alpha\n"
        assert recovered["session_ended"] is True
        assert "not found or expired" in json.loads(_rlm_execute(sid_err, "print(1)"))["error"]
    finally:
        _rlm_end(sid_err)

    # Start with initial code preserves variables across rlm_execute and supports finalize + end_session=True.
    multi = json.loads(_rlm_start(
        path=str(tmp_path),
        query="keep state",
        code="items = read_file('data.txt').splitlines()\nprint('loaded')\nrlm_finalize(len(items))",
    ))
    sid = multi["session_id"]
    try:
        assert multi["error"] is None
        assert multi["stdout"] == "loaded\n"
        assert multi["final"] == {"final": 3}
        assert "re" in multi["allowed_imports"]
        assert "session_ended" not in multi

        mid = json.loads(_rlm_execute(sid, "print(items[1])"))
        assert mid == {"stdout": "beta\n", "error": None}
        assert "session_ended" not in mid

        err_end = json.loads(_rlm_execute(sid, "open('data.txt')", end_session=True))
        assert err_end["session_ended"] is False
        assert err_end["error"] is not None

        done = json.loads(_rlm_execute(sid, "print(items[2])", end_session=True))
        assert done == {"stdout": "gamma\n", "error": None, "session_ended": True}
        assert "not found" in json.loads(_rlm_execute(sid, "print(1)"))["error"]
    finally:
        _rlm_end(sid)


def test_gemini_external_session_installs_llm_tools_and_gate(external_test_roots, tmp_path, monkeypatch):
    import rlm_tools.server as server_module

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("RLM_RECURSE_GATE", "1")
    with patch.object(server_module, "make_scorer", return_value=lambda items, q: [0.9] * len(items)) as make_scorer_mock, \
            patch.object(server_module, "make_llm_query", return_value=lambda prompt, context="": f"gemini:{prompt}") as make_query_mock, \
            patch.object(server_module, "make_recursive_query", return_value=lambda prompt, context="": f"rec:{prompt}") as make_rec_mock:
        started = json.loads(server_module._rlm_start(str(tmp_path), "gemini external"))
        sid = started["session_id"]
        try:
            assert "llm_query(prompt, context='')" in started["available_functions"]
            assert "rlm_recurse(prompt, context='')" in started["available_functions"]
            make_scorer_mock.assert_called_once()
            make_query_mock.assert_called_once()
            assert make_rec_mock.call_args.kwargs["gate"] is not None
        finally:
            server_module._rlm_end(sid)


@pytest.mark.parametrize(
    ("env_key", "env_val"),
    [
        ("OPENAI_API_KEY", "test-openai-key"),
        ("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    ],
)
def test_openai_compat_external_session_installs_llm_tools(external_test_roots, tmp_path, monkeypatch, env_key, env_val):
    import rlm_tools.server as server_module

    monkeypatch.setenv(env_key, env_val)
    with patch.object(server_module, "make_llm_query", return_value=lambda prompt, context="": f"oa:{prompt}") as make_query_mock, \
            patch.object(server_module, "make_recursive_query", return_value=lambda prompt, context="": f"rec:{prompt}") as make_rec_mock:
        started = json.loads(server_module._rlm_start(str(tmp_path), "openai compat external"))
        sid = started["session_id"]
        try:
            assert "llm_query(prompt, context='')" in started["available_functions"]
            assert "rlm_recurse(prompt, context='')" in started["available_functions"]
            make_query_mock.assert_called_once()
            make_rec_mock.assert_called_once()
        finally:
            server_module._rlm_end(sid)
