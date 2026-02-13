import json
import os
import tempfile
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


def test_metadata_includes_file_types():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.py"), "w").close()
        open(os.path.join(tmpdir, "b.py"), "w").close()
        open(os.path.join(tmpdir, "c.txt"), "w").close()

        result = _rlm_start(path=tmpdir, query="test")
        data = json.loads(result)
        assert data["metadata"]["total_files"] == 3
        assert ".py" in data["metadata"]["file_types"]

        _rlm_end(data["session_id"])


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
