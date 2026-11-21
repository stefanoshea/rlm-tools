from evals.conftest import execute
from rlm_tools.sandbox import Sandbox


class TestRobustness:
    def test_invalid_regex_returns_error(self, sandbox):
        result = execute(sandbox, "grep('(')")
        assert result["error"] is not None
        assert "error" in result["error"].lower() or "unterminated" in result["error"].lower()

    def test_missing_file_returns_error(self, sandbox):
        result = execute(sandbox, "read_file('missing.py')")
        assert result["error"] is not None
        assert "No such file" in result["error"] or "FileNotFoundError" in result["error"]

    def test_path_traversal_still_blocked(self, sandbox):
        result = execute(sandbox, "read_file('../../../etc/passwd')")
        assert result["error"] is not None
        assert "PermissionError" in result["error"]

    def test_large_output_truncates(self):
        with Sandbox(base_path="tests/fixtures/rlm_corpus_py", max_output_chars=40) as sandbox:
            result = sandbox.execute("print('x' * 200)")
            assert "... [output truncated]" in result.stdout

    def test_execution_timeout_surfaces_as_error(self):
        with Sandbox(
            base_path="tests/fixtures/rlm_corpus_py",
            max_output_chars=10_000,
            execution_timeout_seconds=1,
        ) as sandbox:
            result = sandbox.execute("while True:\n    pass")
            assert result.error is not None
            assert "timed out" in result.error
