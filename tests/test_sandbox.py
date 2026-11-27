import tempfile

from rlm_tools.sandbox import Sandbox


def test_execute_simple_code():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    result = sandbox.execute("x = 2 + 2\nprint(x)")
    assert result.stdout.strip() == "4"
    assert result.error is None


def test_variables_persist_between_executions():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    sandbox.execute("my_var = 42")
    result = sandbox.execute("print(my_var)")
    assert result.stdout.strip() == "42"


def test_output_truncated():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=50)
    result = sandbox.execute("print('a' * 200)")
    assert "... [output truncated]" in result.stdout
    assert "(50/201 chars, 1/1 lines)" in result.stdout
    assert len(result.stdout) <= 140


def test_blocked_imports():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    result = sandbox.execute("import subprocess")
    assert result.error is not None


def test_no_write_access():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    result = sandbox.execute("open('/tmp/evil.txt', 'w').write('hack')")
    assert result.error is not None


def test_list_variables():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    sandbox.execute("foo = 1\nbar = 'hello'")
    variables = sandbox.list_variables()
    assert "foo" in variables
    assert "bar" in variables


def test_helper_globals_are_not_visible_to_user_code(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    result = sandbox.execute("print(hasattr(read_file, '__globals__'))")
    assert result.stdout.strip() == "False"
    assert result.error is None


def test_helper_globals_cannot_escape_sandbox(tmp_path):
    target = tmp_path / "allowed.txt"
    target.write_text("allowed")

    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    result = sandbox.execute(
        "print(read_file.__globals__['pathlib'].Path('/etc/hosts').read_text())"
    )
    assert result.error is not None


def test_read_file_globals_escape_is_blocked():
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Sandbox(base_path=tmpdir, max_output_chars=10_000)
        result = sandbox.execute(
            "print(read_file.__globals__['pathlib'].Path('/etc/hosts').read_text())"
        )

        assert result.error is not None
        assert "AttributeError" in result.error or "PermissionError" in result.error


def test_object_cannot_expose_helper_proxy_internals(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    result = sandbox.execute("object.__getattribute__(read_file, '_conn')")
    assert result.error is not None


def test_object_cannot_expose_finalize_callback_internals(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    result = sandbox.execute("object.__getattribute__(rlm_finalize, '_callback')")
    assert result.error is not None


def test_mro_recovery_cannot_expose_helper_proxy_internals(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    result = sandbox.execute(
        "(()).__class__.__mro__[1].__getattribute__(read_file, '_conn')"
    )
    assert result.error is not None


def test_mro_recovery_cannot_expose_finalize_callback_internals(tmp_path):
    sandbox = Sandbox(base_path=str(tmp_path), max_output_chars=10_000)
    result = sandbox.execute(
        "(()).__class__.__mro__[1].__getattribute__(rlm_finalize, '_callback')"
    )
    assert result.error is not None
