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
    assert len(result.stdout) <= 80  # 50 + truncation message


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
