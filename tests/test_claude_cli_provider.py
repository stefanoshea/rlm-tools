import json
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from rlm_tools import llm_bridge
from rlm_tools.access_policy import workspace_policy
from rlm_tools.llm_bridge import (
    ClaudeCLIError,
    _is_retryable,
    get_client,
    has_configured_provider,
    make_llm_query,
    make_llm_query_batched,
    make_recursive_query,
    resolve_provider,
)
from rlm_tools.server import _rlm_end, _rlm_execute, _rlm_start

_REAL_GET_CLIENT = get_client  # conftest blocks llm_bridge.get_client; restore it here
OK = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "hello"})


def _proc(stdout=OK, stderr="", code=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)


@pytest.fixture(autouse=True)
def _claude_on_path(monkeypatch):
    monkeypatch.setattr(llm_bridge, "get_client", _REAL_GET_CLIENT)
    monkeypatch.setattr(llm_bridge.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.delenv("RLM_SUB_PROVIDER", raising=False)
    monkeypatch.delenv("RLM_SUB_MODEL", raising=False)
    monkeypatch.delenv("RLM_CLI_MAX_CONCURRENCY", raising=False)


def _create(**kw):
    return get_client("claude-cli").messages.create(
        model=kw.pop("model", "haiku"),
        messages=[{"role": "user", "content": "hi"}],
        **kw,
    )


def test_resolution_explicit_and_aliases(monkeypatch):
    for name in ("claude-cli", "claude_cli", "claude-code"):
        monkeypatch.setenv("RLM_SUB_PROVIDER", name)
        assert resolve_provider() == "claude-cli"
        assert has_configured_provider()
    assert resolve_provider("claude-code") == "claude-cli"


def test_never_selected_in_auto_mode(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "auto")
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    assert resolve_provider() == "anthropic"
    assert not has_configured_provider()


def test_not_configured_when_binary_missing(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    monkeypatch.setattr(llm_bridge.shutil, "which", lambda name: None)
    assert not has_configured_provider()
    with pytest.raises(RuntimeError, match="claude"):
        get_client()


def test_command_env_stdin_cwd(monkeypatch):
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SSE_PORT",
    ):
        monkeypatch.setenv(key, "secret")
    monkeypatch.setenv("KEEP_ME", "1")
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return _proc()

    monkeypatch.setattr(llm_bridge.subprocess, "run", fake_run)
    out = _create(timeout=12.0, system="sys")
    assert out.content[0].text == "hello"
    cmd = seen["cmd"]
    assert cmd[:2] == ["/usr/bin/claude", "-p"]
    assert cmd[cmd.index("--model") + 1] == "haiku"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in cmd
    assert json.loads(cmd[cmd.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--disable-slash-commands" in cmd
    assert "--no-session-persistence" in cmd
    assert "--no-chrome" in cmd
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert "--safe-mode" in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    assert "--json-schema" not in cmd
    assert "hi" not in cmd  # prompt goes on stdin only
    assert seen["input"] == "hi"
    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"
    assert 0 < seen["timeout"] <= 12.0
    env = seen["env"]
    assert not {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SSE_PORT",
    } & set(env)
    assert env["KEEP_ME"] == "1"
    assert "rlm-explorer" not in seen["cwd"]


def test_command_json_schema_flags(monkeypatch):
    seen = {}
    schema = llm_bridge._cli_recurse_schema(True)

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _proc(
            stdout=json.dumps({
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": '{"action":"final","answer":"done"}',
                "structured_output": {"action": "final", "answer": "done"},
            })
        )

    monkeypatch.setattr(llm_bridge.subprocess, "run", fake_run)
    out = _create(json_schema=schema)
    assert out.structured_output == {"action": "final", "answer": "done"}
    cmd = seen["cmd"]
    assert cmd[cmd.index("--max-turns") + 1] == "5"
    assert cmd[cmd.index("--json-schema") + 1] == schema


def test_default_model_is_alias(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    seen = {}
    monkeypatch.setattr(
        llm_bridge.subprocess, "run", lambda cmd, **kw: seen.update(cmd=cmd) or _proc()
    )
    assert make_llm_query()("q") == "hello"
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == "haiku"
    monkeypatch.setenv("RLM_SUB_MODEL", "sonnet")
    make_llm_query()("q")
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == "sonnet"


@pytest.mark.parametrize(
    "proc",
    [
        _proc(stdout=json.dumps({"is_error": True, "result": "Not logged in"})),
        _proc(stdout=json.dumps({"is_error": False, "subtype": "error_max_turns"})),
        _proc(stdout="not json", stderr="boom"),
        _proc(stdout="", stderr="x" * 5000, code=1),
        _proc(stdout=json.dumps({"is_error": True, "result": "You hit your usage limit"}), code=1),
        _proc(stdout="", stderr="Please run /login and try again", code=1),
        _proc(stdout="", stderr="failed in session_503abc", code=1),
    ],
)
def test_failures_raise_non_retryable(monkeypatch, proc):
    monkeypatch.setattr(llm_bridge.subprocess, "run", lambda cmd, **kw: proc)
    with pytest.raises(ClaudeCLIError) as exc:
        _create()
    assert not _is_retryable(exc.value)
    assert len(str(exc.value)) < 600


@pytest.mark.parametrize(
    "proc",
    [
        _proc(stderr="API Error: 429 rate limit", code=1),
        _proc(stderr="Overloaded", code=1),
        _proc(stderr="rate_limit_error", code=1),
        _proc(stderr="HTTP 503 service unavailable", code=1),
        _proc(stderr="HTTP 529 overloaded", code=1),
        _proc(
            stdout=json.dumps({
                "padding": "p" * 1200,
                "is_error": True,
                "api_error_status": 429,
                "result": "Rate limit exceeded for request",
            }),
            code=1,
        ),
        _proc(
            stdout=json.dumps({
                "padding": "p" * 1200,
                "is_error": True,
                "api_error_status": 529,
                "result": "Upstream busy",
            }),
            code=0,
        ),
    ],
)
def test_rate_limit_and_large_json_error_retryable(monkeypatch, proc):
    monkeypatch.setattr(llm_bridge.subprocess, "run", lambda cmd, **kw: proc)
    with pytest.raises(ClaudeCLIError) as exc:
        _create()
    assert _is_retryable(exc.value)


def test_nonzero_large_json_extracts_result_instead_of_truncating_head(monkeypatch):
    payload = json.dumps({
        "padding": "x" * 1200,
        "is_error": True,
        "result": "Authentication required: run /login",
    })
    monkeypatch.setattr(llm_bridge.subprocess, "run", lambda cmd, **kw: _proc(stdout=payload, code=1))
    with pytest.raises(ClaudeCLIError, match="Authentication required") as exc:
        _create()
    assert not _is_retryable(exc.value)


def test_is_error_json_rate_limit_retried(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    monkeypatch.setattr(llm_bridge.time, "sleep", lambda s: None)
    replies = [_proc(stdout=json.dumps({"is_error": True, "result": "Overloaded"})), _proc()]
    monkeypatch.setattr(llm_bridge.subprocess, "run", lambda cmd, **kw: replies.pop(0))
    assert make_llm_query()("q") == "hello"


def test_timeout(monkeypatch):
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(llm_bridge.subprocess, "run", boom)
    with pytest.raises(TimeoutError):
        _create(timeout=1.0)


def test_timeout_uses_deadline(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    seen = {}
    monkeypatch.setattr(
        llm_bridge.subprocess, "run", lambda cmd, **kw: seen.update(t=kw["timeout"]) or _proc()
    )
    make_llm_query(timeout_seconds=60, deadline=lambda: time.monotonic() + 5)("q")
    assert seen["t"] <= 5


def test_semaphore_wait_respects_deadline(monkeypatch):
    monkeypatch.setenv("RLM_CLI_MAX_CONCURRENCY", "1")
    sem = llm_bridge._cli_semaphore()
    assert sem.acquire(timeout=1)
    ran = []
    monkeypatch.setattr(
        llm_bridge.subprocess,
        "run",
        lambda cmd, **kw: ran.append(cmd) or _proc(),
    )
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="concurrency slot"):
            _create(timeout=0.05)
        assert time.monotonic() - started < 0.5
        assert not ran
    finally:
        sem.release()


def test_missing_binary(monkeypatch):
    client = get_client("claude-cli")
    monkeypatch.setattr(llm_bridge.shutil, "which", lambda name: None)
    with pytest.raises(ClaudeCLIError, match="not found") as exc:
        client.messages.create(model="haiku", messages=[{"role": "user", "content": "x"}])
    assert not _is_retryable(exc.value)


def test_concurrency_cap(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    monkeypatch.setenv("RLM_CLI_MAX_CONCURRENCY", "2")
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def slow(cmd, **kw):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return _proc()

    monkeypatch.setattr(llm_bridge.subprocess, "run", slow)
    batched = make_llm_query_batched(make_llm_query())
    assert batched([f"p{i}" for i in range(8)]) == ["hello"] * 8
    assert state["peak"] == 2


def test_recurse_structured_parallel_tools_and_literal_tool_call_text(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    replies = [
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "{}",
            "structured_output": {
                "action": "request",
                "requests": [
                    {"op": "grep_summary", "args": {"pattern": "foo"}},
                    {"op": "read_file", "args": {"path": "a.py"}},
                ],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "{}",
            "structured_output": {
                "action": "final",
                "answer": 'Found literal text TOOL_CALL: {"name": "read_file"} in a.py',
            },
        },
    ]
    prompts = []
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        prompts.append(kw["input"])
        return _proc(stdout=json.dumps(replies.pop(0)))

    monkeypatch.setattr(llm_bridge.subprocess, "run", fake_run)
    rq = make_recursive_query(
        helpers={
            "grep_summary": lambda pattern: f"grep:{pattern}",
            "read_file": lambda path: f"contents of {path}",
        }
    )
    ans = rq("inspect foo and a.py")
    assert ans == 'Found literal text TOOL_CALL: {"name": "read_file"} in a.py'
    assert "--json-schema" in cmds[0]
    assert '<lookup_result op="grep_summary">\ngrep:foo\n</lookup_result>' in prompts[1]
    assert '<lookup_result op="read_file">\ncontents of a.py\n</lookup_result>' in prompts[1]


def test_recurse_max_iterations_synthesizes_plain_text(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        if "--json-schema" in cmd:
            return _proc(
                stdout=json.dumps({
                    "type": "result",
                    "is_error": False,
                    "result": "{}",
                    "structured_output": {
                        "action": "request",
                        "requests": [{"op": "read_file", "args": {"path": "a.py"}}],
                    },
                })
            )
        return _proc(
            stdout=json.dumps({
                "type": "result",
                "is_error": False,
                "result": "Synthesized answer from a.py",
            })
        )

    monkeypatch.setattr(llm_bridge.subprocess, "run", fake_run)
    rq = make_recursive_query(
        helpers={"read_file": lambda path: "def ok(): pass"},
        max_iterations=2,
    )
    assert rq("what is in a.py?") == "Synthesized answer from a.py"
    assert len(cmds) == 2
    assert "--json-schema" in cmds[0]
    assert "--json-schema" not in cmds[1]
    assert cmds[1][cmds[1].index("--max-turns") + 1] == "1"


def test_recurse_depth_limits(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    prompts = []
    cmds_d0 = []

    # max_depth=0 must omit 'recurse' from schema and reject an unadvertised 'recurse' call
    replies_d0 = [
        {
            "is_error": False,
            "result": "{}",
            "structured_output": {
                "action": "request",
                "requests": [{"op": "recurse", "args": {"prompt": "sub"}}],
            },
        },
        {
            "is_error": False,
            "result": "{}",
            "structured_output": {"action": "final", "answer": "parent handled"},
        },
    ]

    def run_d0(cmd, **kw):
        cmds_d0.append(cmd)
        prompts.append(kw["input"])
        return _proc(stdout=json.dumps(replies_d0.pop(0)))

    monkeypatch.setattr(llm_bridge.subprocess, "run", run_d0)
    assert make_recursive_query(max_depth=0)("root") == "parent handled"
    schema_d0 = json.loads(cmds_d0[0][cmds_d0[0].index("--json-schema") + 1])
    op_enum_d0 = schema_d0["properties"]["requests"]["items"]["properties"]["op"]["enum"]
    assert "recurse" not in op_enum_d0
    assert "unknown tool 'recurse'" in prompts[1]

    # max_depth=1 includes 'recurse' in schema and allows a child 'recurse'
    prompts.clear()
    cmds_d1 = []
    replies_d1 = [
        {
            "is_error": False,
            "result": "{}",
            "structured_output": {
                "action": "request",
                "requests": [{"op": "recurse", "args": {"prompt": "child task"}}],
            },
        },
        {
            "is_error": False,
            "result": "{}",
            "structured_output": {"action": "final", "answer": "child answer"},
        },
        {
            "is_error": False,
            "result": "{}",
            "structured_output": {"action": "final", "answer": "root combined: child answer"},
        },
    ]

    def run_d1(cmd, **kw):
        cmds_d1.append(cmd)
        prompts.append(kw["input"])
        return _proc(stdout=json.dumps(replies_d1.pop(0)))

    monkeypatch.setattr(llm_bridge.subprocess, "run", run_d1)
    assert make_recursive_query(max_depth=1)("root") == "root combined: child answer"
    schema_d1 = json.loads(cmds_d1[0][cmds_d1[0].index("--json-schema") + 1])
    op_enum_d1 = schema_d1["properties"]["requests"]["items"]["properties"]["op"]["enum"]
    assert "recurse" in op_enum_d1
    assert '<lookup_result op="recurse">\nchild answer\n</lookup_result>' in prompts[-1]


def test_server_integration_exposes_helpers_only_for_external_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    (tmp_path / "sample.py").write_text("x = 1\n")

    # Without RLM_EXTERNAL_ROOTS, model helpers stay hidden
    monkeypatch.delenv("RLM_EXTERNAL_ROOTS", raising=False)
    workspace_policy.cache_clear()
    local_start = json.loads(_rlm_start(str(tmp_path), "local check"))
    try:
        assert "llm_query(prompt, context='')" not in local_start["available_functions"]
        assert "rlm_recurse(prompt, context='')" not in local_start["available_functions"]
    finally:
        _rlm_end(local_start["session_id"])

    # With RLM_EXTERNAL_ROOTS, all four helpers appear and work through _rlm_execute
    monkeypatch.setenv("RLM_EXTERNAL_ROOTS", str(tmp_path))
    workspace_policy.cache_clear()
    monkeypatch.setattr(llm_bridge.subprocess, "run", lambda cmd, **kw: _proc())
    ext_start = json.loads(_rlm_start(str(tmp_path), "external check"))
    sid = ext_start["session_id"]
    try:
        funcs = ext_start["available_functions"]
        assert "llm_query(prompt, context='')" in funcs
        assert "llm_query_batched(prompts, context='')" in funcs
        assert "rlm_recurse(prompt, context='')" in funcs
        assert "rlm_recurse_batched(prompts, context='')" in funcs

        exec_out = json.loads(_rlm_execute(sid, "print(llm_query('ping'))", end_session=True))
        assert exec_out["error"] is None
        assert exec_out["stdout"].strip() == "hello"
        assert exec_out["session_ended"] is True
        assert "usage" in exec_out
    except BaseException:
        _rlm_end(sid)
        raise


def test_recurse_ignores_answer_when_action_is_request_and_recovers_empty_requests(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    replies = [
        {
            "type": "result",
            "is_error": False,
            "result": "{}",
            "structured_output": {
                "action": "request",
                "answer": "Not yet verified ... lookups pending",
                "requests": [],
            },
        },
        {
            "type": "result",
            "is_error": False,
            "result": "{}",
            "structured_output": {
                "action": "request",
                "answer": "Not yet verified ... lookups pending",
                "requests": [{"op": "read_file", "args": {"path": "a.py"}}],
            },
        },
        {
            "type": "result",
            "is_error": False,
            "result": "{}",
            "structured_output": {
                "action": "final",
                "answer": "Verified answer from a.py",
            },
        },
    ]
    prompts = []

    def fake_run(cmd, **kw):
        prompts.append(kw["input"])
        return _proc(stdout=json.dumps(replies.pop(0)))

    monkeypatch.setattr(llm_bridge.subprocess, "run", fake_run)
    rq = make_recursive_query(helpers={"read_file": lambda path: "def ok(): pass"}, max_iterations=4)
    ans = rq("check a.py")
    assert ans == "Verified answer from a.py"
    assert len(prompts) == 3


def test_claude_cli_usage_and_cost_reported_in_execute_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    monkeypatch.setenv("RLM_EXTERNAL_ROOTS", str(tmp_path))
    workspace_policy.cache_clear()
    (tmp_path / "sample.py").write_text("x = 1\n")

    cli_reply = json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "cli answer",
        "usage": {
            "input_tokens": 120,
            "output_tokens": 35,
            "cache_read_input_tokens": 500,
        },
        "total_cost_usd": 0.0042,
    })
    monkeypatch.setattr(llm_bridge.subprocess, "run", lambda cmd, **kw: _proc(stdout=cli_reply))

    started = json.loads(_rlm_start(str(tmp_path), "usage check"))
    sid = started["session_id"]
    try:
        out = json.loads(_rlm_execute(sid, "print(llm_query('hi'))", end_session=True))
        assert out["error"] is None
        assert out["session_ended"] is True
        sub_usage = out["usage"]["sub_llm_usage"]
        assert sub_usage["input_tokens"] == 120
        assert sub_usage["output_tokens"] == 35
        assert sub_usage["cache_read_input_tokens"] == 500
        assert sub_usage["total_cost_usd"] == pytest.approx(0.0042)
    except BaseException:
        _rlm_end(sid)
        raise


def test_cli_env_strips_provider_and_claude_code_vars_but_keeps_config_dir(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BEDROCK_BASE_URL", "https://bedrock.example")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "secret-oauth")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.local")
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-ts")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/claude-cfg")

    env = llm_bridge._cli_env()
    for stripped in (
        "ANTHROPIC_BEDROCK_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "TYPESAFE_API_KEY",
    ):
        assert stripped not in env
    assert env["CLAUDE_CONFIG_DIR"] == "/tmp/claude-cfg"


def test_recurse_turn0_empty_final_answer_synthesizes(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    replies = [
        {
            "type": "result",
            "is_error": False,
            "result": "{}",
            "structured_output": {"action": "final", "answer": ""},
        },
        {
            "type": "result",
            "is_error": False,
            "result": "Synthesized from context",
        },
    ]
    cmds = []

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        return _proc(stdout=json.dumps(replies.pop(0)))

    monkeypatch.setattr(llm_bridge.subprocess, "run", fake_run)
    rq = make_recursive_query(helpers={}, max_iterations=3)
    assert rq("summarize", context="alpha=1") == "Synthesized from context"
    assert len(cmds) == 2
    assert "--json-schema" not in cmds[1]


def test_recurse_penultimate_empty_requests_synthesizes_without_conflicting_instruction(monkeypatch):
    monkeypatch.setenv("RLM_SUB_PROVIDER", "claude-cli")
    replies = [
        {
            "type": "result",
            "is_error": False,
            "result": "{}",
            "structured_output": {"action": "request", "requests": []},
        },
        {
            "type": "result",
            "is_error": False,
            "result": "Final synthesized answer",
        },
    ]
    prompts = []

    def fake_run(cmd, **kw):
        prompts.append(kw["input"])
        return _proc(stdout=json.dumps(replies.pop(0)))

    monkeypatch.setattr(llm_bridge.subprocess, "run", fake_run)
    rq = make_recursive_query(helpers={}, max_iterations=2)
    assert rq("question") == "Final synthesized answer"
    assert len(prompts) == 2
    assert "Either provide 1-4" not in prompts[1]


def test_default_cli_concurrency_is_8_and_batched_honors_env(monkeypatch):
    assert llm_bridge.DEFAULT_CLI_MAX_CONCURRENCY == 8
    monkeypatch.setenv("RLM_CLI_MAX_CONCURRENCY", "10")
    recorded = {}
    real_pool = llm_bridge.ThreadPoolExecutor

    class SpyPool(real_pool):
        def __init__(self, max_workers=None, *args, **kwargs):
            recorded["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, *args, **kwargs)

    monkeypatch.setattr(llm_bridge, "ThreadPoolExecutor", SpyPool)
    batched = make_llm_query_batched(lambda p, c="": p)
    assert batched(["a", "b"]) == ["a", "b"]
    assert recorded["max_workers"] == 10
