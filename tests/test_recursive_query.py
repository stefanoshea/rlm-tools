"""Tests for the genuinely-recursive sub-agent (rlm_recurse).

The sub-agent runs its own tool-use loop server-side: it greps/reads/globs the
corpus and can spawn depth-limited child agents, returning only a final answer.
Raw file content stays in the sub-agent loop and never reaches the caller.
"""

from unittest.mock import MagicMock

from rlm_tools.llm_bridge import make_recursive_query


class _Block:
    def __init__(self, type, text=None, name=None, input=None, id="tool_1"):
        self.type = type
        self.text = text
        self.name = name
        self.input = input or {}
        self.id = id


class _Resp:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


def _helpers(tool_calls):
    grep_summary = MagicMock(return_value="3 matches in 2 files")
    read_file = MagicMock(return_value="def login(): ...")
    tool_calls["grep_summary"] = grep_summary
    tool_calls["read_file"] = read_file
    return {
        "grep_summary": grep_summary,
        "read_file": read_file,
        "glob_files": MagicMock(return_value=["a.py"]),
        "tree": MagicMock(return_value="."),
    }


def test_recurse_runs_tool_loop_then_returns_final_text():
    tool_calls = {}
    helpers = _helpers(tool_calls)
    client = MagicMock()
    client.messages.create.side_effect = [
        _Resp(
            [_Block("tool_use", name="grep_summary", input={"pattern": "login"})],
            stop_reason="tool_use",
        ),
        _Resp([_Block("text", text="Auth is handled in login().")], stop_reason="end_turn"),
    ]

    recurse = make_recursive_query(client=client, helpers=helpers, model="m", max_depth=2)
    answer = recurse("Where is auth handled?")

    assert answer == "Auth is handled in login()."
    # The sub-agent actually invoked the server-side helper.
    tool_calls["grep_summary"].assert_called_once_with(pattern="login")
    # Two model turns: one to call the tool, one to answer.
    assert client.messages.create.call_count == 2


def test_recurse_offers_recurse_tool_until_depth_exhausted():
    helpers = _helpers({})
    client = MagicMock()
    client.messages.create.return_value = _Resp(
        [_Block("text", text="done")], stop_reason="end_turn"
    )

    recurse = make_recursive_query(client=client, helpers=helpers, model="m", max_depth=1)
    recurse("q")
    tool_names = {
        t["name"] for t in client.messages.create.call_args.kwargs["tools"]
    }
    assert "recurse" in tool_names  # depth 1 -> children allowed

    client.messages.create.reset_mock()
    recurse_leaf = make_recursive_query(
        client=client, helpers=helpers, model="m", max_depth=0
    )
    recurse_leaf("q")
    leaf_tools = {
        t["name"] for t in client.messages.create.call_args.kwargs["tools"]
    }
    assert "recurse" not in leaf_tools  # depth 0 -> no further recursion


def test_recurse_stops_at_iteration_cap():
    helpers = _helpers({})
    client = MagicMock()
    # Always asks for another tool call; must be bounded by max_iterations.
    client.messages.create.return_value = _Resp(
        [_Block("tool_use", name="grep_summary", input={"pattern": "x"})],
        stop_reason="tool_use",
    )

    recurse = make_recursive_query(
        client=client, helpers=helpers, model="m", max_depth=1, max_iterations=3
    )
    recurse("q")
    assert client.messages.create.call_count == 3


def test_recurse_rejects_an_unadvertised_child_call_at_the_depth_limit():
    client = MagicMock()
    client.messages.create.side_effect = [
        _Resp(
            [_Block("tool_use", name="recurse", input={"prompt": "child"})],
            stop_reason="tool_use",
        ),
        _Resp([_Block("text", text="parent answer")], stop_reason="end_turn"),
    ]
    recurse = make_recursive_query(client=client, max_depth=0)

    assert recurse("parent question") == "parent answer"
    messages = client.messages.create.call_args.kwargs["messages"]
    assert "unknown tool" in messages[-1]["content"][0]["content"]
    assert client.messages.create.call_count == 2


def test_recurse_counts_each_model_call():
    helpers = _helpers({})
    client = MagicMock()
    client.messages.create.side_effect = [
        _Resp(
            [_Block("tool_use", name="grep_summary", input={"pattern": "x"})],
            stop_reason="tool_use",
        ),
        _Resp([_Block("text", text="ok")], stop_reason="end_turn"),
    ]
    calls = []
    recurse = make_recursive_query(
        client=client,
        helpers=helpers,
        model="m",
        max_depth=1,
        on_model_call=lambda: calls.append(1),
    )
    recurse("q")
    assert sum(calls) == 2


def test_recurse_skips_final_iteration_tools_and_gate():
    tool_calls = {}
    helpers = _helpers(tool_calls)
    gate_calls = []
    client = MagicMock()
    client.messages.create.side_effect = [
        _Resp(
            [_Block("text", text="partial"), _Block("tool_use", name="grep_summary", input={"pattern": "a"})],
            stop_reason="tool_use",
        ),
        _Resp(
            [_Block("text", text="best answer"), _Block("tool_use", name="read_file", input={"path": "a.py"})],
            stop_reason="tool_use",
        ),
    ]

    recurse = make_recursive_query(
        client=client,
        helpers=helpers,
        model="m",
        max_depth=1,
        max_iterations=2,
        gate=lambda q, text: gate_calls.append(text) or text,
    )
    answer = recurse("q")

    assert answer == "best answer"
    assert tool_calls["grep_summary"].call_count == 1
    assert tool_calls["read_file"].call_count == 0
    assert len(gate_calls) == 1


def test_recurse_empty_child_prompt_returns_tool_error():
    client = MagicMock()
    client.messages.create.side_effect = [
        _Resp([_Block("tool_use", name="recurse", input={"prompt": "   "})], stop_reason="tool_use"),
        _Resp([_Block("text", text="recovered")], stop_reason="end_turn"),
    ]

    recurse = make_recursive_query(client=client, helpers=_helpers({}), model="m", max_depth=1)
    assert recurse("parent") == "recovered"
    messages = client.messages.create.call_args.kwargs["messages"]
    assert "non-empty string" in messages[-1]["content"][0]["content"]
