"""Tests for the genuinely-recursive sub-agent (rlm_recurse).

The sub-agent runs its own tool-use loop server-side: it greps/reads/globs the
corpus and can spawn depth-limited child agents, returning only a final answer.
Raw file content stays in the sub-agent loop and never reaches the caller.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx

from rlm_tools.llm_bridge import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    _OpenAICompatClient,
    make_recursive_query,
)


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


def test_gemini_recurse_loop_depth_gate_and_budget():
    tool_calls = {}
    helpers = _helpers(tool_calls)
    requests = []

    class FakeInteractions:
        def create(self, **kwargs):
            requests.append(kwargs)
            idx = len(requests)
            if idx == 1:
                return SimpleNamespace(
                    id="ix-root-1",
                    status="requires_action",
                    output_text=None,
                    steps=[
                        SimpleNamespace(
                            type="function_call",
                            id="fc-1",
                            name="read_file",
                            arguments={"path": "a.py"},
                        ),
                        SimpleNamespace(
                            type="function_call",
                            id="fc-2",
                            name="recurse",
                            arguments={"prompt": "child question"},
                        ),
                    ],
                )
            if idx == 2:
                # Child explorer (depth=0) has no recurse tool offered.
                tool_names = {t["name"] for t in kwargs["tools"]}
                assert "recurse" not in tool_names
                return SimpleNamespace(
                    id="ix-child-1",
                    status="requires_action",
                    output_text=None,
                    steps=[
                        SimpleNamespace(
                            type="function_call",
                            id="fc-unadv",
                            name="recurse",
                            arguments={"prompt": "too deep"},
                        ),
                    ],
                )
            if idx == 3:
                assert "unknown tool 'recurse'" in kwargs["input"][0]["result"][0]["text"]
                return SimpleNamespace(
                    id="ix-child-2",
                    status="completed",
                    output_text="child finding",
                    steps=[],
                )
            return SimpleNamespace(
                id="ix-root-2",
                status="completed",
                output_text="final gemini synthesis",
                steps=[],
            )

    client = SimpleNamespace(interactions=FakeInteractions())
    gated = []

    def fake_gate(question, text):
        gated.append((question, text))
        return f"GATED:{text}"

    budget_calls = []
    recurse = make_recursive_query(
        client=client,
        helpers=helpers,
        max_depth=1,
        max_iterations=4,
        on_model_call=lambda: budget_calls.append(1),
        gate=fake_gate,
    )
    answer = recurse("root question")
    assert answer == "final gemini synthesis"
    assert sum(budget_calls) == 4
    assert requests[0]["model"] == DEFAULT_GEMINI_MODEL
    assert requests[0]["generation_config"] == {"max_output_tokens": 2048}
    assert "recurse" in {t["name"] for t in requests[0]["tools"]}
    # Gate ran only on read_file output, never on child recurse result.
    assert gated == [("root question", "def login(): ...")]
    assert requests[3]["previous_interaction_id"] == "ix-root-1"
    assert requests[3]["input"][0]["result"][0]["text"] == "GATED:def login(): ..."
    assert requests[3]["input"][1]["result"][0]["text"] == "child finding"

    # Respects max_iterations cap when tool calls keep arriving.
    endless = SimpleNamespace(
        interactions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                id="ix-loop",
                status="requires_action",
                output_text="partial",
                steps=[SimpleNamespace(type="function_call", id="fc", name="grep_summary", arguments={"pattern": "x"})],
            )
        )
    )
    capped_calls = []
    out = make_recursive_query(
        client=endless,
        helpers=helpers,
        max_iterations=2,
        on_model_call=lambda: capped_calls.append(1),
    )("loop?")
    assert out == "partial"
    assert sum(capped_calls) == 2


def test_openai_recurse_loop_depth_gate_malformed_args_and_extra_content():
    tool_calls = {}
    helpers = _helpers(tool_calls)
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        requests.append(payload)
        idx = len(requests)
        if idx == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "type": "function",
                                    "extra_content": {"google": {"thought_signature": "sig-1"}},
                                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                                },
                                {
                                    "id": "call_badjson",
                                    "type": "function",
                                    "function": {"name": "read_file", "arguments": "{not-json"},
                                },
                                {
                                    "id": "call_child",
                                    "type": "function",
                                    "function": {"name": "recurse", "arguments": '{"prompt": "child q"}'},
                                },
                            ],
                        }
                    }]
                },
            )
        if idx == 2:
            # Child explorer (depth=0) omits recurse tool and handles attempted call safely.
            tool_names = {t["function"]["name"] for t in payload["tools"]}
            assert "recurse" not in tool_names
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "call_too_deep",
                                "type": "function",
                                "function": {"name": "recurse", "arguments": '{"prompt": "nope"}'},
                            }],
                        }
                    }]
                },
            )
        if idx == 3:
            assert "unknown tool 'recurse'" in payload["messages"][-1]["content"]
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "message": {"role": "assistant", "content": "child answer"}
                    }]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {"role": "assistant", "content": "final openai synthesis"}
                }]
            },
        )

    client = _OpenAICompatClient(
        api_key="sk-test",
        base_url="http://mock.openai.local/v1",
        transport=httpx.MockTransport(handler),
    )
    gated = []

    def fake_gate(question, text):
        gated.append((question, text))
        return f"GATED:{text}"

    try:
        budget_calls = []
        recurse = make_recursive_query(
            client=client,
            helpers=helpers,
            max_depth=1,
            max_iterations=5,
            on_model_call=lambda: budget_calls.append(1),
            gate=fake_gate,
        )
        answer = recurse("root openai question")
        assert answer == "final openai synthesis"
        assert sum(budget_calls) == 4
        assert requests[0]["model"] == DEFAULT_OPENAI_MODEL
        assert requests[0]["max_tokens"] == 2048
        assert "recurse" in {t["function"]["name"] for t in requests[0]["tools"]}
        assert gated == [("root openai question", "def login(): ...")]

        final_msgs = requests[3]["messages"]
        echoed_assistant = final_msgs[1]
        assert echoed_assistant["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "sig-1"}
        }
        assert final_msgs[2] == {
            "role": "tool",
            "tool_call_id": "call_read",
            "content": "GATED:def login(): ...",
        }
        assert final_msgs[3]["role"] == "tool"
        assert final_msgs[3]["tool_call_id"] == "call_badjson"
        assert "invalid tool arguments" in final_msgs[3]["content"]
        assert final_msgs[4] == {
            "role": "tool",
            "tool_call_id": "call_child",
            "content": "child answer",
        }
    finally:
        client.close()

    # Respects max_iterations cap and non-object JSON arguments recovery.
    loop_reqs = []

    def loop_handler(request: httpx.Request) -> httpx.Response:
        loop_reqs.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "partial openai",
                        "tool_calls": [{
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "grep_summary", "arguments": '["not", "an", "object"]'},
                        }],
                    }
                }]
            },
        )

    loop_client = _OpenAICompatClient(
        api_key="sk-test",
        base_url="http://mock.openai.local/v1",
        transport=httpx.MockTransport(loop_handler),
    )
    try:
        capped = []
        out = make_recursive_query(
            client=loop_client,
            helpers=helpers,
            max_iterations=2,
            on_model_call=lambda: capped.append(1),
        )("loop?")
        assert out == "partial openai"
        assert sum(capped) == 2
        assert "expected JSON object" in loop_reqs[1]["messages"][-1]["content"]
    finally:
        loop_client.close()
