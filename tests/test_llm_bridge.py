import json
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from rlm_tools import llm_bridge
from rlm_tools.llm_bridge import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    _OpenAICompatClient,
    _normalize_openai_endpoint,
    get_client,
    has_configured_provider,
    make_llm_query,
    make_llm_query_batched,
    resolve_provider,
)

_REAL_GET_CLIENT = get_client


def test_llm_query_calls_anthropic():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="YES - handles errors properly")]
    )

    query_fn = make_llm_query(client=mock_client, model="claude-haiku-4-5-20251001")
    result = query_fn("Does this handle errors?", context="some code here")

    assert "YES" in result
    mock_client.messages.create.assert_called_once()


def test_llm_query_without_context():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="42")]
    )

    query_fn = make_llm_query(client=mock_client, model="claude-haiku-4-5-20251001")
    result = query_fn("What is the answer?")

    assert "42" in result
    call_args = mock_client.messages.create.call_args
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
    assert len(messages) == 1
    assert "Context:" not in messages[0]["content"]


def test_llm_query_batched():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="answer")]
    )

    query_fn = make_llm_query(client=mock_client, model="claude-haiku-4-5-20251001")
    batch_fn = make_llm_query_batched(query_fn)

    results = batch_fn(["q1", "q2", "q3"])
    assert len(results) == 3
    assert all(r == "answer" for r in results)


def test_server_import_is_lazy_for_provider_sdks():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rlm_tools.server; "
            "assert 'anthropic' not in sys.modules; "
            "assert 'google.genai' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_resolve_and_has_configured_provider(monkeypatch):
    assert resolve_provider() == "anthropic"
    assert not has_configured_provider()

    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert resolve_provider() == "gemini"
    assert has_configured_provider()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    assert resolve_provider() == "anthropic"
    assert resolve_provider(client=SimpleNamespace(interactions=object())) == "gemini"
    assert resolve_provider(client=MagicMock()) == "anthropic"

    monkeypatch.setenv("RLM_SUB_PROVIDER", "gemini")
    assert resolve_provider() == "gemini"
    assert has_configured_provider()

    monkeypatch.delenv("GEMINI_API_KEY")
    assert not has_configured_provider()

    with pytest.raises(ValueError):
        resolve_provider("unknown-provider")


def test_llm_query_calls_gemini_and_retries(monkeypatch):
    calls = []

    class FakeInteractions:
        def __init__(self):
            self.n = 0

        def create(self, **kwargs):
            calls.append(kwargs)
            self.n += 1
            if self.n == 1:
                raise ConnectionError("transient")
            return SimpleNamespace(
                status="completed",
                output_text=None,
                steps=[
                    SimpleNamespace(
                        type="model_output",
                        content=[SimpleNamespace(type="text", text="YES - safe")],
                    )
                ],
            )

    monkeypatch.setattr("rlm_tools.llm_bridge.time.sleep", lambda _: None)
    fake_client = SimpleNamespace(interactions=FakeInteractions())
    retries = []
    query_fn = make_llm_query(
        client=fake_client,
        system_prompt="Return YES or NO.",
        timeout_seconds=15.0,
        max_retries=2,
        on_retry=lambda: retries.append(1),
    )
    result = query_fn("Is it safe?", context="x = 1")
    assert result == "YES - safe"
    assert len(retries) == 1
    assert len(calls) == 2
    assert calls[1]["model"] == DEFAULT_GEMINI_MODEL
    assert calls[1]["system_instruction"] == "Return YES or NO."
    assert calls[1]["generation_config"] == {"max_output_tokens": 1024}
    assert calls[1]["timeout"] == 15.0
    assert "Context:\nx = 1\n\nQuestion: Is it safe?" == calls[1]["input"]

    failed_client = SimpleNamespace(
        interactions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(status="failed", errors=["bad"], output_text=None, steps=[])
        )
    )
    with pytest.raises(RuntimeError, match="Gemini interaction failed"):
        make_llm_query(client=failed_client, max_retries=1)("q")

    expired = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        make_llm_query(client=fake_client, deadline=lambda: expired)("q")


def test_resolve_and_has_configured_provider_openai(monkeypatch):
    # Priority order: ANTHROPIC > GEMINI > OPENAI in auto mode.
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")
    assert resolve_provider() == "openai"
    assert has_configured_provider()

    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert resolve_provider() == "gemini"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    assert resolve_provider() == "anthropic"

    # Explicit alias & client-shape routing.
    assert resolve_provider("openai-compatible") == "openai"
    assert resolve_provider("openai_compatible") == "openai"
    monkeypatch.setenv("RLM_SUB_PROVIDER", "openai-compatible")
    assert resolve_provider() == "openai"
    assert has_configured_provider()

    monkeypatch.delenv("OPENAI_API_KEY")
    assert not has_configured_provider()

    # Keyless local OpenAI-compatible server (e.g., Ollama / vLLM).
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert has_configured_provider()
    monkeypatch.delenv("RLM_SUB_PROVIDER")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert resolve_provider() == "openai"

    assert resolve_provider(client=SimpleNamespace(chat=SimpleNamespace(completions=object()))) == "openai"
    assert resolve_provider(client=MagicMock()) == "anthropic"


def test_openai_endpoint_normalization_and_validation():
    assert _normalize_openai_endpoint("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"
    assert (
        _normalize_openai_endpoint("https://generativelanguage.googleapis.com/v1beta/openai/")
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert (
        _normalize_openai_endpoint("http://localhost:11434/v1/chat/completions/")
        == "http://localhost:11434/v1/chat/completions"
    )
    with pytest.raises(ValueError, match="http or https"):
        _normalize_openai_endpoint("file:///etc/passwd")
    with pytest.raises(ValueError, match="http or https"):
        _normalize_openai_endpoint("not-a-url")


def test_llm_query_openai_compat_transport_headers_retries_and_tokens(monkeypatch):
    monkeypatch.setattr("rlm_tools.llm_bridge.time.sleep", lambda _: None)
    seen_requests: list[tuple[str, dict[str, str], dict]] = []
    attempt = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        body = json.loads(request.content.decode("utf-8"))
        seen_requests.append((str(request.url), dict(request.headers), body))
        if attempt == 1:
            return httpx.Response(503, text="temporarily overloaded")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "part1-"}, {"text": "part2"}],
                        },
                    }
                ]
            },
        )

    client = _OpenAICompatClient(
        api_key="sk-secret-abc",
        base_url="http://mock.local/v1/",
        transport=httpx.MockTransport(handler),
    )
    try:
        retries = []
        query_fn = make_llm_query(
            client=client,
            system_prompt="Be brief.",
            timeout_seconds=14.0,
            max_retries=2,
            on_retry=lambda: retries.append(1),
        )
        out = query_fn("What does it return?", context="x = 1")
        assert out == "part1-part2"
        assert len(retries) == 1
        assert len(seen_requests) == 2
        url, headers, body = seen_requests[1]
        assert url == "http://mock.local/v1/chat/completions"
        assert headers["authorization"] == "Bearer sk-secret-abc"
        assert body["model"] == DEFAULT_OPENAI_MODEL
        assert body["max_tokens"] == 1024
        assert "max_completion_tokens" not in body
        assert body["messages"][0] == {"role": "system", "content": "Be brief."}
        assert "Context:\nx = 1\n\nQuestion: What does it return?" == body["messages"][1]["content"]

        # Reasoning models use max_completion_tokens; keyless endpoints omit Authorization header.
        keyless_bodies = []
        keyless_headers = []

        def keyless_handler(request: httpx.Request) -> httpx.Response:
            keyless_headers.append(dict(request.headers))
            keyless_bodies.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

        keyless = _OpenAICompatClient(
            api_key=None,
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(keyless_handler),
        )
        try:
            assert make_llm_query(client=keyless, model="openai/o3-mini")("ping") == "ok"
            assert "authorization" not in {k.lower() for k in keyless_headers[0]}
            assert keyless_bodies[0]["max_completion_tokens"] == 1024
            assert "max_tokens" not in keyless_bodies[0]
        finally:
            keyless.close()

        # HTTP errors surface clearly without exposing the bearer token.
        err_client = _OpenAICompatClient(
            api_key="sk-never-leak-me",
            base_url="http://mock.local/v1",
            transport=httpx.MockTransport(lambda req: httpx.Response(401, text="Unauthorized")),
        )
        try:
            with pytest.raises(RuntimeError, match="status=401") as exc_info:
                make_llm_query(client=err_client, max_retries=1)("hello")
            assert "sk-never-leak-me" not in str(exc_info.value)
        finally:
            err_client.close()

        expired = time.monotonic() - 1
        with pytest.raises(TimeoutError):
            make_llm_query(client=client, deadline=lambda: expired)("hello")
    finally:
        client.close()


def test_get_client_anthropic_requires_key_and_constructs_client(monkeypatch):
    import anthropic

    monkeypatch.setattr(llm_bridge, "get_client", _REAL_GET_CLIENT)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RLM_SUB_PROVIDER", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm_bridge.get_client("anthropic")

    constructed = {}

    def fake_anthropic(*, api_key, max_retries):
        constructed["api_key"] = api_key
        constructed["max_retries"] = max_retries
        return SimpleNamespace(messages=MagicMock())

    monkeypatch.setattr(anthropic, "Anthropic", fake_anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = llm_bridge.get_client("anthropic")
    assert hasattr(client, "messages")
    assert constructed == {"api_key": "sk-ant-test", "max_retries": 0}
