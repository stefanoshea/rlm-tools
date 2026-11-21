"""Timeout, retry, and error-surfacing behavior for the sub-LLM bridge."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from anthropic import APIStatusError

from rlm_tools.llm_bridge import make_llm_query, make_llm_query_batched


def test_llm_query_passes_request_timeout():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="ok")]
    )

    query_fn = make_llm_query(
        client=mock_client, model="m", timeout_seconds=12.5
    )
    query_fn("hi")

    call = mock_client.messages.create.call_args
    assert call.kwargs.get("timeout") == 12.5


def test_llm_query_retries_transient_failure():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        ConnectionError("transient"),
        MagicMock(content=[MagicMock(text="recovered")]),
    ]

    query_fn = make_llm_query(client=mock_client, model="m", max_retries=2)
    result = query_fn("hi")

    assert result == "recovered"
    assert mock_client.messages.create.call_count == 2


def test_llm_query_raises_after_exhausting_retries():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = ConnectionError("down")

    query_fn = make_llm_query(client=mock_client, model="m", max_retries=2)
    try:
        query_fn("hi")
        assert False, "expected the error to propagate after retries"
    except ConnectionError:
        pass
    assert mock_client.messages.create.call_count == 2


def test_batched_marks_failed_prompts_distinctly():
    mock_client = MagicMock()

    def create(*args, **kwargs):
        content = kwargs["messages"][0]["content"]
        if "boom" in content:
            raise RuntimeError("kaboom")
        return MagicMock(content=[MagicMock(text="fine")])

    mock_client.messages.create.side_effect = create
    query_fn = make_llm_query(client=mock_client, model="m", max_retries=1)
    batch_fn = make_llm_query_batched(query_fn)

    results = batch_fn(["ok", "boom", "ok"])
    assert results[0] == "fine"
    assert results[2] == "fine"
    assert results[1].startswith("[ERROR]")
    assert "kaboom" in results[1]


@pytest.mark.parametrize("status_code", [400, 401])
def test_non_retryable_status_error_fails_immediately(status_code):
    mock_client = MagicMock()
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    mock_client.messages.create.side_effect = APIStatusError(
        "client error", response=response, body=None
    )

    query_fn = make_llm_query(client=mock_client, model="m", max_retries=3)
    with patch("rlm_tools.llm_bridge.time.sleep") as mock_sleep:
        with pytest.raises(APIStatusError):
            query_fn("hi")

    assert mock_client.messages.create.call_count == 1
    mock_sleep.assert_not_called()


def test_batched_rejects_single_string_prompts():
    batch_fn = make_llm_query_batched(lambda p, c="": p)
    with pytest.raises(TypeError, match="sequence of strings"):
        batch_fn("not a list")


def test_flat_query_returns_empty_text_for_only_non_text_blocks():
    client = MagicMock()
    client.messages.create.return_value.content = [MagicMock(type="tool_use")]
    assert make_llm_query(client=client)("hello") == ""
