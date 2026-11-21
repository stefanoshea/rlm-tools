"""Timeout, retry, and error-surfacing behavior for the sub-LLM bridge."""

from unittest.mock import MagicMock
import time

import pytest

from rlm_tools.llm_bridge import make_llm_query, make_llm_query_batched, make_recursive_query


@pytest.mark.parametrize("factory", [make_llm_query, make_recursive_query])
def test_provider_requests_stop_at_execution_deadline(factory):
    client = MagicMock()
    end = time.monotonic() + 3

    def expired_on_response(**kwargs):
        nonlocal end
        assert 0 < kwargs["timeout"] <= 3
        end = time.monotonic() - 1
        raise ConnectionError("temporary failure")

    client.messages.create.side_effect = expired_on_response
    call = factory(client=client, deadline=lambda: end)
    with pytest.raises(TimeoutError):
        call("count the files")
    assert client.messages.create.call_count == 1


def test_scoring_deadline_precedes_provider_dispatch():
    from rlm_tools.semantic import make_scorer

    client = MagicMock()
    scorer = make_scorer(client=client, deadline=lambda: time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        scorer(["file excerpt"], "relevant?")
    client.system_one.assert_not_called()


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
