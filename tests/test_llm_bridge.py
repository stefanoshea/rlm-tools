from unittest.mock import patch, MagicMock
from rlm_tools.llm_bridge import make_llm_query, make_llm_query_batched


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
