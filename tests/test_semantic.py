"""The scorer must batch candidates into few requests, keep scores aligned with
their items, and charge every request against the session budget. All tests
inject a fake System One client so the suite stays offline."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rlm_tools.semantic import (
    DEFAULT_GEMINI_MODEL,
    has_configured_scorer,
    make_evidence_gate,
    make_scorer,
    resolve_provider,
)


@dataclass
class FakeNoul:
    noul: float


class FakeResponse:
    def __init__(self, nouls):
        self.nouls = nouls


class FakeClient:
    """Scores a candidate by how often 'match' appears in its text."""

    def __init__(self):
        self.calls: list[tuple] = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return FakeResponse({
            f"c{i}": FakeNoul(self._score(text))
            for i, text in state["candidates"].items()
        })

    @staticmethod
    def _score(text: str) -> float:
        return min(1.0, 0.1 + 0.4 * str(text).count("match"))


def test_scores_come_back_in_input_order():
    score_all = make_scorer(client=FakeClient())

    scores = score_all(["nothing", "match match", "one match"], "is it relevant")

    assert scores == [0.1, 0.9, 0.5]


def test_the_question_and_candidates_are_sent_separately():
    client = FakeClient()

    make_scorer(client=client)(["alpha"], "criterion text")
    state, questions = client.calls[0]

    assert state["criterion"] == "criterion text"
    assert state["candidates"] == {"0": "alpha"}
    assert list(questions) == ["c0"]


def test_candidates_are_batched_into_few_requests():
    client = FakeClient()
    score_all = make_scorer(client=client, chunk_size=10)

    scores = score_all([f"item {i}" for i in range(25)], "relevant")

    assert len(scores) == 25
    assert len(client.calls) == 3


def test_every_request_is_reserved_against_the_budget():
    reserved: list[int] = []
    score_all = make_scorer(client=FakeClient(), chunk_size=10, reserve=reserved.append)

    score_all([f"item {i}" for i in range(25)], "relevant")

    assert reserved == [1, 1, 1]


def test_long_candidates_are_truncated_before_sending():
    client = FakeClient()

    make_scorer(client=client, max_item_chars=10)(["x" * 500], "relevant")

    assert client.calls[0][0]["candidates"]["0"] == "x" * 10


def test_empty_input_costs_nothing():
    client = FakeClient()

    assert make_scorer(client=client)([], "relevant") == []
    assert client.calls == []


def test_resolve_and_has_configured_scorer(monkeypatch):
    assert resolve_provider() == "typesafe"
    assert not has_configured_scorer()

    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert resolve_provider() == "gemini"
    assert has_configured_scorer()

    monkeypatch.setenv("TYPESAFE_API_KEY", "t-key")
    assert resolve_provider() == "typesafe"
    assert resolve_provider("gemini") == "gemini"

    monkeypatch.setenv("RLM_GATE_PROVIDER", "gemini")
    assert resolve_provider() == "gemini"

    monkeypatch.delenv("GEMINI_API_KEY")
    assert not has_configured_scorer()

    with pytest.raises(ValueError):
        resolve_provider("unknown")


def test_gemini_decisions_scorer_and_evidence_gate():
    calls = []

    class DynamicDecisions:
        def create(self, **kwargs):
            calls.append(kwargs)
            import json as _json

            parsed = _json.loads(kwargs["input"])
            return SimpleNamespace(
                answers={
                    f"c{idx}": SimpleNamespace(noul=0.9 if "keep" in text else 0.1)
                    for idx, text in parsed["candidates"].items()
                }
            )

    fake_client = SimpleNamespace(decisions=DynamicDecisions())
    reserved = []
    scorer = make_scorer(client=fake_client, chunk_size=2, reserve=reserved.append, deadline=lambda: 9e12)
    scores = scorer(["nothing", "keep one", "keep two"], "what is relevant")
    assert scores == [0.1, 0.9, 0.9]
    assert reserved == [1, 1]
    assert all(c["model"] == DEFAULT_GEMINI_MODEL for c in calls)
    assert all("timeout" in c for c in calls)

    gate = make_evidence_gate(scorer, min_chars=10)
    text = "\n".join(["\n".join(["keep line"] * 40), "\n".join(["noise line"] * 40)])
    gated = gate("what does keep do?", text)
    assert "keep line" in gated
    assert "noise line" not in gated
