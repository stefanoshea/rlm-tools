"""The scorer must batch candidates into few requests, keep scores aligned with
their items, and charge every request against the session budget. All tests
inject a fake System One client so the suite stays offline."""

from dataclasses import dataclass

from rlm_tools.semantic import make_evidence_gate, make_scorer


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


def test_boolean_client_scores_fail_open_before_conversion(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(
        client, "system_one", lambda *_: FakeResponse({"c0": FakeNoul(True), "c1": FakeNoul(False)})
    )
    records = []
    gate = make_evidence_gate(
        make_scorer(client=client), chunk_lines=1, min_chars=0, on_gate=records.append
    )
    text = "alpha\nbeta"
    assert gate("what is relevant?", text) == text
    assert records[0]["outcome"] == "failed_open"
