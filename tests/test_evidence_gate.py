"""The evidence gate must drop irrelevant chunks of a tool result, keep the
surviving ones in order, and never break exploration when scoring fails."""

import pytest

from rlm_tools.llm_bridge import make_recursive_query
from rlm_tools.semantic import ELISION, make_evidence_gate


def _text(blocks: list[str], lines_per_block: int = 40) -> str:
    chunks = ["\n".join([block] * lines_per_block) for block in blocks]
    return "\n".join(chunks)


def test_small_results_pass_through_unjudged():
    calls: list = []

    def score_all(items, question):
        calls.append(items)
        return [1.0] * len(items)

    gate = make_evidence_gate(score_all, min_chars=2_000)
    text = "short result"

    assert gate("question", text) is text
    assert calls == []


def test_irrelevant_chunks_are_dropped_and_order_is_kept():
    def score_all(items, question):
        return [0.9 if "keep" in item else 0.1 for item in items]

    gate = make_evidence_gate(score_all, threshold=0.45, min_chars=10)
    text = _text(["keep one", "drop this", "keep two"])

    result = gate("question", text)

    assert "drop this" not in result
    assert result.index("keep one") < result.index("keep two")
    assert ELISION.strip() in result


def test_everything_relevant_returns_the_original_text():
    def score_all(items, question):
        return [0.9] * len(items)

    gate = make_evidence_gate(score_all, min_chars=10)
    text = _text(["alpha", "beta"])

    assert gate("question", text) is text


def test_nothing_relevant_keeps_everything():
    """No chunk clearing the bar means no signal, not an empty answer."""

    def score_all(items, question):
        return [0.01, 0.02, 0.03]

    gate = make_evidence_gate(score_all, min_chars=10)
    text = _text(["alpha", "beta", "gamma"])

    assert gate("question", text) is text


def test_a_scoring_failure_falls_back_to_the_raw_result():
    def score_all(items, question):
        raise RuntimeError("LLM call limit exceeded")

    gate = make_evidence_gate(score_all, min_chars=10)
    text = _text(["alpha", "beta"])

    assert gate("question", text) is text


@pytest.mark.parametrize("scores", [[], [0.9], [0.9, float("nan")], [0.9, 0.1, 0.9]])
def test_invalid_scores_fall_back_to_the_raw_result(scores):
    gate = make_evidence_gate(lambda items, question: scores, min_chars=10)
    text = _text(["alpha", "beta"])

    assert gate("question", text) is text


def test_leading_elision_is_marked():
    gate = make_evidence_gate(lambda items, question: [0.1, 0.9], min_chars=10)

    assert gate("question", _text(["alpha", "beta"])).startswith(ELISION)


class FakeBlock:
    def __init__(self, kind, **kwargs):
        self.type = kind
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class FakeAnthropic:
    """Answers one tool_use turn, then a final text turn."""

    def __init__(self):
        self.messages = self
        self.sent: list = []

    def create(self, **kwargs):
        self.sent.append(kwargs["messages"])
        if len(self.sent) == 1:
            return FakeResponse(
                [FakeBlock("tool_use", name="read_file", id="t1", input={"path": "a.py"})],
                "tool_use",
            )
        return FakeResponse([FakeBlock("text", text="the answer")], "end_turn")


def test_recursive_query_gates_helper_results_before_they_reach_the_model():
    client = FakeAnthropic()
    helpers = {"read_file": lambda path: _text(["keep me", "drop me"])}

    def score_all(items, question):
        return [0.9 if "keep" in item else 0.1 for item in items]

    query = make_recursive_query(
        client=client,
        helpers=helpers,
        gate=make_evidence_gate(score_all, min_chars=10),
    )

    assert query("what does a.py do?") == "the answer"

    tool_result = client.sent[1][-1]["content"][0]["content"]
    assert "keep me" in tool_result
    assert "drop me" not in tool_result


def test_recurse_results_are_never_gated():
    client = FakeAnthropic()
    gated: list[str] = []

    def gate(question, text):
        gated.append(text)
        return "gated"

    query = make_recursive_query(client=client, helpers={"read_file": lambda path: "x"}, gate=gate)
    query("question")

    assert gated == ["x"]


def test_gate_records_what_it_dropped():
    records: list[dict] = []

    def score_all(items, question):
        return [0.9 if "keep" in item else 0.1 for item in items]

    gate = make_evidence_gate(score_all, min_chars=10, on_gate=records.append)
    gate("question", _text(["keep one", "drop this", "keep two"]))

    record = records[0]
    assert record["outcome"] == "gated"
    assert record["chunks_total"] == 3
    assert record["chunks_kept"] == 2
    assert record["chars_out"] < record["chars_in"]
    assert record["max_dropped_score"] == pytest.approx(0.1)


def test_gate_records_a_failure_as_failed_open():
    records: list[dict] = []

    def score_all(items, question):
        raise RuntimeError("LLM call limit exceeded")

    gate = make_evidence_gate(score_all, min_chars=10, on_gate=records.append)
    gate("question", _text(["alpha", "beta"]))

    assert records[0]["outcome"] == "failed_open"
    assert records[0]["chars_in"] == records[0]["chars_out"]


def test_exhaustive_questions_are_never_gated():
    calls: list = []

    def score_all(items, question):
        calls.append(items)
        return [0.9 if "keep" in item else 0.1 for item in items]

    gate = make_evidence_gate(score_all, min_chars=10)
    text = _text(["keep one", "drop this"])

    for question in [
        "list every environment variable the server reads",
        "what are ALL the sandbox restrictions",
        "describe each guard in turn",
    ]:
        assert gate(question, text) is text

    assert calls == []


def test_a_flat_score_distribution_keeps_everything():
    def score_all(items, question):
        return [0.52, 0.44, 0.48]

    gate = make_evidence_gate(score_all, threshold=0.45, min_separation=0.2, min_chars=10)
    text = _text(["alpha", "beta", "gamma"])

    assert gate(question="what does this do?", text=text) is text


def test_a_clear_distribution_still_gates():
    def score_all(items, question):
        return [0.9, 0.1, 0.05]

    gate = make_evidence_gate(score_all, threshold=0.45, min_separation=0.2, min_chars=10)
    text = _text(["alpha", "beta", "gamma"])

    result = gate("what does this do?", text)

    assert "beta" not in result
    assert "alpha" in result


def test_low_separation_is_recorded():
    records: list[dict] = []

    def score_all(items, question):
        return [0.52, 0.44]

    gate = make_evidence_gate(score_all, min_chars=10, on_gate=records.append)
    gate("what does this do?", _text(["alpha", "beta"]))

    assert records[0]["outcome"] == "passed_through_low_separation"
    assert records[0]["separation"] == pytest.approx(0.08)


def test_boolean_scores_fail_open_and_return_original_text():
    records: list[dict] = []
    gate = make_evidence_gate(lambda items, question: [True, False], min_chars=10, on_gate=records.append)
    text = _text(["alpha", "beta"])

    assert gate("what does this do?", text) is text
    assert records[0]["outcome"] == "failed_open"


def test_dense_lines_split_cleanly_at_max_chunk_chars():
    seen_chunks: list[str] = []

    def score_all(items, question):
        seen_chunks.extend(items)
        return [0.9 if "keep" in item else 0.1 for item in items]

    # 10 lines of 30 chars each -> ~310 chars total, well under 40 lines but > 120 chars.
    lines = [f"keep line {i:02d} " + ("x" * 17) for i in range(6)] + [
        f"drop line {i:02d} " + ("y" * 17) for i in range(6)
    ]
    text = "\n".join(lines)

    gate = make_evidence_gate(score_all, chunk_lines=40, max_chunk_chars=120, min_chars=10)
    gated = gate("which lines keep?", text)

    assert len(seen_chunks) >= 4
    assert all(len(chunk) <= 120 for chunk in seen_chunks)
    for i in range(6):
        assert f"keep line {i:02d}" in gated
        assert f"drop line {i:02d}" not in gated

    # When every split chunk is relevant, nothing is dropped.
    all_keep = "\n".join(f"keep line {i:02d} " + ("x" * 17) for i in range(8))
    assert gate("which lines keep?", all_keep) is all_keep


def test_single_long_line_is_emitted_without_splitting_mid_line():
    seen_chunks: list[str] = []

    def score_all(items, question):
        seen_chunks.extend(items)
        return [0.9 if "keep" in item else 0.1 for item in items]

    long_keep = "keep_" + ("z" * 150)
    noise = "noise_" + ("q" * 150)
    gate = make_evidence_gate(score_all, chunk_lines=40, max_chunk_chars=80, min_chars=10)
    gated = gate("find keep", f"{long_keep}\n{noise}")

    assert seen_chunks == [long_keep, noise]
    assert long_keep in gated
    assert noise not in gated


def test_make_evidence_gate_rejects_invalid_chunk_limits():
    with pytest.raises(ValueError, match="chunk_lines"):
        make_evidence_gate(lambda items, q: [0.5] * len(items), chunk_lines=0)
    with pytest.raises(ValueError, match="max_chunk_chars"):
        make_evidence_gate(lambda items, q: [0.5] * len(items), max_chunk_chars=0)
