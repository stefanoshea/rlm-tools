"""LLM-judge assertion tier for prompt grading.

Substring assertions can't tell a paraphrased-correct answer from a wrong one.
The `llm_judge` kind delegates to an injected judge callable so semantic
correctness can be scored without hard-coded string matching.
"""

from unittest.mock import MagicMock

from evals.grade_eval import grade_output, make_llm_judge


def test_llm_judge_assertion_passes_when_judge_approves():
    seen = {}

    def judge(criteria: str, output: str) -> bool:
        seen["criteria"] = criteria
        seen["output"] = output
        return True

    grading = grade_output(
        output="Authentication is handled by the login service.",
        assertions=[{
            "id": "semantic",
            "kind": "llm_judge",
            "values": ["States that auth lives in the login service"],
        }],
        judge=judge,
    )

    assert grading["score"] == 1.0
    assert seen["criteria"] == "States that auth lives in the login service"
    assert "login service" in seen["output"]


def test_llm_judge_assertion_fails_when_judge_rejects():
    grading = grade_output(
        output="It is handled somewhere in the frontend.",
        assertions=[{
            "id": "semantic",
            "kind": "llm_judge",
            "values": ["States that auth lives in the login service"],
        }],
        judge=lambda criteria, output: False,
    )

    assert grading["score"] == 0.0
    assert grading["assertions"][0]["passed"] is False


def test_llm_judge_without_judge_is_skipped_not_crashed():
    # No judge wired (e.g. offline run): the judge assertion is skipped and
    # surfaced as such rather than raising, so substring assertions still score.
    grading = grade_output(
        output="anything",
        assertions=[
            {"id": "kw", "kind": "contains_all", "values": ["anything"]},
            {"id": "semantic", "kind": "llm_judge", "values": ["something subjective"]},
        ],
    )

    by_id = {a["id"]: a for a in grading["assertions"]}
    assert by_id["kw"]["passed"] is True
    assert by_id["semantic"].get("skipped") is True
    # Skipped judge assertions don't drag the score down.
    assert grading["score"] == 1.0


def test_make_llm_judge_interprets_verdict():
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="YES - the answer satisfies the criteria")]
    )
    judge = make_llm_judge(client=client, model="m")
    assert judge("criteria", "some answer") is True

    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="NO - it does not")]
    )
    assert judge("criteria", "some answer") is False
