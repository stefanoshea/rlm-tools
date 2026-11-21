import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from evals.schemas import EvalSpec

# A judge takes (criteria, output) and returns whether the output satisfies it.
Judge = Callable[[str, str], bool]


def _load_json_answer(text: str) -> Any:
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    return json.loads(text, object_pairs_hook=unique_keys, parse_float=Decimal,
                      parse_constant=invalid_constant)


def _same_json(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_same_json(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_json(a, b) for a, b in zip(actual, expected)))
    return actual == expected


def load_eval_specs(path: str | Path) -> list[EvalSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        cases = data.get("evals", [])
    else:
        cases = data
    specs = [EvalSpec.from_dict(item) for item in cases]
    seen: set[str] = set()
    for spec in specs:
        validate_artifact_name(spec.eval_id)
        if spec.eval_id in seen:
            raise ValueError(f"Duplicate eval ID: {spec.eval_id}")
        seen.add(spec.eval_id)
    return specs


def validate_artifact_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or name in {"", ".", "..", "benchmark.json"}
        or any(separator in name for separator in ("/", "\\", "\0"))
    ):
        raise ValueError(f"Invalid benchmark artifact name: {name!r}")


def make_llm_judge(client=None, model: str | None = None) -> Judge:
    """Build a Judge backed by the sub-LLM.

    The model is asked to answer YES/NO on whether the output meets the
    criteria; the verdict is parsed into a bool. Imported lazily so the eval
    package has no hard dependency on the Anthropic client unless judging is used.
    """
    from rlm_tools.llm_bridge import make_llm_query

    query = make_llm_query(client=client, model=model)

    def judge(criteria: str, output: str) -> bool:
        prompt = (
            "You are grading an answer. Reply with YES if the answer satisfies "
            "the criteria, or NO if it does not. Start your reply with YES or NO.\n\n"
            f"Criteria: {criteria}\n\nAnswer:\n{output}"
        )
        verdict = query(prompt).strip().upper()
        return re.match(r"YES\b", verdict) is not None

    return judge


def _grade_assertion(
    output: str,
    assertion: dict[str, Any],
    judge: Judge | None = None,
) -> dict[str, Any]:
    kind = assertion["kind"]
    values = assertion.get("values", [])
    result = {
        "id": assertion["id"],
        "kind": kind,
        "values": values,
    }

    if kind == "contains_all":
        result["passed"] = all(value in output for value in values)
    elif kind == "contains_any":
        result["passed"] = any(value in output for value in values)
    elif kind == "exact":
        result["passed"] = output.strip() == "".join(values).strip()
    elif kind == "json_equals":
        if len(values) != 1:
            raise ValueError("json_equals requires one JSON-encoded expected answer")
        expected = _load_json_answer(values[0])
        text = output.strip()
        fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text, re.S | re.I)
        try:
            actual = _load_json_answer(fence.group(1) if fence else text)
            result["passed"] = _same_json(actual, expected)
        except (ValueError, RecursionError, ArithmeticError):
            result["passed"] = False
    elif kind == "not_contains":
        result["passed"] = all(value not in output for value in values)
    elif kind == "llm_judge":
        # Semantic grading: a criterion string judged against the output.
        # Without a judge wired (offline runs) the assertion is skipped so it
        # neither crashes nor penalizes the score.
        if judge is None:
            result["passed"] = True
            result["skipped"] = True
        else:
            criteria = " ".join(values)
            result["passed"] = bool(judge(criteria, output))
    else:
        raise ValueError(f"Unsupported assertion kind: {kind}")

    return result


def grade_output(
    output: str,
    assertions: list[dict[str, Any]],
    judge: Judge | None = None,
) -> dict[str, Any]:
    results = [_grade_assertion(output, assertion, judge) for assertion in assertions]
    scored = [r for r in results if not r.get("skipped")]
    passed = sum(1 for result in scored if result["passed"])
    total = len(scored)
    score = 1.0 if total == 0 else passed / total
    return {
        "passed": passed,
        "total": total,
        "score": score,
        "assertions": results,
    }


def grade_eval_case(
    spec: EvalSpec,
    output: str,
    judge: Judge | None = None,
) -> dict[str, Any]:
    graded = grade_output(
        output=output,
        assertions=[assertion.to_dict() for assertion in spec.assertions],
        judge=judge,
    )
    return {
        "eval_id": spec.eval_id,
        "prompt": spec.prompt,
        "score": graded["score"],
        "passed": graded["passed"],
        "total": graded["total"],
        "assertions": graded["assertions"],
    }
