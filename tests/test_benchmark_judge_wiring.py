"""The benchmark runner records advisory judge verdicts when configured."""

import json
from pathlib import Path

import pytest

from evals.run_prompt_benchmark import run_benchmark_from_results


def _write_spec(tmp_path: Path) -> Path:
    spec = {
        "evals": [
            {
                "eval_id": "semantic_case",
                "prompt": "explain X in {target_path}",
                "assertions": [
                    {"id": "anchor", "kind": "contains_any", "values": ["X"]},
                    {"id": "judged", "kind": "llm_judge", "values": ["Is a correct explanation"]},
                ],
            }
        ]
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_judge_approval_does_not_override_a_failed_check(tmp_path):
    spec_path = _write_spec(tmp_path)
    raw = {"with_rlm": {"semantic_case": {"output": "Wrong answer", "total_tokens": 10}}}

    run_benchmark_from_results(
        spec_path, tmp_path / "out", raw, judge=lambda criteria, output: True
    )
    grading = json.loads((tmp_path / "out" / "with_rlm" / "semantic_case" / "run_001" / "grading.json").read_text())
    assert grading["score"] == 0.0
    assert grading["total"] == 1
    assert grading["assertions"][1]["advisory"] is True
    assert grading["assertions"][1]["passed"] is True


def test_judge_error_does_not_prevent_benchmark_scoring(tmp_path):
    spec_path = _write_spec(tmp_path)
    raw = {"with_rlm": {"semantic_case": {"output": "X", "total_tokens": 10}}}

    def unavailable_judge(criteria, output):
        raise RuntimeError("Provider unavailable")

    summary = run_benchmark_from_results(
        spec_path, tmp_path / "out", raw, judge=unavailable_judge,
    )
    grading = json.loads((tmp_path / "out" / "with_rlm" / "semantic_case" / "run_001" / "grading.json").read_text())
    assert summary["configs"][0]["pass_rate"] == 1.0
    assert grading["assertions"][1]["skip_reason"] == "judge_error"


def test_without_judge_llm_judge_is_skipped(tmp_path):
    spec_path = _write_spec(tmp_path)
    raw = {"with_rlm": {"semantic_case": {"output": "X is the thing", "total_tokens": 10}}}

    run_benchmark_from_results(spec_path, tmp_path / "out", raw)
    grading = json.loads((tmp_path / "out" / "with_rlm" / "semantic_case" / "run_001" / "grading.json").read_text())
    # llm_judge skipped (no judge), only the anchor is scored.
    assert grading["total"] == 1
    assert grading["score"] == 1.0


def test_judge_only_case_cannot_receive_a_benchmark_score(tmp_path):
    path = _write_spec(tmp_path)
    spec = json.loads(path.read_text())
    spec["evals"][0]["assertions"].pop(0)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="deterministic assertion"):
        run_benchmark_from_results(path, tmp_path / "out", {})
