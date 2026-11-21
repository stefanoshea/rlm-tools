"""The benchmark runner must thread an optional judge into grading so llm_judge
assertions score when a judge is supplied (RLM_JUDGE=1) and skip otherwise."""

import json
from pathlib import Path

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


def test_judge_scores_llm_judge_assertions(tmp_path):
    spec_path = _write_spec(tmp_path)
    raw = {"with_rlm": {"semantic_case": {"output": "X is the thing", "total_tokens": 10}}}

    run_benchmark_from_results(
        spec_path, tmp_path / "out", raw, judge=lambda criteria, output: True
    )
    grading = json.loads((tmp_path / "out" / "with_rlm" / "semantic_case" / "run_001" / "grading.json").read_text())
    # Both the anchor and the judged assertion pass -> perfect score.
    assert grading["score"] == 1.0
    assert grading["total"] == 2


def test_without_judge_llm_judge_is_skipped(tmp_path):
    spec_path = _write_spec(tmp_path)
    raw = {"with_rlm": {"semantic_case": {"output": "X is the thing", "total_tokens": 10}}}

    run_benchmark_from_results(spec_path, tmp_path / "out", raw)
    grading = json.loads((tmp_path / "out" / "with_rlm" / "semantic_case" / "run_001" / "grading.json").read_text())
    # llm_judge skipped (no judge), only the anchor is scored.
    assert grading["total"] == 1
    assert grading["score"] == 1.0
