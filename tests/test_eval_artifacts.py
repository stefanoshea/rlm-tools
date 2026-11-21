import json
import os
import stat
from pathlib import Path

import pytest

from evals.aggregate_benchmark import aggregate_runs, format_benchmark_summary
from evals.grade_eval import grade_output
from evals.run_prompt_benchmark import run_benchmark, run_benchmark_from_results


def test_grade_output_contains_all():
    grading = grade_output(
        output="DashboardReducer uses apiClient and clock",
        assertions=[{
            "id": "deps",
            "kind": "contains_all",
            "values": ["DashboardReducer", "apiClient", "clock"],
        }],
    )
    assert grading["score"] == 1.0


def test_benchmark_aggregation_reports_mean_and_stddev(tmp_path):
    config_root = tmp_path / "with_rlm" / "sample_eval" / "run_001"
    config_root.mkdir(parents=True)
    (config_root / "grading.json").write_text(json.dumps({"score": 1.0}))
    (config_root / "timing.json").write_text(json.dumps({"total_tokens": 100, "duration_ms": 10}))

    incomplete = config_root.parent / "run_002"
    incomplete.mkdir()
    (incomplete / "timing.json").write_text(json.dumps({"total_tokens": 1000, "duration_ms": 100}))

    benchmark = aggregate_runs(tmp_path)
    assert benchmark["configs"][0]["mean_tokens"] == 100
    assert benchmark["configs"][0]["mean_duration_ms"] == 10
    assert "pass_rate" in benchmark["summary"]
    assert "mean_tokens" in benchmark["summary"]
    assert "stddev_tokens" in benchmark["summary"]


@pytest.fixture
def comparison_spec(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([
        {"eval_id": case, "prompt": case, "assertions": [
            {"id": "answer", "kind": "contains_all", "values": ["correct"]},
        ]}
        for case in ("one", "two")
    ]))
    return spec


def test_missing_usage_cannot_win_an_efficiency_tie(tmp_path, comparison_spec):
    without_usage = {"output": "correct", "duration_ms": 1}
    measured = {"output": "correct", "input_tokens": 10, "output_tokens": 5, "duration_ms": 2}
    results = {
        "first": {"one": without_usage, "two": measured},
        "second": {"one": measured, "two": measured},
    }
    summary = run_benchmark_from_results(comparison_spec, tmp_path, results)
    timing = json.loads((tmp_path / "first" / "one" / "run_001" / "timing.json").read_text())
    assert timing["total_tokens"] is None
    assert timing["input_tokens"] is None
    assert summary["configs"][0]["mean_tokens"] is None
    assert summary["configs"][0]["token_runs"] == 1
    assert summary["configs"][1]["mean_tokens"] == 15
    formatted = format_benchmark_summary(summary)
    assert "No preferred config (incomplete mean tokens data)." in formatted
    assert "- Tokens: N/A" in formatted


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX permissions")
def test_benchmark_creates_private_artifacts(tmp_path, comparison_spec):
    output = tmp_path / "runs"
    run_benchmark_from_results(comparison_spec, output, {"with_rlm": {"one": {"output": "correct"}}})

    for path in [output, *output.rglob("*")]:
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


def test_benchmark_rejects_symlinked_run_directory(tmp_path, comparison_spec):
    case = tmp_path / "runs" / "with_rlm" / "one"
    case.mkdir(parents=True)
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    (case / "run_001").symlink_to(alternate, target_is_directory=True)

    with pytest.raises(ValueError, match="Benchmark artifact directory"):
        run_benchmark_from_results(
            comparison_spec, tmp_path / "runs", {"with_rlm": {"one": {"output": "correct"}}},
        )
    assert not list(alternate.iterdir())


@pytest.mark.parametrize("coverage", ["mismatched cases", "mismatched trials", "missing shared case"])
def test_unmatched_comparison_suppresses_deltas_and_preference(tmp_path, comparison_spec, coverage):
    passing = {"output": "correct", "total_tokens": 1, "duration_ms": 1}
    failing = {"output": "wrong", "total_tokens": 100, "duration_ms": 100}
    first = {"one": passing}
    second = {"one": failing}
    if coverage == "mismatched cases":
        second["two"] = failing
    elif coverage == "mismatched trials":
        first = {"one": [passing, passing], "two": passing}
        second["two"] = failing
    summary = run_benchmark_from_results(comparison_spec, tmp_path, {"first": first, "second": second})
    assert not summary["matched_coverage"]
    formatted = format_benchmark_summary(summary)
    assert "Delta (" not in formatted
    assert "No preferred config (incomplete or unmatched case and trial coverage)." in formatted


def test_unscorable_archived_grade_cannot_win(tmp_path):
    for config, grading in (("first", {"score": 1.0, "total": 0}), ("second", {"score": 0.5})):
        run = tmp_path / config / "case" / "run_001"
        run.mkdir(parents=True)
        (run / "grading.json").write_text(json.dumps(grading))
    summary = aggregate_runs(tmp_path)
    first, second = summary["configs"]
    assert first["scored_runs"] == 0
    assert first["pass_rate"] is None
    assert second["mean_score"] == 0.5
    paired = summary["paired_effects"]
    assert paired["quality_pairs"] == 0
    assert paired["total_pairs"] == 1
    assert paired["median_token_difference"] is None
    assert "No preferred config (incomplete pass rate data)." in format_benchmark_summary(summary)


def test_paired_effects_use_median_for_passing_pairs_with_measured_tokens(tmp_path):
    trials = [
        ((1, 8), (1, 10)),
        ((1, 6), (1, 10)),
        ((1, 9_010), (1, 10)),
        ((1, None), (1, 10)),
        ((1, 5), (0, 10)),
        ((0, 5), (1, 10)),
    ]
    for index, pair in enumerate(trials, start=1):
        for config, (score, tokens) in zip(("first", "second"), pair, strict=True):
            run = tmp_path / config / "case" / f"run_{index:03d}"
            run.mkdir(parents=True)
            (run / "grading.json").write_text(json.dumps({"score": score, "prompt": "same prompt"}))
            (run / "timing.json").write_text(json.dumps({"total_tokens": tokens}))

    summary = aggregate_runs(tmp_path)
    paired = summary["paired_effects"]
    assert paired["quality"] == {"wins": 1, "ties": 4, "losses": 1}
    assert paired["quality_pairs"] == paired["total_pairs"] == 6
    assert paired["median_token_difference"] == -2
    assert paired["passing_pairs_with_tokens"] == 3
    formatted = format_benchmark_summary(summary, claude_cli_tokens=True)
    assert "Median CLI Tokens difference (first - second): -2.0 (3/6 pairs" in formatted


def test_effective_prompt_mismatch_suppresses_comparison(tmp_path, comparison_spec):
    first = {"output": "correct", "total_tokens": 1, "duration_ms": 1}
    second = {"output": "wrong", "total_tokens": 100, "duration_ms": 100}
    results = {
        "first": {"one": {**first, "prompt": "actual prompt A"}, "two": first},
        "second": {"one": {**second, "prompt": "actual prompt B"}, "two": second},
    }
    summary = run_benchmark_from_results(comparison_spec, tmp_path, results)
    assert summary["matched_coverage"]
    assert not summary["matched_prompts"]
    assert "paired_effects" not in summary
    formatted = format_benchmark_summary(summary)
    assert "Delta (" not in formatted
    assert "No preferred config (effective prompts differ across configs)." in formatted


def test_format_benchmark_summary_is_human_readable():
    summary = {
        "configs": [
            {
                "name": "with_rlm",
                "pass_rate": 1.0,
                "mean_score": 1.0,
                "mean_tokens": 42.0,
                "mean_duration_ms": 10.0,
            },
            {
                "name": "without_rlm",
                "pass_rate": 0.5,
                "mean_score": 0.5,
                "mean_tokens": 60.0,
                "mean_duration_ms": 12.0,
            },
        ],
        "notes": ["with_rlm: pass rate below 100%"],
    }
    formatted = format_benchmark_summary(summary)
    assert "Benchmark Summary" in formatted
    assert "with_rlm" in formatted
    assert "Tokens" in formatted
    assert "Delta" in formatted


def test_format_benchmark_summary_prefers_pass_rate_before_tokens():
    summary = {
        "configs": [
            {
                "name": "with_rlm",
                "pass_rate": 1.0,
                "mean_score": 0.92,
                "mean_tokens": 90.0,
                "mean_duration_ms": 11.0,
            },
            {
                "name": "without_rlm",
                "pass_rate": 0.5,
                "mean_score": 0.70,
                "mean_tokens": 40.0,
                "mean_duration_ms": 9.0,
            },
        ],
        "notes": [],
    }
    formatted = format_benchmark_summary(summary)
    assert "Preferred config: with_rlm" in formatted
    assert "pass rate" in formatted


def test_prompt_benchmark_outputs_scored_summary(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "evals": [
            {
                "eval_id": "sample",
                "prompt": "Name the reducer and dependency",
                "assertions": [
                    {"id": "contains_answer", "kind": "contains_all", "values": ["DashboardReducer", "apiClient"]},
                ],
            }
        ]
    }))

    summary = run_benchmark(
        eval_spec_path=spec_path,
        output_dir=tmp_path / "runs",
        executors={
            "with_rlm": lambda prompt: {"output": "DashboardReducer uses apiClient", "total_tokens": 42, "duration_ms": 7},
            "without_rlm": lambda prompt: {"output": "DashboardReducer only", "total_tokens": 30, "duration_ms": 5},
        },
    )
    assert "configs" in summary
    assert "pass_rate" in summary["configs"][0]

    repeated = run_benchmark(
        eval_spec_path=spec_path,
        output_dir=tmp_path / "runs",
        executors={
            "with_rlm": lambda prompt: {"output": "DashboardReducer uses apiClient", "total_tokens": 20},
        },
    )
    assert [config["name"] for config in repeated["configs"]] == ["with_rlm"]
    assert repeated["configs"][0]["mean_tokens"] == 20


def test_run_ab_eval_surfaces_human_readable_summary():
    script = Path("evals/run_ab_eval.py").read_text()
    assert '"benchmark_stdout.txt"' in script
    assert 'print(formatted)' in script


def test_run_ab_eval_defaults_to_long_context_spec_and_fixture():
    script = Path("evals/run_ab_eval.sh").read_text()
    assert 'tests/fixtures/rlm_corpus_longcontext' in script
    assert 'evals/specs/rlm_long_context.json' in script


def test_long_context_spec_uses_target_path_placeholder():
    spec = json.loads(Path("evals/specs/rlm_long_context.json").read_text())
    assert len(spec["evals"]) >= 3
    assert all("{target_path}" in item["prompt"] for item in spec["evals"])
