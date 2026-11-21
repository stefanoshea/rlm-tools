from collections import defaultdict
from collections.abc import Iterable
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from evals.review_benchmark import review_benchmark


TrialMetrics = dict[tuple[str, ...], tuple[float | None, int | None]]


def _stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def aggregate_runs(
    sample_runs_dir: str | Path,
    *,
    grading_files: Iterable[Path] | None = None,
    expected_cases: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(sample_runs_dir)
    configs: list[dict[str, Any]] = []
    files_by_config: dict[str, list[Path]] = defaultdict(list)
    if grading_files is None:
        grading_files = root.glob("*/*/run_*/grading.json")
    for path in grading_files:
        files_by_config[path.relative_to(root).parts[0]].append(path)

    expected_cases = set(expected_cases) if expected_cases is not None else None
    runs_by_config: dict[str, TrialMetrics] = {}
    prompts_by_trial: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for name, config_files in sorted(files_by_config.items()):
        trials: TrialMetrics = {}
        durations = []
        for path in sorted(set(config_files)):
            trial = path.relative_to(root).parts[1:3]
            grading = json.loads(path.read_text())
            timing_path = path.with_name("timing.json")
            timing = json.loads(timing_path.read_text()) if timing_path.is_file() else {}
            prompt = grading.get("prompt")
            if not isinstance(prompt, str) and (output_path := path.with_name("output.json")).is_file():
                prompt = json.loads(output_path.read_text()).get("prompt")
            if isinstance(prompt, str):
                prompts_by_trial[trial].add(prompt)
            score = grading["score"] if grading.get("total", 1) > 0 else None
            total = timing.get("total_tokens")
            tokens = total if type(total) is int and total >= 0 else None
            trials[trial] = score, tokens
            duration = timing.get("duration_ms")
            if type(duration) in (float, int) and math.isfinite(duration) and duration >= 0:
                durations.append(duration)

        runs_by_config[name] = trials
        runs = len(trials)
        scores = [score for score, _ in trials.values() if score is not None]
        tokens = [tokens for _, tokens in trials.values() if tokens is not None]

        config_summary = {
            "name": name,
            "runs": runs,
            "scored_runs": len(scores),
            "token_runs": len(tokens),
            "duration_runs": len(durations),
            "pass_rate": sum(score >= 1.0 for score in scores) / runs if len(scores) == runs else None,
            "mean_score": mean(scores) if len(scores) == runs else None,
            "mean_tokens": mean(tokens) if len(tokens) == runs else None,
            "stddev_tokens": _stddev(tokens) if len(tokens) == runs else None,
            "mean_duration_ms": mean(durations) if len(durations) == runs else None,
            "stddev_duration_ms": _stddev(durations) if len(durations) == runs else None,
        }
        configs.append(config_summary)

    coverages = [trials.keys() for trials in runs_by_config.values()]
    matched_coverage = bool(coverages) and all(
        coverage == coverages[0]
        and (expected_cases is None or {case for case, _ in coverage} == expected_cases)
        for coverage in coverages
    )
    pass_rates = [config["pass_rate"] for config in configs]
    mean_tokens = [config["mean_tokens"] for config in configs]
    summary = {
        "configs": configs,
        "matched_coverage": matched_coverage,
        "matched_prompts": all(len(prompts) == 1 for prompts in prompts_by_trial.values()),
        "summary": {
            "pass_rate": mean(pass_rates) if pass_rates and None not in pass_rates else None,
            "mean_tokens": mean(mean_tokens) if mean_tokens and None not in mean_tokens else None,
            "stddev_tokens": _stddev(mean_tokens) if mean_tokens and None not in mean_tokens else None,
        },
    }
    if len(configs) == 2 and matched_coverage and summary["matched_prompts"]:
        first, second = runs_by_config
        summary["paired_effects"] = _paired_effects(first, second, runs_by_config[first], runs_by_config[second])
    summary["notes"] = review_benchmark(summary)
    return summary


def _paired_effects(
    first: str, second: str, left_trials: TrialMetrics, right_trials: TrialMetrics,
) -> dict[str, Any]:
    outcomes = {"wins": 0, "ties": 0, "losses": 0}
    token_differences = []
    for trial, (left_score, left_tokens) in left_trials.items():
        right_score, right_tokens = right_trials[trial]
        if left_score is None or right_score is None:
            continue
        if left_score > right_score:
            outcomes["wins"] += 1
        elif left_score < right_score:
            outcomes["losses"] += 1
        else:
            outcomes["ties"] += 1
        if left_score >= 1 and right_score >= 1 and left_tokens is not None and right_tokens is not None:
            token_differences.append(left_tokens - right_tokens)

    return {
        "first_config": first,
        "second_config": second,
        "total_pairs": len(left_trials),
        "quality_pairs": sum(outcomes.values()),
        "quality": outcomes,
        "passing_pairs_with_tokens": len(token_differences),
        "median_token_difference": median(token_differences) if token_differences else None,
    }


def _preferred_config(configs: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    if len(configs) < 2:
        return None, None
    for metric, select, reason in (
        ("pass_rate", max, "higher pass rate"),
        ("mean_score", max, "higher mean score"),
        ("mean_tokens", min, "lower token use"),
        ("mean_duration_ms", min, "lower mean duration"),
    ):
        values = [config.get(metric) for config in configs]
        if None in values:
            return None, f"incomplete {metric.replace('_', ' ')} data"
        best = select(values)
        configs = [config for config in configs if config[metric] == best]
        if len(configs) == 1:
            return configs[0], reason
    return None, "equivalent metrics"


def _display(value: float | None, format_spec: str) -> str:
    return "N/A" if value is None else format(value, format_spec)


def format_benchmark_summary(summary: dict[str, Any], *, claude_cli_tokens: bool = False) -> str:
    configs = summary.get("configs", [])
    token_label = "CLI Tokens" if claude_cli_tokens else "Tokens"
    lines = [
        "Benchmark Summary",
        "",
        f"{'Config':<14} {'Pass Rate':>10} {'Mean Score':>11} {token_label:>10} {'Time (ms)':>10}",
        f"{'-' * 14} {'-' * 10} {'-' * 11} {'-' * 10} {'-' * 10}",
    ]

    for config in configs:
        lines.append(
            f"{config['name']:<14} "
            f"{_display(config['pass_rate'], '.1%'):>10} "
            f"{_display(config['mean_score'], '.2f'):>11} "
            f"{_display(config['mean_tokens'], '.1f'):>10} "
            f"{_display(config['mean_duration_ms'], '.1f'):>10}"
        )

    matched_coverage = summary.get("matched_coverage", True)
    matched_prompts = summary.get("matched_prompts", True)
    if len(configs) == 2 and matched_coverage and matched_prompts:
        first, second = configs[0], configs[1]
        lines.extend(["", f"Delta ({first['name']} - {second['name']}):"])
        for metric, label, scale, unit in (
            ("mean_tokens", token_label, 1, ""),
            ("mean_duration_ms", "Time (ms)", 1, ""),
            ("pass_rate", "Pass rate", 100, " pts"),
        ):
            left, right = first[metric], second[metric]
            delta = f"{(left - right) * scale:+.1f}{unit}" if left is not None and right is not None else "N/A"
            lines.append(f"- {label}: {delta}")

    if paired := summary.get("paired_effects"):
        first, second = paired["first_config"], paired["second_config"]
        quality = paired["quality"]
        token_difference = _display(paired["median_token_difference"], "+.1f")
        total = paired["total_pairs"]
        lines.extend([
            "",
            f"Paired outcomes ({first} vs {second})",
            f"- Quality for {first}: {quality['wins']} wins, {quality['ties']} ties, "
            f"{quality['losses']} losses ({paired['quality_pairs']}/{total} scored pairs)",
            f"- Median {token_label} difference ({first} - {second}): {token_difference} "
            f"({paired['passing_pairs_with_tokens']}/{total} pairs with two passing answers and measured totals)",
        ])

    if not matched_coverage:
        winner, reason = None, "incomplete or unmatched case and trial coverage"
    elif not matched_prompts:
        winner, reason = None, "effective prompts differ across configs"
    else:
        winner, reason = _preferred_config(configs)
    if winner:
        lines.extend([
            "",
            f"Preferred config: {winner['name']}",
            f"Reason: {reason} (quality first, efficiency second)",
        ])
    elif reason:
        lines.extend(["", f"No preferred config ({reason})."])

    if claude_cli_tokens:
        lines.extend(["", "CLI tokens exclude separate model usage by MCP helpers and the optional judge."])

    notes = summary.get("notes", [])
    if notes:
        lines.append("")
        lines.append("Notes")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)
