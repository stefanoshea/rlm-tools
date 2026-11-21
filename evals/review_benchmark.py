from typing import Any
import argparse
import json
from pathlib import Path


def review_benchmark(summary: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for config in summary.get("configs", []):
        name = config["name"]
        pass_rate = config.get("pass_rate")
        stddev_tokens = config.get("stddev_tokens")
        mean_tokens = config.get("mean_tokens")
        mean_duration = config.get("mean_duration_ms")
        stddev_duration = config.get("stddev_duration_ms")

        if pass_rate is None:
            notes.append(f"{name}: quality metrics require scorable grades for every run")
        elif pass_rate < 1.0:
            notes.append(f"{name}: pass rate below 100% ({pass_rate:.2%})")
        if mean_tokens is None:
            notes.append(f"{name}: token usage is missing or invalid for at least one run")
        elif stddev_tokens is not None and stddev_tokens > mean_tokens * 0.25:
            notes.append(f"{name}: token variance is high ({stddev_tokens:.1f} stddev)")
        if mean_duration is None:
            notes.append(f"{name}: duration is missing or invalid for at least one run")
        elif stddev_duration is not None and stddev_duration > mean_duration * 0.25:
            notes.append(f"{name}: duration variance is high")

    if summary.get("matched_coverage") is False:
        notes.append("Case and trial coverage is incomplete or differs across configs.")
    if summary.get("matched_prompts") is False:
        notes.append("Effective prompts differ across configs for at least one matched case and trial.")
    if not notes:
        notes.append("No obvious benchmark anomalies detected.")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Review benchmark summary and print analyst notes.")
    parser.add_argument("benchmark_json", help="Path to benchmark.json")
    args = parser.parse_args()

    summary = json.loads(Path(args.benchmark_json).read_text(encoding="utf-8"))
    for note in review_benchmark(summary):
        print(note)


if __name__ == "__main__":
    main()
