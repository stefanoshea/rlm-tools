import json
import os
import time
import argparse
from pathlib import Path
from typing import Any, Callable

from evals.aggregate_benchmark import aggregate_runs, format_benchmark_summary
from evals.grade_eval import Judge, grade_eval_case, load_eval_specs, validate_artifact_name


Executor = Callable[[str], dict[str, Any] | str]


def _normalize_result(result: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(result, str):
        return {"output": result}
    return result


def _timing(payload: dict[str, Any], elapsed_ms: float | None = None) -> dict[str, Any]:
    input_tokens = payload.get("input_tokens")
    output_tokens = payload.get("output_tokens")
    total_tokens = payload.get("total_tokens")
    if total_tokens is None and type(input_tokens) is int and type(output_tokens) is int:
        if input_tokens >= 0 and output_tokens >= 0:
            total_tokens = input_tokens + output_tokens
    return {
        "duration_ms": payload.get("duration_ms", elapsed_ms),
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def run_benchmark(
    eval_spec_path: str | Path,
    output_dir: str | Path,
    executors: dict[str, Executor] | None = None,
    iterations: int = 1,
    judge: Judge | None = None,
) -> dict[str, Any]:
    if not executors:
        raise ValueError("executors are required for run_benchmark()")

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    specs = load_eval_specs(eval_spec_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    grading_files: list[Path] = []

    for config_name, executor in executors.items():
        validate_artifact_name(config_name)
        config_root = root / config_name
        config_root.mkdir(parents=True, exist_ok=True)

        for spec in specs:
            for run_index in range(iterations):
                run_root = config_root / spec.eval_id / f"run_{run_index + 1:03d}"
                run_root.mkdir(parents=True, exist_ok=True)

                started = time.perf_counter()
                raw_result = _normalize_result(executor(spec.prompt))
                elapsed_ms = (time.perf_counter() - started) * 1000

                output_text = raw_result.get("output", "")
                timing = _timing(raw_result, elapsed_ms)
                grading = grade_eval_case(spec, output_text, judge=judge)

                (run_root / "output.json").write_text(json.dumps({
                    "eval_id": spec.eval_id,
                    "prompt": spec.prompt,
                    "output": output_text,
                }, indent=2))
                (run_root / "timing.json").write_text(json.dumps(timing, indent=2))
                grading_path = run_root / "grading.json"
                grading_path.write_text(json.dumps(grading, indent=2))
                grading_files.append(grading_path)

    benchmark = aggregate_runs(root, grading_files=grading_files)
    (root / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
    return benchmark


def run_benchmark_from_results(
    eval_spec_path: str | Path,
    output_dir: str | Path,
    raw_results: dict[str, dict[str, dict[str, Any] | list[dict[str, Any]]]],
    judge: Judge | None = None,
) -> dict[str, Any]:
    specs = {spec.eval_id: spec for spec in load_eval_specs(eval_spec_path)}
    for config_name, by_eval in raw_results.items():
        validate_artifact_name(config_name)
        if not by_eval:
            raise ValueError(f"{config_name}: raw results contain no cases")
        for eval_id, payload in by_eval.items():
            if eval_id not in specs:
                raise ValueError(f"{config_name}: unknown eval ID {eval_id}")
            trials = payload if isinstance(payload, list) else [payload]
            if not trials or any(not isinstance(trial, dict) for trial in trials):
                raise ValueError(f"{config_name}/{eval_id}: expected one or more result objects")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    grading_files: list[Path] = []

    for config_name, by_eval in raw_results.items():
        for eval_id, case_results in by_eval.items():
            spec = specs[eval_id]
            trials = case_results if isinstance(case_results, list) else [case_results]
            for run_index, payload in enumerate(trials, start=1):
                prompt = payload.get("prompt", spec.prompt)
                run_root = root / config_name / eval_id / f"run_{run_index:03d}"
                run_root.mkdir(parents=True, exist_ok=True)
                output_text = payload.get("output", "")
                grading = grade_eval_case(spec, output_text, judge=judge)
                (run_root / "output.json").write_text(json.dumps({
                    "eval_id": eval_id,
                    "prompt": prompt,
                    "output": output_text,
                }, indent=2))
                (run_root / "timing.json").write_text(json.dumps(_timing(payload), indent=2))
                grading["prompt"] = prompt
                grading_path = run_root / "grading.json"
                grading_path.write_text(json.dumps(grading, indent=2))
                grading_files.append(grading_path)

    benchmark = aggregate_runs(root, grading_files=grading_files, expected_cases=specs)
    (root / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
    return benchmark


def judge_from_env() -> Judge | None:
    if os.environ.get("RLM_JUDGE") == "1" and os.environ.get("ANTHROPIC_API_KEY"):
        from evals.grade_eval import make_llm_judge

        return make_llm_judge()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate scored prompt benchmarks for rlm-tools.")
    parser.add_argument("--spec", required=True, help="Path to eval spec JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to write benchmark artifacts")
    parser.add_argument("--raw-results", help="Optional JSON file containing precomputed raw results")
    args = parser.parse_args()

    if args.raw_results:
        raw_results = json.loads(Path(args.raw_results).read_text())
        benchmark = run_benchmark_from_results(args.spec, args.output_dir, raw_results, judge=judge_from_env())
    else:
        raise SystemExit("--raw-results is required when using the CLI")

    print(format_benchmark_summary(benchmark))


if __name__ == "__main__":
    main()
