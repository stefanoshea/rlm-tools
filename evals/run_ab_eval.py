from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from evals.aggregate_benchmark import format_benchmark_summary
from evals.grade_eval import load_eval_specs
from evals.run_prompt_benchmark import judge_from_env, run_benchmark_from_results


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_claude(
    prompt: str,
    workdir: Path,
    target: Path,
    mcp_config: dict,
    tools: str,
    system_prompt: str,
    env: dict[str, str],
    raw_path: Path,
) -> dict[str, Any]:
    config_path = workdir / ".mcp.json"
    config_path.write_text(json.dumps(mcp_config), encoding="utf-8")
    allowed_tools = tools or "mcp__rlm-tools__rlm_start,mcp__rlm-tools__rlm_execute,mcp__rlm-tools__rlm_end"
    command = [
        "claude", "--print", "--output-format", "json",
        "--no-session-persistence", "--strict-mcp-config",
        "--mcp-config", str(config_path), "--tools", tools,
        "--allowedTools", allowed_tools,
        "--add-dir", str(target), "--append-system-prompt", system_prompt,
        "--disable-slash-commands", "--setting-sources", "",
    ]
    if model := env.get("RLM_AB_MODEL"):
        command.extend(["--model", model])

    started = perf_counter()
    with raw_path.open("w", encoding="utf-8") as stdout:
        process = subprocess.run(
            command, input=prompt, stdout=stdout, stderr=subprocess.PIPE,
            cwd=workdir, env=env, encoding="utf-8",
        )
    elapsed_ms = (perf_counter() - started) * 1000
    if process.returncode:
        raise ValueError(f"{raw_path.name}: Claude exited with status {process.returncode}: {process.stderr.strip()}")

    result = json.loads(raw_path.read_text(encoding="utf-8"))
    if result.get("is_error") or not isinstance(result.get("result"), str):
        raise ValueError(f"{raw_path.name}: Claude did not return a successful text result")
    required_tools = set(allowed_tools.split(","))
    denied = {
        event["tool_name"] for event in result.get("permission_denials", [])
        if event.get("tool_name") in required_tools
    }
    if denied:
        raise ValueError(f"{raw_path.name}: benchmark tool access denied: {', '.join(sorted(denied))}")
    usage = result.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    input_parts = (
        usage.get("input_tokens"),
        usage.get("cache_creation_input_tokens", 0),
        usage.get("cache_read_input_tokens", 0),
    )
    input_tokens = sum(input_parts) if all(type(part) is int and part >= 0 for part in input_parts) else None
    output_tokens = usage.get("output_tokens")
    if type(output_tokens) is not int or output_tokens < 0:
        output_tokens = None
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "prompt": prompt,
        "output": result["result"],
        "duration_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def main() -> None:
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)
    try:
        iterations = int(env.get("RLM_AB_ITERATIONS", "1"))
    except ValueError:
        iterations = 0
    if iterations < 1:
        raise SystemExit("RLM_AB_ITERATIONS must be a positive integer.")
    use_gate = env.get("RLM_AB_SEMANTIC") == "1"
    if use_gate:
        if not env.get("TYPESAFE_API_KEY") or not env.get("ANTHROPIC_API_KEY"):
            raise SystemExit("RLM_AB_SEMANTIC=1 requires TYPESAFE_API_KEY and ANTHROPIC_API_KEY.")
        env["RLM_RECURSE_GATE"] = "1"
        print("RLM_AB_SEMANTIC=1 enables TypeSafe evidence gating.")
    else:
        env.pop("TYPESAFE_API_KEY", None)
        env["RLM_RECURSE_GATE"] = "0"

    if not shutil.which("claude"):
        raise SystemExit("Claude CLI is required to run the A/B benchmark.")
    target = Path(env.get("RLM_AB_TARGET_PATH", REPO_ROOT / "tests/fixtures/rlm_corpus_longcontext")).resolve()
    server_path = Path(env.get("RLM_TOOLS_PATH", REPO_ROOT)).resolve()
    spec_path = Path(env.get("RLM_AB_SPEC_PATH", REPO_ROOT / "evals/specs/rlm_long_context.json")).resolve()
    if not target.is_dir() or not server_path.is_dir():
        raise SystemExit(f"Invalid target or server directory: {target}, {server_path}")
    try:
        specs = load_eval_specs(spec_path)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"Invalid benchmark spec {spec_path}: {exc}") from exc
    if not specs:
        raise SystemExit(f"Benchmark spec contains no cases: {spec_path}")

    destination = REPO_ROOT / "evals" / "ab_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f_benchmark")
    destination.mkdir(parents=True)
    archived_spec = destination / "benchmark_spec.json"
    archived_spec.write_text(json.dumps({"evals": [case.to_dict() for case in specs]}, indent=2), encoding="utf-8")
    print(f"Writing benchmark artifacts to {destination}", flush=True)
    system_prompt = (
        "Answer only the question asked and follow its output format. "
        "Use repo-relative paths for any requested file citations. "
        f"The target codebase is at {target}."
    )
    server_command = ["--directory", str(server_path), "run"]
    if use_gate:
        server_command.extend(["--extra", "semantic"])
    configs = {
        "with_rlm": ({"mcpServers": {"rlm-tools": {
            "command": "uv", "args": [*server_command, "rlm-tools"],
        }}}, ""),
        "without_rlm": ({"mcpServers": {}}, "Read,Grep,Glob"),
    }
    trials_by_config: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {spec.eval_id: [] for spec in specs} for name in configs
    }
    try:
        with TemporaryDirectory(prefix="rlm-ab-") as tmpdir:
            for name in configs:
                (Path(tmpdir) / name).mkdir()

            for index, spec in enumerate(specs, start=1):
                prompt = spec.prompt.replace("{target_path}", str(target))
                for trial in range(1, iterations + 1):
                    print(f"Running eval {spec.eval_id} ({trial}/{iterations})", flush=True)
                    # Each case alternates which arm runs first on subsequent trials.
                    arms = configs if (index + trial) % 2 == 0 else reversed(configs)
                    for name in arms:
                        mcp_config, tools = configs[name]
                        trial_suffix = f"_{trial:03d}" if iterations > 1 else ""
                        raw_path = destination / f"{index:03d}_{name}{trial_suffix}_raw.json"
                        trials_by_config[name][spec.eval_id].append(_run_claude(
                            prompt, Path(tmpdir) / name, target, mcp_config, tools,
                            system_prompt, env, raw_path,
                        ))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f"A/B benchmark failed: {exc}. Raw artifacts: {destination}") from exc

    results = {
        name: {eval_id: trials[0] if iterations == 1 else trials for eval_id, trials in by_eval.items()}
        for name, by_eval in trials_by_config.items()
    }
    (destination / "raw_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = run_benchmark_from_results(archived_spec, destination, results, judge=judge_from_env())
    formatted = format_benchmark_summary(summary, claude_cli_tokens=True)
    (destination / "benchmark_stdout.txt").write_text(formatted + "\n", encoding="utf-8")
    print(formatted)


if __name__ == "__main__":
    main()
