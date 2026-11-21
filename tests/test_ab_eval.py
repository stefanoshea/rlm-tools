import json
import subprocess

import pytest

from evals import run_ab_eval


@pytest.mark.parametrize(("use_gate", "iterations"), [(False, 1), (True, 2)])
def test_ab_eval_records_both_arms_with_identical_prompts(tmp_path, monkeypatch, use_gate, iterations):
    target = tmp_path / "fixture with spaces"
    target.mkdir()
    spec = tmp_path / "cases.json"
    spec.write_text(json.dumps([{
        "eval_id": "example",
        "prompt": "Inspect {target_path}.\nReport the result.",
        "assertions": [{"id": "answer", "kind": "contains_all", "values": ["output"]}],
    }]))
    monkeypatch.setattr(run_ab_eval, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_ab_eval.shutil, "which", lambda _: "/stub/claude")
    monkeypatch.setenv("RLM_AB_SPEC_PATH", str(spec))
    monkeypatch.setenv("RLM_AB_TARGET_PATH", str(target))
    monkeypatch.setenv("RLM_AB_MODEL", "example-model")
    monkeypatch.setenv("RLM_AB_ITERATIONS", str(iterations))
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("RLM_AB_SEMANTIC", str(int(use_gate)))
    monkeypatch.setenv("RLM_JUDGE", "0")
    calls = []
    response = 'Two lines of output.\n"Second line."'

    def stub_claude(command, *, input, stdout, cwd, env, **kwargs):
        assert command[command.index("--model") + 1] == "example-model"
        assert command[command.index("--add-dir") + 1] == str(target)
        assert ("TYPESAFE_API_KEY" in env) == use_gate
        assert env["RLM_RECURSE_GATE"] == str(int(use_gate))
        if cwd.name == "with_rlm":
            config = json.loads((cwd / ".mcp.json").read_text())
            assert ("semantic" in config["mcpServers"]["rlm-tools"]["args"]) == use_gate
        tools = command[command.index("--tools") + 1]
        calls.append((cwd.name, tools, input))
        json.dump({
            "result": response,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
            },
        }, stdout)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_ab_eval.subprocess, "run", stub_claude)
    run_ab_eval.main()

    prompt = f"Inspect {target}.\nReport the result."
    first_pair = [("with_rlm", "", prompt), ("without_rlm", "Read,Grep,Glob", prompt)]
    assert calls == (first_pair if iterations == 1 else [*first_pair, *reversed(first_pair)])
    results, = (tmp_path / "evals" / "ab_results").iterdir()
    assert run_ab_eval.load_eval_specs(results / "benchmark_spec.json") == run_ab_eval.load_eval_specs(spec)
    raw_results = json.loads((results / "raw_results.json").read_text())
    for arm in ("with_rlm", "without_rlm"):
        raw_case = raw_results[arm]["example"]
        if iterations == 1:
            assert isinstance(raw_case, dict)
            assert (results / f"001_{arm}_raw.json").is_file()
        else:
            assert len(raw_case) == iterations
        for trial in range(1, iterations + 1):
            if iterations > 1:
                assert (results / f"001_{arm}_{trial:03d}_raw.json").is_file()
            run = results / arm / "example" / f"run_{trial:03d}"
            output = json.loads((run / "output.json").read_text())
            assert output["output"] == response
            assert output["prompt"] == prompt
            timing = json.loads((run / "timing.json").read_text())
            assert timing["input_tokens"] == 60
            assert timing["output_tokens"] == 5
            assert timing["total_tokens"] == 65
    benchmark = json.loads((results / "benchmark.json").read_text())
    assert benchmark["matched_coverage"]
    assert {config["runs"] for config in benchmark["configs"]} == {iterations}
    summary = (results / "benchmark_stdout.txt").read_text()
    assert "Benchmark Summary" in summary
    assert "CLI Tokens" in summary
    assert "separate model usage by MCP helpers and the optional judge" in summary


def test_missing_claude_usage_preserves_the_answer(tmp_path, monkeypatch):
    def stub_claude(command, *, stdout, **kwargs):
        json.dump({"result": "an answer"}, stdout)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_ab_eval.subprocess, "run", stub_claude)
    result = run_ab_eval._run_claude(
        "prompt", tmp_path, tmp_path, {}, "Read", "system prompt", {}, tmp_path / "raw.json",
    )
    assert result["output"] == "an answer"
    assert result["total_tokens"] is None
    assert result["duration_ms"] >= 0


@pytest.mark.parametrize(("tools", "denied"), [("", "mcp__rlm-tools__rlm_start"), ("Read,Grep,Glob", "Grep")])
def test_denied_benchmark_tool_invalidates_an_otherwise_successful_trial(tmp_path, monkeypatch, tools, denied):
    raw = tmp_path / "raw.json"

    def stub_claude(command, *, stdout, **kwargs):
        json.dump({"result": "plausible answer", "is_error": False,
                   "permission_denials": [{"tool_name": denied}]}, stdout)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_ab_eval.subprocess, "run", stub_claude)
    with pytest.raises(ValueError, match="benchmark tool access denied"):
        run_ab_eval._run_claude("prompt", tmp_path, tmp_path, {}, tools, "system", {}, raw)
    assert json.loads(raw.read_text())["permission_denials"][0]["tool_name"] == denied


@pytest.mark.parametrize("value", ["0", "-2", "1.5"])
def test_ab_eval_rejects_invalid_iteration_counts(monkeypatch, value):
    monkeypatch.setenv("RLM_AB_ITERATIONS", value)
    with pytest.raises(SystemExit, match="positive integer"):
        run_ab_eval.main()


def test_failed_claude_invocation_is_not_scored(tmp_path, monkeypatch):
    def stub_claude(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stderr="Could not start")

    monkeypatch.setattr(run_ab_eval.subprocess, "run", stub_claude)
    with pytest.raises(ValueError, match="Could not start"):
        run_ab_eval._run_claude(
            "prompt", tmp_path, tmp_path, {}, "Read", "system prompt", {},
            tmp_path / "raw.json",
        )


@pytest.mark.parametrize("denials", [None, {"tool_name": "Read"}, [{"tool_name": []}]])
def test_malformed_permission_report_invalidates_trial(tmp_path, monkeypatch, denials):
    def stub_claude(command, *, stdout, **kwargs):
        json.dump({"result": "example", "permission_denials": denials}, stdout)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_ab_eval.subprocess, "run", stub_claude)
    with pytest.raises(ValueError, match="Invalid permission_denials"):
        run_ab_eval._run_claude("prompt", tmp_path, tmp_path, {}, "Read", "system", {}, tmp_path / "raw.json")
