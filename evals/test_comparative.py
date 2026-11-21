import json as _json
import pathlib
import re

from evals.conftest import DEFAULT_PY_CORPUS_PATH, DEFAULT_TS_CORPUS_PATH, execute
from evals.metrics import TaskMetric, StepMetric, Timer, format_comparison


PY_CORPUS_PATH = pathlib.Path(DEFAULT_PY_CORPUS_PATH)
TS_CORPUS_PATH = pathlib.Path(DEFAULT_TS_CORPUS_PATH)


def _timed_execute(sandbox, code, step_name=""):
    with Timer() as t:
        result = execute(sandbox, code)
    tool_response = _json.dumps({
        "stdout": result["stdout"],
        "error": result["error"],
    })
    metric = StepMetric(
        name=step_name,
        agent_output_chars=len(code),
        tool_response_chars=len(tool_response),
        elapsed_seconds=t.elapsed,
        error=result.get("error"),
    )
    return result, metric


def _baseline_glob(directory: pathlib.Path, pattern: str) -> str:
    matches = sorted(
        str(f.relative_to(directory))
        for f in directory.glob(pattern)
        if f.is_file()
    )
    return "\n".join(matches)


def _baseline_grep(directory: pathlib.Path, pattern: str, path: str = ".") -> str:
    compiled = re.compile(pattern)
    lines = []
    target = directory / path
    candidates = [target] if target.is_file() else sorted(target.rglob("*"))
    for f in candidates:
        if not f.is_file() or any(part.startswith(".") for part in f.relative_to(directory).parts):
            continue
        try:
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if compiled.search(line):
                    lines.append(f"{f.relative_to(directory)}:{i}: {line.strip()}")
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(lines)


def _baseline_read(filepath: pathlib.Path) -> str:
    return filepath.read_text()


def _baseline_step(name, agent_param_chars, content, elapsed):
    return StepMetric(
        name=name,
        agent_output_chars=agent_param_chars,
        tool_response_chars=len(content),
        elapsed_seconds=elapsed,
    )


class TestPythonSearchComparison:
    def test_context_cost(self, sandbox):
        rlm = TaskMetric(task_name="grep UserService across python corpus (with RLM)")
        code = (
            "r=grep('UserService')\n"
            "print(f\"{len(r)} matches across {len({m['file'] for m in r})} files\")"
        )
        result, step = _timed_execute(sandbox, code, step_name="grep -> summary")
        rlm.steps.append(step)

        baseline = TaskMetric(task_name="grep UserService across python corpus (baseline)")
        with Timer() as t:
            output = _baseline_grep(PY_CORPUS_PATH, "UserService")
        matches = output.splitlines()
        files = {row.split(":", 1)[0] for row in matches}
        assert result["stdout"].strip() == f"{len(matches)} matches across {len(files)} files"
        baseline.steps.append(_baseline_step(
            "Grep tool -> all matching lines",
            agent_param_chars=32,
            content=output,
            elapsed=t.elapsed,
        ))

        print(f"\n{format_comparison(rlm, baseline)}")
        assert not rlm.had_errors
        assert rlm.total_context_chars < baseline.total_context_chars


class TestFileReadComparison:
    TARGET_FILES = [
        "app.py",
        "services/user_service.py",
        "services/auth_service.py",
    ]

    def test_context_cost(self, sandbox):
        rlm = TaskMetric(task_name="read 3 fixture files (with RLM)")

        files_json = _json.dumps(self.TARGET_FILES)
        load_code = (
            f"target_files = {files_json}\n"
            "contents = {f: read_file(f) for f in target_files}\n"
            "print(f'Loaded {len(contents)} files')"
        )
        _, s1 = _timed_execute(sandbox, load_code, step_name="load files into sandbox")
        rlm.steps.append(s1)

        summarize_code = (
            "for f in target_files:\n"
            "    content = contents[f]\n"
            "    print(f'{f}: {len(content.splitlines())} lines')"
        )
        result, s2 = _timed_execute(sandbox, summarize_code, step_name="summarize loaded files")
        rlm.steps.append(s2)

        baseline = TaskMetric(task_name="read 3 fixture files (baseline)")
        expected = []
        for f in self.TARGET_FILES:
            with Timer() as t:
                content = _baseline_read(PY_CORPUS_PATH / f)
            expected.append(f"{f}: {len(content.splitlines())} lines")
            baseline.steps.append(_baseline_step(
                f"Read {f}",
                agent_param_chars=18,
                content=content,
                elapsed=t.elapsed,
            ))

        assert result["stdout"].splitlines() == expected
        print(f"\n{format_comparison(rlm, baseline)}")
        assert not rlm.had_errors
        assert rlm.total_context_chars < baseline.total_context_chars


class TestMultiStepExplorationComparison:
    def test_context_cost(self, sandbox):
        rlm = TaskMetric(task_name="python corpus exploration (with RLM)")

        glob_code = (
            "all_python = glob_files('**/*.py')\n"
            "print(f'{len(all_python)} python files')\n"
            "for f in sorted(all_python):\n"
            "    print(f)"
        )
        _, s1 = _timed_execute(sandbox, glob_code, step_name="glob python files")
        rlm.steps.append(s1)

        read_code = (
            "service_files = [f for f in all_python if f.startswith('services/')]\n"
            "for f in service_files:\n"
            "    content = read_file(f)\n"
            "    print(f'{f}: {len(content)} chars')"
        )
        result, s2 = _timed_execute(sandbox, read_code, step_name="read service files")
        rlm.steps.append(s2)

        baseline = TaskMetric(task_name="python corpus exploration (baseline)")
        with Timer() as t:
            all_python = sorted(
                str(f.relative_to(PY_CORPUS_PATH))
                for f in PY_CORPUS_PATH.rglob("*.py")
                if f.is_file()
            )
            glob_output = "\n".join(all_python)
        baseline.steps.append(_baseline_step(
            "Glob **/*.py",
            agent_param_chars=20,
            content=glob_output,
            elapsed=t.elapsed,
        ))

        with Timer() as t:
            read_output = ""
            expected = []
            for f in all_python:
                if f.startswith("services/"):
                    content = _baseline_read(PY_CORPUS_PATH / f)
                    read_output += content
                    expected.append(f"{f}: {len(content)} chars")
        baseline.steps.append(_baseline_step(
            "Read service files",
            agent_param_chars=24,
            content=read_output,
            elapsed=t.elapsed,
        ))

        assert sorted(result["stdout"].splitlines()) == sorted(expected)
        print(f"\n{format_comparison(rlm, baseline)}")
        assert not rlm.had_errors
        assert rlm.total_context_chars < baseline.total_context_chars


class TestTypeScriptComparison:
    def test_context_cost(self, ts_sandbox):
        rlm = TaskMetric(task_name="grep reducer dependencies (with RLM)")
        code = (
            "deps = grep('apiClient|clock|logger', 'src/dashboard.ts')\n"
            "line_numbers = [str(row['line_number']) for row in deps]\n"
            "print(f\"{len(deps)} dependency lines at {', '.join(line_numbers)}\")"
        )
        result, step = _timed_execute(ts_sandbox, code, step_name="grep ts dependencies")
        rlm.steps.append(step)

        baseline = TaskMetric(task_name="grep reducer dependencies (baseline)")
        with Timer() as t:
            output = _baseline_grep(TS_CORPUS_PATH, r"apiClient|clock|logger", "src/dashboard.ts")
        numbers = [line.split(":", 2)[1] for line in output.splitlines()]
        assert result["stdout"].strip() == f"{len(numbers)} dependency lines at {', '.join(numbers)}"
        baseline.steps.append(_baseline_step(
            "Grep dependency lines",
            agent_param_chars=36,
            content=output,
            elapsed=t.elapsed,
        ))

        print(f"\n{format_comparison(rlm, baseline)}")
        assert not rlm.had_errors
        assert rlm.total_context_chars < baseline.total_context_chars
