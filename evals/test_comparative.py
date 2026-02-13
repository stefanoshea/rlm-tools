"""
Comparative evals: with RLM Tools vs without (baseline).

"Baseline" simulates what an LLM agent receives in its context window when
using standard tools (Read, Grep, Glob) -- full content returned every time,
no persistent state between calls.

"With RLM" measures what the LLM actually receives -- data stays in the
sandbox and only print() output enters the context window. Variables persist
between calls so intermediate results don't need to be re-fetched.

Key metric: total chars entering the LLM's context window for the same task.
We now track both directions of every turn:
  - agent_output_chars: what the agent writes to invoke the tool
  - tool_response_chars: what the tool sends back

Dataset: apple/app (1,800+ Swift files, full iOS app).
"""

import json as _json
import pathlib
import re

from conftest import APPLE_PROJECT_PATH, execute
from metrics import TaskMetric, StepMetric, Timer, format_comparison


APP_PROJECT_PATH = pathlib.Path(APPLE_PROJECT_PATH) / "app"
APP_SRC = APP_PROJECT_PATH / "app"


def _timed_execute(sandbox, code, step_name=""):
    """Execute code in the sandbox and return (result_dict, StepMetric).

    agent_output_chars = len(code) -- the Python the agent writes.
    tool_response_chars = len(json.dumps(result)) -- the full JSON envelope
    (stdout + error + variables list) returned by the tool.
    """
    with Timer() as t:
        result = execute(sandbox, code)
    tool_response = _json.dumps(result)
    metric = StepMetric(
        name=step_name,
        agent_output_chars=len(code),
        tool_response_chars=len(tool_response),
        elapsed_seconds=t.elapsed,
        error=result.get("error"),
    )
    return result, metric


# ---------------------------------------------------------------------------
# Baseline helpers -- simulate what standard tools return into LLM context
# ---------------------------------------------------------------------------

def _baseline_glob(directory: pathlib.Path, pattern: str) -> str:
    """Simulate Glob tool: returns all matching file paths, one per line."""
    matches = sorted(
        str(f.relative_to(APP_PROJECT_PATH))
        for f in directory.glob(pattern)
        if f.is_file()
    )
    return "\n".join(matches)


def _baseline_grep(directory: pathlib.Path, pattern: str) -> str:
    """Simulate Grep tool: returns file:line:content for every match."""
    compiled = re.compile(pattern)
    lines = []
    for f in sorted(directory.rglob("*")):
        if not f.is_file() or any(p.startswith(".") for p in f.relative_to(APP_PROJECT_PATH).parts):
            continue
        try:
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if compiled.search(line):
                    lines.append(f"{f.relative_to(APP_PROJECT_PATH)}:{i}: {line.strip()}")
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(lines)


def _baseline_read(filepath: pathlib.Path) -> str:
    """Simulate Read tool: returns full file content."""
    return filepath.read_text()


def _baseline_step(name, agent_param_chars, content, elapsed):
    """Build a StepMetric for a baseline tool call.

    agent_param_chars: small fixed cost for the tool invocation params.
    content: what the tool returns (full file contents, grep output, etc.).
    """
    return StepMetric(
        name=name,
        agent_output_chars=agent_param_chars,
        tool_response_chars=len(content),
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# 1. Grep across the full app -- 500+ matching lines
# ---------------------------------------------------------------------------

class TestGrepComparison:
    """Grep 'import UIKit' across the entire app source (500+ matches).

    Baseline: 1 turn -- Grep tool returns every matching line.
    RLM: 1 turn -- results stay in sandbox, agent prints summary + sample paths.
    """

    PATTERN = "import UIKit"

    def test_same_results(self, sandbox):
        result = execute(
            sandbox,
            f"import json\nresults = grep('{self.PATTERN}', 'app')\nprint(json.dumps(sorted(set(r['file'] for r in results))))",
        )
        rlm_files = set(_json.loads(result["stdout"]))

        baseline_output = _baseline_grep(APP_SRC, self.PATTERN)
        baseline_files = set()
        for line in baseline_output.splitlines():
            if ":" in line:
                baseline_files.add(line.split(":")[0])

        assert rlm_files == baseline_files, (
            f"Results differ:\n  RLM only: {rlm_files - baseline_files}\n  Baseline only: {baseline_files - rlm_files}"
        )

    def test_context_cost(self, sandbox):
        rlm = TaskMetric(task_name=f"grep '{self.PATTERN}' across app (with RLM)")
        code = (
            f"results = grep('{self.PATTERN}', 'app')\n"
            "files = sorted(set(r['file'] for r in results))\n"
            "print(f'{len(results)} matches across {len(files)} files')\n"
            "for f in files[:15]:\n"
            "    print(f'  {f}')"
        )
        _, step = _timed_execute(sandbox, code, step_name="grep -> summary + file list")
        rlm.steps.append(step)

        baseline = TaskMetric(task_name=f"grep '{self.PATTERN}' across app (baseline)")
        with Timer() as t:
            output = _baseline_grep(APP_SRC, self.PATTERN)
        baseline.steps.append(_baseline_step(
            "Grep tool -> all matching lines",
            agent_param_chars=50,
            content=output,
            elapsed=t.elapsed,
        ))

        print(f"\n{format_comparison(rlm, baseline)}")
        assert not rlm.had_errors
        assert rlm.total_context_chars < baseline.total_context_chars


# ---------------------------------------------------------------------------
# 2. Read 10 large files -- ~1.5M chars baseline
# ---------------------------------------------------------------------------

class TestFileReadComparison:
    """Read the 10 largest source files and summarize them.

    Baseline: 10 turns -- one Read call per file, each returns full content.
    RLM: 2 turns -- load all into sandbox, then summarize from memory.
    """

    TARGET_FILES = [
        "app/User/Domain/AppDomainUser+Shifts.swift",
        "app/User/Domain/AppDomainUser+ShiftTrading.swift",
        "app/ScheduleTab/AppScheduleViewController.swift",
        "app/Dashboard/DashboardInteractor.swift",
        "app/User/Domain/AppDomainUser+Employees.swift",
        "app/Shared/Employees/AppEmployeeDetailsViewController.swift",
        "app/User/Domain/AppDomainUser+Leaves.swift",
        "app/Dashboard/DashboardViewController.swift",
        "app/Dashboard/DashboardBuilder.swift",
        "app/TabBar/AppTabBarController.swift",
    ]

    def test_same_content(self, sandbox):
        for target in self.TARGET_FILES[:3]:
            result = execute(sandbox, f"content = read_file('{target}')\nprint(content[:200])")
            assert result["error"] is None
            native_start = (APP_PROJECT_PATH / target).read_text()[:200]
            assert result["stdout"].strip() == native_start.strip(), (
                f"Content mismatch for {target}"
            )

    def test_context_cost(self, sandbox):
        rlm = TaskMetric(task_name="read 10 large files (with RLM)")

        files_json = _json.dumps(self.TARGET_FILES)
        load_code = (
            f"_large_files = {files_json}\n"
            "for f in _large_files:\n"
            "    key = f.split('/')[-1].replace('.swift','').replace('+','_')\n"
            "    globals()[key] = read_file(f)\n"
            "print(f'Loaded {len(_large_files)} files into sandbox')"
        )
        _, s1 = _timed_execute(sandbox, load_code, step_name="load 10 files into sandbox")
        rlm.steps.append(s1)

        summarize_code = (
            "for f in _large_files:\n"
            "    key = f.split('/')[-1].replace('.swift','').replace('+','_')\n"
            "    c = globals()[key]\n"
            "    lines = c.splitlines()\n"
            "    imports = [l.strip() for l in lines if l.strip().startswith('import ')]\n"
            "    classes = [l.strip() for l in lines if l.strip().startswith('class ') or l.strip().startswith('protocol ')]\n"
            "    print(f'{f.split(\"/\")[-1]}: {len(c):,} chars, {len(lines):,} lines, imports={imports}, types={classes[:5]}')"
        )
        _, s2 = _timed_execute(sandbox, summarize_code, step_name="summarize loaded files")
        rlm.steps.append(s2)

        baseline = TaskMetric(task_name="read 10 large files (baseline)")
        for f in self.TARGET_FILES:
            with Timer() as t:
                content = _baseline_read(APP_PROJECT_PATH / f)
            baseline.steps.append(_baseline_step(
                f"Read {f.split('/')[-1][:25]}",
                agent_param_chars=40,
                content=content,
                elapsed=t.elapsed,
            ))

        print(f"\n{format_comparison(rlm, baseline)}")
        assert rlm.total_context_chars < baseline.total_context_chars


# ---------------------------------------------------------------------------
# 3. Multi-step exploration -- glob 1,800+ files, filter, read
# ---------------------------------------------------------------------------

class TestMultiStepExplorationComparison:
    """Explore the full app: glob all Swift -> filter by module -> read samples.

    Baseline: 2 turns -- Glob returns all paths, then Read returns file contents.
    RLM: 3 turns -- glob + group + read, but only summaries enter context.
    """

    def test_context_cost(self, sandbox):
        rlm = TaskMetric(task_name="full-app exploration (with RLM)")

        glob_code = (
            "all_app_swift = glob_files('**/*.swift')\n"
            "print(f'{len(all_app_swift)} Swift files')\n"
            "for f in sorted(all_app_swift)[:15]:\n"
            "    print(f'  {f}')"
        )
        _, s1 = _timed_execute(sandbox, glob_code, step_name="1. glob all Swift")
        rlm.steps.append(s1)

        group_code = (
            "by_module = {}\n"
            "for f in all_app_swift:\n"
            "    parts = f.split('/')\n"
            "    mod = parts[1] if len(parts) > 2 else 'root'\n"
            "    by_module.setdefault(mod, []).append(f)\n"
            "for mod in sorted(by_module, key=lambda m: -len(by_module[m]))[:10]:\n"
            "    print(f'{mod}: {len(by_module[mod])} files')"
        )
        _, s2 = _timed_execute(sandbox, group_code, step_name="2. group by module")
        rlm.steps.append(s2)

        read_code = (
            "schedule_files = by_module.get('ScheduleTab', [])\n"
            "for f in schedule_files[:5]:\n"
            "    c = read_file(f)\n"
            "    lines = c.splitlines()\n"
            "    imports = [l.strip() for l in lines if l.strip().startswith('import ')]\n"
            "    classes = [l.strip() for l in lines if l.strip().startswith('class ') or l.strip().startswith('protocol ')]\n"
            "    print(f'{f.split(\"/\")[-1]}: {len(lines)} lines, imports={imports}, types={classes[:3]}')"
        )
        _, s3 = _timed_execute(sandbox, read_code, step_name="3. read module samples")
        rlm.steps.append(s3)

        baseline = TaskMetric(task_name="full-app exploration (baseline)")

        with Timer() as t:
            all_swift = sorted(
                str(f.relative_to(APP_PROJECT_PATH))
                for f in APP_PROJECT_PATH.rglob("*.swift")
                if f.is_file() and not any(p.startswith(".") for p in f.relative_to(APP_PROJECT_PATH).parts)
            )
            glob_output = "\n".join(all_swift)
        baseline.steps.append(_baseline_step(
            "1. Glob **/*.swift",
            agent_param_chars=30,
            content=glob_output,
            elapsed=t.elapsed,
        ))

        with Timer() as t:
            schedule = [f for f in all_swift if "/ScheduleTab/" in f]
            read_output = ""
            for f in schedule[:5]:
                read_output += _baseline_read(APP_PROJECT_PATH / f)
        baseline.steps.append(_baseline_step(
            "2. Read 5 module files",
            agent_param_chars=40,
            content=read_output,
            elapsed=t.elapsed,
        ))

        print(f"\n{format_comparison(rlm, baseline)}")
        assert rlm.total_context_chars < baseline.total_context_chars


# ---------------------------------------------------------------------------
# 4. Grep + Read chain -- search protocols, then read definitions
# ---------------------------------------------------------------------------

class TestGrepThenReadComparison:
    """Find all protocol definitions, then read the top files.

    Baseline: 2 turns -- Grep returns 900+ matching lines, then Read returns files.
    RLM: 2 turns -- grep + read, but data stays in sandbox.
    """

    def test_context_cost(self, sandbox):
        rlm = TaskMetric(task_name="find protocols + read (with RLM)")

        grep_code = (
            "protocols = grep('protocol ', 'app')\n"
            "files = sorted(set(r['file'] for r in protocols))\n"
            "print(f'{len(protocols)} protocol lines across {len(files)} files')\n"
            "for f in files[:20]:\n"
            "    count = sum(1 for p in protocols if p['file'] == f)\n"
            "    print(f'  {f}: {count} protocols')"
        )
        _, s1 = _timed_execute(sandbox, grep_code, step_name="1. grep protocols")
        rlm.steps.append(s1)

        read_code = (
            "top_files = sorted(files, key=lambda f: sum(1 for p in protocols if p['file'] == f), reverse=True)[:5]\n"
            "for f in top_files:\n"
            "    c = read_file(f)\n"
            "    count = sum(1 for p in protocols if p['file'] == f)\n"
            "    lines = c.splitlines()\n"
            "    protos = [l.strip() for l in lines if 'protocol ' in l and '{' in l]\n"
            "    print(f'{f.split(\"/\")[-1]}: {count} protocols, {len(lines)} lines')\n"
            "    for p in protos[:5]:\n"
            "        print(f'    {p}')"
        )
        _, s2 = _timed_execute(sandbox, read_code, step_name="2. read top 5 files")
        rlm.steps.append(s2)

        baseline = TaskMetric(task_name="find protocols + read (baseline)")

        with Timer() as t:
            grep_output = _baseline_grep(APP_SRC, r"protocol ")
        baseline.steps.append(_baseline_step(
            "1. Grep all protocols",
            agent_param_chars=50,
            content=grep_output,
            elapsed=t.elapsed,
        ))

        with Timer() as t:
            grep_files: dict[str, int] = {}
            for line in grep_output.splitlines():
                f = line.split(":")[0]
                grep_files[f] = grep_files.get(f, 0) + 1
            top = sorted(grep_files, key=grep_files.get, reverse=True)[:5]
            read_output = ""
            for f in top:
                read_output += _baseline_read(APP_PROJECT_PATH / f)
        baseline.steps.append(_baseline_step(
            "2. Read top 5 files",
            agent_param_chars=40,
            content=read_output,
            elapsed=t.elapsed,
        ))

        print(f"\n{format_comparison(rlm, baseline)}")
        assert rlm.total_context_chars < baseline.total_context_chars


# ---------------------------------------------------------------------------
# 5. Find usages across full codebase -- broad pattern
# ---------------------------------------------------------------------------

class TestFindUsagesComparison:
    """Find all @objc func declarations across the full app.

    Baseline: 1 turn -- Grep returns every matching line.
    RLM: 1 turn -- results in sandbox, agent prints grouped summary + paths.
    """

    def test_context_cost(self, sandbox):
        rlm = TaskMetric(task_name="find @objc funcs (with RLM)")

        code = (
            "objc = grep('@objc func', 'app')\n"
            "files = sorted(set(r['file'] for r in objc))\n"
            "print(f'{len(objc)} @objc funcs across {len(files)} files:')\n"
            "for f in files[:15]:\n"
            "    count = sum(1 for r in objc if r['file'] == f)\n"
            "    funcs = [r['line'].strip() for r in objc if r['file'] == f]\n"
            "    print(f'  {f}:')\n"
            "    for fn in funcs[:3]:\n"
            "        print(f'    {fn}')"
        )
        _, s1 = _timed_execute(sandbox, code, step_name="grep + group by file")
        rlm.steps.append(s1)

        baseline = TaskMetric(task_name="find @objc funcs (baseline)")
        with Timer() as t:
            output = _baseline_grep(APP_SRC, r"@objc func")
        baseline.steps.append(_baseline_step(
            "Grep tool -> all matching lines",
            agent_param_chars=50,
            content=output,
            elapsed=t.elapsed,
        ))

        print(f"\n{format_comparison(rlm, baseline)}")
        assert rlm.total_context_chars < baseline.total_context_chars


# ---------------------------------------------------------------------------
# 6. Understand a module -- tree + glob + read key files
# ---------------------------------------------------------------------------

class TestModuleUnderstandingComparison:
    """Understand the ScheduleTab module: tree it, find key files, read them.

    Baseline: 3 turns -- tree output + Glob file list + Read file contents.
    RLM: 3 turns -- same workflow but data stays in sandbox.
    """

    def test_context_cost(self, sandbox):
        rlm = TaskMetric(task_name="understand ScheduleTab (with RLM)")

        tree_code = "sched_tree = tree('app/ScheduleTab', max_depth=2)\nprint(sched_tree)"
        _, s1 = _timed_execute(sandbox, tree_code, step_name="1. tree module")
        rlm.steps.append(s1)

        glob_code = (
            "sched_swift = glob_files('app/ScheduleTab/**/*.swift')\n"
            "print(f'{len(sched_swift)} Swift files')\n"
            "for f in sorted(sched_swift)[:15]:\n"
            "    print(f'  {f}')"
        )
        _, s2 = _timed_execute(sandbox, glob_code, step_name="2. glob module files")
        rlm.steps.append(s2)

        read_code = (
            "key = [f for f in sched_swift if 'ViewController' in f or 'Builder' in f or 'Interactor' in f][:5]\n"
            "for f in key:\n"
            "    c = read_file(f)\n"
            "    lines = c.splitlines()\n"
            "    imports = [l.strip() for l in lines if l.strip().startswith('import ')]\n"
            "    classes = [l.strip() for l in lines if l.strip().startswith('class ') or l.strip().startswith('protocol ')]\n"
            "    print(f'{f.split(\"/\")[-1]}: {len(lines)} lines, imports={imports}')\n"
            "    for cls in classes[:3]:\n"
            "        print(f'    {cls}')"
        )
        _, s3 = _timed_execute(sandbox, read_code, step_name="3. read key files")
        rlm.steps.append(s3)

        baseline = TaskMetric(task_name="understand ScheduleTab (baseline)")

        sched_dir = APP_PROJECT_PATH / "app" / "ScheduleTab"

        with Timer() as t:
            from rlm_tools.helpers import make_helpers
            helpers, _ = make_helpers(str(APP_PROJECT_PATH))
            tree_output = helpers["tree"]("app/ScheduleTab", max_depth=2)
        baseline.steps.append(_baseline_step(
            "1. tree output",
            agent_param_chars=60,
            content=tree_output,
            elapsed=t.elapsed,
        ))

        with Timer() as t:
            glob_output = _baseline_glob(sched_dir, "**/*.swift")
        baseline.steps.append(_baseline_step(
            "2. Glob file list",
            agent_param_chars=45,
            content=glob_output,
            elapsed=t.elapsed,
        ))

        with Timer() as t:
            sched_files = glob_output.splitlines()
            key_files = [f for f in sched_files if "ViewController" in f or "Builder" in f or "Interactor" in f][:5]
            read_output = ""
            for f in key_files:
                read_output += _baseline_read(APP_PROJECT_PATH / f)
        baseline.steps.append(_baseline_step(
            "3. Read key files",
            agent_param_chars=40,
            content=read_output,
            elapsed=t.elapsed,
        ))

        print(f"\n{format_comparison(rlm, baseline)}")
        assert rlm.total_context_chars < baseline.total_context_chars


# ---------------------------------------------------------------------------
# 7. Session metadata
# ---------------------------------------------------------------------------

class TestSessionMetadata:
    def test_start_returns_valid_metadata(self, apple_path):
        import json
        from rlm_tools.server import _rlm_start, _rlm_end

        result = json.loads(_rlm_start(path=str(apple_path / "app"), query="metadata test"))
        try:
            assert "metadata" in result
            assert result["metadata"]["total_files"] > 1000
            assert ".swift" in result["metadata"]["file_types"]
            assert result["metadata"]["total_lines"] > 100_000
        finally:
            session_id = result.get("session_id")
            if session_id:
                _rlm_end(session_id)
