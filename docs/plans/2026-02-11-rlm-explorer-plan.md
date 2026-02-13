# RLM Tools Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python MCP server that provides Claude Code with a persistent, sandboxed Python REPL for large codebase exploration.

**Architecture:** FastMCP server with session-managed subprocess sandboxes. Each session maintains a persistent Python namespace with built-in helpers for file exploration and sub-LLM queries.

**Tech Stack:** Python 3.10+, mcp[cli] (FastMCP), anthropic SDK, hatchling build

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/rlm_tools/__init__.py`
- Create: `src/rlm_tools/server.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "rlm-tools"
version = "0.1.0"
description = "RLM-style persistent Python REPL for Claude Code codebase exploration"
requires-python = ">=3.10"
dependencies = [
    "mcp[cli]>=1.2.0",
    "anthropic>=0.40.0",
]

[project.scripts]
rlm-tools = "rlm_tools.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rlm_tools"]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
```

**Step 2: Create `src/rlm_tools/__init__.py`**

```python
"""RLM Tools - Persistent Python REPL for Claude Code codebase exploration."""
```

**Step 3: Create minimal server skeleton in `src/rlm_tools/server.py`**

```python
import logging
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("rlm-tools")


@mcp.tool()
async def rlm_start(path: str, query: str) -> str:
    """Start an RLM exploration session."""
    return "not implemented"


@mcp.tool()
async def rlm_execute(session_id: str, code: str) -> str:
    """Execute Python code in an RLM session."""
    return "not implemented"


@mcp.tool()
async def rlm_end(session_id: str) -> str:
    """End an RLM exploration session."""
    return "not implemented"


def main():
    mcp.run(transport="stdio")
```

**Step 4: Verify the project installs and entry point works**

Run: `cd /path/to/rlm-tools && uv sync && uv run rlm-tools --help`
Expected: Server starts (or shows help), proving the entry point is wired up.

**Step 5: Initialize git and commit**

```bash
git init
git add .
git commit -m "feat: project scaffolding with FastMCP server skeleton"
```

---

### Task 2: Session Manager

**Files:**
- Create: `src/rlm_tools/session.py`
- Create: `tests/test_session.py`

**Step 1: Write the failing test**

```python
# tests/test_session.py
import pytest
from rlm_tools.session import SessionManager


def test_create_session():
    manager = SessionManager(max_sessions=5)
    session_id = manager.create(path="/tmp", query="test query")
    assert session_id is not None
    assert manager.get(session_id) is not None


def test_max_sessions_enforced():
    manager = SessionManager(max_sessions=2)
    manager.create(path="/tmp", query="q1")
    manager.create(path="/tmp", query="q2")
    with pytest.raises(RuntimeError, match="max sessions"):
        manager.create(path="/tmp", query="q3")


def test_end_session():
    manager = SessionManager(max_sessions=5)
    session_id = manager.create(path="/tmp", query="test")
    manager.end(session_id)
    assert manager.get(session_id) is None
```

**Step 2: Run tests to verify they fail**

Run: `cd /path/to/rlm-tools && uv run pytest tests/test_session.py -v`
Expected: FAIL - module not found

**Step 3: Implement SessionManager**

```python
# src/rlm_tools/session.py
import uuid
import time
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    path: str
    query: str
    namespace: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    max_output_chars: int = 10_000


class SessionManager:
    def __init__(self, max_sessions: int = 5, timeout_minutes: int = 10):
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._timeout_seconds = timeout_minutes * 60

    def create(self, path: str, query: str, max_output_chars: int = 10_000) -> str:
        self._cleanup_expired()
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(
                f"Cannot create session: max sessions ({self._max_sessions}) reached"
            )
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = Session(
            session_id=session_id,
            path=path,
            query=query,
            max_output_chars=max_output_chars,
        )
        return session_id

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session:
            session.last_used = time.time()
        return session

    def end(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used > self._timeout_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
```

**Step 4: Run tests to verify they pass**

Run: `cd /path/to/rlm-tools && uv run pytest tests/test_session.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/rlm_tools/session.py tests/test_session.py
git commit -m "feat: session manager with create/get/end and expiry"
```

---

### Task 3: Sandbox Execution Engine

**Files:**
- Create: `src/rlm_tools/sandbox.py`
- Create: `tests/test_sandbox.py`

**Step 1: Write the failing tests**

```python
# tests/test_sandbox.py
from rlm_tools.sandbox import Sandbox


def test_execute_simple_code():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    result = sandbox.execute("x = 2 + 2\nprint(x)")
    assert result.stdout.strip() == "4"
    assert result.error is None


def test_variables_persist_between_executions():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    sandbox.execute("my_var = 42")
    result = sandbox.execute("print(my_var)")
    assert result.stdout.strip() == "42"


def test_output_truncated():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=50)
    result = sandbox.execute("print('a' * 200)")
    assert len(result.stdout) <= 80  # 50 + truncation message


def test_blocked_imports():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    result = sandbox.execute("import subprocess")
    assert result.error is not None


def test_no_write_access():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    result = sandbox.execute("open('/tmp/evil.txt', 'w').write('hack')")
    assert result.error is not None


def test_list_variables():
    sandbox = Sandbox(base_path="/tmp", max_output_chars=10_000)
    sandbox.execute("foo = 1\nbar = 'hello'")
    variables = sandbox.list_variables()
    assert "foo" in variables
    assert "bar" in variables
```

**Step 2: Run tests to verify they fail**

Run: `cd /path/to/rlm-tools && uv run pytest tests/test_sandbox.py -v`
Expected: FAIL - module not found

**Step 3: Implement Sandbox**

```python
# src/rlm_tools/sandbox.py
import io
import contextlib
import builtins
from dataclasses import dataclass


ALLOWED_MODULES = frozenset({
    "re", "json", "collections", "math", "os.path", "pathlib",
    "fnmatch", "itertools", "functools", "operator", "string",
    "textwrap", "difflib", "statistics",
})

BLOCKED_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__", "breakpoint",
    "exit", "quit", "input",
})


@dataclass
class ExecutionResult:
    stdout: str
    error: str | None
    variables: list[str]


def _make_restricted_import(allowed: frozenset[str]):
    original_import = builtins.__import__

    def restricted_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level not in allowed:
            raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
        return original_import(name, *args, **kwargs)

    return restricted_import


class Sandbox:
    def __init__(self, base_path: str, max_output_chars: int = 10_000):
        self._base_path = base_path
        self._max_output_chars = max_output_chars
        self._namespace: dict = {}
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        safe_builtins = {
            k: v for k, v in builtins.__dict__.items()
            if k not in BLOCKED_BUILTINS
        }
        safe_builtins["__import__"] = _make_restricted_import(ALLOWED_MODULES)
        self._namespace["__builtins__"] = safe_builtins

    def execute(self, code: str) -> ExecutionResult:
        stdout_capture = io.StringIO()
        error = None

        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, self._namespace)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        stdout = stdout_capture.getvalue()
        if len(stdout) > self._max_output_chars:
            stdout = stdout[:self._max_output_chars] + "\n... [output truncated]"

        return ExecutionResult(
            stdout=stdout,
            error=error,
            variables=self.list_variables(),
        )

    def list_variables(self) -> list[str]:
        return [
            k for k in self._namespace
            if not k.startswith("_") and k != "__builtins__"
        ]
```

**Step 4: Run tests**

Run: `cd /path/to/rlm-tools && uv run pytest tests/test_sandbox.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add src/rlm_tools/sandbox.py tests/test_sandbox.py
git commit -m "feat: sandboxed Python execution with import/write restrictions"
```

---

### Task 4: Built-in Helpers (read_file, grep, glob_files, tree)

**Files:**
- Create: `src/rlm_tools/helpers.py`
- Create: `tests/test_helpers.py`
- Modify: `src/rlm_tools/sandbox.py` (inject helpers into namespace)

**Step 1: Write the failing tests**

```python
# tests/test_helpers.py
import os
import tempfile
from rlm_tools.helpers import make_helpers


def test_read_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "hello.txt")
        with open(test_file, "w") as f:
            f.write("hello world")

        helpers = make_helpers(tmpdir)
        content = helpers["read_file"]("hello.txt")
        assert content == "hello world"


def test_read_file_blocks_path_traversal():
    with tempfile.TemporaryDirectory() as tmpdir:
        helpers = make_helpers(tmpdir)
        try:
            helpers["read_file"]("../../etc/passwd")
            assert False, "Should have raised"
        except PermissionError:
            pass


def test_grep():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "code.py")
        with open(test_file, "w") as f:
            f.write("def hello():\n    pass\ndef world():\n    pass\n")

        helpers = make_helpers(tmpdir)
        results = helpers["grep"]("def.*hello")
        assert len(results) > 0
        assert "hello" in results[0]["line"]


def test_glob_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.py"), "w").close()
        open(os.path.join(tmpdir, "b.py"), "w").close()
        open(os.path.join(tmpdir, "c.txt"), "w").close()

        helpers = make_helpers(tmpdir)
        py_files = helpers["glob_files"]("**/*.py")
        assert len(py_files) == 2


def test_tree():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "src"))
        open(os.path.join(tmpdir, "src", "main.py"), "w").close()

        helpers = make_helpers(tmpdir)
        output = helpers["tree"]()
        assert "src" in output
        assert "main.py" in output
```

**Step 2: Run tests to verify they fail**

Run: `cd /path/to/rlm-tools && uv run pytest tests/test_helpers.py -v`
Expected: FAIL - module not found

**Step 3: Implement helpers**

```python
# src/rlm_tools/helpers.py
import os
import re
import pathlib
import fnmatch


def make_helpers(base_path: str) -> dict:
    base = pathlib.Path(base_path).resolve()

    def _resolve_safe(path: str) -> pathlib.Path:
        resolved = (base / path).resolve()
        if not str(resolved).startswith(str(base)):
            raise PermissionError(f"Access denied: path '{path}' escapes sandbox root")
        return resolved

    def read_file(path: str) -> str:
        target = _resolve_safe(path)
        return target.read_text(errors="replace")

    def grep(pattern: str, path: str = ".") -> list[dict]:
        target = _resolve_safe(path)
        compiled = re.compile(pattern)
        results = []
        search_paths = [target] if target.is_file() else target.rglob("*")
        for file_path in search_paths:
            if not file_path.is_file():
                continue
            try:
                for i, line in enumerate(file_path.read_text(errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        results.append({
                            "file": str(file_path.relative_to(base)),
                            "line_number": i,
                            "line": line.strip(),
                        })
            except (OSError, UnicodeDecodeError):
                continue
        return results

    def glob_files(pattern: str) -> list[str]:
        matches = list(base.glob(pattern))
        return [str(m.relative_to(base)) for m in matches if m.is_file()]

    def tree(path: str = ".", max_depth: int = 3) -> str:
        target = _resolve_safe(path)
        lines = []

        def _walk(dir_path: pathlib.Path, prefix: str, depth: int):
            if depth > max_depth:
                return
            try:
                entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
            except PermissionError:
                return
            for i, entry in enumerate(entries):
                if entry.name.startswith("."):
                    continue
                connector = "└── " if i == len(entries) - 1 else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir():
                    extension = "    " if i == len(entries) - 1 else "│   "
                    _walk(entry, prefix + extension, depth + 1)

        lines.append(str(target.relative_to(base)) if target != base else ".")
        _walk(target, "", 0)
        return "\n".join(lines)

    return {
        "read_file": read_file,
        "grep": grep,
        "glob_files": glob_files,
        "tree": tree,
    }
```

**Step 4: Run tests**

Run: `cd /path/to/rlm-tools && uv run pytest tests/test_helpers.py -v`
Expected: All passed

**Step 5: Inject helpers into Sandbox**

Update `src/rlm_tools/sandbox.py` - add to `_setup_namespace()`:

```python
from rlm_tools.helpers import make_helpers

# In _setup_namespace():
helpers = make_helpers(self._base_path)
self._namespace.update(helpers)
```

**Step 6: Commit**

```bash
git add src/rlm_tools/helpers.py tests/test_helpers.py src/rlm_tools/sandbox.py
git commit -m "feat: built-in helpers for read_file, grep, glob_files, tree"
```

---

### Task 5: LLM Bridge (llm_query)

**Files:**
- Create: `src/rlm_tools/llm_bridge.py`
- Create: `tests/test_llm_bridge.py`
- Modify: `src/rlm_tools/sandbox.py` (inject llm_query into namespace)

**Step 1: Write the failing test**

```python
# tests/test_llm_bridge.py
from unittest.mock import patch, MagicMock
from rlm_tools.llm_bridge import make_llm_query


def test_llm_query_calls_anthropic():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="YES - handles errors properly")]
    )

    query_fn = make_llm_query(client=mock_client, model="claude-haiku-4-5-20251001")
    result = query_fn("Does this handle errors?", context="some code here")

    assert "YES" in result
    mock_client.messages.create.assert_called_once()


def test_llm_query_batched():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="answer")]
    )

    query_fn = make_llm_query(client=mock_client, model="claude-haiku-4-5-20251001")
    # llm_query_batched should be returned alongside llm_query
    from rlm_tools.llm_bridge import make_llm_query_batched
    batch_fn = make_llm_query_batched(query_fn)

    results = batch_fn(["q1", "q2", "q3"])
    assert len(results) == 3
```

**Step 2: Run tests to verify they fail**

Run: `cd /path/to/rlm-tools && uv run pytest tests/test_llm_bridge.py -v`
Expected: FAIL - module not found

**Step 3: Implement LLM bridge**

```python
# src/rlm_tools/llm_bridge.py
import os
import logging
from anthropic import Anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is required for llm_query()"
        )
    return Anthropic(api_key=api_key)


def make_llm_query(
    client: Anthropic | None = None,
    model: str | None = None,
):
    _client = client or get_client()
    _model = model or os.environ.get("RLM_SUB_MODEL", DEFAULT_MODEL)

    def llm_query(prompt: str, context: str = "") -> str:
        messages = []
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        response = _client.messages.create(
            model=_model,
            max_tokens=1024,
            messages=messages,
        )
        return response.content[0].text

    return llm_query


def make_llm_query_batched(llm_query_fn):
    def llm_query_batched(prompts: list[str], context: str = "") -> list[str]:
        return [llm_query_fn(p, context) for p in prompts]

    return llm_query_batched
```

**Step 4: Run tests**

Run: `cd /path/to/rlm-tools && uv run pytest tests/test_llm_bridge.py -v`
Expected: All passed

**Step 5: Inject into sandbox namespace**

Update `src/rlm_tools/sandbox.py` to accept optional llm functions and add them to namespace.

**Step 6: Commit**

```bash
git add src/rlm_tools/llm_bridge.py tests/test_llm_bridge.py src/rlm_tools/sandbox.py
git commit -m "feat: llm_query bridge for sub-LLM semantic analysis"
```

---

### Task 6: Wire Up MCP Tools

**Files:**
- Modify: `src/rlm_tools/server.py`
- Create: `tests/test_server.py`

**Step 1: Write the failing test**

```python
# tests/test_server.py
import json
import tempfile
import os
from rlm_tools.server import session_manager, _rlm_start, _rlm_execute, _rlm_end


def test_full_rlm_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        with open(os.path.join(tmpdir, "example.py"), "w") as f:
            f.write("def hello():\n    return 'world'\n\ndef foo():\n    return 'bar'\n")

        # Start session
        start_result = _rlm_start(path=tmpdir, query="find all functions")
        result_data = json.loads(start_result)
        session_id = result_data["session_id"]
        assert session_id is not None
        assert "metadata" in result_data

        # Execute code
        exec_result = _rlm_execute(
            session_id=session_id,
            code="files = glob_files('**/*.py')\nprint(f'Found {len(files)} Python files')"
        )
        exec_data = json.loads(exec_result)
        assert "Found 1 Python files" in exec_data["stdout"]

        # Variables persist
        exec_result2 = _rlm_execute(
            session_id=session_id,
            code="print(files)"
        )
        exec_data2 = json.loads(exec_result2)
        assert "example.py" in exec_data2["stdout"]

        # End session
        end_result = _rlm_end(session_id=session_id)
        end_data = json.loads(end_result)
        assert end_data["success"] is True
```

**Step 2: Run test to verify it fails**

Run: `cd /path/to/rlm-tools && uv run pytest tests/test_server.py -v`
Expected: FAIL

**Step 3: Implement the full server**

```python
# src/rlm_tools/server.py
import json
import logging
import os
import pathlib
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from rlm_tools.session import SessionManager
from rlm_tools.sandbox import Sandbox
from rlm_tools.llm_bridge import make_llm_query, make_llm_query_batched

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("rlm-tools")

session_manager = SessionManager(
    max_sessions=int(os.environ.get("RLM_MAX_SESSIONS", "5")),
    timeout_minutes=int(os.environ.get("RLM_SESSION_TIMEOUT", "10")),
)

_sandboxes: dict[str, Sandbox] = {}


def _scan_metadata(path: str) -> dict:
    base = pathlib.Path(path)
    files = list(base.rglob("*"))
    files = [f for f in files if f.is_file() and not any(
        part.startswith(".") for part in f.relative_to(base).parts
    )]

    extensions: dict[str, int] = {}
    total_lines = 0
    for f in files:
        ext = f.suffix or "(no ext)"
        extensions[ext] = extensions.get(ext, 0) + 1
        try:
            total_lines += len(f.read_text(errors="replace").splitlines())
        except OSError:
            pass

    return {
        "total_files": len(files),
        "total_lines": total_lines,
        "file_types": dict(sorted(extensions.items(), key=lambda x: -x[1])[:10]),
    }


def _rlm_start(
    path: str,
    query: str,
    max_output_chars: int = 10_000,
) -> str:
    resolved = str(pathlib.Path(path).resolve())
    if not os.path.isdir(resolved):
        return json.dumps({"error": f"Directory not found: {path}"})

    session_id = session_manager.create(
        path=resolved,
        query=query,
        max_output_chars=max_output_chars,
    )
    metadata = _scan_metadata(resolved)

    sandbox = Sandbox(base_path=resolved, max_output_chars=max_output_chars)

    # Inject llm_query if API key is available
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            llm_query_fn = make_llm_query()
            sandbox._namespace["llm_query"] = llm_query_fn
            sandbox._namespace["llm_query_batched"] = make_llm_query_batched(llm_query_fn)
        except Exception as e:
            logger.warning(f"Could not initialize llm_query: {e}")

    _sandboxes[session_id] = sandbox

    return json.dumps({
        "session_id": session_id,
        "metadata": metadata,
        "available_functions": [
            "read_file(path)", "grep(pattern, path='.')",
            "glob_files(pattern)", "tree(path='.', max_depth=3)",
            "llm_query(prompt, context='')", "llm_query_batched(prompts)",
        ],
        "tip": "Use print() to output results. Variables persist between rlm_execute calls.",
    })


def _rlm_execute(session_id: str, code: str) -> str:
    session = session_manager.get(session_id)
    if not session:
        return json.dumps({"error": f"Session '{session_id}' not found or expired"})

    sandbox = _sandboxes.get(session_id)
    if not sandbox:
        return json.dumps({"error": f"Sandbox not found for session '{session_id}'"})

    result = sandbox.execute(code)

    return json.dumps({
        "stdout": result.stdout,
        "error": result.error,
        "variables": result.variables,
    })


def _rlm_end(session_id: str) -> str:
    session_manager.end(session_id)
    _sandboxes.pop(session_id, None)
    return json.dumps({"success": True})


@mcp.tool()
async def rlm_start(
    path: Annotated[str, Field(description="Absolute path to the directory to explore")],
    query: Annotated[str, Field(description="What you want to find or analyze")],
    max_output_chars: Annotated[int, Field(description="Max characters per execute output", ge=100, le=100_000)] = 10_000,
) -> str:
    """Start an RLM exploration session. Returns session_id and codebase metadata. Use rlm_execute to run Python code in the persistent sandbox."""
    return _rlm_start(path, query, max_output_chars)


@mcp.tool()
async def rlm_execute(
    session_id: Annotated[str, Field(description="Session ID from rlm_start")],
    code: Annotated[str, Field(description="Python code to execute. Use print() for output. Variables persist between calls.")],
) -> str:
    """Execute Python code in an RLM session's persistent sandbox. Built-in functions: read_file(path), grep(pattern), glob_files(pattern), tree(), llm_query(prompt, context)."""
    return _rlm_execute(session_id, code)


@mcp.tool()
async def rlm_end(
    session_id: Annotated[str, Field(description="Session ID to end")],
) -> str:
    """End an RLM exploration session and free resources."""
    return _rlm_end(session_id)


def main():
    mcp.run(transport="stdio")
```

**Step 4: Run tests**

Run: `cd /path/to/rlm-tools && uv run pytest tests/ -v`
Expected: All passed

**Step 5: Commit**

```bash
git add src/rlm_tools/server.py tests/test_server.py
git commit -m "feat: wire up rlm_start, rlm_execute, rlm_end MCP tools"
```

---

### Task 7: Integration Test with MCP Inspector

**Files:**
- No new files

**Step 1: Run the MCP inspector to verify tools are registered**

Run: `cd /path/to/rlm-tools && uv run mcp dev src/rlm_tools/server.py`

Test manually:
1. Call `rlm_start` with a real directory path
2. Call `rlm_execute` with code like `files = glob_files("**/*.py")\nprint(files)`
3. Verify variables persist with another `rlm_execute`
4. Call `rlm_end`

**Step 2: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration test fixes"
```

---

### Task 8: Team Distribution Config

**Files:**
- Create: `.mcp.json` (in the rlm-tools repo as an example)

**Step 1: Create example .mcp.json**

```json
{
  "mcpServers": {
    "rlm-tools": {
      "command": "uvx",
      "args": ["rlm-tools"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

**Step 2: Also create a dev .mcp.json example**

For local development (before publishing to PyPI):

```json
{
  "mcpServers": {
    "rlm-tools": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/rlm-tools", "run", "rlm-tools"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

**Step 3: Commit**

```bash
git add .mcp.json
git commit -m "docs: add example .mcp.json for team distribution"
```
