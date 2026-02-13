import json
import logging
import os
import pathlib
import threading
from typing import Annotated, Literal

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


from rlm_tools.helpers import _SKIP_DIRS, _BINARY_EXTENSIONS


def _scan_metadata(path: str) -> dict:
    extensions: dict[str, int] = {}
    total_files = 0
    total_lines = 0
    sampled_lines = 0
    sampled_files = 0
    sample_budget = 500

    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]

        for fname in filenames:
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1] or "(no ext)"
            extensions[ext] = extensions.get(ext, 0) + 1
            total_files += 1

            if ext not in _BINARY_EXTENSIONS:
                try:
                    fpath = os.path.join(dirpath, fname)
                    with open(fpath, errors="replace") as f:
                        file_line_count = sum(1 for _ in f)
                    total_lines += file_line_count

                    if sampled_files < sample_budget:
                        sampled_lines += file_line_count
                        sampled_files += 1
                except OSError:
                    pass

    return {
        "total_files": total_files,
        "total_lines": total_lines,
        "sampled_lines": sampled_lines,
        "sampled_files": sampled_files,
        "file_types": dict(sorted(extensions.items(), key=lambda x: -x[1])[:10]),
    }


def _cleanup_expired_resources() -> None:
    expired_session_ids = session_manager.cleanup_expired()
    for session_id in expired_session_ids:
        _sandboxes.pop(session_id, None)


def _install_session_llm_tools(session, sandbox: Sandbox) -> bool:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return False

    try:
        base_llm_query = make_llm_query()
        base_llm_query_batched = make_llm_query_batched(base_llm_query)
        lock = threading.Lock()

        def _reserve_llm_calls(count: int) -> None:
            if count < 1:
                raise ValueError("count must be >= 1")
            with lock:
                if session.llm_calls_used + count > session.max_llm_calls:
                    raise RuntimeError(
                        "LLM call limit exceeded: "
                        f"{session.llm_calls_used} + {count} > {session.max_llm_calls}"
                    )
                session.llm_calls_used += count

        def llm_query(prompt: str, context: str = "") -> str:
            _reserve_llm_calls(1)
            return base_llm_query(prompt, context)

        def llm_query_batched(prompts: list[str], context: str = "") -> list[str]:
            if not prompts:
                return []
            _reserve_llm_calls(len(prompts))
            return base_llm_query_batched(prompts, context)

        sandbox._namespace["llm_query"] = llm_query
        sandbox._namespace["llm_query_batched"] = llm_query_batched
        return True
    except Exception as e:
        logger.warning(f"Could not initialize llm_query: {e}")
        return False


def _rlm_start(
    path: str,
    query: str,
    max_output_chars: int = 15_000,
    max_llm_calls: int = 50,
    max_execute_calls: int = 50,
    execution_timeout_seconds: int = 30,
    include_guidance: bool = False,
    include_metadata: bool = True,
) -> str:
    _cleanup_expired_resources()

    resolved = str(pathlib.Path(path).resolve())
    if not os.path.isdir(resolved):
        return json.dumps({"error": f"Directory not found: {path}"})

    try:
        session_id = session_manager.create(
            path=resolved,
            query=query,
            max_output_chars=max_output_chars,
            max_llm_calls=max_llm_calls,
            max_execute_calls=max_execute_calls,
        )
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    session = session_manager.get(session_id)
    if not session:
        return json.dumps({"error": f"Failed to create session for path: {path}"})

    metadata = _scan_metadata(resolved) if include_metadata else {}

    sandbox = Sandbox(
        base_path=resolved,
        max_output_chars=max_output_chars,
        execution_timeout_seconds=execution_timeout_seconds,
    )
    has_llm_tools = _install_session_llm_tools(session, sandbox)

    _sandboxes[session_id] = sandbox

    available_functions = [
        "read_file(path)",
        "read_files(paths) -> dict[path, content]",
        "grep(pattern, path='.')",
        "grep_summary(pattern, path='.') -> compact grouped string",
        "grep_read(pattern, path='.', max_files=10, context_lines=0) -> {matches, files, summary}",
        "glob_files(pattern)",
        "tree(path='.', max_depth=3)",
    ]
    if has_llm_tools:
        available_functions.extend([
            "llm_query(prompt, context='')",
            "llm_query_batched(prompts, context='')",
        ])

    response: dict = {
        "session_id": session_id,
        "metadata": metadata,
        "limits": {
            "max_llm_calls": session.max_llm_calls,
            "max_execute_calls": session.max_execute_calls,
            "execution_timeout_seconds": execution_timeout_seconds,
        },
        "available_functions": available_functions,
    }
    if include_guidance:
        response["strategy"] = (
            "You are exploring a codebase through code. "
            "Each rlm_execute call should batch 3-5+ operations into a single Python script. "
            "Pattern: grep/glob to find targets -> read relevant files -> extract/analyze -> store in variables. "
            "Aim for ~5-15 rlm_execute calls total, not 50+. "
            "Use llm_query() for semantic analysis on extracted snippets. "
            "Use llm_query_batched() when analyzing multiple snippets concurrently. "
            "Variables persist between calls — build on previous results."
        )
        response["example"] = (
            "# Good: batch multiple operations in ONE call\n"
            "matches = grep('Reducer', '.')\n"
            "files = list(set(m['file'] for m in matches))[:5]\n"
            "for f in files:\n"
            "    content = read_file(f)\n"
            "    lines = content.split('\\n')\n"
            "    structs = [l.strip() for l in lines if 'struct ' in l or 'class ' in l]\n"
            "    print(f'{f}: {structs}')\n"
        )
    return json.dumps(response)


def _rlm_execute(
    session_id: str,
    code: str,
    detail_level: Literal["compact", "usage", "full"] = "compact",
    max_new_variables: int = 20,
) -> str:
    _cleanup_expired_resources()
    session = session_manager.get(session_id)
    if not session:
        return json.dumps({"error": f"Session '{session_id}' not found or expired"})

    sandbox = _sandboxes.get(session_id)
    if not sandbox:
        return json.dumps({"error": f"Sandbox not found for session '{session_id}'"})

    if session.execute_calls >= session.max_execute_calls:
        return json.dumps({
            "error": (
                "Execution call limit exceeded: "
                f"{session.execute_calls} >= {session.max_execute_calls}"
            )
        })

    session.execute_calls += 1
    result = sandbox.execute(code)

    response: dict = {
        "stdout": result.stdout,
        "error": result.error,
    }

    if detail_level in {"usage", "full"}:
        response["usage"] = {
            "execute_calls_used": session.execute_calls,
            "execute_calls_remaining": session.max_execute_calls - session.execute_calls,
            "llm_calls_used": session.llm_calls_used,
        }

    if detail_level == "full":
        current_vars = set(result.variables)
        previous_vars = getattr(session, "_last_reported_vars", set())
        excluded_vars = {
            "read_file", "read_files",
            "grep", "grep_summary", "grep_read",
            "glob_files", "tree",
            "llm_query", "llm_query_batched",
        }
        new_vars = sorted(
            v for v in (current_vars - previous_vars)
            if v not in excluded_vars
        )
        session._last_reported_vars = current_vars

        response["variables"] = sorted(v for v in current_vars if v not in excluded_vars)
        response["total_variables"] = len(response["variables"])
        response["new_variables"] = new_vars[:max_new_variables]
        if len(new_vars) > max_new_variables:
            response["new_variables_truncated_count"] = len(new_vars) - max_new_variables

    return json.dumps(response)


def _rlm_end(session_id: str) -> str:
    session_manager.end(session_id)
    _sandboxes.pop(session_id, None)
    return json.dumps({"success": True})


@mcp.tool()
async def rlm_start(
    path: Annotated[str, Field(description="Absolute path to the directory to explore")],
    query: Annotated[str, Field(description="What you want to find or analyze")],
    max_output_chars: Annotated[int, Field(description="Max characters per execute output", ge=100, le=100_000)] = 15_000,
    max_llm_calls: Annotated[int, Field(description="Maximum llm_query/llm_query_batched calls for this session", ge=1, le=10_000)] = 50,
    max_execute_calls: Annotated[int, Field(description="Maximum rlm_execute calls for this session", ge=1, le=10_000)] = 50,
    execution_timeout_seconds: Annotated[int, Field(description="Per-rlm_execute timeout in seconds", ge=1, le=300)] = 30,
    include_guidance: Annotated[bool, Field(description="Include strategy/example guidance text in the response (larger payload)")] = False,
    include_metadata: Annotated[bool, Field(description="Scan directory and include file counts/types in response (set false for faster startup)")] = True,
) -> str:
    """Start an RLM exploration session. Returns session_id, metadata, limits, and available functions. Set include_guidance=true to include strategy/example coaching text."""
    return _rlm_start(
        path=path,
        query=query,
        max_output_chars=max_output_chars,
        max_llm_calls=max_llm_calls,
        max_execute_calls=max_execute_calls,
        execution_timeout_seconds=execution_timeout_seconds,
        include_guidance=include_guidance,
        include_metadata=include_metadata,
    )


@mcp.tool()
async def rlm_execute(
    session_id: Annotated[str, Field(description="Session ID from rlm_start")],
    code: Annotated[str, Field(description=(
        "Python code to execute. IMPORTANT: Batch multiple related operations into each call. "
        "A good call does: grep -> read top matches -> extract patterns -> print summary. "
        "A bad call does just one grep or one read_file. Variables persist between calls."
    ))],
    detail_level: Annotated[Literal["compact", "usage", "full"], Field(
        description="Response payload level: compact=stdout+error, usage=add usage metrics, full=add variable details"
    )] = "compact",
    max_new_variables: Annotated[int, Field(
        description="When detail_level=full, cap returned new_variables list to this size",
        ge=1,
        le=200,
    )] = 20,
) -> str:
    """Execute Python in the session sandbox. detail_level controls response payload size (compact, usage, or full)."""
    return _rlm_execute(session_id, code, detail_level, max_new_variables)


@mcp.tool()
async def rlm_end(
    session_id: Annotated[str, Field(description="Session ID to end")],
) -> str:
    """End an RLM exploration session and free resources."""
    return _rlm_end(session_id)


def main():
    mcp.run(transport="stdio")
