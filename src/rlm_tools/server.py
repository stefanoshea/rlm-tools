import json
import logging
import os
import threading
from typing import Annotated, Literal
from weakref import WeakValueDictionary

from anyio import Lock, to_thread
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from rlm_tools.access_policy import workspace_policy
from rlm_tools.helpers import scan_metadata as _scan_metadata
from rlm_tools.session import Session, SessionManager
from rlm_tools.sandbox import Sandbox
from rlm_tools.llm_bridge import (
    make_llm_query,
    make_llm_query_batched,
    make_recursive_query,
)
from rlm_tools.semantic import make_evidence_gate, make_scorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("rlm-tools")

session_manager = SessionManager(
    max_sessions=int(os.environ.get("RLM_MAX_SESSIONS", "5")),
    timeout_minutes=int(os.environ.get("RLM_SESSION_TIMEOUT", "10")),
)

_sandboxes: dict[str, Sandbox] = {}
_mcp_session_locks: WeakValueDictionary[str, Lock] = WeakValueDictionary()


def _cleanup_expired_resources() -> None:
    expired_session_ids = session_manager.cleanup_expired()
    for session_id in expired_session_ids:
        sandbox = _sandboxes.pop(session_id, None)
        if sandbox:
            sandbox.close()


def _make_call_reserver(session):
    """One budget gate for every model call a session makes, sub-LLM or Jev."""
    lock = threading.Lock()

    def reserve(count: int) -> None:
        if count < 1:
            raise ValueError("count must be >= 1")
        with lock:
            if session.llm_calls_used + count > session.max_llm_calls:
                raise RuntimeError(
                    "LLM call limit exceeded: "
                    f"{session.llm_calls_used} + {count} > {session.max_llm_calls}"
                )
            session.llm_calls_used += count

    return reserve


def _make_recurse_gate(session, reserve=None, deadline=None):
    """Build the optional TypeSafe scoring gate for an external-enabled session."""
    if not os.environ.get("TYPESAFE_API_KEY") or os.environ.get("RLM_RECURSE_GATE") != "1":
        return None

    try:
        scorer = make_scorer(reserve=reserve or _make_call_reserver(session), deadline=deadline)
        return make_evidence_gate(scorer)
    except Exception as e:
        logger.warning(f"Could not initialize the evidence gate: {e}")
        return None


def _install_session_llm_tools(session, sandbox: Sandbox) -> bool:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return False

    try:
        _reserve_llm_calls = _make_call_reserver(session)
        gate = _make_recurse_gate(session, reserve=_reserve_llm_calls, deadline=sandbox.execution_deadline)
        base_llm_query = make_llm_query(
            on_retry=lambda: _reserve_llm_calls(1), deadline=sandbox.execution_deadline,
        )
        llm_query_batched = make_llm_query_batched(base_llm_query, on_batch=_reserve_llm_calls)

        def llm_query(prompt: str, context: str = "") -> str:
            _reserve_llm_calls(1)
            return base_llm_query(prompt, context)

        # rlm_recurse is a true recursive explorer: it runs its own tool-use loop
        # server-side and may spawn depth-limited child agents. Every model turn
        # (root or child) is reserved against the session's llm-call budget.
        recursive_query = make_recursive_query(
            helpers=sandbox.helpers,
            max_depth=int(os.environ.get("RLM_RECURSE_MAX_DEPTH", "2")),
            on_model_call=lambda: _reserve_llm_calls(1),
            gate=gate,
            deadline=sandbox.execution_deadline,
        )
        recursive_query_batched = make_llm_query_batched(recursive_query)
    except Exception as e:
        logger.warning(f"Could not initialize llm_query: {e}")
        return False

    sandbox.register_tool("llm_query", llm_query)
    sandbox.register_tool("llm_query_batched", llm_query_batched)
    sandbox.register_tool("rlm_recurse", recursive_query)
    sandbox.register_tool("rlm_recurse_batched", recursive_query_batched)
    return True


def _rlm_start(
    path: str,
    query: str,
    max_output_chars: int = 15_000,
    max_llm_calls: int = 50,
    max_execute_calls: int = 50,
    execution_timeout_seconds: int = 30,
    include_guidance: bool = False,
    include_metadata: bool = False,
    max_memory_bytes: int = 10_000_000,
) -> str:
    _cleanup_expired_resources()

    try:
        policy = workspace_policy()
        root = policy.authorize(path)
    except (OSError, ValueError, RuntimeError) as exc:
        return json.dumps({"error": str(exc)})

    try:
        session_id = session_manager.create(
            path=str(root),
            query=query,
            max_output_chars=max_output_chars,
            max_llm_calls=max_llm_calls,
            max_execute_calls=max_execute_calls,
            max_memory_bytes=max_memory_bytes,
            pin=True,
        )
    except RuntimeError as e:
        return json.dumps({"error": str(e)})

    session = session_manager.get(session_id)
    if not session:
        return json.dumps({"error": f"Failed to create session for path: {path}"})

    try:
        return _initialize_session(
            session, execution_timeout_seconds, include_guidance, include_metadata,
            allow_external=policy.allows_external(root),
        )
    finally:
        session_manager.release(session)


def _initialize_session(
    session: Session,
    execution_timeout_seconds: int,
    include_guidance: bool,
    include_metadata: bool,
    *,
    allow_external: bool,
) -> str:
    session_id = session.session_id
    resolved = session.path
    max_output_chars = session.max_output_chars
    max_memory_bytes = session.max_memory_bytes
    sandbox = None
    try:
        metadata = _scan_metadata(resolved) if include_metadata else {}
        sandbox = Sandbox(
            base_path=resolved,
            base_is_canonical=True,
            max_output_chars=max_output_chars,
            execution_timeout_seconds=execution_timeout_seconds,
            max_memory_bytes=max_memory_bytes,
        )
        has_llm_tools = allow_external and _install_session_llm_tools(session, sandbox)
    except Exception as e:
        session_manager.end(session_id)
        logger.exception("Failed to initialize session for %s", resolved)
        if sandbox is not None:
            try:
                sandbox.close()
            except Exception:
                logger.exception("Failed to close sandbox for session %s", session_id)
        return json.dumps({"error": f"Failed to initialize session: {e}"})

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
            "rlm_recurse(prompt, context='')",
            "rlm_recurse_batched(prompts, context='')",
        ])
    available_functions.append("rlm_finalize(value)")

    response: dict = {
        "session_id": session_id,
        "metadata": metadata,
        "limits": {
            "max_llm_calls": session.max_llm_calls,
            "max_execute_calls": session.max_execute_calls,
            "max_memory_bytes": session.max_memory_bytes,
            "execution_timeout_seconds": execution_timeout_seconds,
        },
        "available_functions": available_functions,
    }
    if include_guidance:
        response["strategy"] = (
            "You are exploring a codebase through code. "
            "Each rlm_execute call should batch 3-5+ operations into a single Python script. "
            "Pattern: tree/grep to find targets -> read only the files that survive -> print a summary. "
            "Aim for ~5-15 rlm_execute calls total, not 50+. "
            "A wide case-insensitive grep_summary beats several narrow greps: it costs one call and "
            "shows you which files matter before you read any of them. "
            "Use llm_query() for semantic analysis on extracted snippets. "
            "Use llm_query_batched() when analyzing multiple snippets concurrently. "
            "Variables persist between calls — build on previous results."
        )
        response["example"] = (
            "# Good: orient, cast a wide net, then read only the survivors\n"
            "print(tree('.', max_depth=3))\n"
            "print(grep_summary(r'(?i)flag|feature_', '.'))\n"
            "for path in ['app/export/pipeline.py', 'app/feature_flags.py']:\n"
            "    print('=' * 20, path)\n"
            "    print(read_file(path)[:2000])\n"
        )
    return json.dumps(response)


def _rlm_execute(
    session_id: str,
    code: str,
    detail_level: Literal["compact", "usage", "full"] = "compact",
    max_new_variables: int = 20,
) -> str:
    _cleanup_expired_resources()
    with session_manager.use(session_id) as session:
        if not session:
            return json.dumps({"error": f"Session '{session_id}' not found or expired"})
        return _execute_session(session, code, detail_level, max_new_variables)


def _execute_session(
    session: Session,
    code: str,
    detail_level: Literal["compact", "usage", "full"],
    max_new_variables: int,
) -> str:
    session_id = session.session_id
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
    session.memory_bytes_used = sandbox.memory_bytes_used

    response: dict = {
        "stdout": result.stdout,
        "error": result.error,
    }
    if result.final is not None:
        response["final"] = result.final

    if detail_level in {"usage", "full"}:
        response["usage"] = {
            "execute_calls_used": session.execute_calls,
            "execute_calls_remaining": session.max_execute_calls - session.execute_calls,
            "llm_calls_used": session.llm_calls_used,
            "memory_bytes_used": result.memory_bytes_used,
        }

    if detail_level == "full":
        current_vars = set(result.variables)
        previous_vars = getattr(session, "_last_reported_vars", set())
        excluded_vars = {
            "read_file", "read_files",
            "grep", "grep_summary", "grep_read",
            "glob_files", "tree",
            "llm_query", "llm_query_batched",
            "rlm_recurse", "rlm_recurse_batched",
            "rlm_finalize",
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
    _cleanup_expired_resources()
    with session_manager.use(session_id) as session:
        if session is not None:
            sandbox = _sandboxes.pop(session_id, None)
            try:
                if sandbox:
                    sandbox.close()
            finally:
                session_manager.end(session_id)
    return json.dumps({"success": True})


async def _run_session_operation(operation, session_id: str, *args) -> str:
    # Keep queued requests off the shared worker pool so other sessions can run.
    with session_manager.pin(session_id) as session:
        if session is None:
            return await to_thread.run_sync(operation, session_id, *args)
        lock = _mcp_session_locks.get(session_id)
        if lock is None:
            lock = _mcp_session_locks[session_id] = Lock()
        async with lock:
            return await to_thread.run_sync(operation, session_id, *args)


@mcp.tool()
async def rlm_start(
    path: Annotated[str, Field(description="Absolute path to the directory to explore")],
    query: Annotated[str, Field(description="What you want to find or analyze")],
    max_output_chars: Annotated[int, Field(description="Max characters per execute output", ge=100, le=100_000)] = 15_000,
    max_llm_calls: Annotated[int, Field(description="Maximum llm_query/llm_query_batched calls for this session", ge=1, le=10_000)] = 50,
    max_execute_calls: Annotated[int, Field(description="Maximum rlm_execute calls for this session", ge=1, le=10_000)] = 50,
    execution_timeout_seconds: Annotated[int, Field(description="Per-rlm_execute timeout in seconds", ge=1, le=300)] = 30,
    include_guidance: Annotated[bool, Field(description="Include strategy/example guidance text in the response (larger payload)")] = False,
    include_metadata: Annotated[bool, Field(description="Scan directory and include file counts/types in response (set false for faster startup)")] = False,
    max_memory_bytes: Annotated[int, Field(description="Maximum in-memory bytes retained for this session", ge=1024, le=100_000_000)] = 10_000_000,
) -> str:
    """Start an RLM exploration session. Returns session_id, metadata, limits, and available functions. Set include_guidance=true to include strategy/example coaching text."""
    return await to_thread.run_sync(
        _rlm_start, path, query, max_output_chars, max_llm_calls, max_execute_calls,
        execution_timeout_seconds, include_guidance, include_metadata, max_memory_bytes,
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
    return await _run_session_operation(_rlm_execute, session_id, code, detail_level, max_new_variables)


@mcp.tool()
async def rlm_end(
    session_id: Annotated[str, Field(description="Session ID to end")],
) -> str:
    """End an RLM exploration session and free resources."""
    return await _run_session_operation(_rlm_end, session_id)


def main():
    workspace_policy()
    try:
        mcp.run(transport="stdio")
    finally:
        for session_id, sandbox in list(_sandboxes.items()):
            try:
                sandbox.close()
            except Exception:
                logger.exception("Failed to close sandbox for session %s", session_id)
            finally:
                _sandboxes.pop(session_id, None)
                session_manager.end(session_id)
