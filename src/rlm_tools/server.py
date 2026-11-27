import json
import logging
import os
import threading
from typing import Annotated, Literal
from weakref import WeakValueDictionary

from anyio import Lock, to_thread
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from rlm_tools._workspace import scan_limits
from rlm_tools.access_policy import workspace_policy
from rlm_tools.helpers import scan_metadata as _scan_metadata
from rlm_tools.session import (
    DEFAULT_MAX_LLM_CALLS,
    Session,
    SessionManager,
    default_max_memory_bytes,
)
from rlm_tools.sandbox import Sandbox
from rlm_tools.llm_bridge import (
    has_configured_provider,
    make_llm_query,
    make_llm_query_batched,
    make_recursive_query,
    resolve_provider,
)
from rlm_tools.semantic import has_configured_scorer, make_evidence_gate, make_scorer
from rlm_tools.worker_runtime import ALLOWED_MODULES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("rlm-tools")
RECURSE_ESTIMATED_CALLS_PER_ITEM = 4

session_manager = SessionManager(
    max_sessions=int(os.environ.get("RLM_MAX_SESSIONS", "16")),
    timeout_minutes=int(os.environ.get("RLM_SESSION_TIMEOUT", "10")),
    idle_evict_seconds=float(os.environ.get("RLM_IDLE_EVICT_SECONDS", "60")),
)

_sandboxes: dict[str, Sandbox] = {}
_mcp_session_locks: WeakValueDictionary[str, Lock] = WeakValueDictionary()


def _close_session(session_id: str) -> None:
    sandbox = _sandboxes.pop(session_id, None)
    try:
        if sandbox:
            sandbox.close()
    finally:
        session_manager.end(session_id)


def _cleanup_expired_resources() -> None:
    expired_session_ids = session_manager.cleanup_expired()
    for session_id in expired_session_ids:
        sandbox = _sandboxes.pop(session_id, None)
        if sandbox:
            try:
                sandbox.close()
            except Exception:
                logger.exception("Failed to close expired sandbox for session %s", session_id)


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

    def check_recurse_batch(num_prompts: int) -> None:
        estimated = num_prompts * RECURSE_ESTIMATED_CALLS_PER_ITEM
        with lock:
            remaining = max(0, session.max_llm_calls - session.llm_calls_used)
        if estimated > remaining:
            max_fit = remaining // RECURSE_ESTIMATED_CALLS_PER_ITEM
            raise RuntimeError(
                f"LLM call budget too small for rlm_recurse_batched: {num_prompts} prompts "
                f"need ~{estimated} calls (~{RECURSE_ESTIMATED_CALLS_PER_ITEM} per item), "
                f"but only {remaining} of {session.max_llm_calls} remain. "
                f"Start a session with max_llm_calls >= {estimated} "
                f"(or split into batches of <= {max_fit})."
            )

    reserve.check_recurse_batch = check_recurse_batch
    return reserve


def _make_recurse_gate(session, reserve=None, deadline=None):
    """Build the optional relevance scoring gate for an external-enabled session."""
    if os.environ.get("RLM_RECURSE_GATE") != "1" or not has_configured_scorer():
        return None

    try:
        scorer = make_scorer(reserve=reserve or _make_call_reserver(session), deadline=deadline)
        return make_evidence_gate(scorer)
    except Exception as e:
        logger.warning(f"Could not initialize the evidence gate: {e}")
        return None


def _default_execution_timeout_seconds(allow_external: bool) -> int:
    raw = os.environ.get("RLM_EXECUTION_TIMEOUT_SECONDS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    if allow_external and has_configured_provider():
        try:
            if resolve_provider() == "claude-cli":
                return 180
        except ValueError:
            pass
        return 60
    return 30


def _install_session_llm_tools(session, sandbox: Sandbox) -> bool:
    if not has_configured_provider():
        return False

    try:
        _reserve_llm_calls = _make_call_reserver(session)
        usage_lock = threading.Lock()

        def record_usage(delta: dict[str, int | float]) -> None:
            with usage_lock:
                for key, value in delta.items():
                    prev = session.sub_llm_usage.get(key, 0)
                    total = prev + value
                    session.sub_llm_usage[key] = round(total, 6) if isinstance(total, float) else total

        def record_child_budget_exhausted() -> None:
            with usage_lock:
                session.child_budget_exhausted += 1

        gate = _make_recurse_gate(session, reserve=_reserve_llm_calls, deadline=sandbox.execution_deadline)
        base_llm_query = make_llm_query(
            on_retry=lambda: _reserve_llm_calls(1),
            deadline=sandbox.execution_deadline,
            on_usage=record_usage,
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
            on_usage=record_usage,
            on_child_budget_exhausted=record_child_budget_exhausted,
        )
        recursive_query_batched = make_llm_query_batched(
            recursive_query,
            check_batch=_reserve_llm_calls.check_recurse_batch,
        )
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
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    max_execute_calls: int = 100,
    execution_timeout_seconds: int | None = None,
    include_guidance: bool = False,
    include_metadata: bool = False,
    max_memory_bytes: int | None = None,
    code: str = "",
) -> str:
    _cleanup_expired_resources()
    has_code = bool(code and code.strip())

    try:
        policy = workspace_policy()
        root = policy.authorize(path)
    except (OSError, ValueError, RuntimeError) as exc:
        return json.dumps({"error": str(exc)})

    allow_external = policy.allows_external(root)
    resolved_timeout = (
        _default_execution_timeout_seconds(allow_external)
        if execution_timeout_seconds is None
        else execution_timeout_seconds
    )
    resolved_memory = (
        default_max_memory_bytes()
        if max_memory_bytes is None
        else max_memory_bytes
    )

    try:
        session_id = session_manager.create(
            path=str(root),
            query=query,
            max_output_chars=max_output_chars,
            max_llm_calls=max_llm_calls,
            max_execute_calls=max_execute_calls,
            max_memory_bytes=resolved_memory,
            pin=True,
        )
    except RuntimeError as e:
        return json.dumps({"error": str(e)})
    finally:
        _cleanup_expired_resources()

    session = session_manager.get(session_id)
    if not session:
        return json.dumps({"error": f"Failed to create session for path: {path}"})

    try:
        return _initialize_session(
            session, resolved_timeout, include_guidance, include_metadata,
            allow_external=allow_external,
            code=code if has_code else "",
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
    code: str = "",
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
        "read_file(path, max_bytes_read=None, start_line=None, end_line=None) -> str",
        "read_files(paths, max_bytes_read=None) -> dict[path, content]",
        "grep(pattern, path='.', max_matches=None, allow_partial=False) -> list[{file, line_number, line}] (raises on workspace truncation unless allow_partial=True; explicit max_matches caps count without error)",
        "grep_summary(pattern, path='.', max_matches=None) -> compact grouped str (starts with TRUNCATED if incomplete)",
        "grep_read(pattern, path='.', max_files=10, context_lines=0, max_matches=None, max_bytes_read=None) -> {matches, files, summary, truncated, limit_hit, files_scanned, files_skipped}",
        "glob_files(pattern, include_ignored=False) -> sorted list[str] (raises on truncation)",
        "tree(path='.', max_depth=3) -> str (starts with TRUNCATED if incomplete)",
        "outline(path) -> {path, total_lines, classes, functions} (Python AST classes, methods, functions with parent/depth [parent=None, depth=0 for module-level], decorators, fields, line spans)",
        "enclosing(path, line) -> {line, qualified, chain, spans}",
        "find_assignments(path, target) -> list[{line_number, kind, target, enclosing, line}]",
    ]
    if has_llm_tools:
        available_functions.extend([
            "llm_query(prompt, context='')",
            "llm_query_batched(prompts, context='')",
            "rlm_recurse(prompt, context='')",
            "rlm_recurse_batched(prompts, context='')",
        ])
    available_functions.append("rlm_finalize(value)")

    active_scan = scan_limits()
    response: dict = {
        "session_id": session_id,
        "metadata": metadata,
        "limits": {
            "max_llm_calls": session.max_llm_calls,
            "max_execute_calls": session.max_execute_calls,
            "max_memory_bytes": session.max_memory_bytes,
            "execution_timeout_seconds": execution_timeout_seconds,
            "max_scan_bytes": active_scan["max_scan_bytes"],
            "max_scan_entries": active_scan["max_scan_entries"],
            "max_scan_seconds": active_scan["max_scan_seconds"],
        },
        "available_functions": available_functions,
        "allowed_imports": sorted(ALLOWED_MODULES),
    }
    if has_llm_tools:
        response["llm_budget_note"] = (
            "For N-item recursion (rlm_recurse_batched), set max_llm_calls ≈ 4×N (~3-4 LLM calls per child)."
        )
    other_sessions = session_manager.list_other_session_ids(session_id)
    if other_sessions:
        response["other_open_sessions"] = other_sessions
    if include_guidance:
        response["strategy"] = (
            "You are exploring a codebase through code. "
            "Each rlm_execute call should batch 3-5+ operations into a single Python script. "
            "Pattern: tree/grep to find targets -> read only the files that survive -> print a summary "
            "or call rlm_finalize(value) to return structured data. "
            "Aim for ~5-15 rlm_execute calls total, not 50+. "
            "A wide case-insensitive grep_summary beats several narrow greps: it costs one call and "
            "shows you which files matter before you read any of them. "
            "Do not import os/glob/pathlib/ast or call open()/exec()/eval()/getattr()/str.format() — use the built-in helpers and f-strings. "
            "Variables persist between calls — build on previous results. "
            "Pass end_session=True on your final rlm_execute call to close without an extra rlm_end call."
        )
        if has_llm_tools:
            response["strategy"] += (
                " Use rlm_recurse_batched() (children can read code) when the child must check code, "
                "giving each child the file path and line span (from outline), not pasted text, "
                "and sizing max_llm_calls ≈ 4×N; use llm_query() / llm_query_batched() only when the "
                "full context is already in the prompt."
            )
        response["example"] = (
            "# Good: orient, cast a wide net, then read only the survivors\n"
            "print(tree('.', max_depth=3))\n"
            "matches = grep(r'(?i)flag|feature_', '.')\n"
            "for path in sorted({m['file'] for m in matches})[:3]:\n"
            "    print('=' * 20, path)\n"
            "    print(read_file(path)[:2000])\n"
        )

    if code:
        exec_response = _execute_session_data(session, code, "compact", 20)
        response["stdout"] = exec_response.get("stdout", "")
        response["error"] = exec_response.get("error")
        if "final" in exec_response:
            response["final"] = exec_response["final"]

    return json.dumps(response)


def _rlm_execute(
    session_id: str,
    code: str,
    detail_level: Literal["compact", "usage", "full"] = "compact",
    max_new_variables: int = 20,
    end_session: bool = False,
) -> str:
    _cleanup_expired_resources()
    with session_manager.use(session_id) as session:
        if not session:
            return json.dumps({"error": f"Session '{session_id}' not found or expired"})
        return _execute_session(session, code, detail_level, max_new_variables, end_session)


def _usage_payload(session: Session, memory_bytes_used: int) -> dict:
    payload: dict = {
        "execute_calls_used": session.execute_calls,
        "execute_calls_remaining": session.max_execute_calls - session.execute_calls,
        "llm_calls_used": session.llm_calls_used,
        "llm_calls_remaining": max(0, session.max_llm_calls - session.llm_calls_used),
        "child_budget_exhausted": session.child_budget_exhausted,
        "memory_bytes_used": memory_bytes_used,
    }
    if session.sub_llm_usage:
        payload["sub_llm_usage"] = dict(session.sub_llm_usage)
    return payload


def _execute_session_data(
    session: Session,
    code: str,
    detail_level: Literal["compact", "usage", "full"],
    max_new_variables: int,
) -> dict:
    session_id = session.session_id
    sandbox = _sandboxes.get(session_id)
    if not sandbox:
        return {"error": f"Sandbox not found for session '{session_id}'"}

    if session.execute_calls >= session.max_execute_calls:
        return {
            "error": (
                "Execution call limit exceeded: "
                f"{session.execute_calls} >= {session.max_execute_calls}"
            )
        }

    session.execute_calls += 1
    exhausted_before = session.child_budget_exhausted
    result = sandbox.execute(code)
    session.memory_bytes_used = sandbox.memory_bytes_used

    response: dict = {
        "stdout": result.stdout,
        "error": result.error,
    }
    if result.final is not None:
        response["final"] = result.final
    if result.error is None:
        warnings: list[str] = []
        remaining_calls = session.max_execute_calls - session.execute_calls
        warn_threshold = max(5, int(session.max_execute_calls * 0.15))
        if 0 < remaining_calls <= warn_threshold:
            warnings.append(f"{remaining_calls} execute calls remaining")
        if session.llm_calls_used > 0:
            remaining_llm = max(0, session.max_llm_calls - session.llm_calls_used)
            llm_warn_threshold = max(5, int(session.max_llm_calls * 0.15))
            if remaining_llm <= llm_warn_threshold:
                warnings.append(f"{remaining_llm} LLM calls remaining (of {session.max_llm_calls})")
        newly_exhausted = session.child_budget_exhausted - exhausted_before
        if newly_exhausted > 0:
            warnings.append(
                f"{newly_exhausted} child(ren) hit the tool output budget (results marked [BUDGET_EXHAUSTED])"
            )
        if warnings:
            response["budget_warning"] = "; ".join(warnings)

    if detail_level in {"usage", "full"}:
        response["usage"] = _usage_payload(session, result.memory_bytes_used)

    if detail_level == "full":
        current_vars = set(result.variables)
        previous_vars = getattr(session, "_last_reported_vars", set())
        excluded_vars = {
            "read_file", "read_files",
            "grep", "grep_summary", "grep_read",
            "glob_files", "tree",
            "outline", "enclosing", "find_assignments",
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

    return response


def _execute_session(
    session: Session,
    code: str,
    detail_level: Literal["compact", "usage", "full"],
    max_new_variables: int,
    end_session: bool = False,
) -> str:
    response = _execute_session_data(session, code, detail_level, max_new_variables)
    if end_session and "error" in response:
        if response["error"] is None:
            if "usage" not in response:
                response["usage"] = _usage_payload(session, session.memory_bytes_used)
            _close_session(session.session_id)
            response["session_ended"] = True
        else:
            response["session_ended"] = False
    return json.dumps(response)


def _rlm_end(session_id: str) -> str:
    _cleanup_expired_resources()
    usage = None
    with session_manager.use(session_id) as session:
        if session is not None:
            usage = _usage_payload(session, session.memory_bytes_used)
            _close_session(session_id)
    payload: dict = {"success": True}
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload)


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
    max_llm_calls: Annotated[int, Field(description="Maximum model calls (llm_query, rlm_recurse, and evidence gate) for this session (default 300; for N-item recursion, set max_llm_calls ≈ 4×N)", ge=1, le=10_000)] = DEFAULT_MAX_LLM_CALLS,
    max_execute_calls: Annotated[int, Field(description="Maximum rlm_execute calls for this session", ge=1, le=10_000)] = 100,
    execution_timeout_seconds: Annotated[int | None, Field(description="Per-rlm_execute timeout in seconds (defaults to 30 local, 60 external API, 180 claude-cli)", ge=1, le=600)] = None,
    include_guidance: Annotated[bool, Field(description="Include strategy/example guidance text in the response (larger payload)")] = False,
    include_metadata: Annotated[bool, Field(description="Scan directory and include file counts/types in response (default false for faster startup)")] = False,
    max_memory_bytes: Annotated[int | None, Field(description="Maximum in-memory bytes retained for this session (default 64 MiB or RLM_MAX_MEMORY_BYTES)", ge=1024, le=512_000_000)] = None,
    code: Annotated[str, Field(description=(
        "Optional initial Python script (e.g., tree/glob_files/grep_summary) to run right after opening the session; "
        "same helpers and restrictions as rlm_execute."
    ))] = "",
) -> str:
    """Start a persistent RLM exploration session once per task (returns session_id, limits, available_functions, allowed_imports). Optionally pass initial code to explore immediately, then use rlm_execute(session_id, ...) for follow-up steps and pass end_session=true on your final rlm_execute call (do not call rlm_start repeatedly)."""
    return await to_thread.run_sync(
        _rlm_start, path, query, max_output_chars, max_llm_calls, max_execute_calls,
        execution_timeout_seconds, include_guidance, include_metadata, max_memory_bytes,
        code,
    )


@mcp.tool()
async def rlm_execute(
    session_id: Annotated[str, Field(description="Session ID from rlm_start")],
    code: Annotated[str, Field(description=(
        "Python code to execute in the persistent sandbox (variables persist across calls). "
        "Built-in helpers: read_file(path, max_bytes_read=None, start_line=None, end_line=None), read_files(paths), "
        "grep(pattern, path='.', max_matches=None, allow_partial=False), "
        "grep_summary(pattern, path='.'), grep_read(pattern, path='.', max_files=10, context_lines=0, max_matches=None, max_bytes_read=None), "
        "glob_files(pattern), tree(path='.', max_depth=3), outline(path), enclosing(path, line), "
        "find_assignments(path, target), rlm_finalize(value). "
        "Allowed imports ONLY: re, json, collections, math, fnmatch, itertools, functools, "
        "operator, string, textwrap, difflib, statistics. "
        "Blocked (never use): os, glob, pathlib, ast, open(), exec(), eval(), getattr(), dir(), type(), "
        "str.format()/format_map() (use f-strings) — use glob_files/read_file/tree/re instead. "
        "Batch multiple operations per call and print only concise summaries rather than raw dumps."
    ))],
    detail_level: Annotated[Literal["compact", "usage", "full"], Field(
        description="Response payload level: compact=stdout+error, usage=add usage metrics, full=add variable details"
    )] = "compact",
    max_new_variables: Annotated[int, Field(
        description="When detail_level=full, cap returned new_variables list to this size",
        ge=1,
        le=200,
    )] = 20,
    end_session: Annotated[bool, Field(
        description="If true, close the session immediately after code succeeds (avoids a separate rlm_end turn). Stays open on error."
    )] = False,
) -> str:
    """Execute Python in the session sandbox and return printed stdout. Set end_session=true on your final step to close the session without a separate rlm_end call."""
    return await _run_session_operation(_rlm_execute, session_id, code, detail_level, max_new_variables, end_session)


@mcp.tool()
async def rlm_end(
    session_id: Annotated[str, Field(description="Session ID to end")],
) -> str:
    """End an RLM exploration session and free resources. Prefer passing end_session=true on your final rlm_execute call to save a turn."""
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
