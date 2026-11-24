"""Live Gemini Interactions API A/B benchmark comparing rlm-tools vs native file tools."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from fnmatch import translate
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable

from mcp.types import LATEST_PROTOCOL_VERSION

from evals.aggregate_benchmark import format_benchmark_summary
from evals.artifacts import ensure_artifact_dir, write_artifact
from evals.grade_eval import load_eval_specs
from evals.heavy_corpus import generate_heavy_corpus
from evals.run_prompt_benchmark import run_benchmark_from_results
from rlm_tools._workspace import _SKIP_DIRS
from rlm_tools.access_policy import is_private_component
from rlm_tools.helpers import _BINARY_EXTENSIONS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_TURNS = 25
DEFAULT_REQUEST_TIMEOUT_SECONDS = 75.0
DEFAULT_CONCURRENCY = 1
DEFAULT_PACE_SECONDS = 3.0
DEFAULT_RETRY_DELAYS = (5.0, 12.0, 25.0, 45.0, 60.0)
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_ALLOWED_SCHEMA_KEYS = {
    "type",
    "description",
    "properties",
    "required",
    "enum",
    "items",
    "minimum",
    "maximum",
}


def load_gemini_api_key(repo_root: Path = REPO_ROOT) -> str:
    """Load GEMINI_API_KEY from the process environment or repo .env file."""
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    env_path = repo_root / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export ") or line.startswith("export\t"):
                line = line[6:].lstrip()
            key, sep, raw_value = line.partition("=")
            if not sep or key.strip() != "GEMINI_API_KEY":
                continue
            value = raw_value.strip()
            quoted = re.match(r'^([\'"])(.*?)\1(?:\s+#.*)?$', value)
            if quoted:
                value = quoted.group(2).strip()
            else:
                value = re.sub(r"\s+#.*$", "", value).strip()
            if value:
                return value

    raise RuntimeError(
        "GEMINI_API_KEY not found in environment or .env; add GEMINI_API_KEY=... to .env"
    )


def sanitize_schema_for_gemini(
    schema: dict[str, Any],
    *,
    _defs: dict[str, Any] | None = None,
    _seen_refs: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Convert MCP/Pydantic JSON schemas to the subset accepted by Gemini tools."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    defs = dict(_defs or {})
    for defs_key in ("$defs", "definitions"):
        if isinstance(schema.get(defs_key), dict):
            defs.update(schema[defs_key])

    ref = schema.get("$ref")
    if isinstance(ref, str) and ref not in _seen_refs:
        ref_name = ref.rsplit("/", 1)[-1]
        if ref_name in defs and isinstance(defs[ref_name], dict):
            merged = dict(defs[ref_name])
            if "description" in schema and "description" not in merged:
                merged["description"] = schema["description"]
            return sanitize_schema_for_gemini(
                merged, _defs=defs, _seen_refs=_seen_refs | {ref}
            )

    for union_key in ("anyOf", "oneOf"):
        variants = schema.get(union_key)
        if isinstance(variants, list):
            non_null = [
                v for v in variants if isinstance(v, dict) and v.get("type") != "null"
            ]
            if len(non_null) == 1:
                chosen = dict(non_null[0])
                if "description" in schema and "description" not in chosen:
                    chosen["description"] = schema["description"]
                return sanitize_schema_for_gemini(
                    chosen, _defs=defs, _seen_refs=_seen_refs
                )

    cleaned: dict[str, Any] = {}
    for key in _ALLOWED_SCHEMA_KEYS:
        if key not in schema:
            continue
        value = schema[key]
        if key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {
                prop_name: sanitize_schema_for_gemini(
                    prop_schema, _defs=defs, _seen_refs=_seen_refs
                )
                for prop_name, prop_schema in value.items()
                if isinstance(prop_schema, dict)
            }
        elif key == "items" and isinstance(value, dict):
            cleaned["items"] = sanitize_schema_for_gemini(
                value, _defs=defs, _seen_refs=_seen_refs
            )
        else:
            cleaned[key] = value

    if cleaned.get("type") == "object" and "properties" not in cleaned:
        cleaned["properties"] = {}
    return cleaned


class StdioRlmClient:
    """Async stdio MCP client that manages an rlm-tools server across benchmark trials."""

    def __init__(self, target: Path):
        self.target = target.resolve()
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._stderr_task: asyncio.Task[None] | None = None
        self._rpc_lock = asyncio.Lock()
        self._next_id = 0
        self._tools: list[dict[str, Any]] = []

    async def _drain_stderr(self) -> None:
        assert self._process and self._process.stderr
        while True:
            line = await self._process.stderr.readline()
            if not line:
                break
            self._stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())

    async def _send(
        self, method: str, params: dict[str, Any], *, notification: bool = False
    ) -> dict[str, Any] | None:
        assert self._process and self._process.stdin and self._process.stdout
        async with self._rpc_lock:
            payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
            if not notification:
                self._next_id += 1
                payload["id"] = self._next_id

            self._process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            await self._process.stdin.drain()
            if notification:
                return None

            while True:
                raw = await self._process.stdout.readline()
                if not raw:
                    stderr_tail = "\n".join(self._stderr_lines)
                    raise RuntimeError(
                        f"rlm-tools MCP server exited unexpectedly during {method}. "
                        f"Stderr:\n{stderr_tail}"
                    )
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                reply = json.loads(line)
                if reply.get("id") != payload["id"]:
                    continue
                if "error" in reply:
                    raise RuntimeError(f"MCP error in {method}: {reply['error']}")
                return reply["result"]

    async def __aenter__(self) -> "StdioRlmClient":
        allowed_keys = ("PATH", "HOME", "USER", "TMPDIR", "LANG", "LC_ALL")
        child_env = {k: os.environ[k] for k in allowed_keys if k in os.environ}
        child_env["RLM_ALLOWED_ROOTS"] = str(self.target)
        child_env["RLM_EXTERNAL_ROOTS"] = ""
        child_env["RLM_RECURSE_GATE"] = "0"
        child_env["RLM_MAX_SESSIONS"] = "16"

        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "rlm_tools",
            cwd=str(self.target),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
            limit=16 * 1024 * 1024,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        await asyncio.wait_for(
            self._send(
                "initialize",
                {
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "gemini-ab-eval", "version": "1.0"},
                },
            ),
            timeout=90.0,
        )
        await self._send("notifications/initialized", {}, notification=True)
        listed = await asyncio.wait_for(self._send("tools/list", {}), timeout=90.0)
        assert listed is not None
        self._tools = listed.get("tools", [])
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._process is not None:
            if self._process.stdin and not self._process.stdin.is_closing():
                self._process.stdin.close()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

    def tool_declarations(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": sanitize_schema_for_gemini(tool.get("inputSchema", {})),
            }
            for tool in self._tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        result = await asyncio.wait_for(
            self._send("tools/call", {"name": name, "arguments": arguments}),
            timeout=90.0,
        )
        assert result is not None
        is_error = bool(result.get("isError", False))
        content = result.get("content", [])
        if isinstance(content, list) and content:
            parts = [
                block.get("text", json.dumps(block))
                if isinstance(block, dict)
                else str(block)
                for block in content
            ]
            return "\n".join(parts), is_error
        return json.dumps(result), is_error


@asynccontextmanager
async def open_stdio_rlm_client(target: Path):
    async with StdioRlmClient(target) as client:
        yield client


class NativeFileTools:
    """Fair native Read/Grep/Glob baseline scoped to the target directory."""

    MAX_READ_LINES = 2000
    MAX_READ_CHARS = 100_000
    MAX_GREP_MATCHES = 2000
    MAX_GREP_CHARS = 120_000
    MAX_GLOB_RESULTS = 2000

    def __init__(self, target: Path):
        self.target = target.resolve()

    def tool_declarations(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "read_file",
                "description": (
                    "Read a text file by repository-relative path, returning numbered lines. "
                    "Use start_line and max_lines to paginate large files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Repository-relative file path to read.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based starting line number (default: 1).",
                            "minimum": 1,
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum lines to return (default: 500, max: 2000).",
                            "minimum": 1,
                            "maximum": 2000,
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "type": "function",
                "name": "grep",
                "description": (
                    "Search files for a Python regular expression pattern. Returns matching "
                    "lines formatted as file:line: text, or matching file paths when files_only=true."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Python regex pattern to search for.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Repository-relative directory or file to search (default: '.').",
                        },
                        "files_only": {
                            "type": "boolean",
                            "description": "Return only unique matching repository-relative file paths.",
                        },
                        "max_matches": {
                            "type": "integer",
                            "description": "Maximum matches or file paths to return (default: 1000, max: 2000).",
                            "minimum": 1,
                            "maximum": 2000,
                        },
                    },
                    "required": ["pattern"],
                },
            },
            {
                "type": "function",
                "name": "glob_files",
                "description": "Find files matching a repository-relative glob pattern (e.g. '**/*.py').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Relative glob pattern such as '**/*.py' or 'services/*.py'.",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        ]

    def _resolve_path(self, raw_path: str) -> tuple[Path, Path]:
        if not raw_path or "\0" in raw_path:
            raise ValueError("Path cannot be empty")
        supplied = Path(raw_path)
        if supplied.is_absolute():
            candidate = supplied
            try:
                rel = candidate.relative_to(self.target)
            except ValueError as exc:
                resolved = candidate.resolve()
                if not resolved.is_relative_to(self.target):
                    raise PermissionError(f"Access denied outside workspace: {raw_path}") from exc
                rel = resolved.relative_to(self.target)
        else:
            if ".." in supplied.parts:
                raise PermissionError(f"Access denied: path traversal in {raw_path}")
            rel = Path(*[p for p in supplied.parts if p != "."])

        cur = self.target
        for part in rel.parts:
            if is_private_component(part) or part in _SKIP_DIRS:
                raise PermissionError(f"Access denied: restricted path component {part!r}")
            cur = cur / part
            if cur.is_symlink():
                raise PermissionError(f"Symbolic links are blocked: {raw_path}")
        resolved = cur.resolve()
        if not resolved.is_relative_to(self.target):
            raise PermissionError(f"Access denied outside workspace: {raw_path}")
        return cur, rel

    def _iter_files(self, base_dir: Path, base_rel: Path):
        if base_dir.is_file():
            if not base_dir.is_symlink() and base_dir.suffix.lower() not in _BINARY_EXTENSIONS:
                yield base_dir, base_rel
            return
        if not base_dir.is_dir():
            raise FileNotFoundError(f"Path not found: {base_rel}")

        def walk(cur_dir: Path, cur_rel: Path):
            entries = sorted(cur_dir.iterdir(), key=lambda p: p.name)
            for entry in entries:
                name = entry.name
                if name.startswith(".") or name in _SKIP_DIRS or is_private_component(name):
                    continue
                if entry.is_symlink():
                    continue
                rel = cur_rel / name if str(cur_rel) != "." else Path(name)
                if entry.is_dir():
                    yield from walk(entry, rel)
                elif entry.is_file() and entry.suffix.lower() not in _BINARY_EXTENSIONS:
                    yield entry, rel

        yield from walk(base_dir, base_rel)

    def read_file(
        self, path: str, start_line: int = 1, max_lines: int = 500
    ) -> tuple[str, bool]:
        try:
            resolved, rel = self._resolve_path(path)
            if not resolved.exists():
                return f"Error: file not found: {path}", True
            if not resolved.is_file():
                return f"Error: not a regular file: {path}", True
            if resolved.suffix.lower() in _BINARY_EXTENSIONS:
                return f"Error: binary file cannot be read as text: {path}", True
            start = max(1, int(start_line))
            limit = max(1, min(int(max_lines), self.MAX_READ_LINES))
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:  # noqa: BLE001 - surface clean tool error to model
            return f"Error: {exc}", True

        total_lines = len(lines)
        if total_lines == 0:
            return f"{rel.as_posix()} (0 lines, empty file)", False
        if start > total_lines:
            return f"Error: start_line {start} exceeds file length ({total_lines} lines)", True

        end = min(total_lines, start + limit - 1)
        out_lines: list[str] = []
        used_chars = 0
        last_shown = start - 1
        truncated_chars = False

        for line_no in range(start, end + 1):
            formatted = f"L{line_no}: {lines[line_no - 1]}"
            added = len(formatted) + (1 if out_lines else 0)
            if out_lines and used_chars + added > self.MAX_READ_CHARS:
                truncated_chars = True
                break
            if not out_lines and len(formatted) > self.MAX_READ_CHARS:
                formatted = formatted[: self.MAX_READ_CHARS - 20] + " ...[line truncated]"
                out_lines.append(formatted)
                last_shown = line_no
                truncated_chars = True
                break
            out_lines.append(formatted)
            used_chars += added
            last_shown = line_no

        text = "\n".join(out_lines)
        if last_shown < total_lines or truncated_chars:
            text += f"\n... [truncated: showing lines {start}-{last_shown} of {total_lines}]"
        return text, False

    def grep(
        self,
        pattern: str,
        path: str = ".",
        files_only: bool = False,
        max_matches: int = 1000,
    ) -> tuple[str, bool]:
        try:
            compiled = re.compile(pattern)
            resolved, rel = self._resolve_path(path or ".")
            if not resolved.exists():
                return f"Error: path not found: {path}", True
            limit = max(1, min(int(max_matches), self.MAX_GREP_MATCHES))
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}", True

        results: list[str] = []
        used_chars = 0
        truncated = False

        for file_path, file_rel in self._iter_files(resolved, rel):
            rel_posix = file_rel.as_posix()
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_no, line in enumerate(content.splitlines(), 1):
                if not compiled.search(line):
                    continue
                if files_only:
                    results.append(rel_posix)
                    break
                rendered = f"{rel_posix}:{line_no}: {line.rstrip()}"
                if len(rendered) > 4000:
                    rendered = rendered[:3980] + " ...[truncated]"
                added = len(rendered) + (1 if results else 0)
                if results and used_chars + added > self.MAX_GREP_CHARS:
                    truncated = True
                    break
                results.append(rendered)
                used_chars += added
                if len(results) >= limit:
                    truncated = True
                    break
            if truncated or len(results) >= limit:
                truncated = truncated or (len(results) >= limit)
                break

        if not results:
            return "No matches found.", False

        output = "\n".join(results)
        if truncated:
            output += f"\n... [truncated: showing first {len(results)} {'files' if files_only else 'matches'}]"
        return output, False

    def glob_files(self, pattern: str) -> tuple[str, bool]:
        if not pattern or "\0" in pattern:
            return "Error: glob pattern cannot be empty", True
        pat_path = PurePosixPath(pattern)
        parts = pat_path.parts
        if pat_path.is_absolute() or ".." in parts:
            return f"Error: glob pattern must stay inside workspace: {pattern}", True
        if any("**" in part and part != "**" for part in parts):
            return "Error: '**' must be a full path segment", True
        if not parts or pattern.endswith("/") or parts[-1] == "**":
            return "No matching files found.", False

        selectors = [
            None if part == "**" else re.compile(translate(part))
            for part in parts
        ]

        def matches(rel: Path) -> bool:
            states = {0}
            for segment in rel.parts:
                next_states: set[int] = set()
                for idx in states:
                    while idx < len(parts) and selectors[idx] is None:
                        next_states.add(idx)
                        idx += 1
                    if idx < len(parts) and selectors[idx] is not None and selectors[idx].match(segment):
                        next_states.add(idx + 1)
                states = next_states
                if not states:
                    return False
            # Handle trailing optional '**' segments (none expected after guard, safe anyway)
            return len(parts) in states

        matched: list[str] = []
        truncated = False
        for _, file_rel in self._iter_files(self.target, Path(".")):
            if matches(file_rel):
                matched.append(file_rel.as_posix())
                if len(matched) >= self.MAX_GLOB_RESULTS:
                    truncated = True
                    break

        if not matched:
            return "No matching files found.", False
        out = "\n".join(matched)
        if truncated:
            out += f"\n... [truncated at {self.MAX_GLOB_RESULTS} files]"
        return out, False

    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        try:
            if name == "read_file":
                return self.read_file(**arguments)
            if name == "grep":
                return self.grep(**arguments)
            if name == "glob_files":
                return self.glob_files(**arguments)
            return f"Error: unknown tool '{name}'", True
        except TypeError as exc:
            return f"Error: invalid arguments for {name}: {exc}", True


def _error_status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    return status if isinstance(status, int) else None


def _error_detail_text(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if body is not None and str(body).strip():
        raw = str(body)
    else:
        raw = str(exc)
    return " ".join(raw.split())


def _format_api_error(exc: Exception) -> str:
    status = _error_status_code(exc)
    detail = _error_detail_text(exc)
    return f"{type(exc).__name__} (status={status}): {detail[:240]}"


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        raw_header = headers.get("retry-after")
        if raw_header is not None:
            try:
                val = float(str(raw_header).strip())
            except ValueError:
                val = None
            if val is not None and 0.0 < val <= 120.0:
                return val

    detail = _error_detail_text(exc)
    match = re.search(r"retry in (\d+(?:\.\d+)?)s\b", detail, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if 0.0 < val <= 120.0:
            return val
    return None


def _is_transient_api_error(exc: Exception) -> bool:
    detail = _error_detail_text(exc).lower()
    if (
        "per day" in detail
        or "daily" in detail
        or re.search(r"retry in \d+h", detail)
    ):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = _error_status_code(exc)
    if status in TRANSIENT_STATUS_CODES:
        return True
    name = type(exc).__name__.lower()
    if any(tok in name for tok in ("timeout", "connect", "network", "transport", "noresponse")):
        return True
    return any(tok in detail for tok in ("429", "rate limit", "resource exhausted", "503", "unavailable"))


class _RequestPacer:
    """Enforces a minimum spacing between consecutive Gemini API requests."""

    def __init__(self, min_interval: float):
        self._min_interval = max(0.0, float(min_interval))
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def wait(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            if self._last_request > 0:
                remaining = self._min_interval - (perf_counter() - self._last_request)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            self._last_request = perf_counter()


async def _create_interaction_with_retry(
    client: Any,
    kwargs: dict[str, Any],
    retry_delays: tuple[float, ...],
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    pacer: _RequestPacer | None = None,
    log_prefix: str = "",
    turn: int = 0,
) -> Any:
    attempts = len(retry_delays) + 1
    call_kwargs = {**kwargs, "timeout": request_timeout}
    for attempt in range(attempts):
        if pacer is not None:
            await pacer.wait()
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(client.interactions.create, **call_kwargs),
                timeout=request_timeout + 30.0,
            )
        except Exception as exc:  # noqa: BLE001 - retried if transient, otherwise raised
            if attempt >= attempts - 1 or not _is_transient_api_error(exc):
                raise
            delay = max(retry_delays[attempt], _retry_after_seconds(exc) or 0.0)
            prefix = f"{log_prefix} " if log_prefix else ""
            print(
                f"  {prefix}turn {turn} attempt {attempt + 1}/{attempts} failed "
                f"({_format_api_error(exc)}); retrying in {delay:.1f}s...",
                flush=True,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("Unreachable retry loop exit")


def _extract_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    fields = (
        "total_input_tokens",
        "total_cached_tokens",
        "total_output_tokens",
        "total_thought_tokens",
        "total_tokens",
        "total_tool_use_tokens",
        "initial_input_tokens",
    )
    out: dict[str, int] = {}
    for field in fields:
        val = getattr(usage, field, None)
        if type(val) is int and val >= 0:
            out[field] = val
    return out


def _extract_output_text(interaction: Any) -> str:
    text = getattr(interaction, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    parts: list[str] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) == "model_output":
            for item in getattr(step, "content", None) or []:
                if getattr(item, "type", None) == "text" and isinstance(getattr(item, "text", None), str):
                    parts.append(item.text)
    return "".join(parts)


async def run_gemini_agent(
    *,
    client: Any,
    model: str,
    prompt: str,
    system_instruction: str,
    tools: list[dict[str, Any]],
    tool_executor: Callable[[str, dict[str, Any]], Any],
    max_turns: int = DEFAULT_MAX_TURNS,
    retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    pacer: _RequestPacer | None = None,
    log_prefix: str = "",
) -> dict[str, Any]:
    """Run a multi-turn Gemini Interactions tool-use loop and collect usage telemetry."""
    started = perf_counter()
    previous_id: str | None = None
    next_input: Any = prompt

    turn_usages: list[dict[str, int]] = []
    steps_trace: list[dict[str, Any]] = []
    tool_calls_count = 0
    tool_result_chars = 0
    final_output = ""
    hit_max_turns = False

    for turn in range(1, max_turns + 1):
        turn_started = perf_counter()
        req_kwargs: dict[str, Any] = {
            "model": model,
            "input": next_input,
            "tools": tools,
            "system_instruction": system_instruction,
        }
        if previous_id is not None:
            req_kwargs["previous_interaction_id"] = previous_id

        interaction = await _create_interaction_with_retry(
            client,
            req_kwargs,
            retry_delays,
            request_timeout=request_timeout,
            pacer=pacer,
            log_prefix=log_prefix,
            turn=turn,
        )
        turn_s = perf_counter() - turn_started
        if getattr(interaction, "status", None) == "failed":
            raise RuntimeError(
                f"Gemini interaction failed on turn {turn}: {getattr(interaction, 'errors', None)}"
            )

        usage_dict = _extract_usage(getattr(interaction, "usage", None))
        turn_usages.append({"turn": turn, **usage_dict})

        steps = getattr(interaction, "steps", None) or []
        function_calls = [s for s in steps if getattr(s, "type", None) == "function_call"]
        text_here = _extract_output_text(interaction)
        if text_here:
            final_output = text_here

        if not function_calls:
            if log_prefix:
                print(
                    f"  {log_prefix} turn {turn}: done "
                    f"in={usage_dict.get('total_input_tokens')} "
                    f"out={usage_dict.get('total_output_tokens')} ({turn_s:.1f}s)",
                    flush=True,
                )
            break

        if log_prefix:
            print(
                f"  {log_prefix} turn {turn}: calls={[fc.name for fc in function_calls]} "
                f"in={usage_dict.get('total_input_tokens')} "
                f"out={usage_dict.get('total_output_tokens')} ({turn_s:.1f}s)",
                flush=True,
            )

        if turn == max_turns:
            hit_max_turns = True
            steps_trace.append({
                "turn": turn,
                "status": getattr(interaction, "status", None),
                "unexecuted_calls": [getattr(fc, "name", "") for fc in function_calls],
            })
            break

        tool_results_input: list[dict[str, Any]] = []
        turn_calls: list[dict[str, Any]] = []
        for fc in function_calls:
            fn_name = fc.name
            fn_args = dict(fc.arguments or {})
            raw_res = tool_executor(fn_name, fn_args)
            if inspect.isawaitable(raw_res):
                result_text, is_error = await raw_res
            else:
                result_text, is_error = raw_res
            tool_calls_count += 1
            tool_result_chars += len(result_text)
            args_serialized = json.dumps(fn_args)
            turn_calls.append({
                "name": fn_name,
                "args_chars": len(args_serialized),
                "result_chars": len(result_text),
                "is_error": is_error,
                "args_preview": args_serialized[:300],
            })
            tool_results_input.append({
                "type": "function_result",
                "name": fn_name,
                "call_id": fc.id,
                "is_error": is_error,
                "result": [{"type": "text", "text": result_text}],
            })

        steps_trace.append({"turn": turn, "calls": turn_calls})
        previous_id = interaction.id
        next_input = tool_results_input

    elapsed_ms = (perf_counter() - started) * 1000.0

    def _sum_field(key: str) -> int | None:
        vals = [u[key] for u in turn_usages if key in u]
        return sum(vals) if len(vals) == len(turn_usages) and turn_usages else None

    input_tokens = _sum_field("total_input_tokens")
    cached_tokens = sum(u.get("total_cached_tokens", 0) for u in turn_usages)
    output_tokens = _sum_field("total_output_tokens")
    thought_tokens = sum(u.get("total_thought_tokens", 0) for u in turn_usages)
    total_tokens = _sum_field("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens + thought_tokens
    input_vals = [u["total_input_tokens"] for u in turn_usages if "total_input_tokens" in u]
    peak_input_tokens = max(input_vals) if input_vals else None
    uncached_input_tokens = (
        max(0, input_tokens - cached_tokens) if input_tokens is not None else None
    )

    return {
        "prompt": prompt,
        "output": final_output,
        "duration_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "thought_tokens": thought_tokens,
        "total_tokens": total_tokens,
        "peak_input_tokens": peak_input_tokens,
        "turns": len(turn_usages),
        "tool_calls": tool_calls_count,
        "tool_result_chars": tool_result_chars,
        "hit_max_turns": hit_max_turns,
        "turn_usage": turn_usages,
        "steps_trace": steps_trace,
    }


def build_detailed_report(
    suite_name: str,
    model: str,
    specs: list[Any],
    raw_results: dict[str, dict[str, Any]],
    destination: Path,
) -> str:
    """Build a markdown table comparing accuracy, tokens, payload chars, turns, and latency."""
    lines = [
        f"# Gemini A/B Benchmark Report ({suite_name})",
        "",
        f"- Model: `{model}`",
        f"- Cases: {len(specs)}",
        "",
        "| Case | Arm | Pass | Total Tokens | Uncached In | Cached | Peak In | Out+Thought | Turns | Calls | Tool Chars | Time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for spec in specs:
        for arm in ("with_rlm", "without_rlm"):
            entry = raw_results[arm][spec.eval_id]
            trials = entry if isinstance(entry, list) else [entry]
            for trial_idx, payload in enumerate(trials, 1):
                label = spec.eval_id if len(trials) == 1 else f"{spec.eval_id}#{trial_idx}"
                grading_path = destination / arm / spec.eval_id / f"run_{trial_idx:03d}" / "grading.json"
                passed_str = "?"
                if grading_path.is_file():
                    score = json.loads(grading_path.read_text(encoding="utf-8")).get("score", 0.0)
                    passed_str = "PASS" if score >= 1.0 else "FAIL"
                out_plus_thought = (payload.get("output_tokens") or 0) + (payload.get("thought_tokens") or 0)
                lines.append(
                    f"| `{label}` | `{arm}` | {passed_str} | "
                    f"{payload.get('total_tokens', 'N/A')} | "
                    f"{payload.get('uncached_input_tokens', 'N/A')} | "
                    f"{payload.get('cached_tokens', 0)} | "
                    f"{payload.get('peak_input_tokens', 'N/A')} | "
                    f"{out_plus_thought} | "
                    f"{payload.get('turns', 0)} | "
                    f"{payload.get('tool_calls', 0)} | "
                    f"{payload.get('tool_result_chars', 0)} | "
                    f"{(payload.get('duration_ms', 0.0) / 1000.0):.1f} |"
                )

    return "\n".join(lines) + "\n"


async def _run_suite(
    *,
    suite_name: str,
    spec_path: Path,
    target: Path,
    model: str,
    iterations: int,
    max_turns: int,
    request_timeout: float,
    concurrency: int,
    pace_seconds: float,
    case_filter: set[str] | None,
    genai_client: Any,
) -> Path:
    specs = load_eval_specs(spec_path)
    if case_filter:
        specs = [s for s in specs if s.eval_id in case_filter]
    if not specs:
        raise SystemExit(f"No matching eval cases in {spec_path}")

    results_root = ensure_artifact_dir(REPO_ROOT / "evals" / "ab_results")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = ensure_artifact_dir(
        results_root / f"{stamp}_gemini_{suite_name}_benchmark", exist_ok=False
    )
    archived_spec = destination / "benchmark_spec.json"
    write_artifact(
        archived_spec,
        json.dumps({"evals": [case.to_dict() for case in specs]}, indent=2),
    )
    print(f"[{suite_name}] Writing benchmark artifacts to {destination}", flush=True)

    system_instruction = (
        "Answer only the question asked and follow its exact output format. "
        "When JSON is requested, return only valid JSON with no extra prose. "
        "Use repo-relative paths for any requested file citations. "
        f"The target codebase is at {target}."
    )

    trials_by_config: dict[str, dict[str, list[dict[str, Any] | None]]] = {
        "with_rlm": {spec.eval_id: [None] * iterations for spec in specs},
        "without_rlm": {spec.eval_id: [None] * iterations for spec in specs},
    }
    native_tools = NativeFileTools(target)
    pacer = _RequestPacer(pace_seconds)

    print(f"[{suite_name}] Starting shared rlm-tools MCP server...", flush=True)
    async with open_stdio_rlm_client(target) as mcp_client:
        mcp_decls = mcp_client.tool_declarations()
        native_decls = native_tools.tool_declarations()
        print(
            f"[{suite_name}] MCP server ready; running {len(specs) * iterations} paired tasks "
            f"(concurrency={concurrency}, pace={pace_seconds}s, request_timeout={request_timeout}s)",
            flush=True,
        )

        async def _run_with_rlm_trial(
            prompt: str, log_prefix: str
        ) -> dict[str, Any]:
            open_sessions: set[str] = set()

            async def tracked_call(
                name: str, arguments: dict[str, Any]
            ) -> tuple[str, bool]:
                text, is_error = await mcp_client.call_tool(name, arguments)
                if not is_error:
                    if name == "rlm_start":
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError:
                            payload = {}
                        sid = payload.get("session_id")
                        if isinstance(sid, str) and sid and not payload.get("session_ended"):
                            open_sessions.add(sid)
                    elif name == "rlm_execute":
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError:
                            payload = {}
                        if payload.get("session_ended") is True:
                            sid = arguments.get("session_id")
                            if isinstance(sid, str):
                                open_sessions.discard(sid)
                    elif name == "rlm_end":
                        sid = arguments.get("session_id")
                        if isinstance(sid, str):
                            open_sessions.discard(sid)
                return text, is_error

            try:
                return await run_gemini_agent(
                    client=genai_client,
                    model=model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    tools=mcp_decls,
                    tool_executor=tracked_call,
                    max_turns=max_turns,
                    request_timeout=request_timeout,
                    pacer=pacer,
                    log_prefix=log_prefix,
                )
            finally:
                for sid in list(open_sessions):
                    try:
                        await mcp_client.call_tool("rlm_end", {"session_id": sid})
                    except Exception:  # noqa: BLE001 - best-effort cleanup
                        pass

        async def _run_pair(index: int, spec: Any, trial: int) -> None:
            prompt = spec.prompt.replace("{target_path}", str(target))
            arms = (
                ("with_rlm", "without_rlm")
                if (index + trial) % 2 == 0
                else ("without_rlm", "with_rlm")
            )
            for arm in arms:
                log_prefix = f"[{suite_name} {spec.eval_id} {trial}/{iterations} {arm}]"
                print(f"Running {log_prefix}", flush=True)
                if arm == "with_rlm":
                    result = await _run_with_rlm_trial(prompt, log_prefix)
                else:
                    result = await run_gemini_agent(
                        client=genai_client,
                        model=model,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        tools=native_decls,
                        tool_executor=native_tools.call_tool,
                        max_turns=max_turns,
                        request_timeout=request_timeout,
                        pacer=pacer,
                        log_prefix=log_prefix,
                    )

                print(
                    f"Completed {log_prefix}: total={result.get('total_tokens')} tok, "
                    f"{result.get('turns')} turns, {result.get('tool_calls')} calls, "
                    f"{result.get('duration_ms', 0.0) / 1000:.1f}s",
                    flush=True,
                )
                suffix = f"_{trial:03d}" if iterations > 1 else ""
                trace_path = destination / f"{index:03d}_{spec.eval_id}_{arm}{suffix}_trace.json"
                write_artifact(trace_path, json.dumps(result, indent=2))
                trials_by_config[arm][spec.eval_id][trial - 1] = result

        pair_args = [
            (index, spec, trial)
            for index, spec in enumerate(specs, start=1)
            for trial in range(1, iterations + 1)
        ]
        if concurrency == 1:
            for index, spec, trial in pair_args:
                await _run_pair(index, spec, trial)
        else:
            sem = asyncio.Semaphore(concurrency)

            async def _run_bounded(index: int, spec: Any, trial: int) -> None:
                async with sem:
                    await _run_pair(index, spec, trial)

            tasks = [
                asyncio.create_task(_run_bounded(index, spec, trial))
                for index, spec, trial in pair_args
            ]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    raw_results = {
        arm: {
            eval_id: trials[0] if iterations == 1 else trials
            for eval_id, trials in by_eval.items()
        }
        for arm, by_eval in trials_by_config.items()
    }
    write_artifact(destination / "raw_results.json", json.dumps(raw_results, indent=2))

    summary = run_benchmark_from_results(archived_spec, destination, raw_results)
    formatted = format_benchmark_summary(summary)
    write_artifact(destination / "benchmark_stdout.txt", formatted + "\n")
    detailed_md = build_detailed_report(suite_name, model, specs, raw_results, destination)
    write_artifact(destination / "detailed_report.md", detailed_md)

    print(formatted, flush=True)
    print("\n" + detailed_md, flush=True)
    return destination


async def _async_main(args: argparse.Namespace) -> None:
    from google import genai

    api_key = load_gemini_api_key(REPO_ROOT)
    genai_client = genai.Client(
        api_key=api_key,
        http_options={"retry_options": {"attempts": 1}},
    )
    case_filter = (
        {c.strip() for c in args.cases.split(",") if c.strip()} if args.cases else None
    )

    if args.suite in {"long_context", "both"}:
        await _run_suite(
            suite_name="long_context",
            spec_path=REPO_ROOT / "evals" / "specs" / "rlm_long_context.json",
            target=(REPO_ROOT / "tests" / "fixtures" / "rlm_corpus_longcontext").resolve(),
            model=args.model,
            iterations=args.iterations,
            max_turns=args.max_turns,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency,
            pace_seconds=args.pace_seconds,
            case_filter=case_filter,
            genai_client=genai_client,
        )

    if args.suite in {"heavy", "both"}:
        with TemporaryDirectory(prefix="rlm-heavy-corpus-") as tmpdir:
            corpus_dir = Path(tmpdir).resolve() / "corpus"
            spec_data = generate_heavy_corpus(corpus_dir)
            spec_path = Path(tmpdir).resolve() / "heavy_spec.json"
            spec_path.write_text(json.dumps(spec_data, indent=2), encoding="utf-8")
            await _run_suite(
                suite_name="heavy",
                spec_path=spec_path,
                target=corpus_dir.resolve(),
                model=args.model,
                iterations=args.iterations,
                max_turns=args.max_turns,
                request_timeout=args.request_timeout,
                concurrency=args.concurrency,
                pace_seconds=args.pace_seconds,
                case_filter=case_filter,
                genai_client=genai_client,
            )

    if args.suite == "custom":
        if not args.spec or not args.target:
            raise SystemExit("--suite custom requires both --spec and --target")
        await _run_suite(
            suite_name="custom",
            spec_path=Path(args.spec).resolve(),
            target=Path(args.target).resolve(),
            model=args.model,
            iterations=args.iterations,
            max_turns=args.max_turns,
            request_timeout=args.request_timeout,
            concurrency=args.concurrency,
            pace_seconds=args.pace_seconds,
            case_filter=case_filter,
            genai_client=genai_client,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live Gemini A/B benchmarks comparing rlm-tools vs native file tools."
    )
    parser.add_argument(
        "--suite",
        choices=["long_context", "heavy", "both", "custom"],
        default="both",
        help="Benchmark suite to run (default: both).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of paired trials per case (default: 1).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max Interactions API turns per case (default: {DEFAULT_MAX_TURNS}).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds for Gemini API calls (default: {DEFAULT_REQUEST_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent case pairs to run in parallel (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=DEFAULT_PACE_SECONDS,
        help=f"Minimum seconds between consecutive Gemini API calls (default: {DEFAULT_PACE_SECONDS}).",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="Optional comma-separated subset of eval_id names to run.",
    )
    parser.add_argument("--spec", help="Eval spec JSON path (for --suite custom).")
    parser.add_argument("--target", help="Target repository path (for --suite custom).")
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.max_turns < 1:
        parser.error("--max-turns must be >= 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.pace_seconds < 0:
        parser.error("--pace-seconds must be >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
