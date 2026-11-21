import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from anthropic import Anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3


def request_timeout(limit: float, deadline: Callable[[], float | None] | None) -> float:
    end = deadline() if deadline is not None else None
    if end is None:
        return limit
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Execution deadline expired before provider request")
    return min(limit, remaining)


def get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is required for llm_query()"
        )
    # Retries are managed by the bridge so each request can be budgeted.
    return Anthropic(api_key=api_key, max_retries=0)


def make_llm_query(
    client: Anthropic | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    on_retry=None,
    *,
    system_prompt: str | None = None,
    deadline: Callable[[], float | None] | None = None,
):
    _client = client or get_client()
    _model = model or os.environ.get("RLM_SUB_MODEL", DEFAULT_MODEL)
    _timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(os.environ.get("RLM_SUB_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    )
    _max_retries = (
        max_retries
        if max_retries is not None
        else int(os.environ.get("RLM_SUB_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    )
    request_options = {"system": system_prompt} if system_prompt is not None else {}

    def llm_query(prompt: str, context: str = "") -> str:
        if not prompt:
            raise ValueError("prompt cannot be empty")

        messages = []
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        attempts = max(1, _max_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            timeout = request_timeout(_timeout, deadline)
            if attempt and on_retry is not None:
                on_retry()
            try:
                response = _client.messages.create(
                    model=_model,
                    max_tokens=1024,
                    messages=messages,
                    timeout=timeout,
                    **request_options,
                )
                if not response.content:
                    return ""
                first = response.content[0]
                return getattr(first, "text", str(first))
            except Exception as e:  # noqa: BLE001 - retried then re-raised below
                last_error = e
                if attempt < attempts - 1:
                    logger.warning(
                        "llm_query attempt %d/%d failed: %s", attempt + 1, attempts, e
                    )
                    time.sleep(request_timeout(min(2 ** min(attempt, 3) * 0.5, 4.0), deadline))
        raise last_error

    return llm_query


def _explore_tool_specs(allow_recurse: bool) -> list[dict]:
    specs = [
        {
            "name": "grep_summary",
            "description": "Search the codebase for a regex pattern. Returns a compact, file-grouped summary of matches.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory or file to search (default '.')"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "read_file",
            "description": "Read a single file's contents by path, relative to the corpus root.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "glob_files",
            "description": "List files matching a glob pattern, relative to the corpus root.",
            "input_schema": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
        {
            "name": "tree",
            "description": "Show the directory structure under a path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer"},
                },
            },
        },
    ]
    if allow_recurse:
        specs.append({
            "name": "recurse",
            "description": (
                "Delegate a focused sub-question to a fresh child explorer agent. "
                "Use this to decompose a large question into independent parts; the "
                "child explores on its own and returns only a concise answer."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        })
    return specs


def make_recursive_query(
    client: Anthropic | None = None,
    helpers: dict | None = None,
    model: str | None = None,
    max_depth: int = 2,
    max_iterations: int = 6,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    on_model_call=None,
    gate=None,
    deadline: Callable[[], float | None] | None = None,
):
    """Build a genuinely recursive explorer.

    Unlike a flat ``llm_query``, the returned callable runs a tool-use loop: the
    sub-model issues grep/read/glob/tree calls that execute server-side against
    ``helpers`` and can spawn depth-limited child explorers via a ``recurse``
    tool. Only the sub-agent's final synthesized text is returned to the caller,
    so the raw file content it reads never enters the caller's context window.

    ``gate(question, text)`` optionally filters each tool result before it joins
    the sub-agent's message history. The history is re-sent on every turn, so
    what the gate drops is saved once per remaining turn rather than once.
    """
    _client = client or get_client()
    _helpers = helpers or {}
    _model = model or os.environ.get("RLM_SUB_MODEL", DEFAULT_MODEL)
    _timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(os.environ.get("RLM_SUB_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    )
    _max_retries = (
        max_retries
        if max_retries is not None
        else int(os.environ.get("RLM_SUB_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    )

    def _create(messages, tools):
        attempts = max(1, _max_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            timeout = request_timeout(_timeout, deadline)
            if on_model_call is not None:
                on_model_call()
            try:
                return _client.messages.create(
                    model=_model,
                    max_tokens=2048,
                    messages=messages,
                    tools=tools,
                    timeout=timeout,
                )
            except Exception as e:  # noqa: BLE001 - retried then re-raised
                last_error = e
                if attempt < attempts - 1:
                    time.sleep(request_timeout(min(2 ** min(attempt, 3) * 0.5, 4.0), deadline))
        raise last_error

    def _run_tool(name: str, args: dict, depth: int) -> str:
        if name == "recurse":
            child_prompt = args.get("prompt", "")
            return _run(child_prompt, context="", depth=depth - 1)
        helper = _helpers.get(name)
        if helper is None:
            return f"[error: unknown tool '{name}']"
        try:
            return str(helper(**args))
        except Exception as e:  # noqa: BLE001 - surfaced to the sub-model as a tool result
            return f"[error: {type(e).__name__}: {e}]"

    def _run(prompt: str, context: str = "", depth: int | None = None) -> str:
        if not prompt:
            raise ValueError("prompt cannot be empty")
        depth = max_depth if depth is None else depth
        tools = _explore_tool_specs(allow_recurse=depth > 0)
        allowed_tools = {spec["name"] for spec in tools}

        content = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
        messages = [{"role": "user", "content": content}]

        last_text = ""
        for _ in range(max_iterations):
            response = _create(messages, tools)
            text_parts = [
                getattr(b, "text", "") for b in response.content
                if getattr(b, "type", None) == "text"
            ]
            if text_parts:
                last_text = "".join(text_parts)

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if getattr(response, "stop_reason", None) != "tool_use" or not tool_uses:
                return last_text

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_uses:
                if block.name not in allowed_tools:
                    result = f"[error: unknown tool '{block.name}']"
                else:
                    result = _run_tool(block.name, dict(block.input), depth)
                    if gate is not None and block.name != "recurse":
                        result = gate(prompt, result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})

        return last_text

    def recursive_query(prompt: str, context: str = "") -> str:
        return _run(prompt, context)

    return recursive_query


def make_llm_query_batched(llm_query_fn, max_workers: int = 8, *, on_batch=None):
    def llm_query_batched(prompts: list[str], context: str = "") -> list[str]:
        if not prompts:
            return []

        if len(prompts) > 64:
            raise ValueError("A model batch accepts at most 64 prompts")
        if on_batch is not None:
            on_batch(len(prompts))

        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(llm_query_fn, prompt, context): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    results[i] = f"[ERROR] {type(e).__name__}: {e}"
        return [results[i] for i in range(len(prompts))]

    return llm_query_batched
