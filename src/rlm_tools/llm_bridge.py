from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from anthropic import Anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
_VALID_PROVIDERS = frozenset({"anthropic", "gemini", "openai"})
_PROVIDER_ALIASES = {
    "openai-compatible": "openai",
    "openai_compatible": "openai",
}


def request_timeout(limit: float, deadline: Callable[[], float | None] | None) -> float:
    end = deadline() if deadline is not None else None
    if end is None:
        return limit
    remaining = end - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Execution deadline expired before provider request")
    return min(limit, remaining)


class _OpenAICompatStatusError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (ValueError, TypeError)):
        return False
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status_code, int):
        return status_code >= 500 or status_code in {408, 409, 429}
    return True


def _extract_text(content) -> str:
    if not content:
        return ""
    text_parts = [
        block.text
        for block in content
        if getattr(block, "type", None) == "text"
        and isinstance(getattr(block, "text", None), str)
    ]
    if text_parts:
        return "".join(text_parts)
    # Lightweight clients may return text blocks without a typed discriminator.
    return "".join(
        block.text for block in content
        if not isinstance(getattr(block, "type", None), str)
        and isinstance(getattr(block, "text", None), str)
    )


def _normalize_provider_name(raw: str) -> str:
    normalized = raw.strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def resolve_provider(provider: str | None = None, client: Any = None) -> str:
    """Resolve the active sub-LLM provider ('anthropic', 'gemini', or 'openai')."""
    if provider is not None:
        normalized = _normalize_provider_name(provider)
        if normalized and normalized != "auto":
            if normalized not in _VALID_PROVIDERS:
                raise ValueError(
                    f"Unsupported sub-LLM provider {provider!r}; expected 'anthropic', 'gemini', 'openai', or 'auto'"
                )
            return normalized

    if client is not None:
        if hasattr(client, "interactions") and not hasattr(client, "messages"):
            return "gemini"
        if hasattr(client, "chat") and not hasattr(client, "messages"):
            return "openai"
        return "anthropic"

    env_provider = _normalize_provider_name(os.environ.get("RLM_SUB_PROVIDER", "auto"))
    if env_provider in {"", "auto"}:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
            return "openai"
        return "anthropic"
    if env_provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unsupported RLM_SUB_PROVIDER {env_provider!r}; expected 'anthropic', 'gemini', 'openai', or 'auto'"
        )
    return env_provider


def has_configured_provider() -> bool:
    """Return True when the configured (or auto-detected) sub-LLM provider has credentials."""
    env_provider = _normalize_provider_name(os.environ.get("RLM_SUB_PROVIDER", "auto"))
    if env_provider in {"", "auto"}:
        return bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_BASE_URL")
        )
    if env_provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if env_provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    if env_provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"))
    return False


def _normalize_openai_endpoint(base_url: str) -> str:
    raw = base_url.strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Invalid OPENAI_BASE_URL {base_url!r}; expected an http or https URL"
        )
    cleaned = raw.rstrip("/")
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


class _OpenAICompatCompletions:
    """Minimal Chat Completions client over httpx for OpenAI and compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        transport: Any = None,
    ):
        import httpx

        self._api_key = (api_key or "").strip()
        self._endpoint = _normalize_openai_endpoint(base_url)
        # Retries are managed by the bridge so each request can be budgeted.
        self._http = httpx.Client(transport=transport)

    def close(self) -> None:
        self._http.close()

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        **extra: Any,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools is not None:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens
        payload.update(extra)

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = self._http.post(
            self._endpoint,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise _OpenAICompatStatusError(
                response.status_code,
                f"OpenAI-compatible API error (status={response.status_code}): {response.text[:400]}",
            )

        data = response.json()
        choices = []
        for choice in data.get("choices") or []:
            msg = choice.get("message") or {}
            tool_calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tc_ns = SimpleNamespace(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=SimpleNamespace(
                        name=fn.get("name", ""),
                        arguments=fn.get("arguments", "{}"),
                    ),
                )
                if "extra_content" in tc and tc["extra_content"] is not None:
                    tc_ns.extra_content = tc["extra_content"]
                tool_calls.append(tc_ns)

            choices.append(
                SimpleNamespace(
                    index=choice.get("index", 0),
                    finish_reason=choice.get("finish_reason"),
                    message=SimpleNamespace(
                        role=msg.get("role", "assistant"),
                        content=msg.get("content"),
                        tool_calls=tool_calls or None,
                    ),
                )
            )
        return SimpleNamespace(choices=choices)


class _OpenAICompatClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        transport: Any = None,
    ):
        completions = _OpenAICompatCompletions(
            api_key=api_key,
            base_url=base_url,
            transport=transport,
        )
        self._completions = completions
        self.chat = SimpleNamespace(completions=completions)

    def close(self) -> None:
        self._completions.close()


def get_client(provider: str | None = None) -> Anthropic | Any:
    """Lazily import and construct the sub-LLM provider client with internal retries disabled."""
    resolved = resolve_provider(provider)
    if resolved == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required for Gemini llm_query()"
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is required for Gemini sub-LLM calls (requires Python 3.11+); "
                "install with the 'gemini' extra (rlm-tools[gemini])"
            ) from exc
        # Retries are managed by the bridge so each request can be budgeted.
        return genai.Client(
            api_key=api_key,
            http_options={"retry_options": {"attempts": 1}},
        )

    if resolved == "openai":
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
        if not api_key and not base_url:
            raise RuntimeError(
                "OPENAI_API_KEY or OPENAI_BASE_URL environment variable is required for OpenAI-compatible llm_query()"
            )
        return _OpenAICompatClient(
            api_key=api_key or None,
            base_url=base_url or DEFAULT_OPENAI_BASE_URL,
        )

    return Anthropic(api_key=api_key, max_retries=0)


def _openai_token_limit_kwarg(model: str, limit: int) -> dict[str, int]:
    leaf = model.rsplit("/", 1)[-1].strip().lower()
    if leaf.startswith(("o1", "o3", "o4", "gpt-5")):
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


def _extract_openai_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_gemini_text(interaction: Any) -> str:
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


_DEFAULT_MODELS = {
    "anthropic": DEFAULT_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
}


def make_llm_query(
    client: Any = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    on_retry=None,
    *,
    system_prompt: str | None = None,
    deadline: Callable[[], float | None] | None = None,
    provider: str | None = None,
):
    _client = client if client is not None else (get_client() if provider is None else get_client(provider))
    _provider = resolve_provider(provider, _client)
    default_model = _DEFAULT_MODELS[_provider]
    _model = model or os.environ.get("RLM_SUB_MODEL") or default_model
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

        user_content = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
        messages = [{"role": "user", "content": user_content}]

        attempts = max(1, _max_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            timeout = request_timeout(_timeout, deadline)
            if attempt and on_retry is not None:
                on_retry()
            try:
                if _provider == "gemini":
                    gemini_kwargs: dict[str, Any] = {
                        "model": _model,
                        "input": user_content,
                        "timeout": timeout,
                        "generation_config": {"max_output_tokens": 1024},
                    }
                    if system_prompt is not None:
                        gemini_kwargs["system_instruction"] = system_prompt
                    response = _client.interactions.create(**gemini_kwargs)
                    if getattr(response, "status", None) == "failed":
                        raise RuntimeError(
                            f"Gemini interaction failed: {getattr(response, 'errors', None)}"
                        )
                    return _extract_gemini_text(response)

                if _provider == "openai":
                    oa_messages = []
                    if system_prompt is not None:
                        oa_messages.append({"role": "system", "content": system_prompt})
                    oa_messages.append({"role": "user", "content": user_content})
                    response = _client.chat.completions.create(
                        model=_model,
                        messages=oa_messages,
                        timeout=timeout,
                        **_openai_token_limit_kwarg(_model, 1024),
                    )
                    choices = getattr(response, "choices", None) or []
                    if not choices:
                        return ""
                    return _extract_openai_text(getattr(choices[0], "message", None))

                response = _client.messages.create(
                    model=_model,
                    max_tokens=1024,
                    messages=messages,
                    timeout=timeout,
                    **request_options,
                )
                return _extract_text(response.content)
            except Exception as e:  # noqa: BLE001 - retried then re-raised below
                if not _is_retryable(e):
                    raise
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


def _gemini_explore_tool_specs(allow_recurse: bool) -> list[dict]:
    return [
        {
            "type": "function",
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["input_schema"],
        }
        for spec in _explore_tool_specs(allow_recurse)
    ]


def _openai_explore_tool_specs(allow_recurse: bool) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
            },
        }
        for spec in _explore_tool_specs(allow_recurse)
    ]


def make_recursive_query(
    client: Any = None,
    helpers: dict | None = None,
    model: str | None = None,
    max_depth: int = 2,
    max_iterations: int = 6,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    on_model_call=None,
    gate=None,
    deadline: Callable[[], float | None] | None = None,
    *,
    provider: str | None = None,
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
    _client = client if client is not None else (get_client() if provider is None else get_client(provider))
    _provider = resolve_provider(provider, _client)
    _helpers = helpers or {}
    default_model = _DEFAULT_MODELS[_provider]
    _model = model or os.environ.get("RLM_SUB_MODEL") or default_model
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

    def _retry_model_call(call_fn: Callable[[float], Any]) -> Any:
        attempts = max(1, _max_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            timeout = request_timeout(_timeout, deadline)
            if on_model_call is not None:
                on_model_call()
            try:
                return call_fn(timeout)
            except Exception as e:  # noqa: BLE001 - retried then re-raised
                if not _is_retryable(e):
                    raise
                last_error = e
                if attempt < attempts - 1:
                    time.sleep(request_timeout(min(2 ** min(attempt, 3) * 0.5, 4.0), deadline))
        raise last_error

    def _run_tool(name: str, args: dict, depth: int) -> str:
        if name == "recurse":
            child_prompt = args.get("prompt")
            if not isinstance(child_prompt, str) or not child_prompt.strip():
                return "[error: recurse requires a non-empty string 'prompt']"
            return _run(child_prompt, context="", depth=depth - 1)
        helper = _helpers.get(name)
        if helper is None:
            return f"[error: unknown tool '{name}']"
        try:
            return str(helper(**args))
        except Exception as e:  # noqa: BLE001 - surfaced to the sub-model as a tool result
            return f"[error: {type(e).__name__}: {e}]"

    def _execute_tool_call(
        name: str, args: dict, depth: int, allowed_tools: set[str], prompt: str
    ) -> str:
        if name not in allowed_tools:
            return f"[error: unknown tool '{name}']"
        result = _run_tool(name, args, depth)
        if gate is not None and name != "recurse":
            result = gate(prompt, result)
        return result

    def _run_anthropic(prompt: str, content: str, depth: int) -> str:
        tools = _explore_tool_specs(allow_recurse=depth > 0)
        allowed_tools = {spec["name"] for spec in tools}
        messages = [{"role": "user", "content": content}]

        last_text = ""
        for iteration in range(max_iterations):
            def _send(timeout: float) -> Any:
                return _client.messages.create(
                    model=_model,
                    max_tokens=2048,
                    messages=messages,
                    tools=tools,
                    timeout=timeout,
                )

            response = _retry_model_call(_send)
            text_parts = [
                getattr(b, "text", "") for b in response.content
                if getattr(b, "type", None) == "text"
            ]
            if text_parts:
                last_text = "".join(text_parts)

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if getattr(response, "stop_reason", None) != "tool_use" or not tool_uses:
                return last_text
            if iteration == max_iterations - 1:
                return last_text

            messages.append({"role": "assistant", "content": response.content})
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _execute_tool_call(
                        block.name,
                        dict(block.input) if isinstance(block.input, dict) else {},
                        depth,
                        allowed_tools,
                        prompt,
                    ),
                }
                for block in tool_uses
            ]
            messages.append({"role": "user", "content": tool_results})

        return last_text

    def _run_gemini(prompt: str, content: str, depth: int) -> str:
        tools = _gemini_explore_tool_specs(allow_recurse=depth > 0)
        allowed_tools = {spec["name"] for spec in tools}
        previous_id: str | None = None
        next_input: Any = content

        last_text = ""
        for iteration in range(max_iterations):
            def _send(timeout: float) -> Any:
                req_kwargs: dict[str, Any] = {
                    "model": _model,
                    "input": next_input,
                    "tools": tools,
                    "generation_config": {"max_output_tokens": 2048},
                    "timeout": timeout,
                }
                if previous_id is not None:
                    req_kwargs["previous_interaction_id"] = previous_id
                interaction = _client.interactions.create(**req_kwargs)
                if getattr(interaction, "status", None) == "failed":
                    raise RuntimeError(
                        f"Gemini interaction failed: {getattr(interaction, 'errors', None)}"
                    )
                return interaction

            interaction = _retry_model_call(_send)
            text_here = _extract_gemini_text(interaction)
            if text_here:
                last_text = text_here

            steps = getattr(interaction, "steps", None) or []
            function_calls = [s for s in steps if getattr(s, "type", None) == "function_call"]
            if not function_calls or iteration == max_iterations - 1:
                return last_text

            tool_results = [
                {
                    "type": "function_result",
                    "name": fc.name,
                    "call_id": fc.id,
                    "result": [
                        {
                            "type": "text",
                            "text": _execute_tool_call(
                                fc.name,
                                dict(getattr(fc, "arguments", None) or {}),
                                depth,
                                allowed_tools,
                                prompt,
                            ),
                        }
                    ],
                }
                for fc in function_calls
            ]
            previous_id = getattr(interaction, "id", None)
            next_input = tool_results

        return last_text

    def _run_openai(prompt: str, content: str, depth: int) -> str:
        tools = _openai_explore_tool_specs(allow_recurse=depth > 0)
        allowed_tools = {spec["function"]["name"] for spec in tools}
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

        last_text = ""
        for iteration in range(max_iterations):
            def _send(timeout: float) -> Any:
                return _client.chat.completions.create(
                    model=_model,
                    messages=messages,
                    tools=tools,
                    timeout=timeout,
                    **_openai_token_limit_kwarg(_model, 2048),
                )

            response = _retry_model_call(_send)
            choices = getattr(response, "choices", None) or []
            if not choices:
                return last_text
            message = choices[0].message
            text_here = _extract_openai_text(message)
            if text_here:
                last_text = text_here

            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls or iteration == max_iterations - 1:
                return last_text

            serialized_tool_calls = []
            for idx, tc in enumerate(tool_calls):
                fn = tc.function
                call_id = getattr(tc, "id", None) or f"call_{idx}"
                entry: dict[str, Any] = {
                    "id": call_id,
                    "type": getattr(tc, "type", "function") or "function",
                    "function": {
                        "name": fn.name,
                        "arguments": getattr(fn, "arguments", "{}") or "{}",
                    },
                }
                extra_content = getattr(tc, "extra_content", None)
                if extra_content is not None:
                    entry["extra_content"] = extra_content
                serialized_tool_calls.append(entry)

            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", None),
                "tool_calls": serialized_tool_calls,
            })

            for idx, tc in enumerate(tool_calls):
                call_id = serialized_tool_calls[idx]["id"]
                fn_name = tc.function.name
                raw_args = getattr(tc.function, "arguments", None)
                if isinstance(raw_args, dict):
                    parsed_args = raw_args
                    parse_err = None
                else:
                    try:
                        loaded = json.loads(raw_args or "{}")
                        if not isinstance(loaded, dict):
                            parse_err = f"[error: invalid tool arguments for '{fn_name}': expected JSON object]"
                            parsed_args = {}
                        else:
                            parsed_args = loaded
                            parse_err = None
                    except Exception as exc:  # noqa: BLE001 - surfaced cleanly as tool result
                        parse_err = f"[error: invalid tool arguments for '{fn_name}': {exc}]"
                        parsed_args = {}

                result = (
                    parse_err
                    if parse_err is not None
                    else _execute_tool_call(fn_name, parsed_args, depth, allowed_tools, prompt)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })

        return last_text

    def _run(prompt: str, context: str = "", depth: int | None = None) -> str:
        if not prompt:
            raise ValueError("prompt cannot be empty")
        depth = max_depth if depth is None else depth
        content = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
        if _provider == "gemini":
            return _run_gemini(prompt, content, depth)
        if _provider == "openai":
            return _run_openai(prompt, content, depth)
        return _run_anthropic(prompt, content, depth)

    def recursive_query(prompt: str, context: str = "") -> str:
        return _run(prompt, context)

    return recursive_query


def make_llm_query_batched(llm_query_fn, max_workers: int = 8, *, on_batch=None):
    def llm_query_batched(prompts: list[str], context: str = "") -> list[str]:
        if isinstance(prompts, (str, bytes)):
            raise TypeError("prompts must be a sequence of strings, not a single string")
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
