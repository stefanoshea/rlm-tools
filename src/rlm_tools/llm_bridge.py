import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic, APIConnectionError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0

RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError, RateLimitError)


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
    max_retries: int | None = None,
    retry_delay: float | None = None,
):
    _client = client or get_client()
    _model = model or os.environ.get("RLM_SUB_MODEL", DEFAULT_MODEL)
    _max_retries = max_retries if max_retries is not None else int(
        os.environ.get("RLM_LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES)
    )
    _retry_delay = retry_delay if retry_delay is not None else float(
        os.environ.get("RLM_LLM_RETRY_DELAY", DEFAULT_RETRY_DELAY)
    )

    def llm_query(prompt: str, context: str = "") -> str:
        if not prompt:
            raise ValueError("prompt cannot be empty")

        messages = []
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        last_exception = None
        for attempt in range(_max_retries + 1):
            try:
                response = _client.messages.create(
                    model=_model,
                    max_tokens=1024,
                    messages=messages,
                )
                if not response.content:
                    return ""
                first = response.content[0]
                return getattr(first, "text", str(first))
            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < _max_retries:
                    delay = _retry_delay * (2 ** attempt)
                    logger.warning(
                        f"LLM request failed (attempt {attempt + 1}/{_max_retries + 1}): "
                        f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"LLM request failed after {_max_retries + 1} attempts: "
                        f"{type(e).__name__}: {e}"
                    )

        raise last_exception

    return llm_query


def make_llm_query_batched(llm_query_fn, max_workers: int = 8):
    def llm_query_batched(prompts: list[str], context: str = "") -> list[str]:
        if not prompts:
            return []

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
