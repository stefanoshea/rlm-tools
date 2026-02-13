import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        if not prompt:
            raise ValueError("prompt cannot be empty")

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
        if not response.content:
            return ""
        first = response.content[0]
        return getattr(first, "text", str(first))

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
