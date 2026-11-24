"""Relevance scoring for the recursive explorer, backed by TypeSafe (Jev).

A sub-agent re-sends every tool result on every turn of its loop, so a large
file read is paid for once per remaining turn. The evidence gate scores a
result in chunks and keeps only the parts that bear on the question.

Scoring is batched: Jev answers many yes/no questions about one shared state in
a single request, so judging a dozen chunks costs about what judging one costs.

The scorer was also exposed to sandbox code as `rank`/`keep`/`judge`/`classify`
helpers. Seven measured agent runs across two corpora never called them once —
`grep_summary` plus a targeted read beat them every time on code — so the
sandbox surface was removed and the scorer kept for the gate, which fires
automatically rather than waiting to be chosen.
"""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from rlm_tools.llm_bridge import request_timeout

DEFAULT_MODEL = "jev-latest"
DEFAULT_GEMINI_MODEL = "gemini-decision-1"
DEFAULT_CHUNK_SIZE = 12
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_ITEM_CHARS = 2_400

DEFAULT_GATE_THRESHOLD = 0.45
DEFAULT_GATE_CHUNK_LINES = 40
DEFAULT_GATE_MIN_CHARS = 2_000
DEFAULT_GATE_MIN_SEPARATION = 0.2
ELISION = "\n[... dropped as irrelevant to the question ...]\n"
_VALID_GATE_PROVIDERS = frozenset({"typesafe", "gemini"})

EXHAUSTIVE_QUESTION = re.compile(
    r"\b(every|all|each|list|enumerate|exhaustive|entire|complete)\b", re.IGNORECASE
)


class SystemOneClient(Protocol):
    def system_one(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> Any: ...


def resolve_provider(provider: str | None = None) -> str:
    """Resolve the relevance scoring provider ('typesafe' or 'gemini')."""
    if provider is not None:
        normalized = provider.strip().lower()
        if normalized and normalized != "auto":
            if normalized not in _VALID_GATE_PROVIDERS:
                raise ValueError(
                    f"Unsupported gate provider {provider!r}; expected 'typesafe', 'gemini', or 'auto'"
                )
            return normalized

    env_provider = os.environ.get("RLM_GATE_PROVIDER", "auto").strip().lower()
    if env_provider in {"", "auto"}:
        if os.environ.get("TYPESAFE_API_KEY"):
            return "typesafe"
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        return "typesafe"
    if env_provider not in _VALID_GATE_PROVIDERS:
        raise ValueError(
            f"Unsupported RLM_GATE_PROVIDER {env_provider!r}; expected 'typesafe', 'gemini', or 'auto'"
        )
    return env_provider


def has_configured_scorer() -> bool:
    """Return True when the configured (or auto-detected) gate provider has credentials."""
    env_provider = os.environ.get("RLM_GATE_PROVIDER", "auto").strip().lower()
    if env_provider in {"", "auto"}:
        return bool(os.environ.get("TYPESAFE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    if env_provider == "typesafe":
        return bool(os.environ.get("TYPESAFE_API_KEY"))
    if env_provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    return False


class _GeminiDecisionsAdapter:
    """Adapt google-genai's v1beta Decisions API to the SystemOneClient protocol."""

    def __init__(self, client: Any, model: str = DEFAULT_GEMINI_MODEL):
        self._client = client
        self._model = model

    def system_one(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> Any:
        shared_input = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        formatted_questions: dict[str, dict[str, Any]] = {}
        for key, question in questions.items():
            if isinstance(question, dict):
                entry = dict(question)
                entry.setdefault("type", "noul")
                formatted_questions[key] = entry
            else:
                instructions = getattr(question, "instructions", None)
                formatted_questions[key] = {
                    "type": "noul",
                    "instructions": instructions if instructions is not None else str(question),
                }

        create_kwargs: dict[str, Any] = {}
        if "timeout" in kwargs:
            create_kwargs["timeout"] = kwargs["timeout"]
        response = self._client.decisions.create(
            model=self._model,
            input=shared_input,
            questions=formatted_questions,
            **create_kwargs,
        )
        return SimpleNamespace(nouls=getattr(response, "answers", None) or {})


def get_client(
    api_key: str | None = None,
    model: str | None = None,
    *,
    provider: str | None = None,
) -> SystemOneClient:
    resolved = resolve_provider(provider)
    if resolved == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required for Gemini relevance scoring"
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is required for Gemini relevance scoring (requires Python 3.11+); "
                "install with the 'gemini' extra (rlm-tools[gemini])"
            ) from exc
        raw_client = genai.Client(
            api_key=key,
            http_options={"api_version": "v1beta", "retry_options": {"attempts": 1}},
        )
        decision_model = model or os.environ.get("GEMINI_DECISION_MODEL") or DEFAULT_GEMINI_MODEL
        return _GeminiDecisionsAdapter(raw_client, model=decision_model)

    from typesafe_sdk import RetryPolicy, TypeSafeClient

    key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError(
            "TYPESAFE_API_KEY environment variable is required for relevance scoring"
        )
    return TypeSafeClient(
        api_key=key,
        retry=RetryPolicy(max_retries=0),
        model=model or os.environ.get("TYPESAFE_DEFAULT_MODEL") or DEFAULT_MODEL,
    )


def _noul(instructions: str) -> Any:
    try:
        from typesafe_sdk import Noul
    except ImportError:
        return {"type": "noul", "instructions": instructions}
    return Noul(instructions=instructions)


def _index_chunks(count: int, size: int) -> list[list[int]]:
    return [list(range(i, min(i + size, count))) for i in range(0, count, size)]


def _is_probability(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def make_scorer(
    client: Any = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
    reserve: Callable[[int], None] | None = None,
    deadline: Callable[[], float | None] | None = None,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> Callable[[list[str], str], list[float]]:
    """Build the batched relevance scorer.

    ``reserve`` is called with the number of API requests the scorer is about to
    make, so the session can charge them against a budget before they run.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if max_item_chars < 1:
        raise ValueError("max_item_chars must be >= 1")
    if client is not None:
        if hasattr(client, "decisions") and not hasattr(client, "system_one"):
            decision_model = model or os.environ.get("GEMINI_DECISION_MODEL") or DEFAULT_GEMINI_MODEL
            _client: SystemOneClient = _GeminiDecisionsAdapter(client, model=decision_model)
        else:
            _client = client
    elif model is None and provider is None:
        _client = get_client()
    else:
        _client = get_client(model=model, provider=provider)

    def score_all(items: list[str], question: str) -> list[float]:
        """Probability per item that ``question`` is true of it, in input order."""
        if not items:
            return []

        texts = [str(item)[:max_item_chars] for item in items]

        def _score_chunk(indexes: list[int]) -> dict[int, float]:
            options = {} if deadline is None else {"timeout": request_timeout(60.0, deadline)}
            if reserve is not None:
                reserve(1)
            response = _client.system_one(
                {
                    "criterion": question,
                    "candidates": {str(i): texts[i] for i in indexes},
                },
                {
                    f"c{i}": _noul(f"Is the criterion true of candidate {i}?")
                    for i in indexes
                },
                **options,
            )
            probabilities = {i: response.nouls[f"c{i}"].noul for i in indexes}
            if not all(map(_is_probability, probabilities.values())):
                raise ValueError("Expected one probability per candidate")
            return {i: float(probability) for i, probability in probabilities.items()}

        batches = _index_chunks(len(items), chunk_size)
        scores: dict[int, float] = {}
        if len(batches) == 1:
            scores.update(_score_chunk(batches[0]))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for batch_scores in executor.map(_score_chunk, batches):
                    scores.update(batch_scores)
        return [scores[i] for i in range(len(items))]

    return score_all


def _split_chunks(
    text: str,
    chunk_lines: int,
    max_chunk_chars: int | None = None,
) -> list[str]:
    """Split text into chunks capped by both line count and character length."""
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    buf_chars = 0
    for line in lines:
        added = len(line) + (1 if buf else 0)
        if buf and (
            len(buf) >= chunk_lines
            or (max_chunk_chars is not None and buf_chars + added > max_chunk_chars)
        ):
            chunks.append("\n".join(buf))
            buf = [line]
            buf_chars = len(line)
        else:
            buf.append(line)
            buf_chars += added
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def make_evidence_gate(
    score_all: Callable[..., list[float]],
    threshold: float = DEFAULT_GATE_THRESHOLD,
    chunk_lines: int = DEFAULT_GATE_CHUNK_LINES,
    min_chars: int = DEFAULT_GATE_MIN_CHARS,
    min_separation: float = DEFAULT_GATE_MIN_SEPARATION,
    on_gate: Callable[[dict], None] | None = None,
    max_chunk_chars: int | None = DEFAULT_MAX_ITEM_CHARS,
) -> Callable[[str, str], str]:
    """Build a gate that drops tool output unrelated to the question.

    The gate never fails a lookup: anything unexpected — a scoring error, an
    exhausted call budget, a result too small to be worth judging, a question
    whose answer could be anywhere — returns the original text untouched.

    ``on_gate`` receives one record per call, so a run can measure what the gate
    dropped and whether the answer survived it.
    """
    if chunk_lines < 1:
        raise ValueError("chunk_lines must be >= 1")
    if max_chunk_chars is not None and max_chunk_chars < 1:
        raise ValueError("max_chunk_chars must be >= 1")

    def _record(**fields) -> None:
        if on_gate is not None:
            on_gate(fields)

    def gate(question: str, text: str) -> str:
        if not text or len(text) < min_chars:
            _record(outcome="passed_through", chars_in=len(text or ""), chars_out=len(text or ""))
            return text

        # An exhaustive question needs every mention, and relevance scores
        # cannot tell the last one from noise. Gating these loses facts.
        if EXHAUSTIVE_QUESTION.search(question):
            _record(outcome="passed_through_exhaustive", chars_in=len(text), chars_out=len(text))
            return text

        chunks = _split_chunks(text, chunk_lines, max_chunk_chars=max_chunk_chars)
        if len(chunks) < 2:
            _record(outcome="passed_through", chars_in=len(text), chars_out=len(text))
            return text

        try:
            scores = score_all(chunks, f"this excerpt helps answer: {question}")
            if len(scores) != len(chunks) or not all(
                map(_is_probability, scores)
            ):
                raise ValueError("Expected one probability per evidence chunk")
        except Exception as e:
            _record(outcome="failed_open", chars_in=len(text), chars_out=len(text), error=str(e))
            return text

        kept = [i for i, score in enumerate(scores) if score >= threshold]
        if not kept:
            _record(
                outcome="passed_through_nothing_relevant",
                chars_in=len(text),
                chars_out=len(text),
                chunks_total=len(chunks),
                max_score=max(scores),
            )
            return text

        dropped_scores = [scores[i] for i in range(len(chunks)) if i not in kept]
        separation = (
            min(scores[i] for i in kept) - max(dropped_scores) if dropped_scores else 1.0
        )
        # A flat distribution means the model found no clear signal, so the
        # threshold would be slicing arbitrarily. Keep everything instead.
        if separation < min_separation:
            _record(
                outcome="passed_through_low_separation",
                chars_in=len(text),
                chars_out=len(text),
                chunks_total=len(chunks),
                separation=separation,
            )
            return text

        if len(kept) == len(chunks):
            _record(
                outcome="kept_all",
                chars_in=len(text),
                chars_out=len(text),
                chunks_total=len(chunks),
                chunks_kept=len(chunks),
            )
            return text

        parts = []
        previous = -1
        for index in kept:
            if index != previous + 1:
                parts.append(ELISION)
            parts.append(chunks[index])
            previous = index
        if kept[-1] != len(chunks) - 1:
            parts.append(ELISION)
        gated = "\n".join(parts)
        _record(
            outcome="gated",
            chars_in=len(text),
            chars_out=len(gated),
            chunks_total=len(chunks),
            chunks_kept=len(kept),
            min_kept_score=min(scores[i] for i in kept),
            max_dropped_score=max(dropped_scores, default=0.0),
        )
        return gated

    return gate
