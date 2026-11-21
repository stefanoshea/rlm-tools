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

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

DEFAULT_MODEL = "jev-latest"
DEFAULT_CHUNK_SIZE = 12
DEFAULT_MAX_WORKERS = 4
DEFAULT_MAX_ITEM_CHARS = 600

DEFAULT_GATE_THRESHOLD = 0.45
DEFAULT_GATE_CHUNK_LINES = 40
DEFAULT_GATE_MIN_CHARS = 2_000
DEFAULT_GATE_MIN_SEPARATION = 0.2
ELISION = "\n[... dropped as irrelevant to the question ...]\n"

EXHAUSTIVE_QUESTION = re.compile(
    r"\b(every|all|each|list|enumerate|exhaustive|entire|complete)\b", re.IGNORECASE
)


class SystemOneClient(Protocol):
    def system_one(self, state: Any, questions: dict[str, Any]) -> Any: ...


def get_client(api_key: str | None = None, model: str | None = None) -> SystemOneClient:
    from typesafe_sdk import TypeSafeClient

    key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError(
            "TYPESAFE_API_KEY environment variable is required for relevance scoring"
        )
    return TypeSafeClient(
        api_key=key,
        model=model or os.environ.get("TYPESAFE_DEFAULT_MODEL", DEFAULT_MODEL),
    )


def _noul(instructions: str) -> Any:
    try:
        from typesafe_sdk import Noul
    except ImportError:
        return {"type": "noul", "instructions": instructions}
    return Noul(instructions=instructions)


def _index_chunks(count: int, size: int) -> list[list[int]]:
    return [list(range(i, min(i + size, count))) for i in range(0, count, size)]


def make_scorer(
    client: SystemOneClient | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
    reserve: Callable[[int], None] | None = None,
) -> Callable[[list[str], str], list[float]]:
    """Build the batched relevance scorer.

    ``reserve`` is called with the number of API requests the scorer is about to
    make, so the session can charge them against a budget before they run.
    """
    _client = client or get_client()

    def score_all(items: list[str], question: str) -> list[float]:
        """Probability per item that ``question`` is true of it, in input order."""
        if not items:
            return []

        texts = [str(item)[:max_item_chars] for item in items]

        def _score_chunk(indexes: list[int]) -> dict[int, float]:
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
            )
            return {i: float(response.nouls[f"c{i}"].noul) for i in indexes}

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


def _split_chunks(text: str, chunk_lines: int) -> list[str]:
    lines = text.splitlines()
    return [
        "\n".join(lines[i:i + chunk_lines])
        for i in range(0, len(lines), chunk_lines)
    ]


def make_evidence_gate(
    score_all: Callable[..., list[float]],
    threshold: float = DEFAULT_GATE_THRESHOLD,
    chunk_lines: int = DEFAULT_GATE_CHUNK_LINES,
    min_chars: int = DEFAULT_GATE_MIN_CHARS,
    min_separation: float = DEFAULT_GATE_MIN_SEPARATION,
    on_gate: Callable[[dict], None] | None = None,
) -> Callable[[str, str], str]:
    """Build a gate that drops tool output unrelated to the question.

    The gate never fails a lookup: anything unexpected — a scoring error, an
    exhausted call budget, a result too small to be worth judging, a question
    whose answer could be anywhere — returns the original text untouched.

    ``on_gate`` receives one record per call, so a run can measure what the gate
    dropped and whether the answer survived it.
    """

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

        chunks = _split_chunks(text, chunk_lines)
        if len(chunks) < 2:
            _record(outcome="passed_through", chars_in=len(text), chars_out=len(text))
            return text

        try:
            scores = score_all(chunks, f"this excerpt helps answer: {question}")
            if len(scores) != len(chunks) or not all(
                math.isfinite(score) and 0 <= score <= 1 for score in scores
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
