"""Measure evidence retained and replayed by the recursive loop with offline clients."""

import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from evals.conftest import DEFAULT_LONG_CONTEXT_CORPUS_PATH
from rlm_tools.helpers import make_helpers
from rlm_tools.llm_bridge import make_recursive_query
from rlm_tools.semantic import make_evidence_gate, make_scorer


@dataclass
class Trace:
    replay_chars: list[int] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    scorer_calls: int = 0


def _read_request(path, call_id):
    block = SimpleNamespace(type="tool_use", name="read_file", input={"path": path}, id=call_id)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


def _explore(question, mode):
    trace = Trace()

    def respond(*, messages, **kwargs):
        evidence = [
            block["content"]
            for message in messages if message["role"] == "user" and isinstance(message["content"], list)
            for block in message["content"] if block["type"] == "tool_result"
        ]
        trace.replay_chars.append(sum(map(len, evidence)))
        if not evidence:
            return _read_request("app/catalog/repository.py", "repository")

        imported = re.search(r"^from (\S+) import PRODUCT_CACHE_TTL$", evidence[0], re.M)
        facts = {}
        if imported:
            source = imported[1].replace(".", "/") + ".py"
            if len(evidence) == 1:
                return _read_request(source, "registry")
            constants = re.findall(r"^(\w+_CACHE_TTL): Final\[int\] = (\d+)$", evidence[1], re.M)
            facts = {name: int(value) for name, value in constants
                     if "List" in question or name == "PRODUCT_CACHE_TTL"}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(facts))],
                               stop_reason="end_turn")

    def score_candidates(state, questions):
        trace.scorer_calls += 1
        if mode == "unavailable":
            raise RuntimeError("offline scorer unavailable")
        probabilities = {}
        for index, excerpt in state["candidates"].items():
            if mode == "uncertain":
                probability = 0.46 if " import PRODUCT_CACHE_TTL" in excerpt else 0.44
            else:
                probability = 0.95 if " import PRODUCT_CACHE_TTL" in excerpt else 0.05
            probabilities[f"c{index}"] = SimpleNamespace(noul=probability)
        return SimpleNamespace(nouls=probabilities)

    gate = None
    if mode != "off":
        scorer = make_scorer(client=SimpleNamespace(system_one=score_candidates))
        gate = make_evidence_gate(scorer, on_gate=lambda event: trace.outcomes.append(event["outcome"]))
    helpers, _ = make_helpers(str(DEFAULT_LONG_CONTEXT_CORPUS_PATH))
    query = make_recursive_query(client=SimpleNamespace(messages=SimpleNamespace(create=respond)),
                                 helpers=helpers, max_retries=1, gate=gate)
    return json.loads(query(question)), trace


@pytest.mark.parametrize(("mode", "exhaustive", "outcome"), [
    ("focused", False, "gated"),
    ("uncertain", False, "passed_through_low_separation"),
    ("unavailable", False, "failed_open"),
    ("focused", True, "passed_through_exhaustive"),
])
def test_replayed_evidence_answers_the_question(mode, exhaustive, outcome):
    question = ("List every cache TTL declared in the registry used by ProductRepository."
                if exhaustive else "Which product cache TTL is declared in the registry used by ProductRepository?")
    expected = {"PRODUCT_CACHE_TTL": 300}
    if exhaustive:
        expected.update(ORDER_CACHE_TTL=60, SESSION_CACHE_TTL=1800)

    baseline, raw = _explore(question, "off")
    answer, gated = _explore(question, mode)
    assert baseline == answer == expected
    assert gated.outcomes == [outcome, "passed_through"]
    assert gated.scorer_calls == (0 if exhaustive else 1)
    assert len(raw.replay_chars) == len(gated.replay_chars) == 3
    if outcome == "gated":
        # The first file's evidence is sent on both subsequent model requests.
        saved = raw.replay_chars[1] - gated.replay_chars[1]
        assert saved > 0
        assert sum(raw.replay_chars) - sum(gated.replay_chars) == 2 * saved
    else:
        assert gated.replay_chars == raw.replay_chars
    print(f"{outcome}: {len(answer)}/{len(expected)} TTLs correct; tool-result characters "
          f"replayed {sum(raw.replay_chars):,} -> {sum(gated.replay_chars):,}; "
          f"{gated.scorer_calls} simulated scoring calls")
