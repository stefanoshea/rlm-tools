"""Paired, scripted context measurements through the public RLM tools.

Each arm answers the same question from the same corpus. The baseline can use
targeted Read and Grep calls and retain everything returned in prior turns.
The measured context is serialized tool arguments plus tool results. The
comparison also shows the existing per-turn framing estimate separately.
Shared user prompts and final answers are outside this tool-context measure.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from evals.metrics import StepMetric, TaskMetric, Timer, format_comparison
from evals.test_comparative import _baseline_grep, _baseline_read
from rlm_tools.server import rlm_end, rlm_execute, rlm_start


def _record(metric: TaskMetric, name: str, args: dict, response: str, elapsed: float):
    metric.steps.append(StepMetric(
        name=name,
        agent_output_chars=len(json.dumps(args)),
        tool_response_chars=len(response),
        elapsed_seconds=elapsed,
    ))


async def _rlm_call(metric, tool, **args):
    with Timer() as timer:
        response = await tool(**args)
    _record(metric, tool.__name__, args, response, timer.elapsed)
    result = json.loads(response)
    assert not result.get("error"), result
    return result


@asynccontextmanager
async def _session(metric: TaskMetric, root: Path, query: str):
    started = await _rlm_call(metric, rlm_start, path=str(root), query=query)
    session_id = started["session_id"]
    try:
        yield session_id
    finally:
        ended = await _rlm_call(metric, rlm_end, session_id=session_id)
        assert ended["success"], ended


def _read(metric: TaskMetric, root: Path, path: str) -> str:
    with Timer() as timer:
        response = _baseline_read(root / path)
    _record(metric, "Read", {"path": path}, response, timer.elapsed)
    return response


def _grep(metric: TaskMetric, root: Path, pattern: str, path: str) -> str:
    with Timer() as timer:
        response = _baseline_grep(root, pattern, path)
    _record(metric, "Grep", {"pattern": pattern, "path": path}, response, timer.elapsed)
    return response


def _payload_chars(metric: TaskMetric) -> int:
    return metric.total_agent_output + metric.total_tool_response


def _report(rlm: TaskMetric, baseline: TaskMetric):
    print(
        f"\n{rlm.task_name}: {_payload_chars(rlm):,} RLM payload chars "
        f"vs {_payload_chars(baseline):,} baseline payload chars"
    )
    print(format_comparison(rlm, baseline))


@pytest.mark.asyncio
async def test_short_lookup_costs_more_with_session_lifecycle(tmp_path):
    (tmp_path / "settings.ini").write_text("retry_limit=7\n", encoding="utf-8")
    query = "What is the retry limit in settings.ini?"
    rlm = TaskMetric(query)
    baseline = TaskMetric(query)

    async with _session(rlm, tmp_path, query) as session_id:
        result = await _rlm_call(
            rlm, rlm_execute, session_id=session_id,
            code="rlm_finalize(int(read_file('settings.ini').split('=')[1]))",
        )
    baseline_answer = int(_read(baseline, tmp_path, "settings.ini").split("=", 1)[1])
    assert result["final"]["final"] == 7
    assert baseline_answer == 7
    assert rlm.num_turns == 3
    assert baseline.num_turns == 1
    _report(rlm, baseline)
    assert _payload_chars(rlm) > _payload_chars(baseline)
    assert rlm.total_context_chars > baseline.total_context_chars


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [100, 500])
async def test_broad_inventory_with_warm_follow_up(tmp_path, size):
    ledger = tmp_path / "services"
    ledger.mkdir()
    # Each 100-record cycle has retry limits 1 through 100. The expected
    # totals below are arithmetic over that specification, independent of
    # either retrieval path or its output.
    cycles = size // 100
    for index in range(size):
        (ledger / f"service_{index:04d}.ini").write_text(
            f"enabled=true\nretry_limit={index % 100 + 1}\n",
            encoding="utf-8",
        )
    query = "Across services, how many retry limits are configured, and what do they sum to?"
    follow_up = "What is the largest retry limit, and how many services exceed 50?"
    rlm = TaskMetric(f"Broad inventory ({size} services)")
    baseline = TaskMetric(rlm.task_name)
    rlm_warm = TaskMetric(f"Warm follow-up ({size} services): {follow_up}")
    baseline_warm = TaskMetric(rlm_warm.task_name)

    async with _session(rlm, tmp_path, query) as session_id:
        cold_result = await _rlm_call(
            rlm, rlm_execute, session_id=session_id,
            code=(
                "matches = grep(r'^retry_limit=', 'services')\n"
                "limits = [int(match['line'].split('=')[1]) for match in matches]\n"
                "rlm_finalize([len(limits), sum(limits)])"
            ),
        )
        warm_result = await _rlm_call(
            rlm_warm, rlm_execute, session_id=session_id,
            code="rlm_finalize([max(limits), sum(limit > 50 for limit in limits)])",
        )

    matches = _grep(baseline, tmp_path, "^retry_limit=", "services")
    baseline_limits = [int(match.rsplit("=", 1)[1]) for match in matches.splitlines()]
    baseline_cold = [len(baseline_limits), sum(baseline_limits)]
    # The baseline already has every required value in its visible transcript.
    baseline_follow_up = [max(baseline_limits), sum(limit > 50 for limit in baseline_limits)]
    assert cold_result["final"]["final"] == [size, cycles * 5050]
    assert baseline_cold == [size, cycles * 5050]
    assert warm_result["final"]["final"] == [100, cycles * 50]
    assert baseline_follow_up == [100, cycles * 50]

    assert rlm.num_turns == 3  # Includes teardown in the initial comparison.
    assert baseline.num_turns == 1
    assert rlm_warm.num_turns == 1
    assert baseline_warm.num_turns == 0
    _report(rlm, baseline)
    _report(rlm_warm, baseline_warm)
    combined = TaskMetric(rlm.task_name, rlm.steps + rlm_warm.steps)
    baseline_combined = TaskMetric(baseline.task_name, baseline.steps + baseline_warm.steps)
    print(
        f"Initial query and follow-up: {_payload_chars(combined):,} RLM payload chars "
        f"vs {_payload_chars(baseline_combined):,} baseline payload chars"
    )
    assert _payload_chars(rlm) < _payload_chars(baseline)
    assert rlm.total_context_chars < baseline.total_context_chars
    assert _payload_chars(rlm_warm) > _payload_chars(baseline_warm) == 0
    assert _payload_chars(combined) < _payload_chars(baseline_combined)
    assert combined.total_context_chars < baseline_combined.total_context_chars
