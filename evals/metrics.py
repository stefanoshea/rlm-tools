import time
from dataclasses import dataclass, field


# Estimated chars per tool-call turn for framing overhead:
# tool call XML/JSON wrapper, agent reasoning between steps, etc.
# Same for both approaches, so it's a wash — but makes totals realistic.
OVERHEAD_PER_TURN = 350


@dataclass
class StepMetric:
    name: str
    # What the agent writes to invoke the tool (code string for RLM, params for baseline)
    agent_output_chars: int = 0
    # What the tool returns (full JSON for RLM, content for baseline)
    tool_response_chars: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None

    @property
    def total_chars(self) -> int:
        return self.agent_output_chars + self.tool_response_chars + OVERHEAD_PER_TURN


@dataclass
class TaskMetric:
    task_name: str
    steps: list[StepMetric] = field(default_factory=list)

    @property
    def total_agent_output(self) -> int:
        return sum(s.agent_output_chars for s in self.steps)

    @property
    def total_tool_response(self) -> int:
        return sum(s.tool_response_chars for s in self.steps)

    @property
    def total_overhead(self) -> int:
        return len(self.steps) * OVERHEAD_PER_TURN

    @property
    def total_context_chars(self) -> int:
        return self.total_agent_output + self.total_tool_response + self.total_overhead

    @property
    def total_elapsed_seconds(self) -> float:
        return sum(s.elapsed_seconds for s in self.steps)

    @property
    def had_errors(self) -> bool:
        return any(s.error for s in self.steps)

    @property
    def num_turns(self) -> int:
        return len(self.steps)


class Timer:
    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


def format_comparison(rlm: TaskMetric, baseline: TaskMetric) -> str:
    lines = [
        f"  {'':40} {'Agent Out':>10} {'Tool Resp':>10} {'Overhead':>10} {'Total':>10} {'Time':>8}",
        f"  {'-' * 92}",
    ]

    for step in rlm.steps:
        lines.append(
            f"  RLM  {step.name:<34} {step.agent_output_chars:>10} {step.tool_response_chars:>10} "
            f"{OVERHEAD_PER_TURN:>10} {step.total_chars:>10} {step.elapsed_seconds:>7.3f}s"
        )
    lines.append(
        f"  {'RLM TOTAL (' + str(rlm.num_turns) + ' turns)':<40} "
        f"{rlm.total_agent_output:>10} {rlm.total_tool_response:>10} "
        f"{rlm.total_overhead:>10} {rlm.total_context_chars:>10} {rlm.total_elapsed_seconds:>7.3f}s"
    )
    lines.append("")

    for step in baseline.steps:
        lines.append(
            f"  BASE {step.name:<34} {step.agent_output_chars:>10} {step.tool_response_chars:>10} "
            f"{OVERHEAD_PER_TURN:>10} {step.total_chars:>10} {step.elapsed_seconds:>7.3f}s"
        )
    lines.append(
        f"  {'BASE TOTAL (' + str(baseline.num_turns) + ' turns)':<40} "
        f"{baseline.total_agent_output:>10} {baseline.total_tool_response:>10} "
        f"{baseline.total_overhead:>10} {baseline.total_context_chars:>10} {baseline.total_elapsed_seconds:>7.3f}s"
    )
    lines.append("")

    savings = _pct_savings(baseline.total_context_chars, rlm.total_context_chars)
    lines.append(f"  Context savings: {savings}")

    return "\n".join(lines)


def _pct_savings(baseline: int, rlm: int) -> str:
    if baseline == 0:
        return "N/A"
    pct = (1 - rlm / baseline) * 100
    return f"{pct:.1f}%"
