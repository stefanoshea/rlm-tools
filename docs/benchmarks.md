# Benchmarks

This document describes how to benchmark `rlm-tools` and includes a publishable baseline result snapshot.

## What Is Measured

The comparative evals measure **context payload size** for equivalent exploration tasks:

- **With RLM Tools:** data stays in sandbox memory; only `print()` output is returned.
- **Baseline:** equivalent `Read`/`Grep`/`Glob` style behavior returns full payloads.

Metric used in `evals/test_comparative.py`:

- `total_context_chars = agent_output_chars + tool_response_chars + per-turn overhead`

Lower is better.

## Reproduce Benchmarks

Prereq: local checkout of the `your-iOS-project` dataset.

```bash
export RLM_EVAL_PROJECT_PATH=/path/to/your-iOS-project
uv run pytest evals/test_comparative.py -q -s
```

For a live client-level A/B comparison (Claude CLI-specific helper script):

```bash
export RLM_EVAL_PROJECT_PATH=/path/to/your-iOS-project
./evals/run_ab_eval.sh
```

Note: `evals/test_comparative.py` is MCP-client agnostic. `evals/run_ab_eval.sh` is currently tailored to Claude CLI.

Optional env vars for A/B script:

- `RLM_TOOLS_PATH` (default: current repo root)
- `RLM_AB_TARGET_PATH` (default: `$RLM_EVAL_PROJECT_PATH/app`)

## Snapshot (February 13, 2026)

Source:

- Command: `uv run pytest evals/test_comparative.py -q -s`
- Commit: `20765a0`
- Dataset: `your-iOS-project/app`

Interpretation note:

- The scenarios below are intentionally heavy payload stress tests.
- In typical day-to-day coding workflows, observed context/token savings are usually closer to **25-35%** (task- and prompting-dependent).

| Scenario | RLM Context Chars | Baseline Context Chars | Savings |
|---|---:|---:|---:|
| Grep full app (`import UIKit`) | 1,644 | 40,045 | 95.9% |
| Read 10 large files | 13,588 | 1,493,720 | 99.1% |
| Multi-step exploration | 5,285 | 136,102 | 96.1% |
| Grep then read | 6,022 | 340,408 | 98.2% |
| Find usages (`@objc func`) | 3,691 | 13,478 | 72.6% |
| Module understanding | 16,925 | 94,745 | 82.1% |
| **Weighted total (all scenarios above)** | **47,155** | **2,118,498** | **97.8%** |

## Reporting Guidance

When publishing benchmark results in GitHub:

1. Include the command(s), commit SHA, and benchmark date.
2. Include dataset path/shape so readers understand scope.
3. Report both raw counts and percentage savings.
4. Keep at least one "harder" scenario (where savings are smaller) to avoid cherry-picking.
