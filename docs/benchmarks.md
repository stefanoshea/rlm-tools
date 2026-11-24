# Benchmarks

This document describes how to benchmark `rlm-tools` and how to interpret the results now that the repo includes fixture-backed scored evals in addition to context-size microbenchmarks.

## Benchmark Lanes

There are now three distinct benchmark lanes:

- **Fixture-backed CI evals**: checked-in corpora, always runnable, scored for correctness
- **Comparative microbenchmarks**: context-size comparisons for equivalent exploration steps
- **Prompt benchmarks**: scored A/B runs between `with_rlm` and `without_rlm` configurations

The default prompt benchmark is now the harder long-context lane in `evals/specs/rlm_long_context.json`, backed by `tests/fixtures/rlm_corpus_longcontext`.

## What Is Measured

The comparative microbenchmarks measure **context payload size** for equivalent exploration tasks:

- **With RLM Tools:** data stays in sandbox memory; only `print()` output is returned.
- **Baseline:** equivalent `Read`/`Grep`/`Glob` style behavior returns full payloads.

Metric used in `evals/test_comparative.py`:

- `total_context_chars = agent_output_chars + tool_response_chars + per-turn overhead`

Lower is better, but it is not sufficient on its own.

The prompt benchmark stack also reports:

- pass rate
- mean Claude CLI tokens
- token variance
- mean duration
- duration variance
- preferred config, chosen by pass rate first and efficiency second
- analyst notes from `evals/review_benchmark.py`

## End-to-end session measurements

The lifecycle evals count the serialized arguments and returned text for every
tool call, including `rlm_start` and `rlm_end`. The scripted baseline can
target a single file or directory and reuse every previous result. Both arms
must produce the exact answer. The table gives tool payload characters from a
local run. Temporary directory names can shift the RLM count slightly.

| Task | RLM characters | Read/Grep characters | RLM / baseline calls |
|---|---:|---:|---:|
| Read one setting from a known file | 896 | 38 | 3 / 1 |
| Count and sum retry limits in 100 files, then answer a follow-up | 1,205 | 4,439 | 4 / 1 |
| Same two questions over 500 files | 1,207 | 22,007 | 4 / 1 |

For the follow-up alone, the baseline already had every value and used zero
additional tool calls. RLM retrieved the answer from session memory in one call
(166–167 additional characters). Savings over the whole task came from keeping
the initial list of matches in the worker. These counts exclude user prompts,
final answers, model reasoning, and transport framing. The test prints a separate
estimate using the existing 350-character allowance per call.

The evidence replay eval follows the product repository's cache import into its
registry. It runs the actual recursive loop twice with a scripted reader, once
with filtering disabled and once with a deterministic scorer. On the focused
question, both arms returned 300 seconds. Cumulative tool-result text across
three model requests fell from 6,719 to 3,859 characters. Uncertain scores and
a scoring failure preserved all 6,719 characters. An exhaustive prompt preserved
all three TTLs and made zero scoring calls. This isolates the loop's replay
behavior. Live scorer accuracy, provider caching, and billed tokens require a
separate model run.

## Reproduce Benchmarks

Run the always-runnable eval suite:

```bash
uv run --locked --extra dev python -m pytest evals -q
```

Run the comparative microbenchmarks:

```bash
uv run --locked --extra dev python -m pytest -q -s evals/test_comparative.py evals/test_session_effectiveness.py evals/test_evidence_effectiveness.py
```

Run a scored prompt benchmark with the Claude CLI and uv installed:

```bash
./evals/run_ab_eval.sh
```

For a two-question live smoke run against the smaller Python and TypeScript fixtures:

```bash
RLM_AB_TARGET_PATH=tests/fixtures ./evals/run_ab_eval.sh evals/specs/rlm_core.json
```

Each benchmark run gets an owner-only, timestamped directory under `evals/ab_results`. It contains:

- `output.json`
- `timing.json`
- `grading.json`
- `benchmark.json` and `benchmark_stdout.txt`
- `raw_results.json` and the raw Claude response for each arm and case
- `benchmark_spec.json` with the exact prompts and assertions used for grading

Optional env vars for A/B script:

- `RLM_TOOLS_PATH` (default: current repo root)
- `RLM_AB_TARGET_PATH` (default: `tests/fixtures/rlm_corpus_longcontext`)
- `RLM_AB_SPEC_PATH` (default: `evals/specs/rlm_long_context.json`)
- `RLM_AB_MODEL` (optional: pin the same model for both arms, e.g. `claude-sonnet-4-6`)
- `RLM_AB_ITERATIONS` sets the number of paired trials per question (default 1). With 11 questions, three iterations run 66 Claude CLI invocations.
- `RLM_JUDGE=1` (optional: record advisory `llm_judge` verdicts alongside deterministic scores; also requires `ANTHROPIC_API_KEY`)
- `RLM_AB_PASSTHROUGH_ENV` lists additional environment variable names, separated by commas, to pass to the Claude CLI if your authentication or proxy setup requires them. Both arms receive those variables.
- `RLM_AB_SEMANTIC=1` enables the semantic extra for the MCP server. The evidence gate for `rlm_recurse` requires both `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY`.

## Spend Policy

Unit tests and evals never call external model APIs. `tests/conftest.py` and
`evals/conftest.py` strip `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `RLM_SUB_PROVIDER`, and
`RLM_GATE_PROVIDER`, and replace both `semantic.get_client` and
`llm_bridge.get_client` with functions that raise, so any test that reaches for a
provider fails instead of billing. `run_ab_eval.sh` and `run_gemini_ab_eval.py` also
sanitize credentials passed to the spawned MCP server and keep `RLM_RECURSE_GATE=0`
unless explicitly enabled.

Spending belongs in the tool itself and in manual verification of the tool, not in the
suites that run on every change.

## Live Gemini A/B Benchmarks

[`evals/run_gemini_ab_eval.py`](../evals/run_gemini_ab_eval.py) runs paired live A/B
evaluations against the Gemini Interactions API (`google-genai>=3.2.0`, requires
Python 3.11+ via the `gemini` extra), comparing the `rlm-tools` stdio MCP server against
scoped native `read_file` / `grep` / `glob_files` tools using the exact same grading specs:

```bash
# Heavy 230-file / 1.28 MB synthetic corpus + long-context fixture suite
uv run --locked --extra gemini python -m evals.run_gemini_ab_eval \
  --suite both --model gemini-3.5-flash-lite \
  --concurrency 1 --pace-seconds 4 --request-timeout 90

# Individual suites: --suite heavy | --suite long_context
```

- `evals/heavy_corpus.py` generates a deterministic 230-file (~1.28 MB) Python codebase
  combining 200 service modules, 10 ~58 KB monolith modules, 12 shared utilities, and
  config files with five `json_equals` cases (`broad_service_inventory`,
  `flag_and_route_audit`, `large_module_audit`, `shared_utility_fanin`,
  `targeted_billing_trace`).
- `long_context` runs the 11 checked-in cases in `evals/specs/rlm_long_context.json`
  against `tests/fixtures/rlm_corpus_longcontext` (~28 KB).

### Measured Before vs. After Results (`gemini-3.5-flash-lite`)

Baseline runs (`2026-11-23`) were compared against post-fix runs (`2026-11-24`) after
adding the explicit sandbox contract to tool schemas, lazy-importing `anthropic`
(~1.5s to 0.280s warm import), and adding `end_session` + initial `code` support:

| Suite & Scope | Run Date | Arm | Pass Rate | Mean Total Tokens | Mean Duration | Tool Calls | `rlm_end` Replay Tokens |
|---|---|---|---:|---:|---:|---:|---:|
| `heavy` (5/5 pairs) | 2026-11-23 (before) | `with_rlm` | 100.0% (5/5) | 59,388.6 | 69.6s | 50 | 47,912 |
| `heavy` (5/5 pairs) | 2026-11-23 (before) | `without_rlm` | 40.0% (2/5) | 469,392.2 | 152.2s | 61 | 0 |
| `heavy` (5/5 pairs) | 2026-11-24 (after) | `with_rlm` | **100.0% (5/5)** | **31,208.0 (-47.5%)** | 121.6s | **22 (-56.0%)** | **0** |
| `heavy` (5/5 pairs) | 2026-11-24 (after) | `without_rlm` | 60.0% (3/5) | 315,676.2 | 95.6s | 71 | 0 |
| `long_context` (11/11 pairs) | 2026-11-23 (before) | `with_rlm` | 81.8% (9/11) | 58,985.8 | 57.0s | 93 | 59,223 |
| `long_context` (11/11 pairs) | 2026-11-23 (before) | `without_rlm` | 100.0% (11/11) | 23,463.6 | 39.4s | 72 | 0 |
| `long_context` (matched 5/11) | 2026-11-23 (before) | `with_rlm` | 100.0% (5/5) | 20,885.8 | 66.8s | 32 | 20,953 |
| `long_context` (matched 5/11) | 2026-11-24 (after, partial*) | `with_rlm` | 80.0% (4/5) | 18,945.4 (-9.3%) | 57.1s | 24 (-25.0%) | 0 |
| `long_context` (matched 5/11) | 2026-11-24 (after, partial*) | `without_rlm` | 100.0% (5/5) | 8,210.4 | 30.3s | 20 | 0 |

\*The post-fix `long_context` run completed the first 5/11 pairs before hitting the Free
Tier `gemini-3.5-flash-lite` 500 requests/day quota limit on case 6 (`refund_notification_channel`);
post-fix `heavy` mean duration (`121.6s`) reflects transient HTTP 503 retries during that run.
Traces from the post-fix runs also showed that exposing `end_session` on `rlm_start` led the
model to prematurely close Turn-1 sessions and call `rlm_start` repeatedly in 4 of the 5 partial
`long_context` cases; `end_session` was subsequently restricted to `rlm_execute` while keeping
optional initial `code` on `rlm_start`.

A follow-up post-refinement check on `gemini-3.5-flash` (which routes to `gemini-3.6-flash`
at a 20 RPD Free Tier limit) verified the updated flow end-to-end on
`heavy/broad_service_inventory`: `with_rlm` passed (`48,828` tokens, 10 turns / 9 tool calls,
0 blocked-import errors, 0 repeated `rlm_start` calls, and clean session close via
`rlm_execute(..., end_session=True)`) before `without_rlm` exhausted the 20 RPD cap on turn 6.
For full multi-case Free Tier benchmark runs (`heavy` + `long_context`), use
`gemini-3.5-flash-lite` (500 RPD limit).

### Crossover Guidance

- **Where `rlm-tools` wins decisively**: broad multi-file inventories, flag/route audits,
  fan-in aggregations, and scanning large files (`heavy`: **100% pass vs 40%–60%**,
  **87.3%–90.1% fewer tokens** than native tools). Native `grep`/`read_file` overflows
  small/medium context windows or loses accuracy under heavy output truncation.
- **Where native `Read`/`Grep`/`Glob` wins**: narrow 1–5 file lookups and small traces
  inside ~28 KB repos (`long_context`: **100% pass, ~2.5× fewer tokens** for native tools),
  where `rlm-tools` session lifecycle and code-writing turns add fixed overhead.

## Benchmarking Against Your Own Repo

The A/B harness is parameterized, so you can point it at any checkout with your own
spec of grounded, verified questions. Keep private specs outside the repo (or ignored)
and pass them by path:

```bash
RLM_AB_TARGET_PATH=/path/to/your/repo \
RLM_AB_SPEC_PATH=/path/to/your/spec.json \
RLM_AB_MODEL=claude-sonnet-4-6 \
./evals/run_ab_eval.sh
```

How the harness keeps the comparison fair:

- Both arms use the same system instructions and the same target directory. The
  harness alternates which arm goes first across cases and repeated trials.
- The with-RLM arm can call only the configured MCP server. The without-RLM arm
  can use the native `Read`, `Grep`, and `Glob` tools.
- Set `RLM_AB_MODEL` to pin one model for both arms.
- CLI token totals include the main Claude session's reported input, cache
  creation, cache reads, and output. Model-backed MCP helpers such as
  `llm_query` and `rlm_recurse`, the TypeSafe evidence gate, and the optional
  semantic judge can incur separate usage outside this metric. A failed Claude
  invocation or a reported denial for one of the arm's assigned tools stops
  the benchmark and leaves its raw artifacts for diagnosis. The comparison
  measures the effect of making the tools available. The result summary does
  not establish which permitted tools the agent chose to use.

The 11 long-context questions require structured JSON answers. The grader
ignores object key order, accepts a single JSON code fence, and checks numeric
values, types, array order, and the complete set of fields. The previous
eight-question prose rubric is preserved in `evals/specs/rlm_long_context_v1.json`
for regrading earlier outputs. Use a run's archived spec when available.

Compare arms on the same questions and trial numbers. The two-arm report gives
paired quality wins, ties, and losses. It also reports the median token
difference for the pairs where both answers passed and both token totals were
measured, along with the number of qualifying pairs. The report withholds a
preference for mismatched coverage or unscorable grades. Missing usage appears
as `N/A` and cannot win a token tiebreak. An exact tie has no preferred arm.
Older summaries may have estimated usage from answer length. Use the provider's
raw usage record to establish token counts for those runs.

**An RLM context-efficiency win requires fewer mean Claude CLI tokens and a pass rate that holds.**
Expect the win to vary sharply by task type — broad reduce/aggregate questions should save
the most, while single-grep lookups may show little or negative savings due to session
overhead. For a custom spec with `llm_judge` assertions, set `RLM_JUDGE=1` and
`ANTHROPIC_API_KEY` to record advisory verdicts. Each scored case requires at least one deterministic assertion; those assertions determine its score.

A good private spec mixes question shapes: a few single-hop lookups, a few multi-hop
traces, and at least one open-ended reduce question. Give each entry grep-verifiable
answer strings (distinctive constant or class names, literal values, a file path).

## Snapshot (Historical Microbenchmark)

Source:

- Command: `uv run pytest evals/test_comparative.py -q -s`
- Commit: `20765a0`
- Dataset: `your-iOS-project/app`

Interpretation note:

- The scenarios below are intentionally payload-heavy stress tests.
- Treat them as microbenchmarks and upper bounds, not as standalone proof of agent quality.
- Use the scored prompt benchmark and fixture-backed evals for outcome quality.

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
3. Report both raw counts and percentage savings for microbenchmarks.
4. Report pass rate and variance for prompt benchmarks.
5. Keep at least one "harder" scenario (where savings are smaller) to avoid cherry-picking.
6. When comparing configs, treat pass rate as the primary outcome and tokens/time as tie-breakers.
