# RLM Tools

<!-- mcp-name: io.github.stefanoshea/rlm-tools -->

Large grep results and file reads can fill an AI coding agent's context window before it has answered the question.

RLM Tools gives your agent a persistent sandbox to explore code in. Local exploration keeps intermediate file reads and search results in the server process. The agent receives the output it asks for. Optional model helpers can send repository content to external providers.

RLM Tools keeps raw context in tool-side memory and exposes a genuinely recursive explorer: `rlm_recurse` runs its own server-side tool-use loop (grep/read/glob/tree) and can spawn depth-limited child agents, so a sub-question is answered by exploring the corpus directly instead of dumping it into the root agent's context. Scored evals and prompt benchmarks measure answer quality alongside efficiency.

```bash
# For Codex / generic MCP clients
uvx --from rlm-tools rlm-tools-setup --write-policy AGENTS.md

# For Claude Code
uvx --from rlm-tools rlm-tools-setup --write-policy CLAUDE.md
```

This auto-detects Claude/Codex on your PATH, registers the MCP server, and writes a policy block that nudges agents toward `rlm-tools` for repo exploration. Use `CLAUDE.md` for Claude Code (it reads this file automatically) or `AGENTS.md` for other clients.

If you prefer manual setup:

```bash
# Claude Code
claude mcp add rlm-tools -- uvx rlm-tools

# Codex
codex mcp add rlm-tools -- uvx rlm-tools
```

## What Changes

**Without RLM Tools** — agent greps for `import UIKit`, gets 500 matches dumped into context. Reads 10 files, burns all their content as tokens. Context window fills up. Agent forgets what it was doing.

**With RLM Tools** — agent runs the same exploration in a server-side Python sandbox. In this example, only the printed summary enters the agent's context:

```python
matches = grep("import UIKit")
by_module = {}
for m in matches:
    module = m["file"].split("/")[0]
    by_module.setdefault(module, []).append(m)
for module, ms in sorted(by_module.items(), key=lambda x: -len(x[1]))[:5]:
    print(f"{module}: {len(ms)} files")
```

500 lines of grep results become 5 lines of summary. The agent sees what it needs, nothing more.

## Real-World Impact

On the four checked-in Python and TypeScript fixture tasks, the RLM scripts used **13.6% to 54.5% fewer estimated context characters** than the baseline file and search operations. The estimate counts script or tool arguments, returned text, and 350 characters of overhead per measured turn. It excludes session startup and teardown. Run `uv run --extra dev python -m pytest evals/test_comparative.py -q -s` for the per-task counts. The prompt benchmark measures root-agent token usage and answer quality separately.

In heavy exploration tasks (reading many files, broad searches), savings go much further. The table below is a **worst-case** comparison: the "Standard Tools" baseline dumps full grep/file payloads into context, which is the upper bound on what naive tool use costs — a careful agent dumps less. Treat these as the ceiling on savings, not the typical case, and look at the [scored evals](#scored-evals) for whether answer quality holds up at those reductions.

| Scenario | Standard Tools | RLM Tools | Saved |
|---|---:|---:|---:|
| Grep across full app | 40,045 chars | 1,644 chars | 95.9% |
| Read 10 large files | 1,493,720 chars | 13,588 chars | 99.1% |
| Multi-step exploration | 136,102 chars | 5,285 chars | 96.1% |
| Grep then read matches | 340,408 chars | 6,022 chars | 98.2% |
| Find all usages of a pattern | 13,478 chars | 3,691 chars | 72.6% |
| Understand a module | 94,745 chars | 16,925 chars | 82.1% |

Full benchmark methodology and reproduction steps: [`docs/benchmarks.md`](docs/benchmarks.md)

## Scored Evals

The repo now carries fixture-backed, always-runnable evals under `evals/` plus checked-in corpora under `tests/fixtures/`.

- `evals/test_correctness.py` checks helper semantics and security boundaries
- `evals/test_task_completion.py` checks multi-step task workflows
- `evals/test_comparative.py` keeps context-size microbenchmarks
- `evals/test_robustness.py` checks error handling and timeout behavior
- `evals/test_session_effectiveness.py` compares a complete session with targeted Read/Grep, including follow-ups where the baseline can reuse its first result
- `evals/test_longcontext_quality.py` checks grounded answers and plausible wrong answers on the long-context fixture
- `evals/test_evidence_effectiveness.py` measures how much tool output the recursive loop replays while checking whether the answer survives filtering
- `evals/test_stdio_workflow.py` exercises independent sessions and compact responses through the actual stdio server

Prompt benchmarks are defined in [`evals/specs/rlm_core.json`](evals/specs/rlm_core.json) and the harder [`evals/specs/rlm_long_context.json`](evals/specs/rlm_long_context.json), graded by [`evals/grade_eval.py`](evals/grade_eval.py), and aggregated with variance notes plus a pass-rate-first preferred-config summary by [`evals/aggregate_benchmark.py`](evals/aggregate_benchmark.py).

## How It Works

Three MCP tools. That's the entire API:

| Tool | Purpose |
|---|---|
| `rlm_start(path, query)` | Open a session on a directory |
| `rlm_execute(session_id, code)` | Run Python in the sandbox |
| `rlm_end(session_id)` | Close session, free resources |

The sandbox provides built-in helpers:

- `read_file(path)` / `read_files(paths)` — Read files into variables (cached across calls)
- `grep(pattern)` / `grep_summary(pattern)` / `grep_read(pattern)` — Search
- `glob_files(pattern)` — Find files by pattern
- `tree(path, max_depth)` — Directory structure
- `llm_query(prompt, context)` / `llm_query_batched(prompts, context)` — Sub-LLM analysis (optional, requires API key)
- `rlm_recurse(prompt, context)` / `rlm_recurse_batched(prompts, context)` — Ask a sub-agent to explore the session directory (optional, requires API key)
- `rlm_finalize(value)` — Return structured final output in `rlm_execute` responses

Variables persist across `rlm_execute` calls within a session. The agent can build up understanding incrementally — search, filter, read, analyze — without any intermediate data touching the context window.

### Evidence Gate

A `rlm_recurse` sub-agent re-sends every tool result on every turn of its loop, so a
large file read is paid for once per remaining turn. When TypeSafe scoring is enabled, each
result is scored in chunks before it joins that history and only the parts bearing on
the question are kept:

```
worker_runtime.py   12,247 -> 1,256 chars   1 of 9 chunks kept
server.py           15,924 -> 1,491 chars   1 of 11 chunks kept
helpers.py           9,200 -> 3,193 chars   2 of 7 chunks kept
```

81% of bytes dropped across those reads with every answer still intact. Scoring is
batched, so judging a dozen chunks costs about what judging one costs.

The gate fails open. A scoring error, an exhausted call budget, a result too small to
judge, a flat score distribution, or an exhaustive question (`every`, `all`, `list`)
all return the original text untouched — exhaustive questions need every mention, and
relevance scores cannot tell the last one from noise.

The gate uses [TypeSafe](https://typesafe.ai) System One models. To enable it, run
the server with the semantic extra (`uvx --from 'rlm-tools[semantic]' rlm-tools`)
and pass both API keys, set `RLM_RECURSE_GATE=1`, and include the repository in `RLM_EXTERNAL_ROOTS`. TypeSafe receives the question and candidate excerpts, including excerpts the gate later drops.

## Works With

RLM Tools is a standard [MCP](https://modelcontextprotocol.io) server. It works with any MCP-compatible client: **Claude Code**, **Codex**, **Cursor**, and others.

### Optional Strict Mode

If your MCP client supports tool permissions or hooks, disable/block default read/grep/glob tools for repository exploration and keep `rlm-tools` enabled. This is the strongest way to force the token-efficient path.

<details>
<summary><strong>Other installation methods</strong></summary>

### JSON MCP config (Cursor, Windsurf, etc.)

```json
{
  "mcpServers": {
    "rlm-tools": {
      "command": "uvx",
      "args": ["rlm-tools"]
    }
  }
}
```

### Setup helper command

```bash
# Auto-detect Claude/Codex CLIs and configure rlm-tools
uvx --from rlm-tools rlm-tools-setup

# Print JSON config snippet for other clients
uvx --from rlm-tools rlm-tools-setup --print-json

# Print policy text instead of writing it
uvx --from rlm-tools rlm-tools-setup --print-policy
```

### Direct run

```bash
uvx rlm-tools
```

### From source

```bash
git clone https://github.com/stefanoshea/rlm-tools.git
cd rlm-tools
uv sync
uv run rlm-tools
```

Then point your MCP client to `command: uv`, `args: ["--directory", "/path/to/rlm-tools", "run", "rlm-tools"]`, and set `RLM_ALLOWED_ROOTS` to the checkout you want to explore.

</details>

## Configuration

Set these variables in the MCP **server** environment and restart the server after changing them. By default, the server allows exploration underneath its launch directory. If a client launches it from your home directory or the filesystem root, configure an explicit workspace. Multiple roots use the platform path separator (`:` on macOS and Linux).

For example, this configuration permits local exploration of two repositories and enables Anthropic helpers only for the second, when the server also has an Anthropic key:

```json
{
  "RLM_ALLOWED_ROOTS": "/work/private:/work/shareable",
  "RLM_EXTERNAL_ROOTS": "/work/shareable"
}
```

A grant covers the named directory and its descendants. Keep private repositories outside roots granted external processing. Anthropic receives the prompts, context, and tool results used by the model helpers. Enabling the optional TypeSafe gate also sends the question and candidate excerpts to TypeSafe. Ordinary read, grep, glob, and tree operations require no provider. Keep real credentials in your client's secret configuration or an environment file outside the explored checkout.

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_ALLOWED_ROOTS` | Server launch directory, except home or filesystem root | Absolute directories the operator permits sessions to explore. An explicitly empty list denies all roots. |
| `RLM_EXTERNAL_ROOTS` | Empty | Subset of allowed directories eligible for external model helpers. |
| `ANTHROPIC_API_KEY` | — | Enables Anthropic model helpers for external-enabled sessions. |
| `TYPESAFE_API_KEY` | — | Credential for optional TypeSafe scoring. |
| `TYPESAFE_DEFAULT_MODEL` | `jev-latest` | System One model used by the evidence gate |
| `RLM_RECURSE_GATE` | `0` | Set to `1` to enable TypeSafe scoring for external-enabled Anthropic sessions. |
| `RLM_SUB_MODEL` | `claude-haiku-4-5-20251001` | Claude model used for `llm_query()`/`rlm_recurse()` |
| `RLM_SUB_TIMEOUT_SECONDS` | `60` | Per-request timeout for sub-LLM calls |
| `RLM_SUB_MAX_RETRIES` | `3` | Retry attempts for transient sub-LLM failures |
| `RLM_RECURSE_MAX_DEPTH` | `2` | Max recursion depth for `rlm_recurse()` child agents |
| `RLM_MAX_SESSIONS` | `5` | Max concurrent sessions |
| `RLM_SESSION_TIMEOUT` | `10` | Session timeout in minutes |

## Security

The stdio server is designed for one local MCP client. It starts each Python worker with an empty environment, a private IPC socket, and isolated Python startup. macOS uses Seatbelt. Linux requires `libseccomp.so.2` and permits the system calls needed for computation and the private socket. Worker startup fails if the OS or launcher cannot apply the policy, including hosts that forbid nested macOS sandboxing. The trusted parent holds repository descriptors and provider credentials.

- A session can open only an operator-approved directory. Helpers open paths relative to a retained directory descriptor, skip symlinks and multiply linked files, and read regular files only. They exclude Git metadata, local MCP configuration, registry token files, and `.env` variants except `.env.example`, `.env.sample`, and `.env.template`. Other files within the root can be returned to the agent.
- Reads, batches, regex searches, directory scans, and IPC messages have size limits. A file read returns at most 4 MiB of source text plus a truncation marker. Grep uses RE2 syntax and a 1 MiB engine memory budget. For patterns that need Python lookaround or backreferences, apply the worker's `re` module to a bounded file read. Narrow the path or pattern when a search reports a limit. Metadata can include `truncated: true`.
- Python imports expose selected public names from approved standard-library modules. Relative imports, private attributes, and wrapper helpers are restricted inside the OS-confined worker.
- Workers enforce an execution timeout, restrict Python imports and builtins, and bound printed output. CPU and address-space backstops depend on OS support. Parent-side model requests use the remaining execution time for their network timeouts. An in-flight filesystem or network operation can still delay a response.
- Model helpers are enabled per root. A local session never registers them, even when the server has provider credentials. Treat any root enabled for external processing as shareable with the selected providers.

## Background

RLM Tools implements an [RLM-style](https://arxiv.org/abs/2512.24601) exploration loop: keep raw data in tool-side memory, send only compact outputs to the model. Built on the [Model Context Protocol](https://modelcontextprotocol.io).

## Development

```bash
git clone https://github.com/stefanoshea/rlm-tools.git
cd rlm-tools
uv sync --locked --extra dev
uv run --no-sync python -m pytest
```

Run fixture-backed evals:

```bash
uv run --extra dev python -m pytest evals -q
```

Run comparative benchmarks:

```bash
uv run --extra dev python -m pytest evals/test_comparative.py -q -s
```

Run scored prompt benchmarks:

```bash
./evals/run_ab_eval.sh
```

By default this now runs the checked-in long-context benchmark corpus at `tests/fixtures/rlm_corpus_longcontext`. Use `RLM_AB_SPEC_PATH` and `RLM_AB_TARGET_PATH` to switch back to lighter smoke-style cases or to point at a different corpus.

## License

MIT
