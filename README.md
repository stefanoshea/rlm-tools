# RLM Tools

<!-- mcp-name: io.github.stefanoshea/rlm-tools -->

RLM Tools gives coding agents a read-only Python sandbox for exploring a repository. An agent can search files, keep results in memory across calls, and return a count, a list of paths, or a short excerpt to the conversation. Local exploration needs no API key. Optional model helpers (`llm_query`, `rlm_recurse`) can send repository content to Anthropic, Gemini, any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, vLLM, LM Studio), or a logged-in local Claude Code CLI (`claude-cli`) when you enable them for that workspace.

## Plugins & Extensions

The plugins register the MCP server and install the `rlm-explore` skill for Codex CLI, Claude Code, and Gemini CLI. The server runs on macOS and Linux and requires `uvx` on the client's `PATH`. Run `uvx --version` in your terminal to check. Linux also requires `libseccomp.so.2`. [Release notes](https://github.com/stefanoshea/rlm-tools/releases) · [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

For Cursor, VS Code, Kiro, and standalone MCP setup, see [other clients and installation methods](#other-clients).

### Codex CLI

Install the plugin once from your terminal.

```sh
codex plugin marketplace add stefanoshea/rlm-tools
codex plugin add rlm-tools@rlm-tools
```

Each time you start Codex, run this from the repository you want to explore. Codex launches the server in the installed plugin directory, so the workspace grant tells it which checkout it can read.

```sh
RLM_ALLOWED_ROOTS="$(pwd -P)" codex --no-daemon
```

### Claude Code

Open Claude Code in your repository and enter these commands at the Claude prompt.

```text
/plugin marketplace add stefanoshea/rlm-tools
/plugin install rlm-tools@rlm-tools
```

The plugin also bundles an `rlm-explorer` subagent (`model: haiku`, restricted to `rlm-tools`) so Claude Code subscribers get parallel, context-isolated sandboxed exploration out of the box. To also run programmatic `llm_query()` / `rlm_recurse()` calls inside sandbox Python scripts on your Claude subscription without an API key, see [Claude Code subscription (`claude-cli`)](#claude-code-subscription-claude-cli).

### Gemini CLI

Install the extension from your terminal.

```sh
gemini extensions install https://github.com/stefanoshea/rlm-tools
```

Restart Gemini in your repository and run `/skills list`. You should see `rlm-explore`, ready to load when an exploration task needs it.

### Try it

Try this prompt in the client you just configured.

> Use RLM Tools to list the top-level directories in this repository, then close the session.

The agent should call `rlm_start`, run `tree(max_depth=1)` through `rlm_execute`, and close the session, ideally by passing `end_session=true` on that last `rlm_execute` call (a separate `rlm_end` also works). If startup returns `Directory outside RLM_ALLOWED_ROOTS`, check that the MCP server received the absolute path to your checkout. [Workspace configuration](#configuration)

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

Live paired A/B runs against Gemini (`gemini-3.5-flash-lite`, exact JSON grading) show where the tool wins and where it does not:

| Repository | Arm | Pass rate | Mean tokens |
|---|---|---:|---:|
| 230 files, 1.28 MB (`heavy`, 5 tasks) | Native `Read`/`Grep`/`Glob` | 60% (3/5) | 315,676 |
| 230 files, 1.28 MB (`heavy`, 5 tasks) | RLM Tools | **100% (5/5)** | **31,208** (−90.1%) |
| ~28 KB (`long_context`, 11 tasks) | Native `Read`/`Grep`/`Glob` | **100% (11/11)** | **23,464** |
| ~28 KB (`long_context`, 11 tasks) | RLM Tools | 81.8% (9/11) | 58,986 |

The `heavy` row is the 2026-11-24 run after the sandbox-contract and `end_session` changes; an earlier run of the same tasks measured native tools at 40% pass and 469,392 tokens against RLM Tools at 100% and 59,389 (−87.3%). The `long_context` row predates those changes, and a later re-run only completed 5 of 11 pairs before hitting a provider quota, so treat it as a baseline. On narrow lookups in small repositories, native tools use about 2.5× fewer tokens because opening and closing a session adds fixed overhead. Route narrow lookups to native tools and use RLM Tools for broad searches, audits, large files, and multi-step analysis; `rlm-tools-setup --write-policy` writes that guidance into `AGENTS.md` or `CLAUDE.md`.

The table below is a **worst-case** comparison where the "Standard Tools" baseline dumps full grep/file payloads into context; treat it as an upper bound and see [`docs/benchmarks.md`](docs/benchmarks.md) plus [scored evals](#scored-evals) for full measured results and crossover guidance.

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
| `rlm_start(path, query, code?)` | Open a session on a directory (optionally run initial code immediately) |
| `rlm_execute(session_id, code, end_session?)` | Run Python in the sandbox (set `end_session=true` on the final step to close) |
| `rlm_end(session_id)` | Explicitly close session and free resources |

The sandbox provides built-in helpers:

- `read_file(path, start_line=None, end_line=None)` / `read_files(paths)` — Read files or line slices into variables (cached across calls)
- `grep(pattern)` / `grep_summary(pattern)` / `grep_read(pattern)` — Search
- `glob_files(pattern, include_ignored=False)` — Find files by pattern (pass `include_ignored=True` to include `.gitignore`d non-secret files)
- `tree(path, max_depth)` — Directory structure
- `outline(path)` / `enclosing(path, line)` / `find_assignments(path, target)` — Python AST outline (with `parent` and `depth` on every class/function), scope, and assignment lookup
- `llm_query(prompt, context)` / `llm_query_batched(prompts, context)` — Sub-LLM analysis when full context is already in the prompt (optional, requires a provider key, `OPENAI_BASE_URL`, or `RLM_SUB_PROVIDER=claude-cli`)
- `rlm_recurse(prompt, context)` / `rlm_recurse_batched(prompts, context)` — Ask a sub-agent to explore the session directory and check code (optional, requires a provider key, `OPENAI_BASE_URL`, or `RLM_SUB_PROVIDER=claude-cli`; `max_llm_calls` defaults to `300`, size `≈ 4×N` for N-item recursion; children hitting `RLM_CHILD_MAX_TOTAL_TOOL_CHARS` return a `[BUDGET_EXHAUSTED]` prefix and increment `usage.child_budget_exhausted`)

The sandbox allows only these imports: `re`, `json`, `collections`, `math`, `fnmatch`, `itertools`, `functools`, `operator`, `string`, `textwrap`, `difflib`, `statistics`. `os`, `glob`, `pathlib`, `ast`, `open()`, `exec()`, `eval()`, `getattr()`, and `str.format()` are blocked. Use the helpers and f-strings instead. `rlm_start` returns the current `allowed_imports` list (and `other_open_sessions` if you already have sessions open).
- `rlm_finalize(value)` — Return structured final output in `rlm_execute` responses

Variables persist across `rlm_execute` calls within a session (and truncated stdout is retained in `_last_stdout` for follow-up paging). The agent can build up understanding incrementally — search, filter, read, analyze — without any intermediate data touching the context window.

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

81% of bytes dropped across those reads with every answer still intact. In a later
direct check with Google's `gemini-decision-1` on three source files (`llm_bridge.py`,
`helpers.py`, `server.py`), the gate dropped 91% of characters (73,068 to 6,381) with every
answer surviving, at roughly one second per gate call. These are single-file measurements of
the gate itself. The gate is off by default and was not enabled in the benchmark runs above.

Chunks are split at 40 lines or 2,400 characters, whichever comes first, and each chunk is
sent to the scorer in full. Scoring is batched (12 chunks per request), so judging a dozen
chunks costs one request.

The gate fails open. A scoring error, an exhausted call budget, a result too small to
judge, a flat score distribution, or an exhaustive question (`every`, `all`, `list`)
all return the original text untouched — exhaustive questions need every mention, and
relevance scores cannot tell the last one from noise.

The gate supports [TypeSafe](https://typesafe.ai) System One models (`uvx --from 'rlm-tools[semantic]' rlm-tools` with `TYPESAFE_API_KEY`) or Google's Decisions API (`uvx --from 'rlm-tools[gemini]' rlm-tools` with `GEMINI_API_KEY` and `gemini-decision-1`, requires Python 3.11+). To enable it, pass a sub-LLM key and scorer key (a single `GEMINI_API_KEY` can power both), set `RLM_RECURSE_GATE=1`, and include the repository in `RLM_EXTERNAL_ROOTS`. The configured provider receives the question and candidate excerpts, including excerpts the gate later drops.

## Works With

RLM Tools is a standard [MCP](https://modelcontextprotocol.io) server. It works with any MCP-compatible client: **Claude Code**, **Codex**, **Cursor**, and others.

### Other clients

- **Cursor, VS Code / GitHub Copilot, and Amazon Kiro** can use the Agent Plugins 1.0 package in [plugins/rlm-tools](plugins/rlm-tools). If your client starts the server from the plugin directory, set `RLM_ALLOWED_ROOTS` to the absolute checkout path in the server environment.
- **Agent Skills clients** such as Windsurf, OpenCode, Cline, and Zed can use the [standalone skill](plugins/rlm-tools/skills/rlm-explore/SKILL.md) alongside a standard MCP registration for `uvx rlm-tools`.

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

To configure a single client and write exploration guidance into the current repository, run the corresponding command from its root.

```sh
# Codex
uvx --from rlm-tools rlm-tools-setup --client codex --write-policy AGENTS.md

# Claude Code
uvx --from rlm-tools rlm-tools-setup --client claude --write-policy CLAUDE.md
```

To register just the server manually, use `codex mcp add rlm-tools -- uvx rlm-tools` or `claude mcp add rlm-tools -- uvx rlm-tools`.

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

For example, this configuration permits local exploration of two repositories and enables sub-LLM helpers only for the second, when the server also has an Anthropic, Gemini, OpenAI-compatible, or `claude-cli` provider configured:

```json
{
  "RLM_ALLOWED_ROOTS": "/work/private:/work/shareable",
  "RLM_EXTERNAL_ROOTS": "/work/shareable"
}
```

A grant covers the named directory and its descendants. Keep private repositories outside roots granted external processing. The configured provider (Anthropic, Gemini, OpenAI, any OpenAI-compatible Chat Completions endpoint such as OpenRouter, Ollama, LM Studio, or vLLM, or the local `claude` CLI) receives the prompts, context, and tool results used by the model helpers. OpenAI-compatible support works out of the box with no extra dependencies. Enabling the optional evidence gate also sends the question and candidate excerpts to TypeSafe or Gemini (`gemini-decision-1`). Ordinary read, grep, glob, and tree operations require no provider and do not import provider SDKs at startup. Keep real credentials in your client's secret configuration or an environment file outside the explored checkout.

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_ALLOWED_ROOTS` | Server launch directory, except home or filesystem root | Absolute directories the operator permits sessions to explore. An explicitly empty list denies all roots. |
| `RLM_EXTERNAL_ROOTS` | Empty | Subset of allowed directories eligible for external model helpers. |
| `ANTHROPIC_API_KEY` | — | Enables Anthropic model helpers for external-enabled sessions. |
| `GEMINI_API_KEY` | — | Enables Gemini Interactions helpers and `gemini-decision-1` gate (`[gemini]` extra, Python 3.11+). |
| `OPENAI_API_KEY` | — | Enables OpenAI (or OpenAI-compatible) `llm_query()` / `rlm_recurse()` helpers. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Custom OpenAI-compatible Chat Completions base URL (supports keyless local Ollama/vLLM/LM Studio endpoints when set alone). |
| `RLM_SUB_PROVIDER` | `auto` | Sub-LLM provider: `auto` (Anthropic, then Gemini, then OpenAI-compatible; never auto-selects `claude-cli`), `anthropic`, `gemini`, `openai` (`openai-compatible`), or `claude-cli` (`claude_cli`, `claude-code`). |
| `TYPESAFE_API_KEY` | — | Credential for optional TypeSafe scoring. |
| `TYPESAFE_DEFAULT_MODEL` | `jev-latest` | System One model used by the TypeSafe evidence gate. |
| `RLM_GATE_PROVIDER` | `auto` | Gate provider: `auto` (TypeSafe if key set, else Gemini), `typesafe`, or `gemini`. |
| `GEMINI_DECISION_MODEL` | `gemini-decision-1` | Gemini Decisions model used when `RLM_GATE_PROVIDER=gemini` (or `auto` with `GEMINI_API_KEY`). |
| `RLM_RECURSE_GATE` | `0` | Set to `1` to enable evidence-gate scoring for external-enabled sessions. |
| `RLM_SUB_MODEL` | `claude-haiku-4-5-20251001` / `gemini-3.5-flash-lite` / `gpt-4.1-mini` / `haiku` | Model used for `llm_query()` / `rlm_recurse()` (defaults by active provider). |
| `RLM_CLI_MAX_CONCURRENCY` | `8` | Max concurrent `claude` subprocesses when `RLM_SUB_PROVIDER=claude-cli`. |
| `RLM_SUB_TIMEOUT_SECONDS` | `60` | Per-request timeout for sub-LLM calls |
| `RLM_SUB_MAX_RETRIES` | `3` | Retry attempts for transient sub-LLM failures |
| `RLM_RECURSE_MAX_DEPTH` | `2` | Max recursion depth for `rlm_recurse()` child agents |
| `RLM_CHILD_MAX_TOOL_CHARS` | `24000` | Max characters per individual tool result returned to an `rlm_recurse()` child |
| `RLM_CHILD_MAX_TOTAL_TOOL_CHARS` | `96000` | Cumulative tool-result character budget per `rlm_recurse()` child run |
| `RLM_MAX_SESSIONS` | `16` | Max concurrent sessions (idle sessions over `RLM_IDLE_EVICT_SECONDS` are LRU-evicted at the cap) |
| `RLM_SESSION_TIMEOUT` | `10` | Session timeout in minutes |
| `RLM_IDLE_EVICT_SECONDS` | `60` | Minimum idle seconds before an inactive session can be evicted at `RLM_MAX_SESSIONS` |
| `RLM_MAX_MEMORY_BYTES` | `67108864` (64 MiB) | Default per-session retained memory cap in bytes |
| `RLM_EXECUTION_TIMEOUT_SECONDS` | `30` (`60` external API, `180` `claude-cli`) | Default per-`rlm_execute` timeout in seconds when not passed to `rlm_start` |
| `RLM_MAX_SCAN_BYTES` | `268435456` (256 MiB) | Max bytes read per workspace scan (`grep*` / `scan_metadata`) |
| `RLM_MAX_SCAN_ENTRIES` | `100000` | Max directory entries visited per workspace scan |
| `RLM_MAX_SCAN_SECONDS` | `15.0` | Max wall-clock seconds per workspace scan |

### Claude Code subscription (`claude-cli`)

To power `llm_query()` and `rlm_recurse()` from an existing Claude Code subscription login without an API key:

1. Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and log in once with `claude` (confirm `claude --version` works on `PATH`).
2. Set these variables on the `rlm-tools` MCP server (either in your MCP client config or via `claude mcp add`), pointing both roots at the parent directory or repository you want external helpers to inspect:

```json
{
  "env": {
    "RLM_SUB_PROVIDER": "claude-cli",
    "RLM_ALLOWED_ROOTS": "/path/to/your/workspace",
    "RLM_EXTERNAL_ROOTS": "/path/to/your/workspace"
  }
}
```

- **Explicit opt-in only:** `RLM_SUB_PROVIDER=claude-cli` (aliases `claude_cli`, `claude-code`) is never selected in `auto` mode.
- **No token extraction:** `rlm-tools` never reads or passes your OAuth token. Instead, it runs `claude -p` as a subprocess with `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, and nested-session variables (`CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`, `CLAUDE_CODE_SSE_PORT`) stripped from the child environment, inside a temporary working directory with tools disabled, an empty strict MCP config (`{"mcpServers":{}}`), `--setting-sources ""`, and `--safe-mode`.
- **Latency & usage limits:** Each call starts a new `claude -p` process (~1–2 seconds of startup overhead per call), so it is slower than direct API calls and counts toward your Claude subscription usage limits. When running `llm_query_batched()` or `rlm_recurse()`, `execution_timeout_seconds` defaults to `180` for `claude-cli`, and you can adjust `RLM_CLI_MAX_CONCURRENCY` (default `8`) or `RLM_SUB_MODEL` (default `haiku`) as needed.

## Security

The stdio server is designed for one local MCP client. It starts each Python worker with an empty environment, a private IPC socket, and isolated Python startup. macOS uses Seatbelt. Linux requires `libseccomp.so.2` and permits the system calls needed for computation and the private socket. Worker startup fails if the OS or launcher cannot apply the policy, including hosts that forbid nested macOS sandboxing. The trusted parent holds repository descriptors and provider credentials.

- A session can open only an operator-approved directory. Helpers open paths relative to a retained directory descriptor, skip symlinks and multiply linked files, and read regular files only. They exclude Git metadata (`.git`), local MCP and agent config (`.mcp.local.json`, `settings.local.json`), registry/auth token files (`.npmrc`, `.pypirc`, `.netrc`, `.mcpregistry_*`), and `.env` variants except `.env.example`, `.env.sample`, and `.env.template` (committed `.mcp.json` files remain readable). Other files within the root can be returned to the agent.
- Reads, batches, regex searches, directory scans, and IPC messages have size limits. A file read returns at most 4 MiB of source text plus a truncation marker. Grep uses RE2 syntax and a 1 MiB engine memory budget. For patterns that need Python lookaround or backreferences, apply the worker's `re` module to a bounded file read. Narrow the path or pattern when a search reports a limit. Metadata can include `truncated: true`.
- Python imports expose selected public names from approved standard-library modules. Relative imports, private attributes, and wrapper helpers are restricted inside the OS-confined worker.
- Workers enforce an execution timeout, restrict Python imports and builtins, and bound printed output. Per-session retained memory (default `64 MiB`) is enforced post-execution with variable eviction backstops, and Linux `RLIMIT_AS` caps runaway virtual allocations in flight (macOS ignores `RLIMIT_AS`). CPU and address-space backstops depend on OS support. Parent-side model requests use the remaining execution time for their network timeouts. An in-flight filesystem or network operation can still delay a response.
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

[![Rlm Tools MCP server](https://mcprush.com/badge/rlm-tools-mcp/card.svg)](https://mcprush.com/to/rlm-tools-mcp?ref=badge)

## License

MIT
