# RLM Tools

Your AI coding agent spends most of its token budget just *reading* your code — not reasoning about it. Every grep, file read, and glob result gets dumped into the conversation. On a large codebase, that's 25-35% of your context (and cost) burned on raw data the model never needed to see.

RLM Tools gives your agent a persistent sandbox to explore code in. Data stays server-side. Only the conclusions come back.

```bash
# Install in one line (Claude Code)
claude mcp add rlm-tools -- uvx rlm-tools

# Or Codex
codex mcp add rlm-tools -- uvx rlm-tools
```

That's it. Your agent automatically uses the sandbox for exploration. No config, no prompting changes.

## What Changes

**Without RLM Tools** — agent greps for `import UIKit`, gets 500 matches dumped into context. Reads 10 files, burns all their content as tokens. Context window fills up. Agent forgets what it was doing.

**With RLM Tools** — agent runs the same exploration in a server-side Python sandbox. Data stays in sandbox memory. Only the `print()` output enters context:

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

In typical coding workflows: **25-35% context reduction.** That means your agent can explore roughly 40-50% more code before hitting context limits.

In heavy exploration tasks (reading many files, broad searches), savings go much further:

| Scenario | Standard Tools | RLM Tools | Saved |
|---|---:|---:|---:|
| Grep across full app | 40,045 chars | 1,644 chars | 95.9% |
| Read 10 large files | 1,493,720 chars | 13,588 chars | 99.1% |
| Multi-step exploration | 136,102 chars | 5,285 chars | 96.1% |
| Grep then read matches | 340,408 chars | 6,022 chars | 98.2% |
| Find all usages of a pattern | 13,478 chars | 3,691 chars | 72.6% |
| Understand a module | 94,745 chars | 16,925 chars | 82.1% |

Full benchmark methodology and reproduction steps: [`docs/benchmarks.md`](docs/benchmarks.md)

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
- `llm_query(prompt, context)` — Sub-LLM analysis (optional, requires API key)

Variables persist across `rlm_execute` calls within a session. The agent can build up understanding incrementally — search, filter, read, analyze — without any intermediate data touching the context window.

## Works With

RLM Tools is a standard [MCP](https://modelcontextprotocol.io) server. It works with any MCP-compatible client: **Claude Code**, **Codex**, **Cursor**, and others.

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

Then point your MCP client to `command: uv`, `args: ["--directory", "/path/to/rlm-tools", "run", "rlm-tools"]`.

</details>

## Configuration

Copy `.env.example` to `.env` to customize. All settings are optional — RLM Tools works out of the box with zero config.

The core exploration features (read, grep, glob, tree) require no API key. The optional `llm_query()` helper calls the Anthropic API for semantic analysis within the sandbox — this is the only feature that requires a key.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for `llm_query()` only. Uses Anthropic's API (Claude). |
| `RLM_SUB_MODEL` | `claude-haiku-4-5-20251001` | Claude model used for `llm_query()` |
| `RLM_MAX_SESSIONS` | `5` | Max concurrent sessions |
| `RLM_SESSION_TIMEOUT` | `10` | Session timeout in minutes |

## Security

The sandbox is read-only and restricted:

- **Imports**: Safe stdlib only (re, json, collections, math, etc.)
- **Builtins**: Blocks exec, eval, compile, `__import__`, breakpoint
- **File access**: Read-only, scoped to session directory, path traversal blocked
- **Execution**: Configurable per-call timeout (default 30s)
- **Rate limits**: Configurable max calls per session

## Background

RLM Tools implements an [RLM-style](https://arxiv.org/abs/2512.24601) exploration loop: keep raw data in tool-side memory, send only compact outputs to the model. Built on the [Model Context Protocol](https://modelcontextprotocol.io).

## Development

```bash
git clone https://github.com/stefanoshea/rlm-tools.git
cd rlm-tools
uv sync --dev
pytest tests
```

Run comparative benchmarks (requires a local project checkout):

```bash
RLM_EVAL_PROJECT_PATH=/path/to/project pytest evals -q -s
```

## License

MIT
