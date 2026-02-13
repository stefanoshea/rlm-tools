# RLM Tools - Design Document

**Goal:** A Python MCP server that gives Claude Code a persistent, sandboxed Python REPL for codebase exploration -- keeping file contents in Python memory (variable space) instead of the conversation context (token space).

**Architecture:** Interactive RLM pattern where Claude is the LLM in the loop. The MCP server provides a persistent Python sandbox with built-in codebase helpers and an `llm_query()` bridge for semantic sub-analysis via Haiku.

**Tech Stack:** Python 3.10+, `mcp[cli]` (FastMCP), `anthropic` SDK, subprocess isolation

---

## MCP Tools

### `rlm_start(path, query, max_output_chars?)`
- Scans directory: file tree, sizes, languages, line counts
- Pre-loads metadata into sandbox namespace
- Returns session_id + metadata summary

### `rlm_execute(session_id, code)`
- Runs Python in session's persistent namespace
- Only print() output returns to Claude
- Truncated at max_output_chars
- Returns stdout, errors, and list of variable names

### `rlm_end(session_id)`
- Tears down session, frees memory

## Sandbox Built-ins
- `read_file(path)` - load file into variable
- `grep(pattern, path=".")` - search files
- `glob_files(pattern)` - find files by pattern
- `tree(path=".")` - directory structure
- `llm_query(prompt, context="")` - sub-LLM call via Anthropic API
- `llm_query_batched(prompts)` - batch sub-queries

## Security
- Read-only filesystem (scoped to session path)
- No network, no process spawning, no writes
- Import allowlist: re, json, collections, math, os.path, pathlib
- 10-minute inactivity timeout, 256MB memory cap per session, max 5 concurrent sessions

## Distribution
- pip-installable, runnable via `uvx rlm-tools`
- Configured via `.mcp.json` in team repos
- `ANTHROPIC_API_KEY` env var for llm_query()
