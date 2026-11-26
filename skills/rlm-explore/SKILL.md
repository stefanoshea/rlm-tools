---
name: rlm-explore
description: Explore large codebases inside a confined Python sandbox (`rlm-tools`) instead of dumping raw files into context. Use for broad cross-repo searches, counting or aggregating patterns across many files, multi-hop call tracing, or read-only inspection of untrusted repositories.
---

Use native read/grep tools for a single known file or narrow lookup. Open an `rlm-tools` session when raw search matches or multi-file reads would bloat context, or when inspecting an untrusted repository in an OS-confined read-only sandbox.

## Workflow

1. **Start one session** with `rlm_start(path, query, code=...)` using the absolute repository root. Keep `include_guidance=False` and `include_metadata=False` for compact startup, and optionally pass an initial `tree()` / `grep_summary()` script in `code=` to save a turn. Check `available_functions` in the response (external model helpers appear only when the root is enabled in `RLM_EXTERNAL_ROOTS`).
2. **Batch 3–5+ operations per `rlm_execute` call.** Orient with `tree()` or a wide case-insensitive `grep_summary()`, filter candidates in Python, inspect surviving files with `read_files()` or `grep_read()`, use `enclosing(path, line)` / `find_assignments(path, target)` on large Python files to avoid nested-function or annotated-assignment mistakes, and `print()` only distilled findings — or call `rlm_finalize(value)` for structured output. Variables and file reads persist across `rlm_execute` turns; aim for ~3–10 turns total.
3. **Use semantic helpers when listed, or `rlm-explorer` in Claude Code.** When `available_functions` includes `llm_query()` / `rlm_recurse()`, delegate independent sub-questions to `rlm_recurse()` / `rlm_recurse_batched()` so child reads stay off the root context, and use `llm_query()` / `llm_query_batched()` for flat analysis on extracted snippets. Note that `RLM_SUB_PROVIDER=claude-cli` defaults `execution_timeout_seconds` to `180` (configurable via `rlm_start` or `RLM_EXECUTION_TIMEOUT_SECONDS`). In Claude Code without external helpers configured, delegate independent multi-file sub-questions in parallel to the bundled `rlm-explorer` subagent.
4. **Close the session** by passing `end_session=True` on your final `rlm_execute` call (or calling `rlm_end(session_id)`) once every required fact is verified from sandbox output.

## Boundaries & Gotchas

- **RE2 regex:** `grep`, `grep_summary`, and `grep_read` use linear-time RE2 (no lookarounds or backreferences). When a pattern needs Python `re` features, narrow candidates first and run `re` inside the worker on bounded `read_file()` slices.
- **Truncation:** Watch for `TRUNCATED` banners or `truncated: true` fields; narrow `path` or the pattern rather than treating truncated output as exhaustive.
- **Scan filtering vs. read-only isolation:** Directory scans and greps honour `.gitignore` and skip binary files; direct `read_file`/`read_files`/`enclosing`/`find_assignments` still work on known paths. Enforced read-only isolation blocks symlinks, hardlinks, `.git`, and `.env*` secrets (except `.env.example`/`.sample`/`.template`) across all helpers. Switch to native edit tools once exploration is complete and edits are needed.
