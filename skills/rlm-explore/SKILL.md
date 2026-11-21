---
name: rlm-explore
description: Explore large codebases inside a confined Python sandbox (`rlm-tools`) instead of dumping raw files into context. Use for broad cross-repo searches, counting or aggregating patterns across many files, multi-hop call tracing, or read-only inspection of untrusted repositories.
---

Use native read/grep tools for a single known file or narrow lookup. Open an `rlm-tools` session when raw search matches or multi-file reads would bloat context, or when inspecting an untrusted repository in an OS-confined read-only sandbox.

## Workflow

1. **Start one session** with `rlm_start(path, query)` using the absolute repository root. Keep `include_guidance=False` and `include_metadata=False` for compact startup. Check `available_functions` in the response (external model helpers appear only when the root is enabled in `RLM_EXTERNAL_ROOTS`).
2. **Batch 3–5+ operations per `rlm_execute` call.** Orient with `tree()` or a wide case-insensitive `grep_summary()`, filter candidates in Python, inspect surviving files with `read_files()` or `grep_read()`, and `print()` only distilled findings — or call `rlm_finalize(value)` for structured output. Variables and file reads persist across `rlm_execute` turns; aim for ~3–10 turns total.
3. **Use semantic helpers when listed.** Delegate independent sub-questions to `rlm_recurse()` / `rlm_recurse_batched()` so child reads stay off the root context; use `llm_query()` / `llm_query_batched()` for flat analysis on extracted snippets.
4. **Close the session** with `rlm_end(session_id)` once every required fact is verified from sandbox output.

## Boundaries & Gotchas

- **RE2 regex:** `grep`, `grep_summary`, and `grep_read` use linear-time RE2 (no lookarounds or backreferences). When a pattern needs Python `re` features, narrow candidates first and run `re` inside the worker on bounded `read_file()` slices.
- **Truncation:** Watch for `truncated` or `[scan truncated]` markers; narrow `path` or the pattern rather than treating truncated output as exhaustive.
- **Read-only isolation:** The sandbox skips symlinks, hardlinks, `.git`, and `.env*` secrets (except `.env.example`/`.sample`/`.template`). Switch to native edit tools once exploration is complete and edits are needed.
