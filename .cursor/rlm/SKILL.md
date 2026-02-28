---
name: rlm
description: Explore and search code in ERP directories (extension and mainconf) via MCP rlm-tools. Use when exploring, searching, or reading code in d:\ERP\extension_code\, d:\ERP\mainconf_code\, or when ERP extension, main configuration, or rlm-tools are mentioned for these paths.
---

# Exploring ERP code (rlm-tools)

When you need to explore, search, or read code in the **d:\ERP\extension_code\** or **d:\ERP\mainconf_code\** directories:

- Use **only** MCP rlm-tools: `rlm_start`, `rlm_execute`, `rlm_end`.
- **Do not** use built-in tools (grep, read_file, codebase_search, etc.) to access these directories.
- When starting work, call `rlm_start` with parameters:
  - **path**: `d:\ERP\extension_code` or `d:\ERP\mainconf_code` (absolute path to the selected directory)
  - **query**: short description of the exploration task
  - **Encoding**: always use **UTF-8**. Do not pass `file_encoding` (so the default utf-8 is used) or pass `file_encoding: "utf-8"` explicitly.
  - **force_rescan** (optional, default `false`): set to `true` to force full directory rescan and rebuild the cache. Use when files have changed significantly since the last session.
- **Caching**: on first start, `rlm_start` performs a full directory scan and saves metadata + file index to `.rlm_cache/` inside the target directory. Subsequent starts load from cache instantly. For `d:\ERP\mainconf\` (50,000+ files) this eliminates the hanging on repeated sessions. Cache is invalidated automatically when the directory's mtime changes, or manually via `force_rescan=true`.
- Perform exploration via `rlm_execute(session_id, code)` with Python code in the sandbox (read_file, grep, glob_files, tree, etc.).
- When finished, call `rlm_end(session_id)` if necessary.

## Rules for working with rlm-tools

- **Session root** — the directory from `path` in rlm_start. All paths in `read_file`, `grep`, `glob_files` are **relative** to this root. Use forward slashes: `Documents/ЗаявкаНаРасходованиеДенежныхСредств/Forms/ФормаДокумента/Ext/Form/Module.bsl`.
- **Available functions** (returned in `rlm_start`): `read_file(path)`, `read_files(paths)`, `grep(pattern, path='.')`, `grep_summary(pattern, path='.')`, `grep_read(pattern, path='.', max_files=10, context_lines=0)`, `glob_files(pattern)`, `tree(path='.', max_depth=3)`, `find_files(substring, case_sensitive=True)`.
- **IMPORTANT: always use `print()` for output** — `rlm_execute` captures only stdout. All functions return values silently; to see results, wrap in `print()`. Example: `print(tree('Documents/...'))`, `print(read_file('...'))`.
- **grep** returns a list of matches; if no matches, an empty list. Process the result in code (e.g., output via `print`). Do not duplicate named arguments in a single call (e.g., `path`).
- **find_files(substring)** — fast search for files by substring in path. Preferred for finding 1C objects by name (e.g., `find_files('ЗаявкаНаРасходование')`).
- **grep_read** returns a `dict` with keys `matches`, `files`, `summary`. `matches` is also a **dict**: `{filepath: [{file, line_number, line}, ...]}` — NOT a list. Iterate as `for filepath, file_matches in results['matches'].items()`. Example:
  ```python
  results = grep_read('pattern', path='Documents/...', max_files=20, context_lines=2)
  for filepath, file_matches in results['matches'].items():
      parts = filepath.replace('\\\\', '/').split('Documents/')[-1]
      print(f'=== Documents/{parts} ===')
      for m in file_matches:
          print(f'  {m["line_number"]}: {m["line"].strip()}')
  ```
- **glob_files**: patterns like `Documents/ИмяДокумента/**/*` or `**/*.bsl`. 1C object names are Latin and Russian words without spaces (e.g., `ЗаявкаНаРасходованиеДенежныхСредств`).
- **Configuration/Extension structure**: at the root `Configuration.xml`, `ConfigDumpInfo.xml`; document forms — `Documents/<DocumentName>/Forms/<FormName>/Ext/Form/Module.bsl` and `Form.xml`. The form module is the main source of procedures with directives (`&НаСервере`, `&После("EventName")`, etc.). Object module — `Documents/<DocumentName>/Ext/ObjectModule.bsl`, manager module — `Ext/ManagerModule.bsl`.
- **Searching for procedures with a directive** (e.g., `&После`): in the module, iterate through lines; a line with a directive is usually a separate line like `&После("EventName")`; the next line is the procedure/function declaration. Search for lines starting with `&` and output the pair "directive + next line".
- In the sandbox, `import os` and apparently other arbitrary imports are **forbidden**; use only the provided functions and standard Python features (regular expressions via `re` are allowed if not forbidden).
- Variables in code between `rlm_execute` calls in the same session are **preserved** — you can split the search into multiple steps.

## Session timeouts and limits

- **Session timeout** — sessions expire after 10 minutes of inactivity (configurable via `RLM_SESSION_TIMEOUT` env var). If `rlm_execute` returns "Session not found or expired", call `rlm_start` again.
- **Execution timeout** — each `rlm_execute` call has a 30-second timeout (configurable via `execution_timeout_seconds` in `rlm_start`). Long-running code will be interrupted with `TimeoutError`. This works on both Windows and Unix.
- **LLM retry** — if `llm_query()` or `llm_query_batched()` fail due to network errors (connection, timeout, rate limiting), they automatically retry up to 3 times with exponential backoff. Configurable via `RLM_LLM_MAX_RETRIES` and `RLM_LLM_RETRY_DELAY` env vars.

## Troubleshooting

- **MCP tool temporarily unavailable** — if `call_mcp_tool` returns "Tool not found", wait a moment and retry. This is a transient MCP server issue, not a code problem.
- **Empty output from `tree()` or `read_file()`** — you forgot `print()`. These functions return strings, not print them automatically.
- **Session expired** — if `rlm_execute` fails with invalid session, call `rlm_start` again to create a new session. Sessions expire after 10 minutes of inactivity.