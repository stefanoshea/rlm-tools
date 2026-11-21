import codecs
import os
import pathlib
import re
import stat
import threading
from collections.abc import Callable
from fnmatch import translate

import re2

from rlm_tools._workspace import (
    MAX_PATH_BYTES,
    MAX_PATH_DEPTH,
    ScanBudget,
    Workspace,
    _SKIP_DIRS,
)
from rlm_tools.access_policy import is_private_component


MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_BATCH_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_RESULTS = 2_000
MAX_CACHE_FILES = 256
MAX_PATTERN_LENGTH = 4_096
REGEX_MEMORY_BYTES = 1024 * 1024
_TRUNCATION = "\n... [read truncated]"
_SCAN_TRUNCATION = "[scan truncated: workspace limit reached]"
_LINE_SEPARATOR = re.compile(r"[\n\v\f\x1c-\x1e\x85\u2028\u2029]")
_PATTERN_ESCAPE = re.compile(r"\\Q.*?(?:\\E|$)|\\.", re.DOTALL)

_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".avif",
    ".pdf", ".zip", ".tar", ".gz", ".xz", ".bz2", ".whl",
    ".o", ".a", ".so", ".dylib", ".dll", ".exe", ".bin", ".wasm",
    ".pyc", ".pyo", ".framework", ".xcassets", ".car", ".nib",
    ".storyboardc", ".momd", ".sqlite", ".db",
    ".ttf", ".otf", ".woff", ".woff2",
    ".mp3", ".mp4", ".mov", ".wav",
}


def _read_bytes(fd: int, limit: int, budget: ScanBudget) -> bytes:
    data = bytearray()
    remaining = min(limit, budget.remaining_bytes)
    while len(data) < remaining and budget.has_time():
        chunk = os.read(fd, min(64 * 1024, remaining - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        budget.remaining_bytes -= len(chunk)
    return bytes(data)


def _decode(data: bytes, complete: bool) -> str:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    return decoder.decode(data, final=complete).replace("\r\n", "\n").replace("\r", "\n")


def _read_text(fd: int, info: os.stat_result, limit: int, budget: ScanBudget):
    read_limit = min(limit + 1, budget.remaining_bytes)
    data = _read_bytes(fd, read_limit, budget)
    complete = (
        len(data) <= limit
        and len(data) >= info.st_size
        and (len(data) < read_limit or read_limit == 0 and info.st_size == 0)
        and budget.has_time()
    )
    # An incomplete multi-byte character at the byte boundary is omitted.
    return _decode(data[:limit], complete), complete


def _text_limit(text: str, limit: int) -> str:
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _search_lines(content: str, complete: bool):
    start = 0
    for separator in _LINE_SEPARATOR.finditer(content):
        yield content[start:separator.start()]
        start = separator.end()
    if complete and start < len(content):
        yield content[start:]


def scan_metadata(base_path: str) -> dict:
    """Return counts for a bounded scan of the canonical workspace root."""
    workspace = Workspace(base_path, canonical=True)
    extensions: dict[str, int] = {}
    total_files = total_lines = sampled_lines = sampled_files = 0
    budget = ScanBudget()
    try:
        with workspace.open(pathlib.Path("."), directory=True) as (root_fd, _):
            for parent_fd, path in workspace.files(root_fd, pathlib.Path("."), budget):
                try:
                    with workspace.open_entry(parent_fd, path.name) as (fd, info):
                        if not stat.S_ISREG(info.st_mode):
                            continue
                        ext = path.suffix or "(no ext)"
                        extensions[ext] = extensions.get(ext, 0) + 1
                        total_files += 1
                        if ext.lower() in _BINARY_EXTENSIONS:
                            continue
                        if budget.remaining_bytes == 0:
                            budget.truncated = True
                            continue
                        content, complete = _read_text(fd, info, MAX_FILE_BYTES, budget)
                        if not complete:
                            budget.truncated = True
                        line_count = content.count("\n") + bool(content and not content.endswith("\n"))
                        total_lines += line_count
                        if sampled_files < 500:
                            sampled_files += 1
                            sampled_lines += line_count
                except OSError:
                    continue
    finally:
        workspace.close()

    metadata = {
        "total_files": total_files,
        "total_lines": total_lines,
        "sampled_lines": sampled_lines,
        "sampled_files": sampled_files,
        "file_types": dict(sorted(extensions.items(), key=lambda item: -item[1])[:10]),
    }
    if budget.truncated:
        metadata["truncated"] = True
    return metadata


def make_helpers(
    base_path: str,
    memory_tracker: Callable[[int], None] | None = None,
    *,
    base_is_canonical: bool = False,
    deadline: Callable[[], float | None] | None = None,
) -> tuple[dict, Workspace]:
    workspace = Workspace(base_path, canonical=base_is_canonical)
    file_cache: dict[pathlib.Path, tuple[tuple[int, ...], str, int]] = {}
    cache_bytes = 0
    cache_lock = threading.Lock()

    def new_budget() -> ScanBudget:
        budget = ScanBudget()
        execution_deadline = deadline() if deadline is not None else None
        if execution_deadline is not None:
            budget.deadline = min(budget.deadline, execution_deadline)
        return budget

    def byte_limit(max_bytes_read: int | None) -> int:
        if max_bytes_read is None:
            return MAX_FILE_BYTES
        if max_bytes_read < 0:
            raise ValueError("max_bytes_read must be non-negative")
        return min(MAX_FILE_BYTES, max_bytes_read)

    def read(path: str, limit: int, budget: ScanBudget) -> str:
        nonlocal cache_bytes
        if not budget.has_time():
            raise TimeoutError("Workspace read deadline exceeded")
        target = workspace.relative(path)
        with workspace.open(target) as (fd, info):
            if not stat.S_ISREG(info.st_mode):
                raise IsADirectoryError(str(target))
            version = (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size)
            with cache_lock:
                cached = file_cache.get(target)
                if cached is not None and cached[0] == version:
                    _, content, size = cached
                    if size <= limit and info.st_size <= limit:
                        return content

            content, complete = _read_text(fd, info, limit, budget)
            size = len(content.encode("utf-8"))
            if not complete or size > limit:
                return _text_limit(content, limit) + _TRUNCATION
            with cache_lock:
                cached = file_cache.get(target)
                old_size = cached[2] if cached is not None else 0
                delta = size - old_size
                if (
                    (cached is not None or len(file_cache) < MAX_CACHE_FILES)
                    and cache_bytes + delta <= MAX_BATCH_BYTES
                ):
                    if memory_tracker is not None:
                        memory_tracker(delta)
                    file_cache[target] = (version, content, size)
                    cache_bytes += delta
                return content

    def read_file(path: str, max_bytes_read: int | None = None) -> str:
        return read(path, byte_limit(max_bytes_read), new_budget())

    def read_files(paths: list[str], max_bytes_read: int | None = None) -> dict[str, str]:
        """Read multiple files at once. Returns {path: content} dict."""
        if isinstance(paths, (str, bytes)):
            raise TypeError("paths must be a list of file paths, not a single string")
        budget = new_budget()
        remaining = MAX_BATCH_BYTES
        limit = byte_limit(max_bytes_read)
        result = {}
        for path in paths:
            if not budget.visit() or len(result) == MAX_RESULTS:
                break
            path_size = len(path.encode("utf-8", errors="surrogatepass"))
            if path_size > MAX_PATH_BYTES:
                raise ValueError("Path exceeds the workspace path limit")
            if remaining < path_size + len(_TRUNCATION):
                break
            try:
                content = read(path, min(limit, remaining - path_size - len(_TRUNCATION)), budget)
            except OSError as exc:
                content = _text_limit(f"[error: {exc}]", remaining - path_size)
            result[path] = content
            remaining -= path_size + len(content.encode("utf-8"))
        return result

    def search(pattern: str, path: str, max_matches: int | None, budget: ScanBudget) -> list[dict]:
        if len(pattern) > MAX_PATTERN_LENGTH:
            raise ValueError("Regex pattern exceeds the length limit")
        options = re2.Options()
        options.max_mem = REGEX_MEMORY_BYTES
        options.log_errors = False
        # Accept Python's end-of-string anchor alongside RE2's native spelling.
        if r"\Z" in pattern:
            pattern = _PATTERN_ESCAPE.sub(lambda m: r"\z" if m[0] == r"\Z" else m[0], pattern)
        try:
            compiled = re2.compile(pattern, options)
        except re2.error as exc:
            raise ValueError(f"Invalid or unsupported RE2 pattern: {exc}") from exc
        target = workspace.relative(path)
        match_limit = MAX_RESULTS if max_matches is None else min(MAX_RESULTS, max_matches)
        automatic_match_limit = max_matches is None or max_matches > MAX_RESULTS
        results = []
        remaining = MAX_RESULT_BYTES
        if match_limit <= 0:
            return results

        def search_file(fd, info, file_path):
            nonlocal remaining
            if not stat.S_ISREG(info.st_mode) or file_path.suffix.lower() in _BINARY_EXTENSIONS:
                return
            content, complete = _read_text(fd, info, MAX_FILE_BYTES, budget)
            if not complete:
                budget.truncated = True
            file_name = str(file_path)
            file_size = len(os.fsencode(file_name))
            for number, line in enumerate(_search_lines(content, complete), 1):
                if not budget.has_time():
                    return
                if compiled.search(line):
                    if remaining <= file_size + 48:
                        remaining = 0
                        budget.truncated = True
                        return
                    line = line.strip()
                    line_limit = min(8192, remaining - file_size - 48)
                    if len(line.encode("utf-8")) > line_limit:
                        if line_limit <= len(_TRUNCATION):
                            remaining = 0
                            budget.truncated = True
                            return
                        line = _text_limit(line, line_limit - len(_TRUNCATION)) + _TRUNCATION
                    remaining -= file_size + len(line.encode("utf-8")) + 48
                    results.append({"file": file_name, "line_number": number, "line": line})
                    if len(results) >= match_limit:
                        if automatic_match_limit:
                            budget.truncated = True
                        return

        try:
            with workspace.open(target) as (fd, info):
                if stat.S_ISREG(info.st_mode):
                    search_file(fd, info, target)
                else:
                    for parent_fd, file_path in workspace.files(fd, target, budget):
                        if len(results) >= match_limit or remaining == 0:
                            break
                        if budget.remaining_bytes == 0:
                            budget.truncated = True
                            break
                        try:
                            with workspace.open_entry(parent_fd, file_path.name) as (file_fd, file_info):
                                search_file(file_fd, file_info, file_path)
                        except TimeoutError:
                            raise
                        except OSError:
                            continue
        except FileNotFoundError:
            pass
        results.sort(key=lambda result: (result["file"], result["line_number"]))
        return results

    def grep(pattern: str, path: str = ".", max_matches: int | None = None) -> list[dict]:
        budget = new_budget()
        results = search(pattern, path, max_matches, budget)
        if budget.truncated:
            raise RuntimeError(_SCAN_TRUNCATION)
        return results

    def grep_summary(pattern: str, path: str = ".", max_matches: int | None = None) -> str:
        """Grep with compact output grouped by file. Returns a formatted string."""
        budget = new_budget()
        results = search(pattern, path, max_matches, budget)
        if not results:
            return _SCAN_TRUNCATION if budget.truncated else "No matches found."

        grouped: dict[str, list[dict]] = {}
        for result in results:
            grouped.setdefault(result["file"], []).append(result)

        lines = [f"{len(results)} matches in {len(grouped)} files:"]
        for file, matches in grouped.items():
            lines.append(f"\n  {file} ({len(matches)} matches):")
            lines.extend(f"    L{match['line_number']}: {match['line']}" for match in matches)
        if budget.truncated:
            lines.append(_SCAN_TRUNCATION)
        return "\n".join(lines)

    def grep_read(
        pattern: str,
        path: str = ".",
        max_files: int = 10,
        context_lines: int = 0,
        max_matches: int | None = None,
        max_bytes_read: int | None = None,
    ) -> dict:
        """Return grouped matches and bounded contents or excerpts of matching files."""
        if max_files < 0:
            raise ValueError("max_files must be non-negative")
        if context_lines < 0:
            raise ValueError("context_lines must be non-negative")
        budget = new_budget()
        results = search(pattern, path, max_matches, budget)
        if not results:
            summary = _SCAN_TRUNCATION if budget.truncated else "No matches found."
            return {"matches": {}, "files": {}, "summary": summary}

        limit = byte_limit(max_bytes_read)
        grouped: dict[str, list[dict]] = {}
        for result in results:
            grouped.setdefault(result["file"], []).append(result)

        file_contents = {}
        remaining = MAX_BATCH_BYTES
        for file_path in list(grouped)[:max_files]:
            if not budget.has_time() or remaining <= len(_TRUNCATION):
                break
            try:
                content = read(file_path, min(limit, remaining - len(_TRUNCATION)), budget)
                if context_lines > 0:
                    truncated_content = content.endswith(_TRUNCATION)
                    if truncated_content:
                        content = content[:-len(_TRUNCATION)]
                    excerpt_limit = remaining - len(_TRUNCATION)
                    excerpts = []
                    windows = iter(grouped[file_path])
                    match = next(windows, None)
                    excerpt_bytes = 0
                    for number, line in enumerate(_search_lines(content, True), 1):
                        if not budget.has_time():
                            break
                        while match is not None and number > match["line_number"] + context_lines:
                            match = next(windows, None)
                        if match is None:
                            break
                        if number >= match["line_number"] - context_lines:
                            excerpt = f"L{number}: {line}"
                            excerpt_bytes += len(excerpt.encode("utf-8")) + 1
                            if excerpt_bytes > excerpt_limit:
                                truncated_content = True
                                break
                            excerpts.append(excerpt)
                    content = "\n".join(excerpts)
                    if truncated_content:
                        content += _TRUNCATION
            except OSError as exc:
                content = _text_limit(f"[error: {exc}]", remaining)
            file_contents[file_path] = content
            remaining -= len(content.encode("utf-8"))

        truncated = len(grouped) - len(file_contents)
        summary = f"{len(results)} matches in {len(grouped)} files"
        if truncated:
            summary += f" (showing {len(file_contents)}, {truncated} more)"
        if budget.truncated:
            summary += f" {_SCAN_TRUNCATION}"

        return {
            "matches": {fp: grouped[fp] for fp in file_contents},
            "files": file_contents,
            "summary": summary,
        }

    def glob_files(pattern: str) -> list[str]:
        if not pattern:
            raise ValueError("Unacceptable pattern: ''")
        if len(pattern) > MAX_PATTERN_LENGTH:
            raise ValueError("Glob pattern exceeds the length limit")
        pattern_path = pathlib.Path(pattern)
        parts = pattern_path.parts
        if pattern_path.is_absolute():
            raise PermissionError("Access denied: absolute glob patterns are not allowed")
        if ".." in parts:
            raise PermissionError("Access denied: glob pattern escapes sandbox root")
        if any("**" in part and part != "**" for part in parts):
            raise ValueError("'**' can only be an entire path component")
        if len(parts) > MAX_PATH_DEPTH:
            raise ValueError("Glob pattern exceeds the workspace depth limit")
        if not parts or pattern.endswith("/") or parts[-1] == "**":
            return []

        selectors = [
            None if part == "**" else re.compile(translate(part))
            for part in parts
        ]

        def positions(path):
            states = {0}
            for name in path.parts:
                advanced = set()
                for index in states:
                    while index < len(parts) and selectors[index] is None:
                        advanced.add(index)
                        index += 1
                    if index < len(parts) and selectors[index].match(name):
                        advanced.add(index + 1)
                states = advanced
                if not states:
                    break
            return states

        def select_directory(path):
            return any(index < len(parts) for index in positions(path))

        matches = []
        budget = new_budget()
        remaining = MAX_RESULT_BYTES
        root = pathlib.Path(".")
        with workspace.open(root, directory=True) as (fd, _):
            for _, path in workspace.files(
                fd, root, budget, include_hidden=True, select_directory=select_directory,
            ):
                if len(parts) not in positions(path):
                    continue
                name = str(path)
                remaining -= len(os.fsencode(name)) + 8
                if remaining < 0 or len(matches) >= MAX_RESULTS:
                    budget.truncated = True
                    break
                matches.append(name)
        if budget.truncated:
            raise RuntimeError(_SCAN_TRUNCATION)
        return sorted(matches)

    def tree(path: str = ".", max_depth: int = 3) -> str:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        target = workspace.relative(path)
        budget = new_budget()
        lines = [str(target)]
        remaining = MAX_RESULT_BYTES
        depth_limit = min(max_depth, MAX_PATH_DEPTH - len(target.parts) - 1)

        def walk(fd, prefix, depth):
            nonlocal remaining
            entries = []
            with os.scandir(fd) as source:
                for entry in source:
                    if not budget.visit():
                        break
                    if (
                        entry.name.startswith(".")
                        or entry.name in _SKIP_DIRS
                        or is_private_component(entry.name)
                    ):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            entries.append((False, entry.name))
                        elif entry.is_file(follow_symlinks=False):
                            if entry.stat(follow_symlinks=False).st_nlink == 1:
                                entries.append((True, entry.name))
                    except OSError:
                        continue
            entries.sort()
            for index, (is_file, name) in enumerate(entries):
                if len(lines) >= MAX_RESULTS or not budget.has_time():
                    budget.truncated = True
                    break
                last = index == len(entries) - 1
                line = prefix + ("└── " if last else "├── ") + name
                remaining -= len(os.fsencode(line)) + 1
                if remaining < 0:
                    budget.truncated = True
                    break
                lines.append(line)
                if is_file or depth >= max_depth:
                    continue
                if depth >= depth_limit or budget.entries <= 0:
                    budget.truncated = True
                else:
                    try:
                        with workspace.open_entry(fd, name, directory=True) as (child_fd, _):
                            walk(child_fd, prefix + ("    " if last else "│   "), depth + 1)
                    except OSError:
                        continue

        with workspace.open(target, directory=True) as (fd, _):
            if depth_limit >= 0:
                walk(fd, "", 0)
            else:
                budget.truncated = True
        if budget.truncated:
            lines.append(_SCAN_TRUNCATION)
        return "\n".join(lines)

    return {
        "read_file": read_file,
        "read_files": read_files,
        "grep": grep,
        "grep_summary": grep_summary,
        "grep_read": grep_read,
        "glob_files": glob_files,
        "tree": tree,
    }, workspace
