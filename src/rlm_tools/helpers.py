import ast
import codecs
import collections
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
)


MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_BATCH_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_RESULTS = 2_000
MAX_CACHE_FILES = 256
MAX_PATTERN_LENGTH = 4_096
REGEX_MEMORY_BYTES = 1024 * 1024
MAX_GREP_LINE_BYTES = 500
MAX_SUMMARY_LINE_CHARS = 300
MAX_AST_BYTES = 1 * 1024 * 1024
MAX_AST_CACHE_FILES = 8
_SNIFF_BYTES = 8 * 1024
_MINIFIED_SNIFF_BYTES = 64 * 1024
_MINIFIED_SIZE_CAP = 1024 * 1024
_TRUNCATION = "\n... [read truncated]"
_LINE_SEPARATOR = re.compile(r"[\n\v\f\x1c-\x1e\x85\u2028\u2029]")
_PATTERN_ESCAPE = re.compile(r"\\Q.*?(?:\\E|$)|\\.", re.DOTALL)
_PUSH_SCOPE = object()
_POP_SCOPE = object()

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
    initial_remaining = budget.remaining_bytes
    allowed = min(limit, initial_remaining)
    probe_limit = allowed + 1
    data = bytearray()
    while len(data) < probe_limit and budget.has_time():
        chunk = os.read(fd, min(64 * 1024, probe_limit - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    budget.remaining_bytes = max(0, initial_remaining - min(len(data), allowed))
    complete = (
        len(data) <= allowed
        and len(data) >= info.st_size
        and budget.has_time()
    )
    if not complete and not budget.truncated:
        if len(data) > initial_remaining or budget.remaining_bytes <= 0:
            budget.mark_truncated("bytes")
        elif budget.has_time():
            budget.mark_truncated("file_bytes")
    # An incomplete multi-byte character at the byte boundary is omitted.
    return _decode(bytes(data[:allowed]), complete), complete


def _text_limit(text: str, limit: int) -> str:
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _clip_matched_line(raw_line: str, match, byte_limit: int) -> str:
    stripped = raw_line.strip()
    stripped_bytes = stripped.encode("utf-8")
    total_b = len(stripped_bytes)
    if total_b <= byte_limit:
        return stripped
    if byte_limit <= 3:
        return _text_limit(stripped, byte_limit)
    leading = len(raw_line) - len(raw_line.lstrip())
    m_start_c = max(0, min(len(stripped), match.start() - leading))
    m_end_c = max(m_start_c, min(len(stripped), match.end() - leading))
    m_start_b = len(stripped[:m_start_c].encode("utf-8"))
    m_end_b = m_start_b + len(stripped[m_start_c:m_end_c].encode("utf-8"))
    center_b = (m_start_b + m_end_b) // 2

    window_b = max(1, byte_limit - 6)
    win_start_b = max(0, center_b - window_b // 2)
    win_end_b = min(total_b, win_start_b + window_b)
    if win_end_b == total_b:
        win_start_b = max(0, win_end_b - window_b)

    # Reclaim unused ellipsis bytes when anchored at start or end of line.
    if win_start_b == 0:
        win_end_b = min(total_b, byte_limit - 3)
    elif win_end_b == total_b:
        win_start_b = max(0, total_b - (byte_limit - 3))

    # Snap byte boundaries to valid UTF-8 codepoint boundaries without expanding.
    while win_start_b < win_end_b and (stripped_bytes[win_start_b] & 0xC0) == 0x80:
        win_start_b += 1
    while win_end_b > win_start_b and win_end_b < total_b and (stripped_bytes[win_end_b] & 0xC0) == 0x80:
        win_end_b -= 1

    excerpt = stripped_bytes[win_start_b:win_end_b].decode("utf-8", errors="ignore")
    if win_start_b > 0:
        excerpt = "..." + excerpt
    if win_end_b < total_b:
        excerpt = excerpt + "..."
    if len(excerpt.encode("utf-8")) > byte_limit:
        excerpt = _text_limit(excerpt, byte_limit - 3) + "..."
    return excerpt


def _search_lines(content: str, complete: bool):
    start = 0
    for separator in _LINE_SEPARATOR.finditer(content):
        yield content[start:separator.start()]
        start = separator.end()
    if complete and start < len(content):
        yield content[start:]


def _split_source_lines(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    curr: ast.AST = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    if isinstance(curr, ast.Name):
        parts.append(curr.id)
        parts.reverse()
        return ".".join(parts)
    return None


def _extract_target_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    stack: list[ast.AST] = [node]
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.Name, ast.Attribute)):
            name = _dotted_name(curr)
            if name:
                names.append(name)
        elif isinstance(curr, (ast.Tuple, ast.List)):
            stack.extend(reversed(curr.elts))
        elif isinstance(curr, (ast.Starred, ast.Subscript)):
            stack.append(curr.value)
    return names


def _target_matches(lhs: str, target: str) -> bool:
    if lhs == target:
        return True
    return "." not in target and lhs.endswith(f".{target}")


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
                            budget.mark_truncated("bytes")
                            continue
                        content, complete = _read_text(fd, info, MAX_FILE_BYTES, budget)
                        if not complete:
                            budget.mark_truncated("bytes" if budget.remaining_bytes <= 0 else "file_bytes")
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
    file_cache: collections.OrderedDict[pathlib.Path, tuple[tuple[int, ...], str, int]] = collections.OrderedDict()
    cache_bytes = 0
    cache_lock = threading.Lock()
    ast_cache: collections.OrderedDict[tuple[pathlib.Path, tuple[int, ...]], tuple[ast.Module, list[str]]] = (
        collections.OrderedDict()
    )

    def evict_cache(bytes_to_free: int | None = None) -> int:
        nonlocal cache_bytes
        freed = 0
        with cache_lock:
            ast_cache.clear()
            while file_cache and (bytes_to_free is None or freed < bytes_to_free):
                _, (_, _, entry_size) = file_cache.popitem(last=False)
                cache_bytes = max(0, cache_bytes - entry_size)
                freed += entry_size
                if memory_tracker is not None and entry_size > 0:
                    memory_tracker(-entry_size)
        workspace.clear_dir_cache()
        return freed

    workspace._evict_file_cache = evict_cache

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

    def _evict_oldest_locked(exclude: pathlib.Path | None = None) -> bool:
        nonlocal cache_bytes
        for key in list(file_cache):
            if key == exclude:
                continue
            _, _, entry_size = file_cache.pop(key)
            cache_bytes = max(0, cache_bytes - entry_size)
            if memory_tracker is not None and entry_size > 0:
                memory_tracker(-entry_size)
            return True
        return False

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
                        file_cache.move_to_end(target)
                        return content

            content, complete = _read_text(fd, info, limit, budget)
            size = len(content.encode("utf-8"))
            if not complete or size > limit:
                return _text_limit(content, limit) + _TRUNCATION
            with cache_lock:
                if size <= MAX_BATCH_BYTES:
                    cached = file_cache.get(target)
                    old_size = cached[2] if cached is not None else 0
                    delta = size - old_size
                    while (
                        (cached is None and len(file_cache) >= MAX_CACHE_FILES)
                        or cache_bytes + delta > MAX_BATCH_BYTES
                    ):
                        if not _evict_oldest_locked(exclude=target):
                            break
                    if (
                        (cached is not None or len(file_cache) < MAX_CACHE_FILES)
                        and cache_bytes + delta <= MAX_BATCH_BYTES
                    ):
                        if memory_tracker is not None:
                            while True:
                                try:
                                    memory_tracker(delta)
                                    break
                                except MemoryError:
                                    if not _evict_oldest_locked(exclude=target):
                                        raise
                        file_cache[target] = (version, content, size)
                        file_cache.move_to_end(target)
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

    def search(
        pattern: str,
        path: str,
        max_matches: int | None,
        budget: ScanBudget,
        *,
        line_bytes_limit: int = MAX_GREP_LINE_BYTES,
    ) -> list[dict]:
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

        def search_file(fd: int, info: os.stat_result, file_path: pathlib.Path) -> None:
            nonlocal remaining
            if not stat.S_ISREG(info.st_mode):
                return
            if file_path.suffix.lower() in _BINARY_EXTENSIONS:
                budget.files_skipped += 1
                return

            sniff_limit = _MINIFIED_SNIFF_BYTES if info.st_size > _MINIFIED_SIZE_CAP else _SNIFF_BYTES
            prefix_buf = bytearray()
            while len(prefix_buf) < sniff_limit and budget.has_time():
                chunk = os.read(fd, min(64 * 1024, sniff_limit - len(prefix_buf)))
                if not chunk:
                    break
                prefix_buf.extend(chunk)
            if not budget.has_time():
                return
            prefix = bytes(prefix_buf)
            if b"\x00" in prefix[:_SNIFF_BYTES]:
                budget.files_skipped += 1
                return
            if info.st_size > _MINIFIED_SIZE_CAP and b"\n" not in prefix and b"\r" not in prefix:
                budget.files_skipped += 1
                return

            if budget.remaining_bytes <= 0 and info.st_size > 0:
                budget.mark_truncated("bytes")
                return

            initial_remaining = budget.remaining_bytes
            allowed = min(MAX_FILE_BYTES, initial_remaining)
            probe_limit = allowed + 1
            if len(prefix) >= probe_limit:
                data = prefix[:probe_limit]
            else:
                rest_buf = bytearray()
                need = probe_limit - len(prefix)
                while len(rest_buf) < need and budget.has_time():
                    chunk = os.read(fd, min(64 * 1024, need - len(rest_buf)))
                    if not chunk:
                        break
                    rest_buf.extend(chunk)
                data = prefix + bytes(rest_buf)
            budget.remaining_bytes = max(0, initial_remaining - min(len(data), allowed))

            complete = (
                len(data) <= allowed
                and len(data) >= info.st_size
                and budget.has_time()
            )
            if not complete:
                if len(data) > initial_remaining or budget.remaining_bytes <= 0:
                    budget.mark_truncated("bytes")
                elif budget.has_time():
                    budget.mark_truncated("file_bytes")

            content = _decode(data[:allowed], complete)
            budget.files_scanned += 1
            file_name = str(file_path)
            file_size = len(os.fsencode(file_name))
            for number, line in enumerate(_search_lines(content, complete), 1):
                if not budget.has_time():
                    return
                match = compiled.search(line)
                if match:
                    if remaining <= file_size + 48:
                        remaining = 0
                        budget.mark_truncated("result_bytes")
                        return
                    line_limit = min(line_bytes_limit, remaining - file_size - 48)
                    if line_limit <= 0:
                        remaining = 0
                        budget.mark_truncated("result_bytes")
                        return
                    clipped = _clip_matched_line(line, match, line_limit)
                    remaining -= file_size + len(clipped.encode("utf-8")) + 48
                    results.append({"file": file_name, "line_number": number, "line": clipped})
                    if len(results) >= match_limit:
                        if automatic_match_limit:
                            budget.mark_truncated("max_matches")
                        return

        try:
            with workspace.open(target) as (fd, info):
                if stat.S_ISREG(info.st_mode):
                    search_file(fd, info, target)
                else:
                    for parent_fd, file_path in workspace.files(fd, target, budget):
                        if len(results) >= match_limit or remaining == 0:
                            break
                        if budget.remaining_bytes <= 0:
                            budget.mark_truncated("bytes")
                            break
                        if file_path.suffix.lower() in _BINARY_EXTENSIONS:
                            budget.files_skipped += 1
                            continue
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

    def grep(
        pattern: str,
        path: str = ".",
        max_matches: int | None = None,
        *,
        allow_partial: bool = False,
    ) -> list[dict]:
        budget = new_budget()
        results = search(pattern, path, max_matches, budget)
        if budget.truncated and not allow_partial:
            raise RuntimeError(budget.format_truncation())
        return results

    def grep_summary(pattern: str, path: str = ".", max_matches: int | None = None) -> str:
        """Grep with compact output grouped by file. Returns a formatted string."""
        budget = new_budget()
        results = search(
            pattern, path, max_matches, budget, line_bytes_limit=MAX_SUMMARY_LINE_CHARS,
        )
        if not results:
            return budget.format_truncation() if budget.truncated else "No matches found."

        grouped: dict[str, list[dict]] = {}
        for result in results:
            grouped.setdefault(result["file"], []).append(result)

        lines: list[str] = []
        if budget.truncated:
            lines.append(budget.format_truncation())
            lines.append(f"{len(results)} matches in {len(grouped)} files so far:")
        else:
            lines.append(f"{len(results)} matches in {len(grouped)} files:")
        for file, matches in grouped.items():
            lines.append(f"\n  {file} ({len(matches)} matches):")
            lines.extend(f"    L{match['line_number']}: {match['line']}" for match in matches)
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
            summary = budget.format_truncation() if budget.truncated else "No matches found."
            return {
                "matches": {},
                "files": {},
                "summary": summary,
                "truncated": budget.truncated,
                "limit_hit": budget.limit_hit,
                "files_scanned": budget.files_scanned,
                "files_skipped": budget.files_skipped,
            }

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

        truncated_files = len(grouped) - len(file_contents)
        summary = f"{len(results)} matches in {len(grouped)} files"
        if truncated_files:
            summary += f" (showing {len(file_contents)}, {truncated_files} more)"
        if budget.truncated:
            summary = f"{budget.format_truncation()}\n{summary}"

        return {
            "matches": {fp: grouped[fp] for fp in file_contents},
            "files": file_contents,
            "summary": summary,
            "truncated": budget.truncated,
            "limit_hit": budget.limit_hit,
            "files_scanned": budget.files_scanned,
            "files_skipped": budget.files_skipped,
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
                if remaining < 0:
                    budget.mark_truncated("result_bytes")
                    break
                if len(matches) >= MAX_RESULTS:
                    budget.mark_truncated("max_matches")
                    break
                matches.append(name)
        if budget.truncated:
            raise RuntimeError(budget.format_truncation())
        return sorted(matches)

    def tree(path: str = ".", max_depth: int = 3) -> str:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        target = workspace.relative(path)
        budget = new_budget()
        lines = [str(target)]
        remaining = MAX_RESULT_BYTES
        depth_limit = min(max_depth, MAX_PATH_DEPTH - len(target.parts) - 1)
        ancestor_layers = workspace.load_ancestor_gitignore_layers(target)

        def walk(fd: int, rel: pathlib.Path, prefix: str, depth: int, layers) -> None:
            nonlocal remaining
            raw_entries, next_layers = workspace.scan_dir_entries(
                fd, rel, budget, include_hidden=False, respect_gitignore=True, ancestor_layers=layers,
            )
            entries = sorted((not is_dir, name) for is_dir, name in raw_entries)
            for index, (is_file, name) in enumerate(entries):
                if len(lines) >= MAX_RESULTS:
                    budget.mark_truncated("max_matches")
                    break
                if not budget.has_time():
                    break
                last = index == len(entries) - 1
                line = prefix + ("└── " if last else "├── ") + name
                remaining -= len(os.fsencode(line)) + 1
                if remaining < 0:
                    budget.mark_truncated("result_bytes")
                    break
                lines.append(line)
                if is_file or depth >= max_depth:
                    continue
                if depth >= depth_limit:
                    budget.mark_truncated("path_depth")
                elif budget.entries <= 0:
                    budget.mark_truncated("entries")
                else:
                    try:
                        with workspace.open_entry(fd, name, directory=True) as (child_fd, _):
                            walk(
                                child_fd,
                                rel / name,
                                prefix + ("    " if last else "│   "),
                                depth + 1,
                                next_layers,
                            )
                    except OSError:
                        continue

        with workspace.open(target, directory=True) as (fd, _):
            if depth_limit < 0:
                budget.mark_truncated("path_depth")
            elif ancestor_layers is not None:
                walk(fd, target, "", 0, ancestor_layers)
        if budget.truncated:
            lines.insert(0, budget.format_truncation())
        return "\n".join(lines)

    def _parse_python_ast(path: str, budget: ScanBudget) -> tuple[ast.Module, list[str]]:
        if not budget.has_time():
            raise TimeoutError("Workspace read deadline exceeded")
        target = workspace.relative(path)
        with workspace.open(target) as (_, info):
            if not stat.S_ISREG(info.st_mode):
                raise IsADirectoryError(str(target))
            if info.st_size > MAX_AST_BYTES:
                raise ValueError(f"File '{path}' exceeds AST size limit ({MAX_AST_BYTES} bytes)")
            version = (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size)
        cache_key = (target, version)
        with cache_lock:
            cached = ast_cache.get(cache_key)
            if cached is not None:
                ast_cache.move_to_end(cache_key)
                return cached
        content = read(path, MAX_AST_BYTES, budget)
        if content.endswith(_TRUNCATION):
            raise ValueError(f"File '{path}' exceeds AST size limit ({MAX_AST_BYTES} bytes)")
        try:
            parsed = ast.parse(content, filename=str(target))
        except (SyntaxError, RecursionError, ValueError, MemoryError) as exc:
            raise ValueError(f"Could not parse Python AST for '{path}': {exc}") from exc
        lines = _split_source_lines(content)
        with cache_lock:
            ast_cache[cache_key] = (parsed, lines)
            ast_cache.move_to_end(cache_key)
            while len(ast_cache) > MAX_AST_CACHE_FILES:
                ast_cache.popitem(last=False)
        return parsed, lines

    def enclosing(path: str, line: int) -> dict:
        """Return the enclosing class/function chain for a 1-based line in a Python file."""
        if type(line) is not int or line < 1:
            raise ValueError("line must be a positive integer")
        budget = new_budget()
        parsed, lines = _parse_python_ast(path, budget)
        if line > max(1, len(lines)):
            raise ValueError(f"Line {line} is out of range (file has {len(lines)} lines)")

        chain: list[str] = []
        spans: list[dict] = []
        kind_map = {
            ast.ClassDef: "class",
            ast.FunctionDef: "function",
            ast.AsyncFunctionDef: "async_function",
        }

        stack: list[ast.AST] = list(reversed(list(ast.iter_child_nodes(parsed))))
        steps = 0
        while stack:
            steps += 1
            if (steps & 0x3FF) == 0 and not budget.has_time():
                raise TimeoutError("AST inspection deadline exceeded")
            child = stack.pop()
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                dec_lines = [getattr(d, "lineno", child.lineno) for d in child.decorator_list]
                start_line = min([child.lineno, *dec_lines])
                end_line = getattr(child, "end_lineno", child.lineno) or child.lineno
                if start_line <= line <= end_line:
                    chain.append(child.name)
                    spans.append({
                        "kind": kind_map[type(child)],
                        "name": child.name,
                        "start_line": start_line,
                        "end_line": end_line,
                    })
                    stack = list(reversed(list(ast.iter_child_nodes(child))))
                    continue
            else:
                stack.extend(reversed(list(ast.iter_child_nodes(child))))

        return {
            "line": line,
            "qualified": ".".join(chain) if chain else "<module>",
            "chain": chain,
            "spans": spans,
        }

    def find_assignments(path: str, target: str) -> list[dict]:
        """Find Assign, AnnAssign, AugAssign, and NamedExpr sites for a target name in a Python file."""
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty string")
        needle = target.strip()
        budget = new_budget()
        parsed, lines = _parse_python_ast(path, budget)
        hits: list[dict] = []
        seen: set[tuple[int, str, str]] = set()
        scope_stack: list[str] = []

        def record(line_no: int, kind: str, lhs: str) -> None:
            key = (line_no, kind, lhs)
            if key in seen:
                return
            seen.add(key)
            src = lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else ""
            if len(src) > MAX_SUMMARY_LINE_CHARS:
                src = src[:MAX_SUMMARY_LINE_CHARS - 3] + "..."
            hits.append({
                "line_number": line_no,
                "kind": kind,
                "target": lhs,
                "enclosing": ".".join(scope_stack) if scope_stack else "<module>",
                "line": src,
            })

        stack: list[object] = [parsed]
        steps = 0
        while stack:
            steps += 1
            if (steps & 0x3FF) == 0 and not budget.has_time():
                raise TimeoutError("AST inspection deadline exceeded")
            node = stack.pop()
            if node is _POP_SCOPE:
                scope_stack.pop()
                continue
            if isinstance(node, tuple) and node[0] is _PUSH_SCOPE:
                scope_stack.append(node[1])
                continue
            assert isinstance(node, ast.AST)

            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                stack.append(_POP_SCOPE)
                for stmt in reversed(node.body):
                    stack.append(stmt)
                stack.append((_PUSH_SCOPE, node.name))
                outer_nodes: list[ast.AST] = list(node.decorator_list)
                if isinstance(node, ast.ClassDef):
                    outer_nodes.extend(node.bases)
                    outer_nodes.extend(node.keywords)
                else:
                    outer_nodes.extend(node.args.defaults)
                    outer_nodes.extend(d for d in node.args.kw_defaults if d is not None)
                for outer in reversed(outer_nodes):
                    stack.append(outer)
                continue

            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    for lhs in _extract_target_names(tgt):
                        if _target_matches(lhs, needle):
                            record(node.lineno, "assign", lhs)
            elif isinstance(node, ast.AnnAssign):
                for lhs in _extract_target_names(node.target):
                    if _target_matches(lhs, needle):
                        record(node.lineno, "ann_assign", lhs)
            elif isinstance(node, ast.AugAssign):
                for lhs in _extract_target_names(node.target):
                    if _target_matches(lhs, needle):
                        record(node.lineno, "aug_assign", lhs)
            elif isinstance(node, ast.NamedExpr):
                for lhs in _extract_target_names(node.target):
                    if _target_matches(lhs, needle):
                        record(node.lineno, "named_expr", lhs)

            stack.extend(reversed(list(ast.iter_child_nodes(node))))

        hits.sort(key=lambda item: item["line_number"])
        return hits

    return {
        "read_file": read_file,
        "read_files": read_files,
        "grep": grep,
        "grep_summary": grep_summary,
        "grep_read": grep_read,
        "glob_files": glob_files,
        "tree": tree,
        "enclosing": enclosing,
        "find_assignments": find_assignments,
    }, workspace
