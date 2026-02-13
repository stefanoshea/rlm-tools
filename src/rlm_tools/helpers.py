import os
import pathlib
import re


_SKIP_DIRS = {
    ".git", ".build", ".swiftpm", "DerivedData", "Build", "Pods",
    "node_modules", ".venv", "venv", "__pycache__", ".tox", ".mypy_cache",
    "Carthage", ".cache", "xcuserdata",
}

_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz",
    ".xz", ".bz2", ".o", ".a", ".dylib", ".framework", ".xcassets",
    ".car", ".nib", ".storyboardc", ".momd", ".sqlite", ".db",
}


def _walk_files(root: pathlib.Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]
        for fname in filenames:
            if fname.startswith("."):
                continue
            yield pathlib.Path(dirpath) / fname


def make_helpers(base_path: str) -> tuple[dict, callable]:
    base = pathlib.Path(base_path).resolve()
    _file_cache: dict[str, str] = {}

    def _resolve_safe(path: str) -> pathlib.Path:
        resolved = (base / path).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise PermissionError(f"Access denied: path '{path}' escapes sandbox root")
        return resolved

    def read_file(path: str) -> str:
        target = _resolve_safe(path)
        cache_key = str(target)
        if cache_key in _file_cache:
            return _file_cache[cache_key]
        content = target.read_text(errors="replace")
        _file_cache[cache_key] = content
        return content

    def read_files(paths: list[str]) -> dict[str, str]:
        """Read multiple files at once. Returns {path: content} dict."""
        result = {}
        for path in paths:
            try:
                result[path] = read_file(path)
            except (OSError, PermissionError) as e:
                result[path] = f"[error: {e}]"
        return result

    def grep(pattern: str, path: str = ".") -> list[dict]:
        target = _resolve_safe(path)
        compiled = re.compile(pattern)
        results = []

        if target.is_file():
            search_paths = [target]
        else:
            search_paths = _walk_files(target)

        for file_path in search_paths:
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()
            if ext in _BINARY_EXTENSIONS:
                continue
            try:
                for i, line in enumerate(file_path.read_text(errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        results.append({
                            "file": str(file_path.relative_to(base)),
                            "line_number": i,
                            "line": line.strip(),
                        })
            except (OSError, UnicodeDecodeError):
                continue
        return results

    def grep_summary(pattern: str, path: str = ".") -> str:
        """Grep with compact output grouped by file. Returns a formatted string."""
        results = grep(pattern, path)
        if not results:
            return "No matches found."

        grouped: dict[str, list[dict]] = {}
        for r in results:
            grouped.setdefault(r["file"], []).append(r)

        lines = [f"{len(results)} matches in {len(grouped)} files:"]
        for file, matches in grouped.items():
            lines.append(f"\n  {file} ({len(matches)} matches):")
            for m in matches:
                lines.append(f"    L{m['line_number']}: {m['line']}")
        return "\n".join(lines)

    def grep_read(
        pattern: str,
        path: str = ".",
        max_files: int = 10,
        context_lines: int = 0,
    ) -> dict:
        """Grep then auto-read matching files. Returns match info + file contents.

        Args:
            pattern: Regex pattern to search for.
            path: Directory or file to search in.
            max_files: Maximum number of matching files to read (default 10).
            context_lines: Lines of context around each match (default 0).

        Returns:
            Dict with 'matches' (grouped by file) and 'files' (full contents).
        """
        results = grep(pattern, path)
        if not results:
            return {"matches": {}, "files": {}, "summary": "No matches found."}

        grouped: dict[str, list[dict]] = {}
        for r in results:
            grouped.setdefault(r["file"], []).append(r)

        file_paths = list(grouped.keys())[:max_files]
        file_contents = {}
        for fp in file_paths:
            try:
                content = read_file(fp)
                if context_lines > 0:
                    content_lines = content.splitlines()
                    relevant = set()
                    for m in grouped[fp]:
                        line_idx = m["line_number"] - 1
                        start = max(0, line_idx - context_lines)
                        end = min(len(content_lines), line_idx + context_lines + 1)
                        for i in range(start, end):
                            relevant.add(i)
                    excerpts = []
                    for i in sorted(relevant):
                        excerpts.append(f"L{i+1}: {content_lines[i]}")
                    file_contents[fp] = "\n".join(excerpts)
                else:
                    file_contents[fp] = content
            except (OSError, PermissionError) as e:
                file_contents[fp] = f"[error: {e}]"

        truncated = len(grouped) - len(file_paths) if len(grouped) > max_files else 0
        summary = f"{len(results)} matches in {len(grouped)} files"
        if truncated:
            summary += f" (showing {max_files}, {truncated} more)"

        return {
            "matches": {fp: grouped[fp] for fp in file_paths},
            "files": file_contents,
            "summary": summary,
        }

    def glob_files(pattern: str) -> list[str]:
        matches = list(base.glob(pattern))
        safe_matches: list[str] = []
        for match in matches:
            if not match.is_file():
                continue
            parts = match.relative_to(base).parts
            if any(part in _SKIP_DIRS or part.startswith(".") for part in parts[:-1]):
                continue
            try:
                safe_matches.append(str(match.resolve().relative_to(base)))
            except ValueError:
                continue
        return safe_matches

    def tree(path: str = ".", max_depth: int = 3) -> str:
        target = _resolve_safe(path)
        lines = []

        def _walk(dir_path: pathlib.Path, prefix: str, depth: int):
            if depth > max_depth:
                return
            try:
                entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name))
            except PermissionError:
                return
            visible = [e for e in entries if not e.name.startswith(".") and e.name not in _SKIP_DIRS]
            for i, entry in enumerate(visible):
                connector = "└── " if i == len(visible) - 1 else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir():
                    extension = "    " if i == len(visible) - 1 else "│   "
                    _walk(entry, prefix + extension, depth + 1)

        lines.append(str(target.relative_to(base)) if target != base else ".")
        _walk(target, "", 0)
        return "\n".join(lines)

    return {
        "read_file": read_file,
        "read_files": read_files,
        "grep": grep,
        "grep_summary": grep_summary,
        "grep_read": grep_read,
        "glob_files": glob_files,
        "tree": tree,
    }, _resolve_safe
