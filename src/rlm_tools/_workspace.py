"""Descriptor-relative access to a workspace and bounded traversal."""
import collections
import errno
import os
import pathlib
import re
import stat
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from weakref import finalize

import re2

from rlm_tools.access_policy import is_private_component


MAX_PATH_DEPTH = 64
MAX_PATH_BYTES = 4096
MAX_SCAN_ENTRIES = 100_000
MAX_SCAN_BYTES = 256 * 1024 * 1024
MAX_WORK_SECONDS = 15.0

_MAX_GITIGNORE_BYTES = 256 * 1024
_MAX_GITIGNORE_RULES = 2_000
_MAX_GITIGNORE_LINE_LENGTH = 4_096
_GITIGNORE_REGEX_MEMORY_BYTES = 1024 * 1024
_MAX_DIR_CACHE_ENTRIES = 512

_SKIP_DIRS = {
    ".git", ".build", ".swiftpm", "DerivedData", "Build", "Pods",
    "node_modules", ".venv", "venv", "__pycache__", ".tox", ".mypy_cache",
    "Carthage", ".cache", "xcuserdata",
    "htmlcov", "coverage", "dist", "build", "target",
    ".pytest_cache", ".ruff_cache", ".next", ".nuxt", ".svelte-kit",
    "__pypackages__", ".eggs",
}


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def scan_limits() -> dict[str, int | float]:
    """Return active workspace scan limits (env overrides over module defaults)."""
    return {
        "max_scan_bytes": _positive_int_env("RLM_MAX_SCAN_BYTES", MAX_SCAN_BYTES),
        "max_scan_entries": _positive_int_env("RLM_MAX_SCAN_ENTRIES", MAX_SCAN_ENTRIES),
        "max_scan_seconds": _positive_float_env("RLM_MAX_SCAN_SECONDS", MAX_WORK_SECONDS),
    }


@dataclass(frozen=True)
class _GitIgnoreRule:
    regex: Any
    negated: bool
    dir_only: bool
    anchored: bool

    def matches(self, rel_path: str, name: str, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False
        target = rel_path if self.anchored else name
        return self.regex.match(target) is not None


def _strip_unescaped_trailing_spaces(line: str) -> str:
    end = len(line)
    while end > 0 and line[end - 1] == " ":
        backslashes = 0
        idx = end - 2
        while idx >= 0 and line[idx] == "\\":
            backslashes += 1
            idx -= 1
        if backslashes % 2 == 1:
            break
        end -= 1
    return line[:end]


def _has_unescaped_slash(pattern: str) -> bool:
    idx = 0
    while idx < len(pattern):
        ch = pattern[idx]
        if ch == "\\":
            idx += 2
            continue
        if ch == "/":
            return True
        idx += 1
    return False


def _escape_char_class_atom(ch: str) -> str:
    if ch in {"\\", "]", "[", "-", "^"}:
        return "\\" + ch
    return ch


def _translate_gitignore_char_class(body: str) -> str:
    """Translate a gitignore [...] body to an RE2 character class that never matches '/'."""
    if body.startswith(("!", "^")):
        inner = body[1:]
        if inner.startswith(("]", "-")):
            inner = "\\" + inner
        return f"[^/{inner}]"

    tokens: list[tuple[str, bool]] = []
    idx = 0
    while idx < len(body):
        if body[idx] == "\\" and idx + 1 < len(body):
            tokens.append((body[idx + 1], True))
            idx += 2
        else:
            tokens.append((body[idx], False))
            idx += 1

    parts: list[str] = []
    i = 0
    slash_ord = ord("/")
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 1] == ("-", False):
            lo = tokens[i][0]
            hi = tokens[i + 2][0]
            i += 3
            lo_ord = ord(lo)
            hi_ord = ord(hi)
            if lo_ord > hi_ord:
                raise re2.error("invalid character class range")
            if lo_ord <= slash_ord <= hi_ord:
                if lo_ord < slash_ord:
                    parts.append(f"{_escape_char_class_atom(lo)}-{_escape_char_class_atom('.')}")
                if hi_ord > slash_ord:
                    parts.append(f"{_escape_char_class_atom('0')}-{_escape_char_class_atom(hi)}")
            else:
                parts.append(f"{_escape_char_class_atom(lo)}-{_escape_char_class_atom(hi)}")
        else:
            ch = tokens[i][0]
            i += 1
            if ch != "/":
                parts.append(_escape_char_class_atom(ch))

    if not parts:
        # An empty positive class (e.g. [/]) can never match any path character.
        return "(?:a^)"
    return "[" + "".join(parts) + "]"


def _compile_gitignore_glob(pattern: str) -> Any:
    out: list[str] = ["^"]
    idx = 0
    length = len(pattern)
    while idx < length:
        ch = pattern[idx]
        if ch == "\\":
            if idx + 1 < length:
                out.append(re.escape(pattern[idx + 1]))
                idx += 2
            else:
                out.append(re.escape(ch))
                idx += 1
            continue
        if ch == "*" and idx + 1 < length and pattern[idx + 1] == "*":
            prev_is_sep = idx == 0 or pattern[idx - 1] == "/"
            next_idx = idx + 2
            if next_idx == length and prev_is_sep:
                out.append(".*")
                idx = next_idx
                continue
            if next_idx < length and pattern[next_idx] == "/" and prev_is_sep:
                out.append("(?:.*/)?")
                idx = next_idx + 1
                continue
            out.append("[^/]*")
            idx = next_idx
            continue
        if ch == "*":
            out.append("[^/]*")
            idx += 1
            continue
        if ch == "?":
            out.append("[^/]")
            idx += 1
            continue
        if ch == "[":
            end = idx + 1
            if end < length and pattern[end] in {"!", "^"}:
                end += 1
            if end < length and pattern[end] == "]":
                end += 1
            while end < length and pattern[end] != "]":
                if pattern[end] == "\\" and end + 1 < length:
                    end += 2
                else:
                    end += 1
            if end < length and pattern[end] == "]":
                body = pattern[idx + 1:end]
                out.append(_translate_gitignore_char_class(body))
                idx = end + 1
                continue
            out.append(re.escape(ch))
            idx += 1
            continue
        out.append(re.escape(ch))
        idx += 1
    out.append("$")
    options = re2.Options()
    options.max_mem = _GITIGNORE_REGEX_MEMORY_BYTES
    options.log_errors = False
    return re2.compile("".join(out), options)


def _parse_gitignore_text(text: str) -> tuple[_GitIgnoreRule, ...]:
    rules: list[_GitIgnoreRule] = []
    for raw_line in text.splitlines():
        if len(rules) >= _MAX_GITIGNORE_RULES:
            break
        if len(raw_line) > _MAX_GITIGNORE_LINE_LENGTH:
            continue
        line = raw_line.rstrip("\r")
        if not line or line.startswith("#"):
            continue
        line = _strip_unescaped_trailing_spaces(line)
        if not line:
            continue
        negated = False
        if line.startswith("!"):
            negated = True
            line = line[1:]
            if not line:
                continue
        dir_only = False
        while line.endswith("/") and not line.endswith("\\/"):
            dir_only = True
            line = line[:-1]
        if not line:
            continue
        anchored = line.startswith("/")
        if anchored:
            line = line.lstrip("/")
        if not line:
            continue
        if not anchored:
            anchored = _has_unescaped_slash(line)
        try:
            compiled = _compile_gitignore_glob(line)
        except re2.error:
            continue
        rules.append(_GitIgnoreRule(
            regex=compiled,
            negated=negated,
            dir_only=dir_only,
            anchored=anchored,
        ))
    return tuple(rules)


def _is_gitignored(
    entry_parts: tuple[str, ...],
    name: str,
    is_dir: bool,
    layers: tuple[tuple[tuple[str, ...], tuple[_GitIgnoreRule, ...]], ...],
) -> bool:
    ignored = False
    for base_parts, rules in layers:
        rel_str = "/".join(entry_parts[len(base_parts):])
        for rule in rules:
            if rule.matches(rel_str, name, is_dir):
                ignored = not rule.negated
    return ignored


@dataclass
class ScanBudget:
    deadline: float | None = None
    entries: int | None = None
    remaining_bytes: int | None = None
    truncated: bool = False
    limit_hit: str | None = None
    files_scanned: int = 0
    files_skipped: int = 0
    oversized_files: list[str] = field(default_factory=list)
    oversized_count: int = 0
    max_bytes: int = field(init=False)
    max_entries: int = field(init=False)
    max_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        limits = scan_limits()
        self.max_bytes = int(limits["max_scan_bytes"])
        self.max_entries = int(limits["max_scan_entries"])
        self.max_seconds = float(limits["max_scan_seconds"])
        if self.entries is None:
            self.entries = self.max_entries
        if self.remaining_bytes is None:
            self.remaining_bytes = self.max_bytes
        if self.deadline is None:
            self.deadline = time.monotonic() + self.max_seconds

    def mark_truncated(self, reason: str) -> None:
        self.truncated = True
        if self.limit_hit is None:
            self.limit_hit = reason

    def record_oversized(self, path: str, size_bytes: int) -> None:
        self.oversized_count += 1
        if len(self.oversized_files) < 5:
            self.oversized_files.append(f"{path} ({size_bytes / (1024 * 1024):.2f} MiB)")

    def has_time(self) -> bool:
        if time.monotonic() >= self.deadline:
            self.mark_truncated("seconds")
            return False
        return True

    def visit(self) -> bool:
        if not self.has_time():
            return False
        if self.entries <= 0:
            self.mark_truncated("entries")
            return False
        self.entries -= 1
        return True

    def format_truncation(self) -> str:
        reason = self.limit_hit or "limit"
        if reason == "bytes":
            mib = self.max_bytes / (1024 * 1024)
            cap = f"{mib:.0f} MiB" if mib >= 1 else f"{self.max_bytes} B"
            detail = f"{cap} byte scan limit"
        elif reason == "entries":
            detail = f"{self.max_entries:,} entry scan limit"
        elif reason == "seconds":
            detail = f"{self.max_seconds:g}s scan time limit"
        elif reason == "max_matches":
            detail = "max_matches result limit"
        elif reason == "result_bytes":
            detail = "result size limit"
        elif reason == "file_bytes":
            if self.oversized_files:
                names = ", ".join(self.oversized_files)
                extra = (
                    f" (+{self.oversized_count - len(self.oversized_files)} more)"
                    if self.oversized_count > len(self.oversized_files)
                    else ""
                )
                detail = f"file read size limit: {self.oversized_count} file(s) exceeded 4 MiB — {names}{extra}"
            else:
                detail = "file read size limit"
        elif reason == "path_depth":
            detail = f"{MAX_PATH_DEPTH}-level path depth limit"
        else:
            detail = f"{reason} workspace limit"
        if self.files_scanned or self.files_skipped:
            progress = f"after {self.files_scanned} files scanned, {self.files_skipped} skipped"
        else:
            visited = max(0, self.max_entries - (self.entries or 0))
            progress = f"after {visited:,} entries visited"
        return (
            f"TRUNCATED ({detail}; scan truncated {progress}"
            " — pass allow_partial=True or narrow path=, max_depth=, or pattern to complete)"
        )


class Workspace:
    def __init__(self, base_path: str, *, canonical: bool = False):
        self.base = pathlib.Path(base_path)
        self._supplied_base = self.base if canonical else pathlib.Path(os.path.abspath(base_path))
        if not canonical:
            self.base = self.base.resolve()
        if (
            not self.base.is_absolute()
            or ".." in self.base.parts
            or self.base.anchor != os.sep
        ):
            raise ValueError("Workspace root must be a canonical absolute path")
        if any(is_private_component(part) for part in self.base.parts[1:]):
            raise PermissionError("Private configuration directories cannot be explored")
        if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
            raise RuntimeError("Descriptor-relative workspace access is unavailable on this platform")

        self._directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        self._file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        root_fd = os.open(os.sep, self._directory_flags)
        try:
            for part in self.base.parts[1:]:
                fd = os.open(part, self._directory_flags, dir_fd=root_fd)
                os.close(root_fd)
                root_fd = fd
        except BaseException:
            os.close(root_fd)
            raise
        self._root_fd = root_fd
        self._close_lock = threading.Lock()
        self._finalize = finalize(self, os.close, root_fd)
        self._dir_cache: collections.OrderedDict[
            tuple[tuple[str, ...], bool, bool],
            tuple[
                tuple[int, int, int, int],
                tuple[int, int, int, int, int] | None,
                list[tuple[bool, str]],
                tuple[_GitIgnoreRule, ...],
            ],
        ] = collections.OrderedDict()
        self._dir_cache_lock = threading.Lock()
        self._evict_file_cache = None

    def clear_dir_cache(self) -> None:
        with self._dir_cache_lock:
            self._dir_cache.clear()

    def close(self) -> None:
        with self._close_lock:
            self._finalize()

    def relative(self, path: str) -> pathlib.Path:
        if len(path) > MAX_PATH_BYTES or len(os.fsencode(path)) > MAX_PATH_BYTES:
            raise ValueError("Path exceeds the workspace path limit")
        supplied = pathlib.Path(path)
        if supplied.is_absolute():
            if supplied.is_relative_to(self.base):
                supplied = supplied.relative_to(self.base)
            elif supplied.is_relative_to(self._supplied_base):
                supplied = supplied.relative_to(self._supplied_base)
            else:
                raise PermissionError(f"Access denied: path '{path}' escapes sandbox root")
        parts: list[str] = []
        for part in supplied.parts:
            if is_private_component(part):
                raise PermissionError(f"Access denied: private workspace path {part!r}")
            if part == "..":
                if not parts:
                    raise PermissionError(f"Access denied: path '{path}' escapes sandbox root")
                parts.pop()
            else:
                parts.append(part)
        if len(parts) > MAX_PATH_DEPTH:
            raise ValueError("Path exceeds the workspace depth limit")
        return pathlib.Path(*parts)

    def __call__(self, path: str) -> pathlib.Path:
        return self.base / self.relative(path)

    @contextmanager
    def open(self, relative: pathlib.Path, *, directory: bool = False):
        with self._close_lock:
            if not self._finalize.alive:
                raise OSError("Workspace is closed")
            # Open a new directory description to keep concurrent scans'
            # directory stream positions independent.
            fd = os.open(".", self._directory_flags, dir_fd=self._root_fd)
        try:
            for part in relative.parts[:-1]:
                next_fd = os.open(part, self._directory_flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            if relative.parts:
                with self.open_entry(fd, relative.name, directory=directory) as opened:
                    yield opened
            else:
                yield fd, os.fstat(fd)
        finally:
            os.close(fd)

    @contextmanager
    def open_entry(self, parent_fd: int, name: str, *, directory: bool = False):
        # Check the entry before opening to avoid devices and special files in
        # normal use. Check the opened descriptor again before any read.
        mode = os.stat(name, dir_fd=parent_fd, follow_symlinks=False).st_mode
        if stat.S_ISLNK(mode):
            raise PermissionError(f"Symbolic links are blocked: {name}")
        if directory and not stat.S_ISDIR(mode):
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), name)
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise OSError(f"Expected a regular file: {name}")
        flags = self._directory_flags if stat.S_ISDIR(mode) else self._file_flags
        fd = os.open(name, flags, dir_fd=parent_fd)
        try:
            info = os.fstat(fd)
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise OSError(f"Expected a regular file: {name}")
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise PermissionError(f"Files with multiple hard links are blocked: {name}")
            yield fd, info
        finally:
            os.close(fd)

    def _read_gitignore_rules(self, dir_fd: int) -> tuple[_GitIgnoreRule, ...]:
        try:
            with self.open_entry(dir_fd, ".gitignore") as (gi_fd, info):
                if not stat.S_ISREG(info.st_mode):
                    return ()
                chunks = bytearray()
                while len(chunks) < _MAX_GITIGNORE_BYTES:
                    chunk = os.read(gi_fd, min(64 * 1024, _MAX_GITIGNORE_BYTES - len(chunks)))
                    if not chunk:
                        break
                    chunks.extend(chunk)
        except OSError:
            return ()
        return _parse_gitignore_text(chunks.decode("utf-8", errors="replace"))

    def _dir_gitignore_sig(self, dir_fd: int) -> tuple[int, int, int, int, int] | None:
        try:
            st = os.stat(".gitignore", dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode):
            return None
        return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)

    def load_ancestor_gitignore_layers(
        self, relative: pathlib.Path,
    ) -> tuple[tuple[tuple[str, ...], tuple[_GitIgnoreRule, ...]], ...] | None:
        """Load .gitignore rules from workspace root down to the parent of relative.

        Returns None if any directory prefix along relative (including relative itself)
        is excluded by an ancestor .gitignore rule.
        """
        parts = relative.parts
        if not parts:
            return ()
        with self._close_lock:
            if not self._finalize.alive:
                raise OSError("Workspace is closed")
            fd = os.open(".", self._directory_flags, dir_fd=self._root_fd)
        layers: list[tuple[tuple[str, ...], tuple[_GitIgnoreRule, ...]]] = []
        try:
            prefix: tuple[str, ...] = ()
            for idx, part in enumerate(parts):
                rules = self._read_gitignore_rules(fd)
                if rules:
                    layers.append((prefix, rules))
                if layers and _is_gitignored(parts[: idx + 1], part, True, tuple(layers)):
                    return None
                if idx == len(parts) - 1:
                    break
                mode = os.stat(part, dir_fd=fd, follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                    break
                next_fd = os.open(part, self._directory_flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
                prefix = prefix + (part,)
        except OSError:
            pass
        finally:
            os.close(fd)
        return tuple(layers)

    def scan_dir_entries(
        self,
        fd: int,
        relative: pathlib.Path,
        budget: ScanBudget,
        *,
        include_hidden: bool = False,
        include_hidden_dirs: bool = False,
        respect_gitignore: bool = True,
        ancestor_layers: tuple[tuple[tuple[str, ...], tuple[_GitIgnoreRule, ...]], ...] = (),
    ) -> tuple[list[tuple[bool, str]], tuple[tuple[tuple[str, ...], tuple[_GitIgnoreRule, ...]], ...]]:
        """Return filtered sorted-ready (is_dir, name) entries and updated gitignore layers."""
        try:
            dir_st = os.fstat(fd)
            dir_sig = (dir_st.st_dev, dir_st.st_ino, dir_st.st_mtime_ns, dir_st.st_ctime_ns)
        except OSError:
            return [], ancestor_layers
        gi_sig = self._dir_gitignore_sig(fd) if respect_gitignore else None
        cache_key = (relative.parts, include_hidden, include_hidden_dirs)

        cached_raw: list[tuple[bool, str]] | None = None
        local_rules: tuple[_GitIgnoreRule, ...] = ()
        from_cache = False
        with self._dir_cache_lock:
            cached = self._dir_cache.get(cache_key)
            if cached is not None and cached[0] == dir_sig and (not respect_gitignore or cached[1] == gi_sig):
                self._dir_cache.move_to_end(cache_key)
                cached_raw = cached[2]
                local_rules = cached[3]
                from_cache = True

        if cached_raw is None:
            raw: list[tuple[bool, str]] = []
            scan_complete = True
            try:
                with os.scandir(fd) as entries:
                    for entry in entries:
                        if not budget.visit():
                            scan_complete = False
                            break
                        if is_private_component(entry.name) or (entry.name.startswith(".") and not include_hidden):
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name in _SKIP_DIRS or (entry.name.startswith(".") and not include_hidden_dirs):
                                    continue
                                raw.append((True, entry.name))
                            elif entry.is_file(follow_symlinks=False):
                                if entry.stat(follow_symlinks=False).st_nlink == 1:
                                    raw.append((False, entry.name))
                        except OSError:
                            continue
            except OSError:
                return [], ancestor_layers
            if respect_gitignore and gi_sig is not None:
                local_rules = self._read_gitignore_rules(fd)
            if scan_complete:
                with self._dir_cache_lock:
                    self._dir_cache[cache_key] = (dir_sig, gi_sig, raw, local_rules)
                    self._dir_cache.move_to_end(cache_key)
                    while len(self._dir_cache) > _MAX_DIR_CACHE_ENTRIES:
                        self._dir_cache.popitem(last=False)
            cached_raw = raw

        layers = ancestor_layers + ((relative.parts, local_rules),) if (respect_gitignore and local_rules) else ancestor_layers

        filtered: list[tuple[bool, str]] = []
        for is_dir, name in cached_raw:
            if from_cache and not budget.visit():
                break
            entry_parts = relative.parts + (name,)
            if len(os.fsencode(os.path.join(*entry_parts))) > MAX_PATH_BYTES:
                budget.mark_truncated("path_depth")
                continue
            if respect_gitignore and layers and _is_gitignored(entry_parts, name, is_dir, layers):
                if not is_dir:
                    budget.files_skipped += 1
                continue
            if from_cache and not is_dir:
                try:
                    st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                        continue
                except OSError:
                    continue
            filtered.append((is_dir, name))
        return filtered, layers

    def files(
        self,
        fd,
        relative,
        budget,
        *,
        include_hidden=False,
        include_hidden_dirs: bool = False,
        select_directory=None,
        respect_gitignore: bool = True,
        _ancestor_layers=None,
    ):
        """Yield each regular file while its parent directory is open."""
        if len(relative.parts) >= MAX_PATH_DEPTH:
            budget.mark_truncated("path_depth")
            return
        if _ancestor_layers is None:
            if respect_gitignore:
                _ancestor_layers = self.load_ancestor_gitignore_layers(relative)
                if _ancestor_layers is None:
                    return
            else:
                _ancestor_layers = ()
        entries, layers = self.scan_dir_entries(
            fd,
            relative,
            budget,
            include_hidden=include_hidden,
            include_hidden_dirs=include_hidden_dirs,
            respect_gitignore=respect_gitignore,
            ancestor_layers=_ancestor_layers,
        )
        for is_dir, name in entries:
            if not budget.has_time():
                break
            path = relative / name
            if is_dir:
                if select_directory is not None and not select_directory(path):
                    continue
                try:
                    with self.open_entry(fd, name, directory=True) as (child_fd, _):
                        yield from self.files(
                            child_fd,
                            path,
                            budget,
                            include_hidden=include_hidden,
                            include_hidden_dirs=include_hidden_dirs,
                            select_directory=select_directory,
                            respect_gitignore=respect_gitignore,
                            _ancestor_layers=layers,
                        )
                except OSError:
                    continue
            else:
                yield fd, path
