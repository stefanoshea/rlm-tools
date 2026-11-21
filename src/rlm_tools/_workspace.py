"""Descriptor-relative access to a workspace and bounded traversal."""
import errno
import os
import pathlib
import stat
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from weakref import finalize

from rlm_tools.access_policy import is_private_component


MAX_PATH_DEPTH = 64
MAX_PATH_BYTES = 4096
MAX_SCAN_ENTRIES = 10_000
MAX_SCAN_BYTES = 32 * 1024 * 1024
MAX_WORK_SECONDS = 3.0

_SKIP_DIRS = {
    ".git", ".build", ".swiftpm", "DerivedData", "Build", "Pods",
    "node_modules", ".venv", "venv", "__pycache__", ".tox", ".mypy_cache",
    "Carthage", ".cache", "xcuserdata",
}


@dataclass
class ScanBudget:
    deadline: float = field(default_factory=lambda: time.monotonic() + MAX_WORK_SECONDS)
    entries: int = MAX_SCAN_ENTRIES
    remaining_bytes: int = MAX_SCAN_BYTES
    truncated: bool = False

    def has_time(self) -> bool:
        if time.monotonic() >= self.deadline:
            self.truncated = True
            return False
        return True

    def visit(self) -> bool:
        if not self.has_time() or self.entries <= 0:
            self.truncated = True
            return False
        self.entries -= 1
        return True


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

    def files(self, fd, relative, budget, *, include_hidden=False, select_directory=None):
        """Yield each regular file while its parent directory is open."""
        if len(relative.parts) >= MAX_PATH_DEPTH:
            budget.truncated = True
            return
        try:
            with os.scandir(fd) as entries:
                for entry in entries:
                    if not budget.visit():
                        break
                    if is_private_component(entry.name) or (entry.name.startswith(".") and not include_hidden):
                        continue
                    path = relative / entry.name
                    if len(os.fsencode(path)) > MAX_PATH_BYTES:
                        budget.truncated = True
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in _SKIP_DIRS or entry.name.startswith("."):
                                continue
                            if select_directory is not None and not select_directory(path):
                                continue
                            with self.open_entry(fd, entry.name, directory=True) as (child_fd, _):
                                yield from self.files(
                                    child_fd, path, budget, include_hidden=include_hidden,
                                    select_directory=select_directory,
                                )
                        elif entry.is_file(follow_symlinks=False):
                            if entry.stat(follow_symlinks=False).st_nlink == 1:
                                yield fd, path
                    except OSError:
                        continue
        except OSError:
            return
