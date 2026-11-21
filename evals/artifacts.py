"""Write benchmark artifacts with owner-only permissions."""

import os
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile


def ensure_artifact_dir(path: Path, *, exist_ok: bool = True) -> Path:
    try:
        path.mkdir(mode=0o700)
    except FileNotFoundError:
        ensure_artifact_dir(path.parent)
        path.mkdir(mode=0o700, exist_ok=exist_ok)
    except FileExistsError:
        if not exist_ok:
            raise

    info = path.lstat()
    owner = os.geteuid() if hasattr(os, "geteuid") else info.st_uid
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != owner or info.st_mode & 0o022:
        raise ValueError(f"Benchmark artifact directory must be owned by the current user and protected from writes by others: {path}")
    return path


def write_artifact(path: Path, content: str) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as destination:
            temporary_path = Path(destination.name)
            destination.write(content)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _private_file_descriptor(path: str, flags: int) -> int:
    return os.open(path, flags, 0o600)


def open_raw_artifact(path: Path):
    # Exclusive creation protects the raw stream's destination during a run.
    return open(path, "x", encoding="utf-8", opener=_private_file_descriptor)
