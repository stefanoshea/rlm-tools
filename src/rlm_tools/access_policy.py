import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def is_private_component(name: str) -> bool:
    name = name.casefold()
    if name in {".env.example", ".env.sample", ".env.template"}:
        return False
    return (
        name in {".git", ".env", ".mcp.local.json"}
        or name.startswith((".env.", ".mcpregistry_"))
    )


def _configured_directories(variable: str, default: tuple[Path, ...]) -> tuple[Path, ...]:
    value = os.environ.get(variable)
    if value is None:
        return default
    directories = []
    for entry in value.split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError(f"{variable} requires existing absolute directories")
        directories.append(path.resolve(strict=True))
    return tuple(directories)


@dataclass(frozen=True)
class WorkspacePolicy:
    allowed: tuple[Path, ...]
    external: tuple[Path, ...]

    @classmethod
    def from_environment(cls):
        cwd = Path.cwd()
        default = () if cwd in {Path(cwd.anchor), Path.home()} else (cwd,)
        allowed = _configured_directories("RLM_ALLOWED_ROOTS", default)
        external = _configured_directories("RLM_EXTERNAL_ROOTS", ())
        if any(not any(path.is_relative_to(root) for root in allowed) for path in external):
            raise ValueError("RLM_EXTERNAL_ROOTS must be inside RLM_ALLOWED_ROOTS")
        return cls(allowed, external)

    def authorize(self, path: str) -> Path:
        target = Path(path).resolve()
        if not any(target.is_relative_to(root) for root in self.allowed):
            raise PermissionError("Directory outside RLM_ALLOWED_ROOTS")
        if any(is_private_component(part) for part in target.parts):
            raise PermissionError("Private configuration directories cannot be explored")
        if not target.is_dir():
            raise NotADirectoryError("Session root must be an existing directory")
        return target

    def allows_external(self, path: Path) -> bool:
        return any(path.is_relative_to(root) for root in self.external)


@lru_cache(maxsize=1)
def workspace_policy() -> WorkspacePolicy:
    return WorkspacePolicy.from_environment()
