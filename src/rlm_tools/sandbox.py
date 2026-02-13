import io
import contextlib
import builtins
import pathlib
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from rlm_tools.helpers import make_helpers


ALLOWED_MODULES = frozenset({
    "re", "json", "collections", "math",
    "fnmatch", "itertools", "functools", "operator", "string",
    "textwrap", "difflib", "statistics",
})

BLOCKED_BUILTINS = frozenset({
    "exec", "eval", "compile", "__import__", "breakpoint",
    "exit", "quit", "input",
})


@dataclass
class ExecutionResult:
    stdout: str
    error: str | None
    variables: list[str]


def _make_restricted_import(allowed: frozenset[str]):
    original_import = builtins.__import__

    def restricted_import(name, *args, **kwargs):
        if name not in allowed and name.split(".")[0] not in allowed:
            raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
        return original_import(name, *args, **kwargs)

    return restricted_import


class Sandbox:
    def __init__(
        self,
        base_path: str,
        max_output_chars: int = 15_000,
        execution_timeout_seconds: int = 30,
    ):
        self._base_path = base_path
        self._max_output_chars = max_output_chars
        self._execution_timeout_seconds = execution_timeout_seconds
        self._namespace: dict = {}
        self._resolve_safe = None
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        safe_builtins = {
            k: v for k, v in builtins.__dict__.items()
            if k not in BLOCKED_BUILTINS
        }
        safe_builtins["__import__"] = _make_restricted_import(ALLOWED_MODULES)

        original_open = builtins.open

        def restricted_open(file, mode="r", *args, **kwargs):
            if any(c in mode for c in "wxa+"):
                raise PermissionError(f"Write access denied in sandbox (mode='{mode}')")

            if self._resolve_safe is None:
                raise RuntimeError("Sandbox path resolver was not initialized")

            if isinstance(file, int):
                raise PermissionError("File descriptor access is not allowed in sandbox")

            # Keep read access scoped to the sandbox root.
            safe_path = self._resolve_safe(str(pathlib.Path(file)))
            return original_open(safe_path, mode, *args, **kwargs)

        safe_builtins["open"] = restricted_open

        self._namespace["__builtins__"] = safe_builtins

        helpers, self._resolve_safe = make_helpers(self._base_path)
        self._namespace.update(helpers)

    @contextmanager
    def _execution_timeout(self):
        if (
            self._execution_timeout_seconds <= 0
            or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "SIGALRM")
        ):
            yield
            return

        def _raise_timeout(_signum, _frame):
            raise TimeoutError(
                f"Execution timed out after {self._execution_timeout_seconds} seconds"
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self._execution_timeout_seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    def execute(self, code: str) -> ExecutionResult:
        stdout_capture = io.StringIO()
        error = None

        try:
            with contextlib.redirect_stdout(stdout_capture):
                with self._execution_timeout():
                    exec(code, self._namespace)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        stdout = stdout_capture.getvalue()
        if len(stdout) > self._max_output_chars:
            stdout = stdout[:self._max_output_chars] + "\n... [output truncated]"

        return ExecutionResult(
            stdout=stdout,
            error=error,
            variables=self.list_variables(),
        )

    def list_variables(self) -> list[str]:
        return [
            k for k in self._namespace
            if not k.startswith("_") and k != "__builtins__"
        ]
