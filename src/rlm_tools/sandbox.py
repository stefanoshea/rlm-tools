import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rlm_tools.helpers import make_helpers
from rlm_tools.worker_protocol import (
    WorkerChannel,
    WorkerProtocolError,
    receive_message,
    send_message,
)


_WORKER_RESPONSE_GRACE_SECONDS = 1
_WORKER_STARTUP_TIMEOUT_SECONDS = 10


@dataclass
class ExecutionResult:
    stdout: str
    error: str | None
    variables: list[str]
    final: dict | None = None
    memory_bytes_used: int = 0


class Sandbox:
    def __init__(
        self,
        base_path: str,
        max_output_chars: int = 15_000,
        execution_timeout_seconds: int = 30,
        max_memory_bytes: int = 10_000_000,
        max_address_space_bytes: int | None = None,
        cpu_seconds: int = 600,
        *,
        base_is_canonical: bool = False,
    ):
        self._base_path = base_path
        self._max_output_chars = max_output_chars
        if execution_timeout_seconds <= 0:
            raise ValueError("Sandbox execution_timeout_seconds must be positive")
        self._execution_timeout_seconds = execution_timeout_seconds
        self._max_memory_bytes = max_memory_bytes
        # OS-level backstop ceiling for the worker's virtual address space, well
        # above the precise per-session byte cap. Stops a single runaway
        # allocation from OOMing the host before the post-exec accounting runs.
        # No-op where the platform ignores RLIMIT_AS (e.g. macOS).
        if max_address_space_bytes is None:
            max_address_space_bytes = max(1_073_741_824, max_memory_bytes * 8 + 536_870_912)
        self._max_address_space_bytes = max_address_space_bytes
        self._cpu_seconds = cpu_seconds
        self._helper_memory_bytes_used = 0
        self._worker_memory_bytes_used = 0
        self._worker_unavailable_error = "RuntimeError: Sandbox worker is unavailable"
        self._last_variables: list[str] = []
        self._worker_lock = threading.Lock()
        self._execution_deadline: float | None = None
        self._helpers, self._workspace = make_helpers(
            self._base_path,
            memory_tracker=self._track_memory,
            base_is_canonical=base_is_canonical,
            deadline=self.execution_deadline,
        )
        self._tools: dict[str, object] = dict(self._helpers)
        self._parent_conn: WorkerChannel | None = None
        self._process: subprocess.Popen | None = None

        try:
            if sys.platform not in {"darwin", "linux"}:
                raise RuntimeError(f"OS confinement is unavailable on {sys.platform}")
            parent, child = socket.socketpair()
            self._parent_conn = WorkerChannel(parent)
            with child:
                self._process = subprocess.Popen(
                    [
                        sys.executable, "-I", "-S", "-B",
                        str(Path(__file__).resolve().with_name("_worker_entry.py")),
                        str(child.fileno()),
                        str(self._max_output_chars),
                        str(self._execution_timeout_seconds),
                        str(self._max_memory_bytes),
                        str(self._max_address_space_bytes),
                        str(self._cpu_seconds),
                    ],
                    cwd="/",
                    env={},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    pass_fds=(child.fileno(),),
                )
            deadline = time.monotonic() + _WORKER_STARTUP_TIMEOUT_SECONDS
            message = receive_message(self._parent_conn, deadline)
            if message.get("type") != "ready":
                raise RuntimeError(message.get("error", "Unexpected worker startup response"))
            self._register_tools(list(self._helpers))
        except (OSError, EOFError, ValueError, RuntimeError) as exc:
            self.close()
            raise RuntimeError(f"Could not start a confined sandbox worker: {exc}") from exc
        except BaseException:
            self.close()
            raise

    def _register_tools(self, names: list[str]) -> None:
        deadline = time.monotonic() + _WORKER_STARTUP_TIMEOUT_SECONDS
        send_message(self._parent_conn, {"type": "register_tools", "names": names}, deadline)
        message = receive_message(self._parent_conn, deadline)
        if message.get("type") != "ack":
            raise WorkerProtocolError("Failed to register tools with worker")

    def register_tool(self, name: str, func) -> None:
        with self._worker_lock:
            try:
                self._register_tools([name])
            except (EOFError, OSError, WorkerProtocolError):
                self._close_worker(graceful=False)
                raise
            self._tools[name] = func

    @property
    def helpers(self) -> dict:
        """The server-side exploration helpers (read_file, grep, ...)."""
        return dict(self._helpers)

    def _handle_helper_call(self, message: dict, response_deadline: float) -> float:
        name = message.get("name")
        args = message.get("args")
        kwargs = message.get("kwargs")
        if not isinstance(name, str) or not isinstance(args, list) or not isinstance(kwargs, dict):
            raise WorkerProtocolError("Invalid worker helper request")
        tool = self._tools.get(name)
        try:
            if tool is None:
                raise RuntimeError(f"Unknown helper '{name[:100]}'")
            value = tool(*args, **kwargs)
            response = {"type": "helper_result", "value": value, "error": None}
        except Exception as exc:
            response = {
                "type": "helper_result", "value": None,
                "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            }

        # A synchronous helper must finish its reply before the worker can
        # surface its own timeout without leaving the frame stream out of sync.
        response_deadline = max(response_deadline, time.monotonic() + _WORKER_RESPONSE_GRACE_SECONDS)
        try:
            send_message(self._parent_conn, response, response_deadline)
        except (TypeError, ValueError, RecursionError):
            send_message(self._parent_conn, {
                "type": "helper_result", "value": None,
                "error": "RuntimeError: Helper result exceeds IPC limits or cannot be encoded",
            }, response_deadline)
        return response_deadline

    def execution_deadline(self) -> float | None:
        """Return the active execution's absolute monotonic deadline, if any."""
        return self._execution_deadline

    def execute(self, code: str) -> ExecutionResult:
        with self._worker_lock:
            if self._parent_conn.closed:
                return self._unavailable_result()
            deadline = time.monotonic() + self._execution_timeout_seconds
            self._execution_deadline = deadline
            try:
                return self._execute(code, deadline)
            except TimeoutError:
                return self._expire_execution()
            except (EOFError, OSError, WorkerProtocolError):
                self._close_worker(graceful=False)
                return self._unavailable_result()
            finally:
                self._execution_deadline = None

    def _unavailable_result(self, error: str | None = None) -> ExecutionResult:
        return ExecutionResult(
            stdout="",
            error=error or self._worker_unavailable_error,
            variables=self.list_variables(),
            memory_bytes_used=self.memory_bytes_used,
        )

    def _expire_execution(self) -> ExecutionResult:
        self._worker_unavailable_error = "RuntimeError: Sandbox worker was closed after an execution timeout"
        self._close_worker(graceful=False)
        return self._unavailable_result(
            f"TimeoutError: Worker did not return after {self._execution_timeout_seconds} seconds. "
            "Worker state was discarded."
        )

    def _execute(self, code: str, deadline: float) -> ExecutionResult:
        response_deadline = deadline + _WORKER_RESPONSE_GRACE_SECONDS
        try:
            send_message(self._parent_conn, {"type": "execute", "code": code}, deadline)
        except WorkerProtocolError as exc:
            return self._unavailable_result(f"ValueError: {exc}")

        while True:
            if not self._parent_conn.poll(max(0, response_deadline - time.monotonic())):
                return self._expire_execution()

            message = receive_message(self._parent_conn, response_deadline)
            kind = message.get("type")

            if kind == "helper_call":
                if time.monotonic() >= deadline:
                    return self._expire_execution()
                response_deadline = self._handle_helper_call(message, response_deadline)
                continue

            if kind != "execute_result":
                raise WorkerProtocolError("Unexpected worker message")
            self._validate_execute_result(message)
            self._last_variables = message["variables"]
            self._worker_memory_bytes_used = message.get("memory_bytes_used", 0)
            memory_bytes_used = self.memory_bytes_used
            if memory_bytes_used > self._max_memory_bytes:
                self._worker_unavailable_error = (
                    "RuntimeError: Sandbox worker was closed after exceeding the session memory limit"
                )
                self._close_worker()
                return ExecutionResult(
                    stdout=message["stdout"],
                    error=(
                        "MemoryError: Session memory limit exceeded: "
                        f"{memory_bytes_used} > {self._max_memory_bytes}; worker state was discarded"
                    ),
                    variables=[],
                    memory_bytes_used=memory_bytes_used,
                )
            return ExecutionResult(
                stdout=message["stdout"],
                error=message["error"],
                variables=message["variables"],
                final=message.get("final"),
                memory_bytes_used=self.memory_bytes_used,
            )

    @staticmethod
    def _validate_execute_result(message: dict) -> None:
        variables = message.get("variables")
        error = message.get("error")
        final = message.get("final")
        memory = message.get("memory_bytes_used")
        if (
            not isinstance(message.get("stdout"), str)
            or (error is not None and not isinstance(error, str))
            or not isinstance(variables, list)
            or not all(isinstance(name, str) for name in variables)
            or (final is not None and not isinstance(final, dict))
            or type(memory) is not int
            or memory < 0
        ):
            raise WorkerProtocolError("Invalid worker execution result")
        message.setdefault("error", None)

    def list_variables(self) -> list[str]:
        return list(self._last_variables)

    @property
    def memory_bytes_used(self) -> int:
        return self._helper_memory_bytes_used + self._worker_memory_bytes_used

    def _track_memory(self, value: int, from_bytes: bool = False) -> None:
        size = value if from_bytes else value
        if self.memory_bytes_used + size > self._max_memory_bytes:
            raise MemoryError(
                f"Session memory limit exceeded: {self.memory_bytes_used + size} > {self._max_memory_bytes}"
            )
        self._helper_memory_bytes_used += size

    def close(self) -> None:
        with self._worker_lock:
            try:
                self._close_worker()
            finally:
                self._workspace.close()

    def _close_worker(self, graceful: bool = True) -> None:
        if self._parent_conn is not None and not self._parent_conn.closed:
            try:
                if graceful and self._process is not None and self._process.poll() is None:
                    send_message(
                        self._parent_conn, {"type": "close"},
                        time.monotonic() + _WORKER_RESPONSE_GRACE_SECONDS,
                    )
                    self._process.wait(timeout=_WORKER_RESPONSE_GRACE_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass
            finally:
                self._parent_conn.close()
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=_WORKER_RESPONSE_GRACE_SECONDS)
        self._last_variables = []
        self._worker_memory_bytes_used = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
