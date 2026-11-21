import multiprocessing
import threading
import time
from dataclasses import dataclass

from rlm_tools.helpers import make_helpers
from rlm_tools.worker_protocol import receive_message, send_message
from rlm_tools.worker_runtime import worker_main


_WORKER_RESPONSE_GRACE_SECONDS = 1


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
    ):
        self._base_path = base_path
        self._max_output_chars = max_output_chars
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
        self._helpers, _ = make_helpers(self._base_path, memory_tracker=self._track_memory)
        self._tools: dict[str, object] = dict(self._helpers)

        ctx = multiprocessing.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._process = ctx.Process(
            target=worker_main,
            args=(
                child_conn,
                self._max_output_chars,
                self._execution_timeout_seconds,
                self._max_memory_bytes,
                self._max_address_space_bytes,
                self._cpu_seconds,
            ),
        )
        try:
            try:
                self._process.start()
            finally:
                child_conn.close()
            self._register_tools(list(self._helpers))
        except BaseException:
            self.close()
            raise

    def _register_tools(self, names: list[str]) -> None:
        send_message(self._parent_conn, {
            "type": "register_tools",
            "names": names,
        })
        message = receive_message(self._parent_conn)
        if message["type"] != "ack":
            raise RuntimeError("Failed to register tools with worker")

    def register_tool(self, name: str, func) -> None:
        with self._worker_lock:
            self._register_tools([name])
            self._tools[name] = func

    @property
    def helpers(self) -> dict:
        """The server-side exploration helpers (read_file, grep, ...)."""
        return dict(self._helpers)

    def _handle_helper_call(self, message: dict) -> None:
        name = message["name"]
        tool = self._tools.get(name)
        if tool is None:
            send_message(self._parent_conn, {
                "type": "helper_result",
                "value": None,
                "error": f"RuntimeError: Unknown helper '{name}'",
            })
            return

        try:
            value = tool(*message["args"], **message["kwargs"])
            send_message(self._parent_conn, {
                "type": "helper_result",
                "value": value,
                "error": None,
            })
        except Exception as e:
            send_message(self._parent_conn, {
                "type": "helper_result",
                "value": None,
                "error": f"{type(e).__name__}: {e}",
            })

    def execute(self, code: str) -> ExecutionResult:
        with self._worker_lock:
            if self._parent_conn.closed:
                return self._unavailable_result()
            try:
                return self._execute(code)
            except (EOFError, OSError):
                self._close_worker()
                return self._unavailable_result()

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

    def _execute(self, code: str) -> ExecutionResult:
        send_message(self._parent_conn, {"type": "execute", "code": code})

        deadline = None
        if self._execution_timeout_seconds > 0:
            deadline = time.monotonic() + self._execution_timeout_seconds
        response_deadline = None if deadline is None else deadline + _WORKER_RESPONSE_GRACE_SECONDS

        while True:
            if response_deadline is not None and not self._parent_conn.poll(max(0, response_deadline - time.monotonic())):
                return self._expire_execution()

            message = receive_message(self._parent_conn)
            kind = message["type"]

            if kind == "helper_call":
                if deadline is not None and time.monotonic() >= deadline:
                    return self._expire_execution()
                self._handle_helper_call(message)
                if response_deadline is not None:
                    # The worker must drain a synchronous helper response before
                    # its own execution timer can surface a timeout.
                    response_deadline = max(response_deadline, time.monotonic() + _WORKER_RESPONSE_GRACE_SECONDS)
                continue

            if kind != "execute_result":
                raise RuntimeError(f"Unexpected worker message '{kind}'")

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
        if not getattr(self, "_process", None):
            return
        with self._worker_lock:
            self._close_worker()

    def _close_worker(self, graceful: bool = True) -> None:
        if self._parent_conn.closed:
            return
        try:
            if graceful and self._process.is_alive():
                send_message(self._parent_conn, {"type": "close"})
                self._process.join(timeout=1)
        except (EOFError, OSError):
            pass
        finally:
            self._parent_conn.close()
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1)
            self._last_variables = []
            self._worker_memory_bytes_used = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
