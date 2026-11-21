import ast
import builtins
import contextlib
import io
import signal
import sys
import threading
from contextlib import contextmanager
from multiprocessing.connection import Connection
from types import ModuleType

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX platforms
    resource = None

from rlm_tools.worker_protocol import ExecuteResponse, receive_message, send_message


ALLOWED_MODULES = frozenset({
    "re", "json", "collections", "math",
    "fnmatch", "itertools", "functools", "operator", "string",
    "textwrap", "difflib", "statistics",
})

BLOCKED_BUILTINS = frozenset({
    "compile", "eval", "exec", "__import__", "breakpoint",
    "delattr", "dir", "exit", "getattr", "globals",
    "input", "locals", "open", "quit", "setattr",
    "object", "type", "vars",
})

# Method names that reach arbitrary attributes by string at runtime, bypassing
# the AST dot-attribute guard (e.g. "{0.__class__}".format(obj)).
BLOCKED_METHODS = frozenset({"format", "format_map"})

# Members of otherwise-safe stdlib modules that defeat the AST guard by resolving
# attributes from a string name. attrgetter/methodcaller can be aliased, so a
# name-based AST check cannot catch them — the primitive must not exist.
SANITIZED_MODULE_ATTRS: dict[str, tuple[str, ...]] = {
    "operator": ("attrgetter", "methodcaller"),
    "string": ("Formatter",),
}


class _OutputCapture(io.StringIO):
    def __init__(self, limit: int):
        super().__init__()
        self._limit = max(0, limit)
        self.truncated = False

    def write(self, text: str) -> int:
        remaining = self._limit - self.tell()
        if len(text) > remaining:
            self.truncated = True
            text_to_keep = text[:remaining]
        else:
            text_to_keep = text
        super().write(text_to_keep)
        return len(text)


def apply_resource_limits(
    max_address_space_bytes: int | None,
    cpu_seconds: int | None,
) -> dict[str, int]:
    """Best-effort OS-level resource backstop for the worker process.

    Layered *under* the per-call SIGALRM timeout and post-exec memory accounting:
    it catches runaway allocations and CPU loops that evade those (e.g. a tight
    C-level loop, or a multi-gigabyte allocation that would OOM before the
    post-exec check runs). Limits are cumulative for the process lifetime, so
    callers should pass generous session-scoped values, not per-call ones.

    Returns the limits actually applied. Silently skips any the platform does not
    support (notably RLIMIT_AS on macOS), since this is defense in depth.
    """
    applied: dict[str, int] = {}
    if resource is None:
        return applied

    requested = [
        ("RLIMIT_AS", max_address_space_bytes),
        ("RLIMIT_CPU", cpu_seconds),
    ]
    for name, value in requested:
        if value is None or value <= 0:
            continue
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            _soft, hard = resource.getrlimit(limit)
            target = value
            if hard != resource.RLIM_INFINITY:
                target = min(value, hard)
            resource.setrlimit(limit, (target, hard))
            applied[name] = target
        except (ValueError, OSError):
            continue
    return applied


def _make_restricted_import(allowed: frozenset[str]):
    original_import = builtins.__import__

    def restricted_import(name, *args, **kwargs):
        if name not in allowed and name.split(".")[0] not in allowed:
            raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
        module = original_import(name, *args, **kwargs)
        forbidden = SANITIZED_MODULE_ATTRS.get(name.split(".")[0])
        if forbidden:
            target = sys.modules.get(name.split(".")[0], module)
            for attr in forbidden:
                target.__dict__.pop(attr, None)
        return module

    return restricted_import


class HelperProxy:
    __slots__ = ("_conn", "_name")

    def __init__(self, conn: Connection, name: str):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_name", name)

    def __getattribute__(self, name: str):
        if name in {"_conn", "_name"}:
            return object.__getattribute__(self, name)
        if name.startswith("_") or name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(name)

    def __call__(self, *args, **kwargs):
        conn = object.__getattribute__(self, "_conn")
        # Read the entire reply before delivering an execution timeout. A
        # partial read would leave the worker pipe out of sync.
        previous_mask = None
        if hasattr(signal, "pthread_sigmask") and hasattr(signal, "SIGALRM"):
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        try:
            send_message(conn, {
                "type": "helper_call",
                "name": object.__getattribute__(self, "_name"),
                "args": args,
                "kwargs": kwargs,
            })
            message = receive_message(conn)
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if message["type"] != "helper_result":
            raise RuntimeError(f"Unexpected helper response: {message['type']}")
        if message["error"] is not None:
            raise RuntimeError(message["error"])
        return message["value"]


class LocalTool:
    __slots__ = ("_callback",)

    def __init__(self, callback):
        object.__setattr__(self, "_callback", callback)

    def __getattribute__(self, name: str):
        if name == "_callback":
            return object.__getattribute__(self, name)
        if name.startswith("_") or name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(name)

    def __call__(self, *args, **kwargs):
        callback = object.__getattribute__(self, "_callback")
        return callback(*args, **kwargs)


class WorkerRuntime:
    def __init__(
        self,
        conn: Connection,
        max_output_chars: int,
        execution_timeout_seconds: int,
        max_memory_bytes: int,
    ):
        self._conn = conn
        self._max_output_chars = max_output_chars
        self._execution_timeout_seconds = execution_timeout_seconds
        self._max_memory_bytes = max_memory_bytes
        self._namespace: dict = {}
        self._final_result: dict | None = None
        self._hidden_names = {"__builtins__", "rlm_finalize"}
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        safe_builtins = {
            k: v for k, v in builtins.__dict__.items()
            if k not in BLOCKED_BUILTINS
        }
        safe_builtins["__import__"] = _make_restricted_import(ALLOWED_MODULES)
        self._namespace["__builtins__"] = safe_builtins
        self._namespace["rlm_finalize"] = LocalTool(self._finalize)

    def register_tools(self, names: list[str]) -> None:
        for name in names:
            self._namespace[name] = HelperProxy(self._conn, name)
            self._hidden_names.add(name)

    def _finalize(self, value):
        self._final_result = {"final": value}
        return self._final_result

    def _validate_code(self, code: str) -> None:
        tree = ast.parse(code, mode="exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    raise PermissionError(
                        f"Access to private attribute '{node.attr}' is not allowed in the sandbox"
                    )
                if node.attr in BLOCKED_METHODS:
                    raise PermissionError(
                        f"'{node.attr}' is not allowed in the sandbox "
                        "(it can resolve attributes by name); use an f-string instead"
                    )

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

    def execute(self, code: str) -> ExecuteResponse:
        stdout_capture = _OutputCapture(self._max_output_chars)
        error = None
        self._final_result = None

        try:
            with contextlib.redirect_stdout(stdout_capture):
                with self._execution_timeout():
                    self._validate_code(code)
                    exec(code, self._namespace)
        except (Exception, SystemExit, KeyboardInterrupt) as e:
            error = f"{type(e).__name__}: {e}"

        stdout = stdout_capture.getvalue()
        if stdout_capture.truncated:
            stdout += "\n... [output truncated]"

        memory_bytes_used = self._measure_user_memory(limit=self._max_memory_bytes)
        if memory_bytes_used > self._max_memory_bytes and error is None:
            error = (
                "MemoryError: Session memory limit exceeded: "
                f"{memory_bytes_used} > {self._max_memory_bytes}"
            )

        return ExecuteResponse(
            stdout=stdout,
            error=error,
            variables=self.list_variables(),
            final=self._final_result if memory_bytes_used <= self._max_memory_bytes else None,
            memory_bytes_used=memory_bytes_used,
        )

    def list_variables(self) -> list[str]:
        return [
            key for key in self._namespace
            if not key.startswith("_") and key not in self._hidden_names
        ]

    def _measure_user_memory(self, limit: int | None = None) -> int:
        seen: set[int] = set()
        total = 0
        for name, value in self._namespace.items():
            if name in self._hidden_names:
                continue
            remaining = None if limit is None else limit - total
            total += self._deep_size(value, seen, remaining)
            # Stop walking the object graph once we know the cap is blown; the
            # exact over-limit figure does not matter, only that it is over.
            if limit is not None and total > limit:
                return total
        if self._final_result is not None:
            remaining = None if limit is None else limit - total
            total += self._deep_size(self._final_result, seen, remaining)
        return total

    def _deep_size(self, value, seen: set[int], limit: int | None) -> int:
        size = 0
        stack = [iter((value,))]
        while stack:
            try:
                item = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            obj_id = id(item)
            if obj_id in seen or isinstance(item, ModuleType):
                continue
            seen.add(obj_id)
            size += sys.getsizeof(item)
            if limit is not None and size > limit:
                break
            stack.append(iter(self._children(item)))
        return size

    def _children(self, value):
        if isinstance(value, dict):
            for key, item in dict.items(value):
                yield key
                yield item
        elif isinstance(value, list):
            yield from list.__iter__(value)
        elif isinstance(value, tuple):
            yield from tuple.__iter__(value)
        elif isinstance(value, set):
            yield from set.__iter__(value)
        elif isinstance(value, frozenset):
            yield from frozenset.__iter__(value)
        elif not isinstance(value, (str, bytes, bytearray)):
            if hasattr(value, "__dict__"):
                yield vars(value)
            slots = getattr(type(value), "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if hasattr(value, slot):
                    yield getattr(value, slot)


def worker_main(
    conn: Connection,
    max_output_chars: int,
    execution_timeout_seconds: int,
    max_memory_bytes: int,
    max_address_space_bytes: int | None = None,
    cpu_seconds: int | None = None,
) -> None:
    apply_resource_limits(
        max_address_space_bytes=max_address_space_bytes,
        cpu_seconds=cpu_seconds,
    )
    runtime = WorkerRuntime(
        conn=conn,
        max_output_chars=max_output_chars,
        execution_timeout_seconds=execution_timeout_seconds,
        max_memory_bytes=max_memory_bytes,
    )

    while True:
        try:
            message = receive_message(conn)
        except EOFError:
            break
        kind = message["type"]

        if kind == "register_tools":
            runtime.register_tools(message["names"])
            send_message(conn, {"type": "ack"})
            continue

        if kind == "execute":
            result = runtime.execute(message["code"])
            response = {
                "type": "execute_result",
                "stdout": result.stdout,
                "error": result.error,
                "variables": result.variables,
                "final": result.final,
                "memory_bytes_used": result.memory_bytes_used,
            }
            try:
                send_message(conn, response)
            except (TypeError, ValueError, RecursionError) as exc:
                response["error"] = result.error or f"{type(exc).__name__}: {exc}"
                response["final"] = None
                send_message(conn, response)
            continue

        if kind == "close":
            break

        send_message(conn, {
            "type": "execute_result",
            "stdout": "",
            "error": f"RuntimeError: Unknown worker message '{kind}'",
            "variables": runtime.list_variables(),
            "final": None,
            "memory_bytes_used": runtime._measure_user_memory(),
        })
