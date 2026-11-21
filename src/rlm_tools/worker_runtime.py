import ast
import builtins
import collections
import contextlib
import functools
import io
import signal
import sys
import threading
from contextlib import contextmanager
from multiprocessing.connection import Connection
from types import (
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MethodDescriptorType,
    ModuleType,
    WrapperDescriptorType,
)

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
    "help", "license", "credits", "copyright",
    "input", "locals", "open", "quit", "setattr",
    "object", "type", "vars",
})

_ALLOWED_DUNDER_BUILTINS = frozenset({"__build_class__", "__debug__", "__name__"})

# Method names that reach arbitrary attributes by string at runtime, bypassing
# the AST dot-attribute guard (e.g. "{0.__class__}".format(obj)).
BLOCKED_METHODS = frozenset({"format", "format_map"})

# Members of otherwise-safe stdlib modules that defeat the AST guard by resolving
# attributes from a string name or wrapping functions to expose __wrapped__.
SANITIZED_MODULE_ATTRS: dict[str, tuple[str, ...]] = {
    "functools": ("update_wrapper", "wraps"),
    "operator": ("attrgetter", "methodcaller"),
    "string": ("Formatter",),
}


def _type_namespaces(cls):
    # Read class metadata through the built-in descriptors.
    mro = type.__dict__["__mro__"].__get__(cls)
    for base in mro:
        yield type.__dict__["__dict__"].__get__(base)


def _shallow_size(value) -> int:
    native_lookup = True
    value_type = type(value)
    for namespace in _type_namespaces(value_type):
        if "__sizeof__" not in namespace:
            continue
        descriptor = namespace["__sizeof__"]
        descriptor_type = type(descriptor)
        if (
            (descriptor_type is MethodDescriptorType or descriptor_type is WrapperDescriptorType)
            and issubclass(value_type, descriptor.__objclass__)
        ):
            if native_lookup:
                return sys.getsizeof(value)
            # Preserve the base storage estimate without running a Python override.
            return descriptor(value)
        native_lookup = False
    return object.__sizeof__(value)


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
    sanitized_cache: dict[str, ModuleType] = {}

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise ImportError("Relative imports are not allowed in the sandbox")
        if name not in allowed:
            raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
        proxy = sanitized_cache.get(name)
        if proxy is None:
            module = original_import(name)
            candidates = (
                list(module.__all__)
                if hasattr(module, "__all__")
                else [k for k in vars(module) if not k.startswith("_")]
            )
            forbidden = set(SANITIZED_MODULE_ATTRS.get(name, ()))
            proxy = ModuleType(name)
            exported: list[str] = []
            for attr in candidates:
                if attr.startswith("_") or attr in forbidden or not hasattr(module, attr):
                    continue
                value = getattr(module, attr)
                if isinstance(value, ModuleType):
                    continue
                setattr(proxy, attr, value)
                exported.append(attr)
            proxy.__all__ = exported
            sanitized_cache[name] = proxy
        return proxy

    return restricted_import


class HelperProxy:
    __slots__ = ("_conn", "_name")

    def __init__(self, conn: Connection, name: str):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_name", name)

    def __getattribute__(self, name: str):
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
            and (not k.startswith("_") or k in _ALLOWED_DUNDER_BUILTINS)
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
            elif isinstance(node, ast.MatchClass):
                for attr in node.kwd_attrs:
                    if attr.startswith("_"):
                        raise PermissionError(
                            f"Access to private attribute '{attr}' is not allowed in the sandbox"
                        )
                    if attr in BLOCKED_METHODS:
                        raise PermissionError(
                            f"'{attr}' is not allowed in the sandbox "
                            "(it can resolve attributes by name); use an f-string instead"
                        )
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name.startswith("_"):
                        raise PermissionError(
                            f"Import of private name '{alias.name}' is not allowed in the sandbox"
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
        except BaseException as e:
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
            if obj_id in seen or issubclass(type(item), ModuleType):
                continue
            seen.add(obj_id)
            size += _shallow_size(item)
            if limit is not None and size > limit:
                break
            stack.append(iter(self._children(item)))
        return size

    def _children(self, value):
        value_type = type(value)
        if issubclass(value_type, dict):
            for key, item in dict.items(value):
                yield key
                yield item
        elif issubclass(value_type, list):
            yield from list.__iter__(value)
        elif issubclass(value_type, tuple):
            yield from tuple.__iter__(value)
        elif issubclass(value_type, set):
            yield from set.__iter__(value)
        elif issubclass(value_type, frozenset):
            yield from frozenset.__iter__(value)
        elif issubclass(value_type, collections.deque):
            yield from collections.deque.__iter__(value)
        elif issubclass(value_type, functools.partial):
            for attr in ("func", "args", "keywords"):
                yield functools.partial.__dict__[attr].__get__(value)
        elif value_type is FunctionType:
            yield value.__defaults__
            yield value.__kwdefaults__
            if value.__closure__:
                for cell in value.__closure__:
                    try:
                        yield cell.cell_contents
                    except ValueError:
                        pass

        if value_type is str or value_type is bytes or value_type is bytearray:
            return
        has_instance_dict = False
        for namespace in _type_namespaces(value_type):
            dict_descriptor = namespace.get("__dict__")
            if not has_instance_dict and type(dict_descriptor) is GetSetDescriptorType:
                try:
                    yield dict_descriptor.__get__(value)
                except AttributeError:
                    pass
                has_instance_dict = True
            if "__slots__" in namespace:
                for descriptor in namespace.values():
                    if type(descriptor) is MemberDescriptorType:
                        try:
                            yield descriptor.__get__(value)
                        except AttributeError:
                            pass


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
