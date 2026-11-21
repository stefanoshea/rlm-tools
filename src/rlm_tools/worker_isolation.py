"""Apply OS restrictions after trusted worker initialization."""

import ctypes
import errno
import sys


def _confine_macos() -> None:
    sandbox = ctypes.CDLL("/usr/lib/libsandbox.1.dylib")
    sandbox.sandbox_init.argtypes = (
        ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p),
    )
    sandbox.sandbox_init.restype = ctypes.c_int
    sandbox.sandbox_free_error.argtypes = (ctypes.c_char_p,)
    sandbox.sandbox_free_error.restype = None

    error = ctypes.c_char_p()
    if sandbox.sandbox_init(b"(version 1) (deny default (with no-log))", 0, ctypes.byref(error)):
        try:
            reason = error.value.decode(errors="replace") if error.value else "unknown error"
        finally:
            if error:
                sandbox.sandbox_free_error(error)
        raise RuntimeError(f"Seatbelt could not confine the worker: {reason}")


# Python execution, memory management, clocks, and local signal handling.
# No syscall below resolves a path, creates a process, or connects a socket.
_LINUX_RUNTIME_SYSCALLS = (
    "brk", "clock_gettime", "clock_gettime64", "clock_nanosleep", "clock_nanosleep_time64",
    "exit", "exit_group", "futex", "futex_time64", "getpid", "gettid", "getrandom",
    "gettimeofday", "madvise", "mmap", "mmap2", "mprotect", "mremap", "munmap",
    "nanosleep", "restart_syscall", "rt_sigaction", "rt_sigprocmask", "rt_sigreturn",
    "setitimer", "sigaltstack",
)


class _SyscallArgument(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint),
        ("operation", ctypes.c_int),
        ("value", ctypes.c_uint64),
        ("mask_value", ctypes.c_uint64),
    ]


def _confine_linux(channel_fd: int) -> None:
    try:
        seccomp = ctypes.CDLL("libseccomp.so.2")
    except OSError as exc:
        raise RuntimeError("Linux worker confinement requires libseccomp.so.2") from exc

    seccomp.seccomp_init.argtypes = (ctypes.c_uint32,)
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_release.argtypes = (ctypes.c_void_p,)
    seccomp.seccomp_release.restype = None
    seccomp.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add_array.argtypes = (
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_SyscallArgument),
    )
    seccomp.seccomp_rule_add_array.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = (ctypes.c_void_p,)
    seccomp.seccomp_load.restype = ctypes.c_int

    action_errno = 0x00050000
    action_allow = 0x7FFF0000
    compare_equal = 4
    context = seccomp.seccomp_init(action_errno | errno.EPERM)
    if not context:
        raise RuntimeError("Could not initialize the worker seccomp filter")

    def allow(name: str, argument: _SyscallArgument | None = None) -> None:
        number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
        if number < 0:
            # Syscalls absent on the native ABI have no rule to install.
            return
        count = int(argument is not None)
        args = ctypes.byref(argument) if argument is not None else None
        if seccomp.seccomp_rule_add_array(context, action_allow, number, count, args) != 0:
            raise RuntimeError(f"Could not configure worker seccomp syscall {name}")

    try:
        for name in _LINUX_RUNTIME_SYSCALLS:
            allow(name)
        channel = _SyscallArgument(0, compare_equal, channel_fd, 0)
        for name in ("close", "recv", "recvfrom", "send", "sendto"):
            allow(name, channel)
        # libseccomp sets no_new_privs before loading the filter by default.
        if seccomp.seccomp_load(context) != 0:
            raise RuntimeError("Could not install the worker seccomp filter")
    finally:
        seccomp.seccomp_release(context)


def confine_worker(channel_fd: int) -> None:
    if sys.platform == "darwin":
        _confine_macos()
    elif sys.platform == "linux":
        _confine_linux(channel_fd)
    else:
        raise RuntimeError(f"OS confinement is unavailable on {sys.platform}")
