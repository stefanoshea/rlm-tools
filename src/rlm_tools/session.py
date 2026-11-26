import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field


DEFAULT_MAX_MEMORY_BYTES = 64 * 1024 * 1024


def default_max_memory_bytes() -> int:
    raw = os.environ.get("RLM_MAX_MEMORY_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_MEMORY_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_MEMORY_BYTES
    return value if value >= 1024 else DEFAULT_MAX_MEMORY_BYTES


@dataclass
class Session:
    session_id: str
    path: str
    query: str
    _last_reported_vars: set = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    max_output_chars: int = 15_000
    max_llm_calls: int = 50
    llm_calls_used: int = 0
    max_execute_calls: int = 50
    execute_calls: int = 0
    memory_bytes_used: int = 0
    max_memory_bytes: int = field(default_factory=default_max_memory_bytes)
    sub_llm_usage: dict[str, int | float] = field(default_factory=dict)
    _operation_ready: threading.Condition = field(default_factory=threading.Condition, repr=False, compare=False)
    _next_operation: int = field(default=0, init=False, repr=False, compare=False)
    _current_operation: int = field(default=0, init=False, repr=False, compare=False)
    _active_operations: int = field(default=0, init=False, repr=False, compare=False)


class SessionManager:
    def __init__(
        self,
        max_sessions: int = 16,
        timeout_minutes: int = 10,
        idle_evict_seconds: float = 60.0,
    ):
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._timeout_seconds = timeout_minutes * 60
        self._idle_evict_seconds = max(0.0, float(idle_evict_seconds))
        self._lock = threading.Lock()
        self._expired_session_ids: list[str] = []

    def create(
        self,
        path: str,
        query: str,
        max_output_chars: int = 15_000,
        max_llm_calls: int = 50,
        max_execute_calls: int = 50,
        max_memory_bytes: int | None = None,
        *,
        pin: bool = False,
    ) -> str:
        resolved_memory = default_max_memory_bytes() if max_memory_bytes is None else max_memory_bytes
        with self._lock:
            self._cleanup_expired_locked()
            if len(self._sessions) >= self._max_sessions:
                now = time.time()
                evictable = [
                    s for s in self._sessions.values()
                    if not s._active_operations and (now - s.last_used) >= self._idle_evict_seconds
                ]
                if evictable:
                    victim = min(evictable, key=lambda s: s.last_used)
                    self._sessions.pop(victim.session_id, None)
                    self._expired_session_ids.append(victim.session_id)
                else:
                    summary_items = []
                    for sid, s in sorted(self._sessions.items(), key=lambda kv: kv[1].last_used):
                        idle = int(max(0.0, now - s.last_used))
                        state = f"active={s._active_operations}" if s._active_operations else f"idle={idle}s"
                        summary_items.append(f"{sid} ({s.path}, {state})")
                    summary = "; ".join(summary_items[:10])
                    if len(summary_items) > 10:
                        summary += f"; +{len(summary_items) - 10} more"
                    raise RuntimeError(
                        f"Cannot create session: max sessions ({self._max_sessions}) reached "
                        f"(no idle sessions >= {int(self._idle_evict_seconds)}s to evict). "
                        f"Open sessions: [{summary}]. Close finished sessions with end_session=true or rlm_end."
                    )
            session_id = uuid.uuid4().hex[:12]
            self._sessions[session_id] = Session(
                session_id=session_id,
                path=path,
                query=query,
                max_output_chars=max_output_chars,
                max_llm_calls=max_llm_calls,
                max_execute_calls=max_execute_calls,
                max_memory_bytes=resolved_memory,
            )
            if pin:
                session = self._sessions[session_id]
                session._active_operations = 1
                session._next_operation = 1
            return session_id

    def _unpin(self, session: Session) -> None:
        with self._lock:
            session._active_operations -= 1
            session.last_used = time.time()

    def release(self, session: Session) -> None:
        self._unpin(session)
        with session._operation_ready:
            session._current_operation += 1
            session._operation_ready.notify_all()

    @contextmanager
    def pin(self, session_id: str):
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)
            if session is not None:
                session._active_operations += 1
        try:
            yield session
        finally:
            if session is not None:
                self._unpin(session)

    @contextmanager
    def use(self, session_id: str):
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)
            if session is not None:
                operation = session._next_operation
                session._next_operation += 1
                session._active_operations += 1

        if session is None:
            yield None
            return

        try:
            with session._operation_ready:
                session._operation_ready.wait_for(lambda: session._current_operation == operation)
            with self._lock:
                current = self._sessions.get(session_id)
            yield session if current is session else None
        finally:
            self.release(session)

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._cleanup_expired_locked()
            session = self._sessions.get(session_id)
            if session:
                session.last_used = time.time()
            return session

    def end(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> list[str]:
        with self._lock:
            self._cleanup_expired_locked()
            expired, self._expired_session_ids = self._expired_session_ids, []
            return expired

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if not s._active_operations and now - s.last_used > self._timeout_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
        self._expired_session_ids.extend(expired)
