import uuid
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field


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
    max_memory_bytes: int = 10_000_000
    _operation_ready: threading.Condition = field(default_factory=threading.Condition, repr=False, compare=False)
    _next_operation: int = field(default=0, init=False, repr=False, compare=False)
    _current_operation: int = field(default=0, init=False, repr=False, compare=False)
    _active_operations: int = field(default=0, init=False, repr=False, compare=False)


class SessionManager:
    def __init__(self, max_sessions: int = 5, timeout_minutes: int = 10):
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._timeout_seconds = timeout_minutes * 60
        self._lock = threading.Lock()
        self._expired_session_ids: list[str] = []

    def create(
        self,
        path: str,
        query: str,
        max_output_chars: int = 15_000,
        max_llm_calls: int = 50,
        max_execute_calls: int = 50,
        max_memory_bytes: int = 10_000_000,
        *,
        pin: bool = False,
    ) -> str:
        with self._lock:
            self._cleanup_expired_locked()
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(
                    f"Cannot create session: max sessions ({self._max_sessions}) reached"
                )
            session_id = uuid.uuid4().hex[:12]
            self._sessions[session_id] = Session(
                session_id=session_id,
                path=path,
                query=query,
                max_output_chars=max_output_chars,
                max_llm_calls=max_llm_calls,
                max_execute_calls=max_execute_calls,
                max_memory_bytes=max_memory_bytes,
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
