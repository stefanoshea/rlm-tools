import uuid
import time
import threading
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


class SessionManager:
    def __init__(self, max_sessions: int = 5, timeout_minutes: int = 10):
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._timeout_seconds = timeout_minutes * 60
        self._lock = threading.Lock()

    def create(
        self,
        path: str,
        query: str,
        max_output_chars: int = 15_000,
        max_llm_calls: int = 50,
        max_execute_calls: int = 50,
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
            )
            return session_id

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
            return self._cleanup_expired_locked()

    def _cleanup_expired_locked(self) -> list[str]:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used > self._timeout_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
        return expired
