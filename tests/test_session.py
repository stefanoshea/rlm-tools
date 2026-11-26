import pytest
from rlm_tools.session import SessionManager


def test_create_session():
    manager = SessionManager(max_sessions=5)
    session_id = manager.create(path="/tmp", query="test query")
    assert session_id is not None
    assert manager.get(session_id) is not None


def test_max_sessions_enforced():
    manager = SessionManager(max_sessions=2, idle_evict_seconds=60.0)
    s1 = manager.create(path="/tmp/one", query="q1")
    s2 = manager.create(path="/tmp/two", query="q2")
    with pytest.raises(RuntimeError, match="max sessions") as exc_info:
        manager.create(path="/tmp/three", query="q3")
    assert s1 in str(exc_info.value)
    assert s2 in str(exc_info.value)


def test_lru_idle_eviction_at_max_sessions():
    manager = SessionManager(max_sessions=2, timeout_minutes=10, idle_evict_seconds=60.0)
    s1 = manager.create(path="/tmp/one", query="q1")
    s2 = manager.create(path="/tmp/two", query="q2")
    manager.get(s1).last_used -= 120
    manager.get(s2).last_used -= 70

    # Active sessions must never be evicted even if idle >= 60s.
    with manager.use(s1):
        s3 = manager.create(path="/tmp/three", query="q3")
        assert manager.get(s1) is not None
        assert manager.get(s2) is None
        assert manager.get(s3) is not None
        assert manager.cleanup_expired() == [s2]


def test_default_max_memory_env_override(monkeypatch):
    from rlm_tools.session import DEFAULT_MAX_MEMORY_BYTES, default_max_memory_bytes

    monkeypatch.delenv("RLM_MAX_MEMORY_BYTES", raising=False)
    assert default_max_memory_bytes() == DEFAULT_MAX_MEMORY_BYTES == 64 * 1024 * 1024

    monkeypatch.setenv("RLM_MAX_MEMORY_BYTES", "12345678")
    assert default_max_memory_bytes() == 12_345_678


def test_end_session():
    manager = SessionManager(max_sessions=5)
    session_id = manager.create(path="/tmp", query="test")
    manager.end(session_id)
    assert manager.get(session_id) is None


def test_session_carries_memory_limit():
    manager = SessionManager(max_sessions=1)
    session_id = manager.create(path="/tmp", query="test", max_memory_bytes=2048)
    session = manager.get(session_id)
    assert session is not None
    assert session.max_memory_bytes == 2048


def test_expired_sessions_are_reported_even_after_get_removes_them():
    manager = SessionManager()
    session_id = manager.create(path="/tmp", query="test")
    session = manager.get(session_id)
    session.last_used -= 601

    assert manager.get(session_id) is None
    assert manager.cleanup_expired() == [session_id]
    assert manager.cleanup_expired() == []


def test_starting_and_active_sessions_cannot_expire():
    manager = SessionManager(timeout_minutes=1)
    session_id = manager.create(path="/tmp", query="test", pin=True)
    session = manager.get(session_id)
    session.last_used -= 61
    assert manager.cleanup_expired() == []
    manager.release(session)

    with manager.use(session_id) as active:
        assert active is session
        session.last_used -= 61
        assert manager.cleanup_expired() == []

    assert manager.cleanup_expired() == []
    session.last_used -= 61
    assert manager.cleanup_expired() == [session_id]
