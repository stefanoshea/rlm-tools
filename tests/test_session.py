import pytest
from rlm_tools.session import SessionManager


def test_create_session():
    manager = SessionManager(max_sessions=5)
    session_id = manager.create(path="/tmp", query="test query")
    assert session_id is not None
    assert manager.get(session_id) is not None


def test_max_sessions_enforced():
    manager = SessionManager(max_sessions=2)
    manager.create(path="/tmp", query="q1")
    manager.create(path="/tmp", query="q2")
    with pytest.raises(RuntimeError, match="max sessions"):
        manager.create(path="/tmp", query="q3")


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
