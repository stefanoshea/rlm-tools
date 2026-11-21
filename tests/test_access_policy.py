import json
import os
from unittest.mock import patch

import pytest

from rlm_tools.access_policy import WorkspacePolicy, workspace_policy


def test_launch_directory_is_default_and_credentials_do_not_grant_egress(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RLM_ALLOWED_ROOTS")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    policy = workspace_policy()
    assert policy.authorize(".") == tmp_path.resolve()
    assert not policy.allows_external(tmp_path)
    with pytest.raises(PermissionError):
        policy.authorize(str(tmp_path.parent))


def test_home_launch_requires_explicit_roots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RLM_ALLOWED_ROOTS")
    assert not WorkspacePolicy.from_environment().allowed


def test_separate_local_and_external_roots(tmp_path, monkeypatch):
    local, shared = tmp_path / "local", tmp_path / "shared"
    local.mkdir()
    shared.mkdir()
    monkeypatch.setenv("RLM_ALLOWED_ROOTS", os.pathsep.join(map(str, (local, shared))))
    monkeypatch.setenv("RLM_EXTERNAL_ROOTS", str(shared))
    policy = workspace_policy()
    assert not policy.allows_external(policy.authorize(str(local)))
    assert policy.allows_external(policy.authorize(str(shared)))
    with pytest.raises(PermissionError):
        policy.authorize(str(tmp_path))


def test_invalid_configuration_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_ALLOWED_ROOTS", "relative")
    with pytest.raises(ValueError, match="absolute"):
        WorkspacePolicy.from_environment()
    monkeypatch.setenv("RLM_ALLOWED_ROOTS", "")
    monkeypatch.setenv("RLM_EXTERNAL_ROOTS", str(tmp_path))
    with pytest.raises(ValueError, match="RLM_EXTERNAL_ROOTS"):
        WorkspacePolicy.from_environment()


def test_rejected_root_does_not_start_worker_or_scan(tmp_path, monkeypatch):
    from rlm_tools import server

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.setenv("RLM_ALLOWED_ROOTS", str(repo))
    with patch.object(server, "Sandbox") as sandbox, patch.object(server, "_scan_metadata") as scan:
        for path in (repo / ".git", tmp_path):
            result = json.loads(server._rlm_start(str(path), "count files", include_metadata=True))
            assert "error" in result
        sandbox.assert_not_called()
        scan.assert_not_called()


def test_local_session_never_initializes_providers(tmp_path, monkeypatch):
    from rlm_tools import server

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("RLM_RECURSE_GATE", "1")
    with patch.object(server, "make_scorer") as scorer, patch.object(server, "make_llm_query") as provider:
        started = json.loads(server._rlm_start(str(tmp_path), "count files"))
        try:
            assert not any("llm_query" in name for name in started["available_functions"])
            scorer.assert_not_called()
            provider.assert_not_called()
        finally:
            server._rlm_end(started["session_id"])
