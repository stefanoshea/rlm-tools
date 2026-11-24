"""Test-wide guard against live model calls.

Every semantic and sub-LLM test injects a fake client. This fixture makes that
the only possible path: provider keys are stripped from the environment and the
real client factories raise, so a test that forgets to inject fails loudly
instead of quietly spending tokens.
"""

import os
import tempfile
from pathlib import Path

import pytest

import rlm_tools.llm_bridge as llm_bridge
import rlm_tools.semantic as semantic
from rlm_tools.access_policy import workspace_policy


@pytest.fixture(autouse=True)
def test_workspace_roots(monkeypatch, tmp_path_factory):
    roots = (Path(tempfile.gettempdir()).resolve(), tmp_path_factory.getbasetemp().resolve())
    monkeypatch.setenv("RLM_ALLOWED_ROOTS", os.pathsep.join(map(str, roots)))
    monkeypatch.delenv("RLM_EXTERNAL_ROOTS", raising=False)
    monkeypatch.delenv("RLM_RECURSE_GATE", raising=False)
    workspace_policy.cache_clear()
    yield
    workspace_policy.cache_clear()


@pytest.fixture
def external_test_roots(monkeypatch, test_workspace_roots):
    monkeypatch.setenv("RLM_EXTERNAL_ROOTS", os.environ["RLM_ALLOWED_ROOTS"])


@pytest.fixture(autouse=True)
def no_live_model_calls(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("RLM_SUB_PROVIDER", raising=False)
    monkeypatch.delenv("RLM_GATE_PROVIDER", raising=False)

    def _blocked(*args, **kwargs):
        raise RuntimeError("live model calls are disabled in tests; inject a client")

    monkeypatch.setattr(semantic, "get_client", _blocked)
    monkeypatch.setattr(llm_bridge, "get_client", _blocked)
