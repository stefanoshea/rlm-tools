"""Test-wide guard against live model calls.

Every semantic test injects a fake client. This fixture makes that the only
possible path: the TypeSafe key is stripped from the environment and the real
client factory raises, so a test that forgets to inject fails loudly instead of
quietly spending tokens.
"""

import pytest

import rlm_tools.llm_bridge as llm_bridge
import rlm_tools.semantic as semantic


@pytest.fixture(autouse=True)
def no_live_model_calls(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _blocked(*args, **kwargs):
        raise RuntimeError("live model calls are disabled in tests; inject a client")

    monkeypatch.setattr(semantic, "get_client", _blocked)
    monkeypatch.setattr(llm_bridge, "get_client", _blocked)
