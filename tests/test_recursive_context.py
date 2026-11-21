import json
import os
import tempfile
from unittest.mock import patch

from rlm_tools.server import _rlm_end, _rlm_execute, _rlm_start


def test_rlm_recurse_uses_named_context_variable(external_test_roots):
    # rlm_recurse is now a recursive explorer; verify the prompt and the named
    # context variable are threaded through to it and the result is finalizable.
    with tempfile.TemporaryDirectory() as tmpdir, \
            patch("rlm_tools.server.make_llm_query") as make_query, \
            patch("rlm_tools.server.make_recursive_query") as make_recursive:
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        make_query.return_value = lambda prompt, context="": f"flat:{prompt}:{context}"
        make_recursive.return_value = lambda prompt, context="": f"recursed:{prompt}:{context}"

        start = json.loads(_rlm_start(path=tmpdir, query="analyze"))
        sid = start["session_id"]
        try:
            result = json.loads(_rlm_execute(
                sid,
                "snippet = 'class DashboardReducer: pass'\n"
                "response = rlm_recurse('What type is this?', snippet)\n"
                "rlm_finalize(response)\n"
                "print(response)",
            ))
            assert result["error"] is None
            assert result["final"] == {"final": "recursed:What type is this?:class DashboardReducer: pass"}
        finally:
            _rlm_end(sid)
            os.environ.pop("ANTHROPIC_API_KEY", None)
