import os
import pathlib
import time

import pytest

from rlm_tools.sandbox import Sandbox

import rlm_tools.semantic as semantic
import rlm_tools.llm_bridge as llm_bridge


@pytest.fixture(autouse=True)
def no_live_model_calls(monkeypatch):
    """Offline evals must inject model clients."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("RLM_SUB_PROVIDER", raising=False)
    monkeypatch.delenv("RLM_GATE_PROVIDER", raising=False)
    monkeypatch.setenv("RLM_RECURSE_GATE", "0")

    def _blocked(*args, **kwargs):
        raise RuntimeError("live model calls are disabled in evals")

    monkeypatch.setattr(semantic, "get_client", _blocked)
    monkeypatch.setattr(llm_bridge, "get_client", _blocked)


FIXTURES_ROOT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"
DEFAULT_PY_CORPUS_PATH = FIXTURES_ROOT / "rlm_corpus_py"
DEFAULT_TS_CORPUS_PATH = FIXTURES_ROOT / "rlm_corpus_ts"
DEFAULT_MIXED_CORPUS_PATH = FIXTURES_ROOT / "rlm_corpus_mixed"
DEFAULT_LONG_CONTEXT_CORPUS_PATH = FIXTURES_ROOT / "rlm_corpus_longcontext"

APPLE_PROJECT_PATH = os.environ.get("RLM_EVAL_PROJECT_PATH", "")


def execute(sandbox: Sandbox, code: str) -> dict:
    result = sandbox.execute(code)
    return {
        "stdout": result.stdout,
        "error": result.error,
        "variables": result.variables,
    }


@pytest.fixture(scope="session")
def fixture_root():
    return FIXTURES_ROOT


@pytest.fixture(scope="session")
def sandbox():
    sb = Sandbox(base_path=str(DEFAULT_PY_CORPUS_PATH), max_output_chars=50_000)
    try:
        yield sb
    finally:
        sb.close()


@pytest.fixture(scope="session")
def ts_sandbox():
    sb = Sandbox(base_path=str(DEFAULT_TS_CORPUS_PATH), max_output_chars=50_000)
    try:
        yield sb
    finally:
        sb.close()


@pytest.fixture(scope="session")
def mixed_path():
    return DEFAULT_MIXED_CORPUS_PATH


@pytest.fixture(scope="session")
def external_apple_path():
    if not APPLE_PROJECT_PATH:
        pytest.skip("Skipping external dataset evals: set RLM_EVAL_PROJECT_PATH")

    path = pathlib.Path(APPLE_PROJECT_PATH)
    if not path.is_dir():
        pytest.skip(
            "Skipping external dataset evals: set RLM_EVAL_PROJECT_PATH to a valid checkout "
            f"(current: {APPLE_PROJECT_PATH})"
        )
    return path


# --- Timing report plugin ---

_timings: list[tuple[str, float, str]] = []


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    item._start_time = time.perf_counter()
    item._report_status = "UNKNOWN"


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    elapsed = time.perf_counter() - item._start_time
    status = getattr(item, "_report_status", "ERROR")
    _timings.append((item.nodeid, elapsed, status))


@pytest.hookimpl()
def pytest_runtest_makereport(item, call):
    if call.excinfo:
        if call.excinfo.errisinstance(pytest.skip.Exception):
            status = "SKIPPED"
        elif call.when == "setup":
            status = "ERROR"
        else:
            status = "FAILED"
    elif call.when == "call":
        status = "PASSED"
    else:
        return

    item._report_status = status


def pytest_terminal_summary(terminalreporter, config):
    if not _timings:
        return

    terminalreporter.section("Eval Timing Report")

    by_module: dict[str, list[tuple[str, float, str]]] = {}
    for nodeid, elapsed, status in _timings:
        module = nodeid.split("::")[0]
        by_module.setdefault(module, []).append((nodeid, elapsed, status))

    total = 0.0
    for module, tests in by_module.items():
        module_total = sum(t[1] for t in tests)
        total += module_total
        terminalreporter.write_line(f"\n  {module} ({module_total:.3f}s)")
        for nodeid, elapsed, status in sorted(tests, key=lambda x: -x[1]):
            short_name = "::".join(nodeid.split("::")[1:])
            terminalreporter.write_line(f"    {elapsed:.3f}s  {status}  {short_name}")

    terminalreporter.write_line(f"\n  Total eval time: {total:.3f}s")
