import os
import pathlib
import tempfile
import time
import atexit
import shutil

import pytest

from rlm_tools.sandbox import Sandbox

APPLE_PROJECT_PATH = os.environ.get(
    "RLM_EVAL_PROJECT_PATH",
    str(pathlib.Path(__file__).resolve().parent.parent.parent / "your-iOS-project"),
)


def _resolve_override(base: pathlib.Path, value: str) -> pathlib.Path:
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _swift_file_count(root: pathlib.Path) -> int:
    try:
        return sum(1 for _ in root.rglob("*.swift"))
    except OSError:
        return 0


def _detect_app_root(project_root: pathlib.Path) -> pathlib.Path | None:
    override = os.environ.get("RLM_EVAL_APP_ROOT")
    if override:
        candidate = _resolve_override(project_root, override)
        return candidate if candidate.is_dir() else None

    candidates: list[tuple[int, pathlib.Path]] = []
    for child in project_root.iterdir():
        if not child.is_dir() or child.name.startswith(".") or child.name == "Packages":
            continue
        score = _swift_file_count(child)
        if score > 0:
            candidates.append((score, child))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _detect_package_root(project_root: pathlib.Path) -> pathlib.Path | None:
    override = os.environ.get("RLM_EVAL_PACKAGE_ROOT")
    if override:
        candidate = _resolve_override(project_root, override)
        return candidate if candidate.is_dir() else None

    packages_dir = project_root / "Packages"
    if not packages_dir.is_dir():
        return None

    candidates: list[tuple[int, pathlib.Path]] = []
    for child in packages_dir.iterdir():
        sources = child / "Sources"
        if not child.is_dir() or not sources.is_dir():
            continue
        score = _swift_file_count(sources)
        if score > 0:
            candidates.append((score, child))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def _detect_tests_root(project_root: pathlib.Path, app_root: pathlib.Path) -> pathlib.Path | None:
    override = os.environ.get("RLM_EVAL_TESTS_ROOT")
    if override:
        candidate = _resolve_override(project_root, override)
        return candidate if candidate.is_dir() else None

    for candidate in (project_root / "appTests", app_root.parent / "appTests"):
        if candidate.is_dir():
            return candidate
    return None


def _build_overlay_root(
    project_root: pathlib.Path,
    app_root: pathlib.Path,
    package_root: pathlib.Path | None,
    tests_root: pathlib.Path | None,
) -> pathlib.Path:
    overlay_root = pathlib.Path(tempfile.mkdtemp(prefix="rlm_tools_evals_"))
    os.symlink(app_root, overlay_root / "app")

    if tests_root:
        os.symlink(tests_root, overlay_root / "appTests")

    packages_alias = overlay_root / "Packages"
    packages_alias.mkdir()
    if package_root:
        os.symlink(package_root, packages_alias / "DashboardPackage")

    return overlay_root


_overlay_root: pathlib.Path | None = None
_real_project_root = pathlib.Path(APPLE_PROJECT_PATH).resolve()
if _real_project_root.is_dir():
    _app_root = _detect_app_root(_real_project_root)
    _package_root = _detect_package_root(_real_project_root)
    if _app_root:
        _tests_root = _detect_tests_root(_real_project_root, _app_root)
        _overlay_root = _build_overlay_root(
            project_root=_real_project_root,
            app_root=_app_root,
            package_root=_package_root,
            tests_root=_tests_root,
        )
        APPLE_PROJECT_PATH = str(_overlay_root)


@atexit.register
def _cleanup_overlay_root() -> None:
    if _overlay_root and _overlay_root.exists():
        shutil.rmtree(_overlay_root, ignore_errors=True)


def execute(sandbox: Sandbox, code: str) -> dict:
    result = sandbox.execute(code)
    return {
        "stdout": result.stdout,
        "error": result.error,
        "variables": result.variables,
    }


@pytest.fixture(scope="session")
def apple_path():
    path = pathlib.Path(APPLE_PROJECT_PATH)
    if not path.is_dir():
        pytest.skip(
            "Skipping evals: set RLM_EVAL_PROJECT_PATH to a local your-iOS-project checkout "
            f"(current: {APPLE_PROJECT_PATH})"
        )
    return path


@pytest.fixture(scope="session")
def sandbox(apple_path):
    return Sandbox(base_path=str(apple_path / "app"), max_output_chars=50_000)


@pytest.fixture(scope="session")
def package_sandbox(apple_path):
    return Sandbox(base_path=str(apple_path / "Packages" / "DashboardPackage"), max_output_chars=50_000)


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
