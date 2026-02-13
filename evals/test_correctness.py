import json as _json
import pathlib

from conftest import APPLE_PROJECT_PATH, execute

APP_PROJECT_PATH = pathlib.Path(APPLE_PROJECT_PATH) / "app"


class TestGlobCorrectness:
    def test_glob_finds_known_swift_files(self, sandbox):
        result = execute(sandbox, "swift_files = glob_files('**/*.swift')\nprint(len(swift_files))")
        assert result["error"] is None
        count = int(result["stdout"].strip())
        assert count > 100, f"Expected many Swift files, got {count}"

    def test_glob_includes_known_file(self, sandbox):
        result = execute(
            sandbox,
            "import json\nfiles = glob_files('**/DashboardBuilder.swift')\nprint(json.dumps(files))",
        )
        assert result["error"] is None
        files = _json.loads(result["stdout"])
        matching = [f for f in files if f.endswith("Dashboard/DashboardBuilder.swift")]
        assert len(matching) >= 1, f"DashboardBuilder.swift not found: {files}"

    def test_glob_matches_native_for_known_pattern(self, sandbox):
        result = execute(
            sandbox,
            "import json\nfiles = glob_files('app/Dashboard/*.swift')\nprint(json.dumps(sorted(files)))",
        )
        assert result["error"] is None
        rlm_files = set(_json.loads(result["stdout"]))

        native_dir = APP_PROJECT_PATH / "app" / "Dashboard"
        native_files = set(
            str(f.relative_to(APP_PROJECT_PATH))
            for f in native_dir.glob("*.swift")
            if f.is_file()
        )
        assert rlm_files == native_files, (
            f"Mismatch:\n  RLM only: {rlm_files - native_files}\n  Native only: {native_files - rlm_files}"
        )


class TestGrepCorrectness:
    def test_grep_finds_known_class(self, sandbox):
        result = execute(
            sandbox,
            "import json\nresults = grep('class DashboardBuilder', 'app/Dashboard')\nprint(json.dumps(results))",
        )
        assert result["error"] is None
        matches = _json.loads(result["stdout"])
        files_matched = {m["file"] for m in matches}
        assert any("DashboardBuilder.swift" in f for f in files_matched), (
            f"Expected DashboardBuilder.swift in results, got: {files_matched}"
        )

    def test_grep_finds_known_protocol(self, sandbox):
        result = execute(
            sandbox,
            "import json\nresults = grep('protocol DashboardBuilding', 'app/Dashboard')\nprint(json.dumps(results))",
        )
        assert result["error"] is None
        matches = _json.loads(result["stdout"])
        assert len(matches) >= 1, "protocol DashboardBuilding not found"
        assert any("DashboardBuilder.swift" in m["file"] for m in matches)

    def test_grep_scoped_to_subdirectory(self, sandbox):
        result = execute(
            sandbox,
            "import json\nresults = grep('import', 'app/Dashboard')\nprint(json.dumps([r['file'] for r in results[:20]]))",
        )
        assert result["error"] is None
        files = _json.loads(result["stdout"])
        for f in files:
            assert f.startswith("app/Dashboard"), (
                f"Scoped grep returned file outside scope: {f}"
            )


class TestReadFileCorrectness:
    def test_read_file_returns_correct_content(self, sandbox):
        target_file = "app/AppDelegate.swift"
        result = execute(sandbox, f"content = read_file('{target_file}')\nprint(content[:500])")
        assert result["error"] is None

        native_content = (APP_PROJECT_PATH / target_file).read_text()[:500]
        assert result["stdout"].strip() == native_content.strip()

    def test_read_file_contains_expected_class(self, sandbox):
        result = execute(
            sandbox,
            "content = read_file('app/AppDelegate.swift')\nprint('AppDelegate' in content)",
        )
        assert result["error"] is None
        assert "True" in result["stdout"]


class TestTreeCorrectness:
    def test_tree_shows_known_directories(self, sandbox):
        result = execute(sandbox, "output = tree('.', max_depth=1)\nprint(output)")
        assert result["error"] is None
        assert "app" in result["stdout"]
        assert "appTests" in result["stdout"]

    def test_tree_shows_dashboard_contents(self, sandbox):
        result = execute(sandbox, "output = tree('app/Dashboard', max_depth=1)\nprint(output)")
        assert result["error"] is None
        assert "DashboardBuilder.swift" in result["stdout"]


class TestPackageExploration:
    def test_package_tree(self, package_sandbox):
        result = execute(package_sandbox, "output = tree('.', max_depth=2)\nprint(output)")
        assert result["error"] is None
        assert "Sources" in result["stdout"]

    def test_package_glob(self, package_sandbox):
        result = execute(
            package_sandbox,
            "sources = glob_files('Sources/**/*.swift')\nprint(len(sources))",
        )
        assert result["error"] is None
        count = int(result["stdout"].strip())
        assert count >= 5, f"Expected DashboardPackage sources, got {count}"


class TestSecurityBoundaries:
    def test_path_traversal_blocked(self, sandbox):
        result = execute(sandbox, "content = read_file('../../../etc/passwd')\nprint(content)")
        assert result["error"] is not None
        assert "PermissionError" in result["error"]

    def test_write_operations_blocked(self, sandbox):
        result = execute(sandbox, "f = open('test_write.txt', 'w')\nf.write('hack')\nf.close()")
        assert result["error"] is not None
        assert "PermissionError" in result["error"]

    def test_dangerous_import_blocked(self, sandbox):
        result = execute(sandbox, "import subprocess\nsubprocess.run(['ls'])")
        assert result["error"] is not None
        assert "ImportError" in result["error"]
