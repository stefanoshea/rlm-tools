import json as _json

from evals.conftest import execute


def test_fixture_corpus_exists(fixture_root):
    assert (fixture_root / "rlm_corpus_py" / "app.py").is_file()
    assert (fixture_root / "rlm_corpus_ts" / "src" / "dashboard.ts").is_file()
    assert (fixture_root / "rlm_corpus_longcontext" / "app" / "export" / "pipeline.py").is_file()


class TestPythonCorpusCorrectness:
    def test_glob_finds_python_files(self, sandbox):
        result = execute(sandbox, "python_files = glob_files('**/*.py')\nprint(len(python_files))")
        assert result["error"] is None
        assert int(result["stdout"].strip()) >= 3

    def test_glob_includes_user_service(self, sandbox):
        result = execute(
            sandbox,
            "import json\nfiles = glob_files('services/*.py')\nprint(json.dumps(sorted(files)))",
        )
        assert result["error"] is None
        files = _json.loads(result["stdout"])
        assert "services/user_service.py" in files

    def test_grep_finds_user_service_class(self, sandbox):
        result = execute(
            sandbox,
            "import json\nresults = grep('class UserService')\nprint(json.dumps(results))",
        )
        assert result["error"] is None
        matches = _json.loads(result["stdout"])
        assert any(match["file"] == "services/user_service.py" for match in matches)

    def test_read_file_returns_expected_content(self, sandbox):
        result = execute(sandbox, "content = read_file('app.py')\nprint('DashboardController' in content)")
        assert result["error"] is None
        assert result["stdout"].strip() == "True"

    def test_tree_shows_services_directory(self, sandbox):
        result = execute(sandbox, "output = tree('.', max_depth=2)\nprint(output)")
        assert result["error"] is None
        assert "services" in result["stdout"]
        assert "user_service.py" in result["stdout"]


class TestTypeScriptCorpusCorrectness:
    def test_ts_glob_finds_dashboard_file(self, ts_sandbox):
        result = execute(
            ts_sandbox,
            "import json\nfiles = glob_files('src/*.ts')\nprint(json.dumps(sorted(files)))",
        )
        assert result["error"] is None
        files = _json.loads(result["stdout"])
        assert "src/dashboard.ts" in files

    def test_ts_grep_finds_dashboard_reducer(self, ts_sandbox):
        result = execute(
            ts_sandbox,
            "import json\nresults = grep('class DashboardReducer', 'src')\nprint(json.dumps(results))",
        )
        assert result["error"] is None
        matches = _json.loads(result["stdout"])
        assert any(match["file"] == "src/dashboard.ts" for match in matches)


class TestSecurityBoundaries:
    def test_path_traversal_blocked(self, sandbox):
        result = execute(sandbox, "content = read_file('../../../etc/passwd')\nprint(content)")
        assert result["error"] is not None
        assert "PermissionError" in result["error"]

    def test_write_operations_blocked(self, sandbox):
        result = execute(sandbox, "f = open('test_write.txt', 'w')\nf.write('hack')\nf.close()")
        assert result["error"] is not None
        assert "NameError" in result["error"] or "PermissionError" in result["error"]

    def test_dangerous_import_blocked(self, sandbox):
        result = execute(sandbox, "import subprocess\nsubprocess.run(['ls'])")
        assert result["error"] is not None
        assert "ImportError" in result["error"]
