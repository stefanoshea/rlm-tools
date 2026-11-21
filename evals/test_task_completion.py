import json as _json

from evals.conftest import execute


class TestMultiStepSearchWorkflows:
    def test_find_and_read_service_files(self, sandbox):
        result1 = execute(
            sandbox,
            "service_files = glob_files('services/*.py')\nprint(len(service_files))",
        )
        assert result1["error"] is None
        assert int(result1["stdout"].strip()) == 2

        result2 = execute(
            sandbox,
            "sample = read_file(service_files[0])\nprint('class ' in sample)",
        )
        assert result2["error"] is None
        assert result2["stdout"].strip() == "True"

    def test_find_class_and_read_definition(self, sandbox):
        result1 = execute(
            sandbox,
            "import json\nservice_defs = grep('class UserService')\nprint(json.dumps(service_defs))",
        )
        assert result1["error"] is None
        defs = _json.loads(result1["stdout"])
        assert len(defs) == 1

        result2 = execute(
            sandbox,
            "content = read_file('services/user_service.py')\nprint('load_user_profile' in content)",
        )
        assert result2["error"] is None
        assert result2["stdout"].strip() == "True"

    def test_explore_typescript_directory(self, ts_sandbox):
        result1 = execute(ts_sandbox, "tree_output = tree('src', max_depth=1)\nprint(tree_output)")
        assert result1["error"] is None
        assert "dashboard.ts" in result1["stdout"]

        result2 = execute(
            ts_sandbox,
            "import json\nts_files = glob_files('src/*.ts')\nprint(json.dumps(sorted(ts_files)))",
        )
        assert result2["error"] is None
        files = _json.loads(result2["stdout"])
        assert files == ["src/dashboard.ts", "src/deps.ts"]


class TestVariablePersistence:
    def test_variables_persist_across_three_calls(self, sandbox):
        execute(sandbox, "step1_data = glob_files('services/*.py')")
        execute(sandbox, "step2_count = len(step1_data)")
        result3 = execute(sandbox, "print(f'Found {step2_count} service files')")

        assert result3["error"] is None
        assert "Found 2 service files" in result3["stdout"]
        assert "step1_data" in result3["variables"]
        assert "step2_count" in result3["variables"]

    def test_variable_mutation_persists(self, sandbox):
        execute(sandbox, "accumulator_eval = []")
        execute(sandbox, "accumulator_eval.append('first')")
        execute(sandbox, "accumulator_eval.append('second')")
        result = execute(sandbox, "print(','.join(accumulator_eval))")

        assert result["error"] is None
        assert result["stdout"].strip() == "first,second"

    def test_computed_results_reusable(self, sandbox):
        execute(sandbox, "all_python = glob_files('**/*.py')")
        execute(sandbox, "service_only = [f for f in all_python if f.startswith('services/')]")
        result = execute(
            sandbox,
            "print(f'{len(service_only)} service files out of {len(all_python)} total')",
        )
        assert result["error"] is None
        assert "service files out of" in result["stdout"]


class TestSearchChains:
    def test_find_type_then_find_usages(self, sandbox):
        result1 = execute(
            sandbox,
            "import json\nservice_defs = grep('class UserService')\nprint(json.dumps(service_defs))",
        )
        assert result1["error"] is None

        result2 = execute(
            sandbox,
            "import json\nusages = grep('UserService')\nusage_files = sorted(set(r['file'] for r in usages))\nprint(json.dumps(usage_files))",
        )
        assert result2["error"] is None
        files = _json.loads(result2["stdout"])
        assert files == ["app.py", "services/user_service.py"]

    def test_find_import_then_trace_dependencies(self, ts_sandbox):
        result1 = execute(
            ts_sandbox,
            "imports = grep('import', 'src')\nprint(len(imports))",
        )
        assert result1["error"] is None
        assert int(result1["stdout"].strip()) >= 1

        result2 = execute(
            ts_sandbox,
            "import json\ndeps = grep('apiClient|clock|logger', 'src/dashboard.ts')\nprint(json.dumps(sorted(set(r['line'].strip() for r in deps))))",
        )
        assert result2["error"] is None
        lines = _json.loads(result2["stdout"])
        assert any("apiClient" in line for line in lines)
        assert any("clock" in line for line in lines)
