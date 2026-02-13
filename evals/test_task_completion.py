import json as _json

from conftest import execute


class TestMultiStepSearchWorkflows:
    def test_find_and_read_dashboard_files(self, sandbox):
        result1 = execute(
            sandbox,
            "dashboard_files = glob_files('app/Dashboard/*.swift')\nprint(len(dashboard_files))",
        )
        assert result1["error"] is None
        count = int(result1["stdout"].strip())
        assert count >= 5, f"Expected at least 5 Dashboard Swift files, got {count}"

        result2 = execute(sandbox, "sample = read_file(dashboard_files[0])\nprint(len(sample))")
        assert result2["error"] is None
        assert "dashboard_files" in result2["variables"]
        assert "sample" in result2["variables"]
        assert int(result2["stdout"].strip()) > 0

    def test_find_protocol_and_read_definition(self, sandbox):
        result1 = execute(
            sandbox,
            "import json\nprotocol_defs = grep('protocol DashboardBuilding', 'app/Dashboard')\nprint(json.dumps(protocol_defs))",
        )
        assert result1["error"] is None
        defs = _json.loads(result1["stdout"])
        assert len(defs) >= 1

        protocol_file = defs[0]["file"]
        result2 = execute(sandbox, f"content = read_file('{protocol_file}')\nprint(len(content.splitlines()))")
        assert result2["error"] is None
        assert "content" in result2["variables"]

    def test_explore_test_directory(self, sandbox):
        result1 = execute(sandbox, "test_tree = tree('appTests/Dashboard', max_depth=1)\nprint(test_tree)")
        assert result1["error"] is None
        assert "DashboardBuilderTests.swift" in result1["stdout"]

        result2 = execute(
            sandbox,
            "import json\ntest_files = glob_files('appTests/Dashboard/*.swift')\nprint(json.dumps(test_files))",
        )
        assert result2["error"] is None
        files = _json.loads(result2["stdout"])
        filenames = [f.split("/")[-1] for f in files]
        assert "DashboardBuilderTests.swift" in filenames

    def test_package_source_exploration(self, package_sandbox):
        result1 = execute(package_sandbox, "pkg_tree = tree('.', max_depth=2)\nprint(pkg_tree)")
        assert result1["error"] is None
        assert "Sources" in result1["stdout"]

        result2 = execute(
            package_sandbox,
            "pkg_sources = glob_files('Sources/**/*.swift')\nprint(len(pkg_sources))",
        )
        assert result2["error"] is None
        assert int(result2["stdout"].strip()) >= 5


class TestVariablePersistence:
    def test_variables_persist_across_three_calls(self, sandbox):
        execute(sandbox, "step1_data = glob_files('app/Dashboard/*.swift')")
        execute(sandbox, "step2_count = len(step1_data)")
        result3 = execute(sandbox, "print(f'Found {step2_count} files from step1')")

        assert result3["error"] is None
        assert "Found" in result3["stdout"]
        assert "files from step1" in result3["stdout"]
        assert "step1_data" in result3["variables"]
        assert "step2_count" in result3["variables"]

    def test_variable_mutation_persists(self, sandbox):
        execute(sandbox, "import json\naccumulator_eval = []")
        execute(sandbox, "accumulator_eval.append('first')")
        execute(sandbox, "accumulator_eval.append('second')")
        result = execute(sandbox, "print(json.dumps(accumulator_eval))")

        assert result["error"] is None
        items = _json.loads(result["stdout"])
        assert items == ["first", "second"]

    def test_computed_results_reusable(self, sandbox):
        execute(sandbox, "all_app_swift = glob_files('app/**/*.swift')")
        execute(sandbox, "dash_only = [f for f in all_app_swift if 'Dashboard' in f]")
        result = execute(
            sandbox,
            "print(f'{len(dash_only)} dashboard files out of {len(all_app_swift)} total')",
        )
        assert result["error"] is None
        assert "dashboard files out of" in result["stdout"]


class TestSearchChains:
    def test_find_type_then_find_its_usages(self, sandbox):
        result1 = execute(
            sandbox,
            "import json\nbuilder_defs = grep('class DashboardBuilder', 'app/Dashboard')\nprint(json.dumps(builder_defs[:3]))",
        )
        assert result1["error"] is None

        result2 = execute(
            sandbox,
            "import json\nusages = grep('DashboardBuilder', 'app')\nusage_files = list(set(r['file'] for r in usages))\nprint(json.dumps(usage_files[:10]))",
        )
        assert result2["error"] is None
        files = _json.loads(result2["stdout"])
        assert len(files) >= 2, f"Expected DashboardBuilder used in multiple files: {files}"

    def test_find_import_then_trace_reducers(self, sandbox):
        result1 = execute(
            sandbox,
            "import json\ntca_users = grep('import ComposableArchitecture', 'app')\ntca_files = list(set(r['file'] for r in tca_users))\nprint(len(tca_files))",
        )
        assert result1["error"] is None
        tca_count = int(result1["stdout"].strip())
        assert tca_count >= 1

        result2 = execute(
            sandbox,
            "import json\nreducers = grep('@Reducer', 'app')\nreducer_files = list(set(r['file'] for r in reducers))\nprint(json.dumps(reducer_files[:10]))",
        )
        assert result2["error"] is None
        reducer_files = _json.loads(result2["stdout"])
        assert len(reducer_files) >= 1
