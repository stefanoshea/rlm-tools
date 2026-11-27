import ast
import json
from pathlib import Path
import re

import pytest

from evals.grade_eval import load_eval_specs
from evals.monorepo_benchmark import PLANTED_BUG_SPECS, generate_monorepo_corpus
from evals.run_audit_benchmark import parse_transcript_metrics, score_audit_findings
from rlm_tools.helpers import make_helpers


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_monorepo_corpus_and_helpers_contract(tmp_path):
    manifest = generate_monorepo_corpus(tmp_path, seed=1234, scale="small")

    assert manifest["tool_count"] == 50
    assert len(manifest["planted_bugs"]) == 7
    assert manifest["scheduling_suite_bytes"] >= 250_000
    assert sum(1 for bug in manifest["planted_bugs"] if bug["in_mega_suite"]) == 2

    # Every generated Python file must parse cleanly.
    py_files = list(tmp_path.rglob("*.py"))
    assert len(py_files) == manifest["total_py_files"]
    for py_file in py_files:
        ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

    # Generated evals conform to EvalSpec validation rules.
    spec_path = tmp_path / "eval_specs.json"
    spec_path.write_text(json.dumps({"evals": manifest["evals"]}), encoding="utf-8")
    specs = load_eval_specs(spec_path)
    assert [s.eval_id for s in specs] == [
        "tool_contract_audit",
        "long_functions_module_level",
        "monolith_outline_counts",
        "model_field_inventory",
    ]

    helpers, _ = make_helpers(str(tmp_path))

    # Monolith outline matches AST ground truth with parent/depth fields.
    mono_outline = helpers["outline"]("services/svc_monolith/legacy_core.py")
    mono_methods = [m for cls in mono_outline["classes"] for m in cls["methods"]]
    mono_mod_funcs = [fn for fn in mono_outline["functions"] if fn["depth"] == 0]
    mono_nested_funcs = [fn for fn in mono_outline["functions"] if fn["depth"] >= 1]
    assert len(mono_mod_funcs) + len(mono_methods) + len(mono_nested_funcs) == manifest["ground_truth"]["legacy_core"]["total_defs"]
    assert len(mono_mod_funcs) == manifest["ground_truth"]["legacy_core"]["module_level_functions"]
    assert len(mono_methods) == manifest["ground_truth"]["legacy_core"]["methods"]
    assert len(mono_nested_funcs) == manifest["ground_truth"]["legacy_core"]["nested_functions"]
    assert all(fn["parent"] is None for fn in mono_mod_funcs)
    assert all(fn["parent"] is not None for fn in mono_nested_funcs)
    assert any(fn["depth"] == 2 and "." in fn["parent"] for fn in mono_nested_funcs)

    # Worker outline distinguishes module-level >=150-line defs from nested >=150-line defs.
    found_module_long: list[dict[str, str]] = []
    found_nested_long: list[dict[str, str]] = []
    for rel in helpers["glob_files"]("services/svc_workers/*.py"):
        if rel.endswith("__init__.py"):
            continue
        out = helpers["outline"](rel)
        for fn in out["functions"]:
            if fn["lines"] >= 150:
                if fn["depth"] == 0 and fn["parent"] is None:
                    found_module_long.append({"file": rel, "function": fn["name"]})
                elif fn["depth"] >= 2:
                    found_nested_long.append({"file": rel, "function": fn["name"]})
    found_module_long.sort(key=lambda r: (r["file"], r["function"]))
    expected_module_long = [
        {"file": r["file"], "function": r["function"]}
        for r in manifest["ground_truth"]["long_functions_module_level"]
    ]
    assert found_module_long == expected_module_long
    assert len(found_nested_long) == len(manifest["ground_truth"]["long_functions_nested"])

    # Verify all 43 clean tools have distinct domain-accurate implementations (no placeholder stub)
    # and the 5 scheduling handlers are interleaved across the >270 KB mega-suite file.
    handler_paths = sorted(helpers["glob_files"]("services/svc_agent_tools/handlers/*_handlers.py"))
    assert len(handler_paths) == 10
    for hp in handler_paths:
        src = (tmp_path / hp).read_text(encoding="utf-8")
        assert "def <" not in src
        assert "row.get('workspace_id', workspace_id) == workspace_id" not in src
    sched_outline = helpers["outline"]("services/svc_agent_tools/handlers/scheduling_handlers.py")
    sched_tool_lines = [
        fn["start_line"] for fn in sched_outline["functions"]
        if any(d.get("name") == "register_tool" for d in fn["decorators"])
    ]
    assert len(sched_tool_lines) == 5
    assert sched_tool_lines[-1] - sched_tool_lines[0] > 2_000

    # Sandbox & policy checks.
    for blocked in (".env", ".npmrc", ".pypirc", ".claude/settings.local.json", ".mcp.local.json"):
        with pytest.raises(PermissionError, match="private"):
            helpers["read_file"](blocked)
    assert "APP_ENV=development" in helpers["read_file"](".env.example")
    assert helpers["glob_files"](".github/workflows/*.yml") == [".github/workflows/ci.yml"]
    assert helpers["glob_files"]("services/svc_api_gateway/**") == [
        "services/svc_api_gateway/.env.example",
        "services/svc_api_gateway/__init__.py",
        "services/svc_api_gateway/app.py",
    ]
    assert "services/svc_api_gateway/.mcp.json" in helpers["glob_files"](
        "services/svc_api_gateway/**", include_ignored=True
    )


def test_parse_transcript_metrics_sums_all_turns_and_deduplicates_message_ids():
    events = [
        {
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "cache_creation_input_tokens": 500,
                    "cache_read_input_tokens": 0,
                },
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "mcp__rlm-tools__rlm_start", "input": {}},
                ],
            },
        },
        # Duplicate chunk for msg_1 with updated output count should replace, not double-count.
        {
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 40,
                    "cache_creation_input_tokens": 500,
                    "cache_read_input_tokens": 0,
                },
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "mcp__rlm-tools__rlm_start", "input": {}},
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "[BUDGET_EXHAUSTED] partial answer"},
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "id": "msg_2",
                "role": "assistant",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 600,
                },
                "content": [
                    {"type": "tool_use", "id": "tu_2", "name": "mcp__rlm-tools__rlm_execute", "input": {"code": "res = rlm_recurse_batched(prompts)"}},
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": json.dumps({
                            "result": json.dumps({
                                "stdout": "done",
                                "error": None,
                                "usage": {
                                    "llm_calls_used": 148,
                                    "child_budget_exhausted": 2,
                                    "sub_llm_usage": {"total_cost_usd": 0.42},
                                },
                            })
                        }),
                    },
                ],
            },
        },
        {
            "type": "result",
            "result": "final text",
            "total_cost_usd": 0.12,
            "duration_ms": 4200,
            "usage": {"input_tokens": 50, "output_tokens": 20, "cache_read_input_tokens": 600},
        },
    ]

    metrics = parse_transcript_metrics(events)
    assert metrics["turns_counted"] == 2
    assert metrics["input_tokens"] == 150
    assert metrics["output_tokens"] == 60
    assert metrics["cache_creation_input_tokens"] == 500
    assert metrics["cache_read_input_tokens"] == 600
    assert metrics["total_tokens"] == 1310
    assert metrics["rlm_start_count"] == 1
    assert metrics["rlm_execute_count"] == 1
    assert metrics["rlm_recurse_batched_calls"] == 1
    assert metrics["llm_calls_used"] == 148
    assert metrics["child_budget_exhausted_count"] == 2
    assert metrics["sub_llm_cost_usd"] == pytest.approx(0.42)
    assert metrics["second_session_opened"] is False
    assert metrics["llm_cap_hit"] is False
    assert metrics["child_budget_exhausted_seen"] is True
    assert metrics["total_cost_usd"] == pytest.approx(0.12)
    assert metrics["result_text"] == "final text"


def test_score_audit_findings_recall_and_false_positives():
    planted = PLANTED_BUG_SPECS
    reported_payload = {
        "contradictions": [
            {"tool": bug["item"], "file_line": "f:1", "claim": "c", "actual": "a"}
            for bug in planted[:6]
        ]
        + [{"tool": "get_payroll_summary", "file_line": "f:2", "claim": "c", "actual": "a"}]
    }
    output = f"Findings:\n```json\n{json.dumps(reported_payload)}\n```"

    scored = score_audit_findings(output, planted)
    assert scored["recall_count"] == 6
    assert scored["total_bugs"] == 7
    assert scored["passed_target"] is True
    assert scored["missed"] == [planted[6]["item"]]
    assert scored["false_positives"] == ["get_payroll_summary"]


def test_oss_realism_prefect_spec_and_patch_valid():
    oss_dir = REPO_ROOT / "evals" / "oss_realism"
    spec = json.loads((oss_dir / "prefect_spec.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", spec["commit_sha"])
    assert len(spec["planted_bugs"]) == 7
    assert spec["endpoint_count"] == 50

    patch_path = oss_dir / spec["patch_file"]
    assert patch_path.is_file()
    patch_text = patch_path.read_text(encoding="utf-8")
    assert "diff --git " in patch_text
    for bug in spec["planted_bugs"]:
        assert bug["file"] in patch_text
        assert f"def {bug['item']}(" in patch_text
