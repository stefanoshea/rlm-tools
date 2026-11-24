"""Offline tests for the heavy corpus generator, native tools, and Gemini A/B runner."""

from __future__ import annotations

import asyncio
from collections import Counter
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from evals.grade_eval import grade_eval_case, load_eval_specs
from evals.heavy_corpus import generate_heavy_corpus
from evals.run_gemini_ab_eval import (
    DEFAULT_CONCURRENCY,
    NativeFileTools,
    _format_api_error,
    _is_transient_api_error,
    _retry_after_seconds,
    open_stdio_rlm_client,
    load_gemini_api_key,
    run_gemini_agent,
    sanitize_schema_for_gemini,
)
from rlm_tools.sandbox import Sandbox
from rlm_tools.server import mcp


def _independent_ground_truth(root: Path) -> dict[str, dict]:
    """Independently re-parse every file on disk to cross-check heavy_corpus."""
    valid_flags = set(json.loads((root / "config" / "flags.json").read_text(encoding="utf-8")).keys())

    active_count = 0
    active_retry_sum = 0
    team_counts: Counter[str] = Counter()
    slowest_service = ""
    slowest_timeout = -1
    unknown_flags: list[dict[str, str]] = []
    routes: dict[str, list[str]] = {}
    util_importers: Counter[str] = Counter()

    for svc_file in sorted((root / "services").glob("svc_*.py")):
        rel = svc_file.relative_to(root).as_posix()
        text = svc_file.read_text(encoding="utf-8")
        team = re.search(r'^OWNER_TEAM = "([^"]+)"', text, re.M).group(1)
        deprecated = re.search(r"^DEPRECATED = (True|False)", text, re.M).group(1) == "True"
        retry_limit = int(re.search(r"^RETRY_LIMIT = (\d+)", text, re.M).group(1))
        timeout_ms = int(re.search(r"^TIMEOUT_MS = (\d+)", text, re.M).group(1))
        flag = re.search(r'^FEATURE_FLAG = "([^"]+)"', text, re.M).group(1)
        route = re.search(r'^@route\("([^"]+)"\)', text, re.M).group(1)

        if not deprecated:
            active_count += 1
            active_retry_sum += retry_limit
            team_counts[team] += 1
            if timeout_ms > slowest_timeout:
                slowest_timeout = timeout_ms
                slowest_service = rel
        if flag not in valid_flags:
            unknown_flags.append({"file": rel, "flag": flag})
        routes.setdefault(route, []).append(rel)
        for mod in set(re.findall(r"^from (shared\.util_\d+) import ", text, re.M)):
            util_importers[mod] += 1

    violations: list[dict[str, str]] = []
    for mod_file in sorted((root / "modules").glob("engine_*.py")):
        rel = mod_file.relative_to(root).as_posix()
        text = mod_file.read_text(encoding="utf-8")
        for mod in set(re.findall(r"^from (shared\.util_\d+) import ", text, re.M)):
            util_importers[mod] += 1

        lines = text.splitlines()
        audited = False
        current_fn: str | None = None
        for line in lines:
            if line.strip() == "@audit_required":
                audited = True
                continue
            fn_match = re.match(r"^def (task_\d+_\d+)\(", line)
            if fn_match:
                current_fn = fn_match.group(1) if audited else None
                audited = False
                continue
            if line and not line.startswith((" ", "\t")):
                audited = False
                current_fn = None
                continue
            if current_fn is not None:
                code_part = line.split("#", 1)[0]
                if "legacy_db_query(" in code_part:
                    violations.append({"file": rel, "function": current_fn})
                    current_fn = None

    entrypoint_text = (root / "entrypoint.py").read_text(encoding="utf-8")
    assert "from router import dispatch_billing" in entrypoint_text
    router_text = (root / "router.py").read_text(encoding="utf-8")
    handler_match = re.search(r"^from (handlers\.\w+) import (\w+)", router_text, re.M)
    handler_file = root / (handler_match.group(1).replace(".", "/") + ".py")
    handler_class = handler_match.group(2)
    handler_text = handler_file.read_text(encoding="utf-8")
    policy_match = re.search(r"^from (policies\.\w+) import (\w+)", handler_text, re.M)
    policy_rel = policy_match.group(1).replace(".", "/") + ".py"
    policy_constant = policy_match.group(2)
    policy_text = (root / policy_rel).read_text(encoding="utf-8")
    limit_match = re.search(
        rf'^{policy_constant}: int = get_limit\("([^"]+)", (\d+)\)',
        policy_text,
        re.M,
    )
    config_key = limit_match.group(1)
    fallback_limit = int(limit_match.group(2))
    yaml_section, yaml_key = config_key.split(".", 1)
    yaml_text = (root / "config" / "limits.yaml").read_text(encoding="utf-8")
    sec_block = yaml_text.split(f"{yaml_section}:\n", 1)[1].split("\n\n", 1)[0]
    configured_limit = int(re.search(rf"^\s+{re.escape(yaml_key)}:\s*(\d+)", sec_block, re.M).group(1))

    return {
        "broad_service_inventory": {
            "active_count": active_count,
            "active_retry_sum": active_retry_sum,
            "top_3_teams": [t for t, _ in team_counts.most_common(3)],
            "slowest_active_service": slowest_service,
            "slowest_timeout_ms": slowest_timeout,
        },
        "flag_and_route_audit": {
            "unknown_flag_services": sorted(unknown_flags, key=lambda x: x["file"]),
            "duplicate_routes": {k: sorted(v) for k, v in sorted(routes.items()) if len(v) > 1},
        },
        "large_module_audit": {
            "violations": sorted(violations, key=lambda x: (x["file"], x["function"])),
        },
        "shared_utility_fanin": {
            "top_5_utilities": [
                {"module": mod, "importers": cnt}
                for mod, cnt in util_importers.most_common(5)
            ],
        },
        "targeted_billing_trace": {
            "handler_class": handler_class,
            "policy_constant": policy_constant,
            "config_key": config_key,
            "configured_limit": configured_limit,
            "fallback_limit": fallback_limit,
            "policy_file": policy_rel,
        },
    }


def test_heavy_corpus_ground_truth_and_grading(tmp_path):
    corpus_dir = tmp_path / "heavy"
    spec_data = generate_heavy_corpus(corpus_dir)
    spec_path = tmp_path / "heavy_spec.json"
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")

    specs = load_eval_specs(spec_path)
    assert len(specs) == 5

    expected = _independent_ground_truth(corpus_dir)
    assert len(expected["large_module_audit"]["violations"]) == 7
    assert len(expected["flag_and_route_audit"]["unknown_flag_services"]) == 3
    assert len(expected["flag_and_route_audit"]["duplicate_routes"]) == 2

    for spec in specs:
        candidate = json.dumps(expected[spec.eval_id])
        graded = grade_eval_case(spec, candidate)
        assert graded["score"] == 1.0, (spec.eval_id, graded)


def test_sandbox_solves_all_heavy_cases_within_limits(tmp_path):
    corpus_dir = tmp_path / "heavy_rlm"
    spec_data = generate_heavy_corpus(corpus_dir)
    spec_path = tmp_path / "heavy_spec.json"
    spec_path.write_text(json.dumps(spec_data), encoding="utf-8")
    specs = {s.eval_id: s for s in load_eval_specs(spec_path)}

    sandbox = Sandbox(base_path=str(corpus_dir.resolve()), max_output_chars=50_000)
    try:
        solve_code = """
import json, re
from collections import Counter

flags = set(json.loads(read_file('config/flags.json')).keys())
svc_files = sorted(glob_files('services/*.py'))
svc_contents = read_files(svc_files)

active_count = 0
active_retry_sum = 0
team_counts = Counter()
slowest = (-1, '')
unknown_flags = []
routes = {}
utils = Counter()

for f, text in svc_contents.items():
    team = re.search(r'^OWNER_TEAM = "([^"]+)"', text, re.M).group(1)
    deprecated = re.search(r'^DEPRECATED = (True|False)', text, re.M).group(1) == 'True'
    retry = int(re.search(r'^RETRY_LIMIT = (\\d+)', text, re.M).group(1))
    timeout = int(re.search(r'^TIMEOUT_MS = (\\d+)', text, re.M).group(1))
    flag = re.search(r'^FEATURE_FLAG = "([^"]+)"', text, re.M).group(1)
    route = re.search(r'^@route\\("([^"]+)"\\)', text, re.M).group(1)
    if not deprecated:
        active_count += 1
        active_retry_sum += retry
        team_counts[team] += 1
        if timeout > slowest[0]:
            slowest = (timeout, f)
    if flag not in flags:
        unknown_flags.append({'file': f, 'flag': flag})
    routes.setdefault(route, []).append(f)
    for m in set(re.findall(r'^from (shared\\.util_\\d+) import ', text, re.M)):
        utils[m] += 1

mod_files = sorted(glob_files('modules/*.py'))
mod_contents = read_files(mod_files)
violations = []
for f, text in mod_contents.items():
    for m in set(re.findall(r'^from (shared\\.util_\\d+) import ', text, re.M)):
        utils[m] += 1
    audited = False
    cur_fn = None
    for line in text.splitlines():
        if line.strip() == '@audit_required':
            audited = True
            continue
        m = re.match(r'^def (task_\\d+_\\d+)\\(', line)
        if m:
            cur_fn = m.group(1) if audited else None
            audited = False
            continue
        if line and not line.startswith((' ', '\\t')):
            audited = False
            cur_fn = None
            continue
        if cur_fn is not None and 'legacy_db_query(' in line.split('#', 1)[0]:
            violations.append({'file': f, 'function': cur_fn})
            cur_fn = None

ep = read_file('entrypoint.py')
r_mod = re.search(r'^from (router) import ', ep, re.M).group(1)
rt = read_file(r_mod + '.py')
hm = re.search(r'^from (handlers\\.\\w+) import (\\w+)', rt, re.M)
ht = read_file(hm.group(1).replace('.', '/') + '.py')
pm = re.search(r'^from (policies\\.\\w+) import (\\w+)', ht, re.M)
pfile = pm.group(1).replace('.', '/') + '.py'
pt = read_file(pfile)
lm = re.search(pm.group(2) + r': int = get_limit\\("([^"]+)", (\\d+)\\)', pt, re.M)
key_parts = lm.group(1).split('.', 1)
yt = read_file('config/limits.yaml')
sec = yt.split(key_parts[0] + ':\\n', 1)[1]
val = int(re.search(r'^\\s+' + key_parts[1] + r':\\s*(\\d+)', sec, re.M).group(1))

answers = {
    'broad_service_inventory': {
        'active_count': active_count,
        'active_retry_sum': active_retry_sum,
        'top_3_teams': [t for t, _ in team_counts.most_common(3)],
        'slowest_active_service': slowest[1],
        'slowest_timeout_ms': slowest[0],
    },
    'flag_and_route_audit': {
        'unknown_flag_services': sorted(unknown_flags, key=lambda x: x['file']),
        'duplicate_routes': {k: sorted(v) for k, v in sorted(routes.items()) if len(v) > 1},
    },
    'large_module_audit': {
        'violations': sorted(violations, key=lambda x: (x['file'], x['function'])),
    },
    'shared_utility_fanin': {
        'top_5_utilities': [{'module': m, 'importers': c} for m, c in utils.most_common(5)],
    },
    'targeted_billing_trace': {
        'handler_class': hm.group(2),
        'policy_constant': pm.group(2),
        'config_key': lm.group(1),
        'configured_limit': val,
        'fallback_limit': int(lm.group(2)),
        'policy_file': pfile,
    },
}
print(json.dumps(answers))
"""
        res = sandbox.execute(solve_code)
        assert res.error is None, res.error
        answers = json.loads(res.stdout)
        for eval_id, ans in answers.items():
            assert grade_eval_case(specs[eval_id], json.dumps(ans))["score"] == 1.0
    finally:
        sandbox.close()


def test_native_file_tools_behavior_and_security(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "\n".join(f"line_{i} = {i}" for i in range(1, 26)) + "\n", encoding="utf-8"
    )
    (tmp_path / "pkg" / "other.txt").write_text("hello world\nline_10\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=12345\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret_git\n", encoding="utf-8")
    outside = tmp_path.parent / f"outside_{tmp_path.name}.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (tmp_path / "escape_link.txt").symlink_to(outside)

        tools = NativeFileTools(tmp_path)
        text, err = tools.read_file("pkg/mod.py", start_line=5, max_lines=3)
        assert not err
        assert "L5: line_5 = 5" in text
        assert "L7: line_7 = 7" in text
        assert "showing lines 5-7 of 25" in text

        grep_out, err = tools.grep(r"line_1\b", path=".")
        assert not err
        assert "pkg/mod.py:1: line_1 = 1" in grep_out
        assert ".env" not in grep_out
        assert ".git" not in grep_out

        files_only_out, err = tools.grep("line_10", files_only=True)
        assert not err
        assert files_only_out.splitlines() == ["pkg/mod.py", "pkg/other.txt"]

        glob_out, err = tools.glob_files("**/*.py")
        assert not err
        assert glob_out == "pkg/mod.py"

        for bad_call in (
            ("read_file", {"path": "../secret"}),
            ("read_file", {"path": ".env"}),
            ("read_file", {"path": "escape_link.txt"}),
            ("grep", {"pattern": "x", "path": ".git"}),
            ("glob_files", {"pattern": "../*.py"}),
        ):
            msg, is_err = tools.call_tool(*bad_call)
            assert is_err, bad_call
            assert "Error:" in msg
    finally:
        outside.unlink(missing_ok=True)


def test_sanitize_schema_for_gemini():
    for tool in mcp._tool_manager.list_tools():
        cleaned = sanitize_schema_for_gemini(tool.parameters)
        serialized = json.dumps(cleaned)
        for forbidden in ('"title"', '"default"', '"additionalProperties"', '"anyOf"', '"$defs"'):
            assert forbidden not in serialized

    complex_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"Item": {"type": "string", "title": "ItemTitle"}},
        "type": "object",
        "title": "Root",
        "additionalProperties": False,
        "properties": {
            "count": {
                "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}],
                "default": None,
                "description": "Optional count",
            },
            "tags": {
                "type": "array",
                "items": {"$ref": "#/$defs/Item"},
            },
        },
        "required": ["tags"],
    }
    cleaned = sanitize_schema_for_gemini(complex_schema)
    assert cleaned == {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1, "description": "Optional count"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["tags"],
    }


def test_load_gemini_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_gemini_api_key(tmp_path)

    (tmp_path / ".env").write_text(
        "# comment\nexport GEMINI_API_KEY='from-dotenv' # trailing\n", encoding="utf-8"
    )
    assert load_gemini_api_key(tmp_path) == "from-dotenv"

    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert load_gemini_api_key(tmp_path) == "from-env"


@pytest.mark.asyncio
async def test_stdio_rlm_client_sanitizes_env_and_runs_session(tmp_path, monkeypatch):
    (tmp_path / "sample.py").write_text("VALUE = 42\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("VALUE = 99\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")

    async with open_stdio_rlm_client(tmp_path) as client:
        decls = client.tool_declarations()
        assert {d["name"] for d in decls} == {"rlm_start", "rlm_execute", "rlm_end"}

        async def run_one(filename: str) -> str:
            start_txt, err = await client.call_tool(
                "rlm_start", {"path": str(tmp_path.resolve()), "query": filename}
            )
            assert not err
            sid = json.loads(start_txt)["session_id"]

            exec_txt, err = await client.call_tool(
                "rlm_execute",
                {"session_id": sid, "code": f"print(read_file({filename!r}).strip())"},
            )
            assert not err
            out = json.loads(exec_txt)["stdout"]

            end_txt, err = await client.call_tool("rlm_end", {"session_id": sid})
            assert not err and json.loads(end_txt)["success"]
            return out

        out1, out2 = await asyncio.gather(run_one("sample.py"), run_one("other.py"))
        assert out1 == "VALUE = 42\n"
        assert out2 == "VALUE = 99\n"

        # rlm_start(code=...) runs initial code and keeps state for rlm_execute(..., end_session=True).
        start2_txt, err = await client.call_tool(
            "rlm_start",
            {
                "path": str(tmp_path.resolve()),
                "query": "two-step",
                "code": "first = read_file('sample.py').strip()\nprint(first)",
            },
        )
        assert not err
        start2 = json.loads(start2_txt)
        assert start2["stdout"] == "VALUE = 42\n"
        sid2 = start2["session_id"]
        exec2_txt, err = await client.call_tool(
            "rlm_execute",
            {
                "session_id": sid2,
                "code": "print(first, read_file('other.py').strip())",
                "end_session": True,
            },
        )
        assert not err
        exec2 = json.loads(exec2_txt)
        assert exec2["stdout"] == "VALUE = 42 VALUE = 99\n"
        assert exec2["session_ended"] is True

        closed_txt, _ = await client.call_tool(
            "rlm_execute", {"session_id": sid2, "code": "print(1)"}
        )
        assert "not found" in json.loads(closed_txt)["error"]


@pytest.mark.asyncio
async def test_run_gemini_agent_tool_loop_and_transient_retries():
    calls_made = []

    class Transient429(Exception):
        status_code = 429

    class FakeInteractions:
        def __init__(self):
            self.n = 0

        def create(self, **kwargs):
            calls_made.append(kwargs)
            self.n += 1
            if self.n == 1:
                raise Transient429("rate limited")
            if self.n == 2:
                raise TimeoutError("stalled request")
            if self.n == 3:
                return SimpleNamespace(
                    id="ix-1",
                    status="requires_action",
                    output_text=None,
                    usage=SimpleNamespace(
                        total_input_tokens=100,
                        total_cached_tokens=20,
                        total_output_tokens=15,
                        total_thought_tokens=5,
                        total_tokens=120,
                    ),
                    steps=[
                        SimpleNamespace(
                            type="function_call",
                            id="fc-1",
                            name="read_file",
                            arguments={"path": "a.py"},
                        ),
                        SimpleNamespace(
                            type="function_call",
                            id="fc-2",
                            name="read_file",
                            arguments={"path": "b.py"},
                        ),
                    ],
                )
            return SimpleNamespace(
                id="ix-2",
                status="completed",
                output_text='{"ok": true}',
                usage=SimpleNamespace(
                    total_input_tokens=160,
                    total_cached_tokens=80,
                    total_output_tokens=10,
                    total_thought_tokens=0,
                    total_tokens=170,
                ),
                steps=[],
            )

    fake_client = SimpleNamespace(interactions=FakeInteractions())

    def fake_exec(name, args):
        return f"contents of {args['path']}", False

    res = await run_gemini_agent(
        client=fake_client,
        model="gemini-4-hydrogen",
        prompt="Inspect files",
        system_instruction="Be concise.",
        tools=[{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
        tool_executor=fake_exec,
        retry_delays=(0.0, 0.0, 0.0),
        request_timeout=42.5,
    )

    assert res["output"] == '{"ok": true}'
    assert res["turns"] == 2
    assert res["tool_calls"] == 2
    assert res["input_tokens"] == 260
    assert res["cached_tokens"] == 100
    assert res["uncached_input_tokens"] == 160
    assert res["output_tokens"] == 25
    assert res["thought_tokens"] == 5
    assert res["total_tokens"] == 290
    assert res["peak_input_tokens"] == 160
    assert not res["hit_max_turns"]
    assert len(calls_made) == 4
    assert all(call["timeout"] == 42.5 for call in calls_made)
    assert calls_made[3]["previous_interaction_id"] == "ix-1"
    assert calls_made[3]["system_instruction"] == "Be concise."
    assert len(calls_made[3]["input"]) == 2


def test_api_error_classification_and_retry_after():
    assert DEFAULT_CONCURRENCY == 1

    class ApiErr(Exception):
        def __init__(self, status_code, body="", headers=None):
            super().__init__(body)
            self.status_code = status_code
            self.body = body
            self.headers = headers or {}

    daily_429 = ApiErr(
        429,
        "Rate limit exceeded for model gemini-4-hydrogen (limit: 20 requests per day on Free Tier). "
        "Please retry in 21h19m54s.",
    )
    assert not _is_transient_api_error(daily_429)
    assert _retry_after_seconds(daily_429) is None
    assert "status=429" in _format_api_error(daily_429)

    rpm_429 = ApiErr(429, "Rate limit exceeded per minute. Please retry in 14.5s.")
    assert _is_transient_api_error(rpm_429)
    assert _retry_after_seconds(rpm_429) == 14.5

    header_503 = ApiErr(503, "Service unavailable", headers={"retry-after": "9"})
    assert _is_transient_api_error(header_503)
    assert _retry_after_seconds(header_503) == 9.0
    assert _is_transient_api_error(TimeoutError("timed out"))
