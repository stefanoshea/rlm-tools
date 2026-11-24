"""Deterministic heavy/broad-aggregation Python corpus and eval spec generator."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any


TEAMS = (
    "checkout",
    "ledger",
    "fulfillment",
    "identity",
    "notifications",
    "analytics",
    "search",
    "compliance",
)
TEAM_WEIGHTS = (34, 26, 20, 14, 10, 7, 5, 3)

UTILITIES = tuple(f"shared.util_{i:02d}" for i in range(12))
UTILITY_WEIGHTS = (42, 34, 27, 21, 16, 11, 8, 6, 4, 3, 2, 1)

UNKNOWN_FLAG_OVERRIDES = {
    17: "flag_ghost_alpha",
    89: "flag_ghost_beta",
    163: "flag_ghost_gamma",
}

DUPLICATE_ROUTE_OVERRIDES = {
    12: "/api/v1/orders/export",
    144: "/api/v1/orders/export",
    45: "/api/v1/billing/sync",
    178: "/api/v1/billing/sync",
}

# Exactly 7 (module_idx, func_idx) targets where an @audit_required function
# calls legacy_db_query(...) on an executable code line.
AUDIT_VIOLATION_TARGETS = {
    (0, 12),
    (2, 7),
    (3, 31),
    (5, 19),
    (6, 4),
    (8, 28),
    (9, 40),
}


def _weighted_sample_without_replacement(
    rng: random.Random,
    items: tuple[str, ...],
    weights: tuple[int, ...],
    k: int,
) -> list[str]:
    pool = list(zip(items, weights))
    chosen: list[str] = []
    for _ in range(k):
        total = sum(w for _, w in pool)
        pick = rng.uniform(0, total)
        upto = 0.0
        for idx, (item, weight) in enumerate(pool):
            upto += weight
            if upto >= pick or idx == len(pool) - 1:
                chosen.append(item)
                pool.pop(idx)
                break
    return chosen


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_utility_module(idx: int) -> str:
    return "\n".join([
        f'"""Shared utility module util_{idx:02d}."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "",
        f"def helper_{idx:02d}(payload: dict[str, Any]) -> dict[str, Any]:",
        f'    """Normalize payload metadata for utility {idx:02d}."""',
        "    normalized = dict(payload)",
        f'    normalized["util_tag"] = "util_{idx:02d}"',
        "    return normalized",
        "",
        "",
        f"def validate_{idx:02d}(record_id: str, limit: int = 100) -> bool:",
        f"    return bool(record_id) and 0 < limit <= 10_000",
        "",
    ])


def _render_service_file(
    idx: int,
    *,
    owner_team: str,
    deprecated: bool,
    retry_limit: int,
    timeout_ms: int,
    feature_flag: str,
    route_path: str,
    imported_utils: list[str],
) -> str:
    import_lines = [
        f"from {mod} import helper_{mod.split('_')[-1]}"
        for mod in sorted(imported_utils)
    ]
    helper_calls = [
        f"    ctx = helper_{mod.split('_')[-1]}(ctx)"
        for mod in sorted(imported_utils)
    ]

    padding_blocks: list[str] = []
    for step in range(1, 5):
        padding_blocks.extend([
            f"def _stage_{step}_transform_{idx:03d}(records: list[dict], threshold: int) -> list[dict]:",
            f'    """Stage {step} deterministic payload filter for service {idx:03d}."""',
            "    output = []",
            "    for position, item in enumerate(records):",
            "        weight = int(item.get('weight', 1)) + position",
            f"        if weight >= (threshold % {step + 7}):",
            "            enriched = {",
            '                "service": f"svc_' + f'{idx:03d}' + '",',
            f'                "stage": {step},',
            '                "record_id": item.get("id", "unknown"),',
            '                "score": weight * 2,',
            f'                "team": "{owner_team}",',
            "            }",
            "            output.append(enriched)",
            "    return output",
            "",
        ])

    lines = [
        f'"""Service module svc_{idx:03d} owned by {owner_team}."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        *import_lines,
        "",
        "",
        "def route(path: str):",
        "    def decorator(fn):",
        "        fn._route = path",
        "        return fn",
        "    return decorator",
        "",
        "",
        f'OWNER_TEAM = "{owner_team}"',
        f"DEPRECATED = {deprecated}",
        f"RETRY_LIMIT = {retry_limit}",
        f"TIMEOUT_MS = {timeout_ms}",
        f'FEATURE_FLAG = "{feature_flag}"',
        "",
        "",
        f'@route("{route_path}")',
        f"def handle_svc_{idx:03d}(request: dict[str, Any]) -> dict[str, Any]:",
        '    """Primary endpoint handler for this microservice."""',
        '    ctx: dict[str, Any] = {"team": OWNER_TEAM, "flag": FEATURE_FLAG, "input": request}',
        *helper_calls,
        "    records = request.get('records', [])",
        "    stage1 = _stage_1_transform_" + f"{idx:03d}(records, RETRY_LIMIT)",
        "    stage2 = _stage_2_transform_" + f"{idx:03d}(stage1, TIMEOUT_MS)",
        "    stage3 = _stage_3_transform_" + f"{idx:03d}(stage2, RETRY_LIMIT + 1)",
        "    stage4 = _stage_4_transform_" + f"{idx:03d}(stage3, TIMEOUT_MS // 10)",
        "    return {",
        f'        "service": "svc_{idx:03d}",',
        '        "deprecated": DEPRECATED,',
        '        "count": len(stage4),',
        '        "ctx": ctx,',
        "    }",
        "",
        "",
        *padding_blocks,
    ]
    return "\n".join(lines)


def _render_engine_module(
    mod_idx: int,
    *,
    imported_utils: list[str],
    rng: random.Random,
    num_functions: int = 45,
) -> tuple[str, list[str]]:
    import_lines = [
        f"from {mod} import helper_{mod.split('_')[-1]}"
        for mod in sorted(imported_utils)
    ]
    primary_helper = f"helper_{sorted(imported_utils)[0].split('_')[-1]}"

    lines = [
        f'"""Execution engine monolith {mod_idx:02d}."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        *import_lines,
        "",
        "",
        "def audit_required(fn):",
        "    fn._audit_required = True",
        "    return fn",
        "",
        "",
        "def legacy_db_query(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:",
        '    """Legacy SQL helper scheduled for deprecation."""',
        '    return [{"sql": sql, "params": params or {}}]',
        "",
        "",
    ]

    violations: list[str] = []
    for fn_idx in range(num_functions):
        fn_name = f"task_{mod_idx:02d}_{fn_idx:02d}"
        is_violation = (mod_idx, fn_idx) in AUDIT_VIOLATION_TARGETS
        # Give deterministic coverage of @audit_required, comment distractors,
        # and non-audited legacy calls across the 10 monoliths.
        is_audited = is_violation or ((mod_idx * 7 + fn_idx) % 7 == 0) or (fn_idx in (2, 14, 27))
        comment_distractor = (not is_violation) and is_audited and ((mod_idx + fn_idx) % 2 == 0)
        unaudited_legacy_call = (not is_audited) and ((mod_idx * 5 + fn_idx) % 10 == 1)

        if is_violation:
            violations.append(fn_name)

        if is_audited:
            lines.append("@audit_required")
        lines.append(f"def {fn_name}(payload: dict[str, Any]) -> dict[str, Any]:")
        lines.append(f'    """Process batch workflow item {mod_idx:02d}/{fn_idx:02d}."""')
        lines.append(f"    state = {primary_helper}(payload)")
        lines.append("    items = list(state.get('items', []))")
        lines.append("    accumulator = 0")
        lines.append("    checksum = 17")
        lines.append("    for offset in range(1, 6):")
        lines.append(f"        factor = (offset + {fn_idx + 1}) * {(mod_idx + 2)}")
        lines.append("        accumulator += factor")
        lines.append("        checksum = (checksum * 31 + factor) % 1000003")

        if comment_distractor:
            lines.append("    # TODO(cleanup): formerly called legacy_db_query('SELECT * FROM jobs') here")
            lines.append("    # Removed direct legacy_db_query(table='audit_log') during migration")

        if is_violation:
            lines.append(
                f"    rows = legacy_db_query('SELECT id FROM audit_events WHERE mod = {mod_idx}', payload)"
            )
            lines.append("    accumulator += len(rows)")
        elif unaudited_legacy_call:
            lines.append(
                f"    rows = legacy_db_query('SELECT id FROM temp_events WHERE slot = {fn_idx}', payload)"
            )
            lines.append("    accumulator += len(rows)")
        else:
            lines.append("    rows = [{'id': i, 'score': accumulator + i} for i in range(len(items))]")
            lines.append("    accumulator += len(rows)")

        for block_i in range(4):
            lines.extend([
                f"    bucket_{block_i} = (accumulator + checksum + {block_i * 13 + rng.randint(1, 9)}) % 997",
                f"    if bucket_{block_i} > 500:",
                f"        state['flag_{block_i}'] = bucket_{block_i}",
                "    else:",
                f"        state['flag_{block_i}'] = bucket_{block_i} + 100",
            ])

        lines.extend([
            "    return {",
            f'        "task": "{fn_name}",',
            '        "accumulator": accumulator,',
            '        "checksum": checksum,',
            '        "state": state,',
            "    }",
            "",
            "",
        ])

    return "\n".join(lines), violations


def _write_billing_chain(root: Path) -> None:
    _write(
        root / "config" / "limits.yaml",
        "\n".join([
            "# Service rate limits",
            "billing:",
            "  v1_rate_limit_per_min: 100",
            "  v2_rate_limit_per_min: 420",
            "payments:",
            "  default_rate_limit_per_min: 60",
            "",
        ]),
    )
    _write(
        root / "policies" / "rate_limits.py",
        "\n".join([
            '"""Rate-limit policy constants loaded from config/limits.yaml."""',
            "",
            "from __future__ import annotations",
            "",
            "from pathlib import Path",
            "",
            "",
            "def _load_yaml_limits() -> dict[str, dict[str, int]]:",
            '    cfg_path = Path(__file__).resolve().parents[1] / "config" / "limits.yaml"',
            '    sections: dict[str, dict[str, int]] = {}',
            "    current_section: str | None = None",
            '    for raw_line in cfg_path.read_text(encoding="utf-8").splitlines():',
            '        stripped = raw_line.strip()',
            '        if not stripped or stripped.startswith("#"):',
            "            continue",
            '        if not raw_line.startswith(" ") and stripped.endswith(":"):',
            "            current_section = stripped[:-1]",
            "            sections.setdefault(current_section, {})",
            '        elif current_section and ":" in stripped:',
            '            key, value = stripped.split(":", 1)',
            "            sections[current_section][key.strip()] = int(value.strip())",
            "    return sections",
            "",
            "",
            "def get_limit(dotted_key: str, default: int) -> int:",
            '    section, key = dotted_key.split(".", 1)',
            "    return _load_yaml_limits().get(section, {}).get(key, default)",
            "",
            "",
            'BILLING_V1_RATE_LIMIT: int = get_limit("billing.v1_rate_limit_per_min", 90)',
            'BILLING_V2_RATE_LIMIT: int = get_limit("billing.v2_rate_limit_per_min", 150)',
            'PAYMENTS_DEFAULT_RATE_LIMIT: int = get_limit("payments.default_rate_limit_per_min", 30)',
            "",
        ]),
    )
    _write(
        root / "handlers" / "billing_v1.py",
        "\n".join([
            '"""Deprecated v1 billing handler (kept only for reference)."""',
            "",
            "from policies.rate_limits import BILLING_V1_RATE_LIMIT",
            "",
            "",
            "class BillingV1Handler:",
            "    rate_limit = BILLING_V1_RATE_LIMIT",
            "",
            "    def process(self, account_id: str) -> dict:",
            '        return {"handler": "v1", "account_id": account_id, "limit": self.rate_limit}',
            "",
        ]),
    )
    _write(
        root / "handlers" / "billing_v2.py",
        "\n".join([
            '"""Active v2 billing handler."""',
            "",
            "from policies.rate_limits import BILLING_V2_RATE_LIMIT",
            "",
            "",
            "class BillingV2Handler:",
            "    rate_limit = BILLING_V2_RATE_LIMIT",
            "",
            "    def process(self, account_id: str) -> dict:",
            '        return {"handler": "v2", "account_id": account_id, "limit": self.rate_limit}',
            "",
        ]),
    )
    _write(
        root / "router.py",
        "\n".join([
            '"""Active request router."""',
            "",
            "# Note: BillingV1Handler was retired in Q1; do not route new traffic to v1.",
            "from handlers.billing_v2 import BillingV2Handler",
            "",
            "",
            "def dispatch_billing(account_id: str) -> dict:",
            "    handler = BillingV2Handler()",
            "    return handler.process(account_id)",
            "",
        ]),
    )
    _write(
        root / "entrypoint.py",
        "\n".join([
            '"""Application entrypoint."""',
            "",
            "from router import dispatch_billing",
            "",
            "",
            "def run_billing_cycle(account_id: str) -> dict:",
            "    return dispatch_billing(account_id)",
            "",
        ]),
    )
    _write(
        root / "docs" / "legacy_billing.md",
        "\n".join([
            "# Legacy Billing Notes",
            "",
            "Historically, traffic hit `BillingV1Handler` with `BILLING_V1_RATE_LIMIT`",
            "(key `billing.v1_rate_limit_per_min`, fallback 90).",
            "Do not confuse that legacy flow with the active `entrypoint.py` path.",
            "",
        ]),
    )


def generate_heavy_corpus(dest: Path, *, seed: int = 42) -> dict[str, Any]:
    """Generate a deterministic ~1.5 MB Python corpus and exact-graded eval spec."""
    root = dest.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # 1. Feature flags registry (20 known flags)
    flags_payload = {f"flag_{i:02d}": (i % 3 != 0) for i in range(20)}
    _write(root / "config" / "flags.json", json.dumps(flags_payload, indent=2) + "\n")

    # 2. Shared utility modules
    for i in range(12):
        _write(root / "shared" / f"util_{i:02d}.py", _render_utility_module(i))

    # 3. 200 service files
    active_count = 0
    active_retry_sum = 0
    active_team_counts: Counter[str] = Counter()
    active_timeouts: list[tuple[int, str]] = []
    unknown_flag_services: list[dict[str, str]] = []
    route_to_files: dict[str, list[str]] = {}
    utility_importers: Counter[str] = Counter()

    # Reserve strictly decreasing unique timeouts so slowest_active_service has no ties.
    timeout_pool = [1200 + i * 25 for i in range(200)]
    rng.shuffle(timeout_pool)

    for idx in range(200):
        rel_path = f"services/svc_{idx:03d}.py"
        owner_team = rng.choices(TEAMS, weights=TEAM_WEIGHTS, k=1)[0]
        deprecated = rng.random() < 0.20
        retry_limit = rng.randint(1, 15)
        timeout_ms = timeout_pool[idx]
        feature_flag = UNKNOWN_FLAG_OVERRIDES.get(idx, f"flag_{rng.randint(0, 19):02d}")
        route_path = DUPLICATE_ROUTE_OVERRIDES.get(idx, f"/api/v1/svc_{idx:03d}/run")
        num_utils = rng.randint(1, 4)
        imported_utils = _weighted_sample_without_replacement(
            rng, UTILITIES, UTILITY_WEIGHTS, num_utils
        )

        _write(
            root / rel_path,
            _render_service_file(
                idx,
                owner_team=owner_team,
                deprecated=deprecated,
                retry_limit=retry_limit,
                timeout_ms=timeout_ms,
                feature_flag=feature_flag,
                route_path=route_path,
                imported_utils=imported_utils,
            ),
        )

        if not deprecated:
            active_count += 1
            active_retry_sum += retry_limit
            active_team_counts[owner_team] += 1
            active_timeouts.append((timeout_ms, rel_path))
        if feature_flag not in flags_payload:
            unknown_flag_services.append({"file": rel_path, "flag": feature_flag})
        route_to_files.setdefault(route_path, []).append(rel_path)
        for mod in imported_utils:
            utility_importers[mod] += 1

    # 4. 10 large monolith modules
    violations: list[dict[str, str]] = []
    for mod_idx in range(10):
        rel_path = f"modules/engine_{mod_idx:02d}.py"
        num_utils = rng.randint(2, 3)
        imported_utils = _weighted_sample_without_replacement(
            rng, UTILITIES, UTILITY_WEIGHTS, num_utils
        )
        source, mod_violations = _render_engine_module(
            mod_idx, imported_utils=imported_utils, rng=rng
        )
        _write(root / rel_path, source)
        for mod in imported_utils:
            utility_importers[mod] += 1
        for fn_name in mod_violations:
            violations.append({"file": rel_path, "function": fn_name})

    # 5. Multi-hop billing trace chain
    _write_billing_chain(root)

    # Assert strict uniqueness for ordered tie-free expected answers.
    team_ranking = active_team_counts.most_common()
    assert len(team_ranking) >= 4
    top_3_counts = [count for _, count in team_ranking[:3]]
    assert len(set(top_3_counts)) == 3 and team_ranking[2][1] > team_ranking[3][1], (
        f"Team ranking tie detected: {team_ranking}"
    )

    active_timeouts.sort(reverse=True)
    assert len(active_timeouts) >= 2 and active_timeouts[0][0] > active_timeouts[1][0]
    slowest_timeout_ms, slowest_active_service = active_timeouts[0]

    util_ranking = utility_importers.most_common()
    assert len(util_ranking) >= 6
    top_5_counts = [count for _, count in util_ranking[:5]]
    assert len(set(top_5_counts)) == 5 and util_ranking[4][1] > util_ranking[5][1], (
        f"Utility fan-in tie detected: {util_ranking}"
    )

    unknown_flag_services.sort(key=lambda item: item["file"])
    duplicate_routes = {
        route: sorted(files)
        for route, files in sorted(route_to_files.items())
        if len(files) > 1
    }
    violations.sort(key=lambda item: (item["file"], item["function"]))

    expected_answers = {
        "broad_service_inventory": {
            "active_count": active_count,
            "active_retry_sum": active_retry_sum,
            "top_3_teams": [team for team, _ in team_ranking[:3]],
            "slowest_active_service": slowest_active_service,
            "slowest_timeout_ms": slowest_timeout_ms,
        },
        "flag_and_route_audit": {
            "unknown_flag_services": unknown_flag_services,
            "duplicate_routes": duplicate_routes,
        },
        "large_module_audit": {
            "violations": violations,
        },
        "shared_utility_fanin": {
            "top_5_utilities": [
                {"module": mod, "importers": count}
                for mod, count in util_ranking[:5]
            ],
        },
        "targeted_billing_trace": {
            "handler_class": "BillingV2Handler",
            "policy_constant": "BILLING_V2_RATE_LIMIT",
            "config_key": "billing.v2_rate_limit_per_min",
            "configured_limit": 420,
            "fallback_limit": 150,
            "policy_file": "policies/rate_limits.py",
        },
    }

    eval_cases = [
        {
            "eval_id": "broad_service_inventory",
            "prompt": (
                "In the codebase at `{target_path}`, inspect all Python service files under `services/`. "
                "Consider only non-deprecated services (`DEPRECATED = False`). Compute: "
                "the count of non-deprecated services, the sum of their `RETRY_LIMIT` integers, "
                "the top 3 `OWNER_TEAM` names by non-deprecated service count (from highest to lowest), "
                "and the non-deprecated service with the largest `TIMEOUT_MS`. "
                "Return a JSON object with exactly these five fields: "
                "`active_count` (integer), `active_retry_sum` (integer), `top_3_teams` (array of 3 strings), "
                "`slowest_active_service` (repository-relative path, e.g. `services/svc_000.py`), and "
                "`slowest_timeout_ms` (integer)."
            ),
            "assertions": [
                {
                    "id": "complete_inventory",
                    "kind": "json_equals",
                    "values": [json.dumps(expected_answers["broad_service_inventory"])],
                }
            ],
        },
        {
            "eval_id": "flag_and_route_audit",
            "prompt": (
                "In the codebase at `{target_path}`, audit all service files under `services/` against "
                "`config/flags.json`. Find: (1) every service whose `FEATURE_FLAG` string is not a key in "
                "`config/flags.json`, and (2) every `@route(\"...\")` path declared by more than one service file. "
                "Return a JSON object with exactly two fields: "
                "`unknown_flag_services` (array of `{\"file\": ..., \"flag\": ...}` objects sorted by `file` ascending, "
                "using repository-relative paths) and `duplicate_routes` (object mapping each duplicated route string "
                "to an alphabetically sorted array of repository-relative file paths)."
            ),
            "assertions": [
                {
                    "id": "complete_audit",
                    "kind": "json_equals",
                    "values": [json.dumps(expected_answers["flag_and_route_audit"])],
                }
            ],
        },
        {
            "eval_id": "large_module_audit",
            "prompt": (
                "In the codebase at `{target_path}`, inspect all Python modules under `modules/`. "
                "Find every top-level function decorated with `@audit_required` that calls `legacy_db_query(...)` "
                "on an executable code line inside its body (ignore comments mentioning `legacy_db_query` and ignore "
                "functions without `@audit_required`). Return a JSON object with exactly one field: "
                "`violations` (array of `{\"file\": ..., \"function\": ...}` objects sorted by `file` ascending "
                "then `function` ascending, using repository-relative file paths)."
            ),
            "assertions": [
                {
                    "id": "complete_violations",
                    "kind": "json_equals",
                    "values": [json.dumps(expected_answers["large_module_audit"])],
                }
            ],
        },
        {
            "eval_id": "shared_utility_fanin",
            "prompt": (
                "In the codebase at `{target_path}`, count how many distinct files under `services/` and `modules/` "
                "import each shared utility module (`shared.util_00` through `shared.util_11`). "
                "Return a JSON object with exactly one field: `top_5_utilities` (array of the 5 most-imported "
                "utilities ordered from highest `importers` count to lowest, each an object with `module` string "
                "like `shared.util_00` and `importers` integer)."
            ),
            "assertions": [
                {
                    "id": "complete_fanin",
                    "kind": "json_equals",
                    "values": [json.dumps(expected_answers["shared_utility_fanin"])],
                }
            ],
        },
        {
            "eval_id": "targeted_billing_trace",
            "prompt": (
                "In the codebase at `{target_path}`, start from `entrypoint.py` and trace the active billing "
                "dispatch handler, the rate-limit constant it uses, and the YAML config key and fallback for "
                "that constant. Return a JSON object with exactly these six fields: "
                "`handler_class` (string), `policy_constant` (string), `config_key` (dotted YAML key string), "
                "`configured_limit` (integer), `fallback_limit` (integer), and `policy_file` (repository-relative "
                "path defining the constant)."
            ),
            "assertions": [
                {
                    "id": "complete_trace",
                    "kind": "json_equals",
                    "values": [json.dumps(expected_answers["targeted_billing_trace"])],
                }
            ],
        },
    ]

    return {"evals": eval_cases}
