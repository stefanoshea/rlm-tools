"""Deterministic ~2,900-file synthetic Python monorepo generator and benchmark spec.

Models the structural and semantic failure modes tested in the 0.2.5/0.2.6 benchmarks:
- ~2,900 Python files across services and shared packages
- 50 agent tools split across descriptions, models, helpers, and multi-line handlers
- 7 subtle description-vs-code contradictions (2 inside a >270 KB mega-suite)
- A ~7,400-line monolith module with 150 defs including nested-in-method defs
- 292 worker files with 39 module-level >=150-line functions and 5 nested >=150-line defs
- 120 model files with >=347 class fields and __init__ attributes
- Sandbox & policy traps (.env*, .npmrc, .pypirc, .mcp.json/.gitignore, dot-dirs, >4 MiB file)
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import random
from typing import Any


DOMAINS = (
    "payroll",
    "compliance",
    "directory",
    "leave",
    "incidents",
    "billing",
    "analytics",
    "webhooks",
    "onboarding",
    "scheduling",
)

DOMAIN_TOOLS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "payroll": (
        (
            "list_payroll_runs",
            "List payroll runs sorted newest-first by created_at descending. Supports optional status filter and limit.",
            "Sorts payroll runs ascending (reverse=False) instead of newest-first descending.",
        ),
        (
            "get_payroll_summary",
            "Return aggregate gross pay, tax withholdings, and headcount for a finalized payroll_run_id.",
            "",
        ),
        (
            "validate_bank_routing",
            "Validate a 9-digit ABA routing number checksum without persisting any records.",
            "",
        ),
        (
            "list_pay_periods",
            "Return open and closed pay periods for a pay_schedule_id within start_date and end_date (inclusive).",
            "",
        ),
        (
            "calculate_prorated_salary",
            "Compute prorated base compensation for an employee based on active days in the pay period.",
            "",
        ),
    ),
    "compliance": (
        (
            "search_audit_events",
            "Search compliance audit events matching actor_id or action. Accepts limit from 1 up to 100 records.",
            "Delegated helper clamp_audit_limit silently caps limit at 25 instead of 100.",
        ),
        (
            "get_policy_acknowledgements",
            "Return policy acknowledgement timestamps for a given policy_id and employee_ids list.",
            "",
        ),
        (
            "check_document_retention",
            "Check whether an artifact satisfies the minimum retention_days requirement for its category.",
            "",
        ),
        (
            "list_expiring_certifications",
            "List active employee certifications expiring within horizon_days (default 30).",
            "",
        ),
        (
            "verify_i9_completeness",
            "Inspect an onboarding packet and report missing required verification fields.",
            "",
        ),
    ),
    "directory": (
        (
            "export_team_roster",
            "Export active employee roster rows. department_id is optional and defaults to all departments when omitted.",
            "Handler raises ValueError('department_id is required') whenever department_id is None.",
        ),
        (
            "lookup_employee_profile",
            "Fetch directory profile fields for employee_id, omitting deactivated records unless include_inactive=True.",
            "",
        ),
        (
            "list_reporting_chain",
            "Return the upward manager chain from employee_id to the root executive, stopping on cycle detection.",
            "",
        ),
        (
            "find_department_leads",
            "List designated leads for each department_id in departments.",
            "",
        ),
        (
            "resolve_work_location",
            "Resolve timezone and office address metadata for a given location_code.",
            "",
        ),
    ),
    "leave": (
        (
            "preview_leave_accrual",
            "Read-only dry-run preview of projected leave accrual balances with no side effects or state writes.",
            "Handler mutates state by calling record_accrual_mutation(...) on every invocation.",
        ),
        (
            "list_leave_requests",
            "List leave requests filtered by employee_id and optional status.",
            "",
        ),
        (
            "check_holiday_overlap",
            "Count working days in a leave window excluding configured company holidays.",
            "",
        ),
        (
            "get_leave_policy_caps",
            "Return maximum carryover and annual accrual caps for a leave_type.",
            "",
        ),
        (
            "summarize_team_absences",
            "Summarize approved absences per team_id over a date window.",
            "",
        ),
    ),
    "incidents": (
        (
            "list_open_incidents",
            "List active incidents. When include_archived=False (default), archived incidents are excluded.",
            "Handler accepts include_archived but never filters out status == 'archived' records.",
        ),
        (
            "get_incident_timeline",
            "Return chronological audit timeline entries for incident_id.",
            "",
        ),
        (
            "compute_mttr_stats",
            "Compute mean and p95 time-to-resolution in minutes for resolved incidents.",
            "",
        ),
        (
            "list_oncall_escalations",
            "Return escalation targets ordered by tier ascending for a service_name.",
            "",
        ),
        (
            "validate_postmortem_fields",
            "Check that a postmortem draft contains root_cause, impact, and action_items.",
            "",
        ),
    ),
    "billing": (
        (
            "get_invoice_breakdown",
            "Return line-item subtotal, tax, and total cents for invoice_id.",
            "",
        ),
        (
            "list_delinquent_accounts",
            "Return accounts whose overdue_days exceeds grace_days (strictly greater than grace_days).",
            "",
        ),
        (
            "estimate_seat_proration",
            "Compute prorated seat charge in integer cents for mid-cycle seat additions.",
            "",
        ),
        (
            "list_credit_memos",
            "List issued credit memos for account_id sorted newest-first.",
            "",
        ),
        (
            "check_entitlement_quota",
            "Return remaining quota units and whether usage has reached hard_cap.",
            "",
        ),
    ),
    "analytics": (
        (
            "compute_retention_cohort",
            "Calculate weekly retention percentages for cohorts starting from start_week.",
            "",
        ),
        (
            "summarize_feature_adoption",
            "Aggregate unique active workspaces per feature_key over window_days.",
            "",
        ),
        (
            "rank_slow_endpoints",
            "Return top N endpoints sorted by p99_ms descending.",
            "",
        ),
        (
            "export_funnel_metrics",
            "Compute step-to-step conversion rates for funnel_steps in order.",
            "",
        ),
        (
            "detect_usage_anomalies",
            "Flag metrics whose z-score exceeds threshold_sigma.",
            "",
        ),
    ),
    "webhooks": (
        (
            "list_webhook_endpoints",
            "List configured webhook subscriptions for workspace_id, redacting signing_secret.",
            "",
        ),
        (
            "inspect_delivery_attempts",
            "Return recent webhook delivery attempts and HTTP status codes for endpoint_id.",
            "",
        ),
        (
            "verify_webhook_signature",
            "Verify HMAC-SHA256 signature header against raw payload bytes using constant-time comparison.",
            "",
        ),
        (
            "list_dead_letter_events",
            "Return undelivered events in dead-letter queue for endpoint_id up to limit.",
            "",
        ),
        (
            "compute_backoff_schedule",
            "Compute exponential retry delay seconds capped at max_delay_seconds.",
            "",
        ),
    ),
    "onboarding": (
        (
            "get_checklist_progress",
            "Return completed and pending onboarding task IDs for hire_id.",
            "",
        ),
        (
            "list_pending_equipment",
            "List unshipped hardware requests ordered by start_date ascending.",
            "",
        ),
        (
            "validate_ssn_last4",
            "Check that ssn_last4 consists of exactly four ASCII digits.",
            "",
        ),
        (
            "assign_mentor_candidates",
            "Rank eligible mentors with capacity > 0 by fewest active mentees ascending.",
            "",
        ),
        (
            "summarize_cohort_readiness",
            "Aggregate percentage of complete checklists for start_date.",
            "",
        ),
    ),
    "scheduling": (
        (
            "query_shift_coverage",
            "Return scheduled shifts between start_date and end_date (both inclusive: start_date <= d <= end_date).",
            "Handler checks d < end_date (exclusive upper bound) instead of d <= end_date.",
        ),
        (
            "summarize_overtime_window",
            "Summarize overtime per employee with durations in integer minutes and UTC-aware window timestamps.",
            "Delegated helper format_overtime_summary returns duration in seconds and strips tzinfo (naive local timestamp).",
        ),
        (
            "find_unassigned_shifts",
            "List open shifts needing coverage for location_id sorted by starts_at ascending.",
            "",
        ),
        (
            "check_rest_period_compliance",
            "Verify minimum rest_hours gap between consecutive shifts for an employee.",
            "",
        ),
        (
            "compute_labor_cost_forecast",
            "Estimate regular and overtime labor cost cents for draft_shifts.",
            "",
        ),
    ),
}

PLANTED_BUG_SPECS: list[dict[str, Any]] = [
    {
        "item": "list_payroll_runs",
        "domain": "payroll",
        "handler_file": "services/svc_agent_tools/handlers/payroll_handlers.py",
        "helper_file": "services/svc_agent_tools/helpers/payroll_helpers.py",
        "in_mega_suite": False,
        "bug_kind": "inverted_sort_order",
        "description_claim": "Sorted newest-first by created_at descending.",
        "actual_behavior": "Sorts ascending (reverse=False).",
    },
    {
        "item": "search_audit_events",
        "domain": "compliance",
        "handler_file": "services/svc_agent_tools/handlers/compliance_handlers.py",
        "helper_file": "services/svc_agent_tools/helpers/compliance_helpers.py",
        "in_mega_suite": False,
        "bug_kind": "silent_helper_cap",
        "description_claim": "Accepts limit from 1 up to 100 records.",
        "actual_behavior": "clamp_audit_limit in helpers/compliance_helpers.py caps limit at 25.",
    },
    {
        "item": "export_team_roster",
        "domain": "directory",
        "handler_file": "services/svc_agent_tools/handlers/directory_handlers.py",
        "helper_file": "services/svc_agent_tools/helpers/directory_helpers.py",
        "in_mega_suite": False,
        "bug_kind": "missing_optional_default",
        "description_claim": "department_id is optional and defaults to all departments.",
        "actual_behavior": "Raises ValueError('department_id is required') when department_id is None.",
    },
    {
        "item": "preview_leave_accrual",
        "domain": "leave",
        "handler_file": "services/svc_agent_tools/handlers/leave_handlers.py",
        "helper_file": "services/svc_agent_tools/helpers/leave_helpers.py",
        "in_mega_suite": False,
        "bug_kind": "mutating_readonly_tool",
        "description_claim": "Read-only dry-run preview with no side effects or state writes.",
        "actual_behavior": "Calls record_accrual_mutation(...) to mutate ledger state.",
    },
    {
        "item": "list_open_incidents",
        "domain": "incidents",
        "handler_file": "services/svc_agent_tools/handlers/incidents_handlers.py",
        "helper_file": "services/svc_agent_tools/helpers/incidents_helpers.py",
        "in_mega_suite": False,
        "bug_kind": "ignored_filter_parameter",
        "description_claim": "When include_archived=False, archived incidents are excluded.",
        "actual_behavior": "Accepts include_archived parameter without filtering status == 'archived'.",
    },
    {
        "item": "query_shift_coverage",
        "domain": "scheduling",
        "handler_file": "services/svc_agent_tools/handlers/scheduling_handlers.py",
        "helper_file": "services/svc_agent_tools/helpers/scheduling_helpers.py",
        "in_mega_suite": True,
        "bug_kind": "off_by_one_bound",
        "description_claim": "end_date is inclusive (start_date <= d <= end_date).",
        "actual_behavior": "Filters strictly d < end_date (exclusive upper bound).",
    },
    {
        "item": "summarize_overtime_window",
        "domain": "scheduling",
        "handler_file": "services/svc_agent_tools/handlers/scheduling_handlers.py",
        "helper_file": "services/svc_agent_tools/helpers/scheduling_helpers.py",
        "in_mega_suite": True,
        "bug_kind": "unit_and_timezone_mismatch",
        "description_claim": "Durations in integer minutes and UTC-aware window timestamps.",
        "actual_behavior": "format_overtime_summary returns seconds and strips tzinfo (naive timestamp).",
    },
]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_agent_tools_service(root: Path) -> dict[str, Any]:
    base = root / "services" / "svc_agent_tools"
    _write(base / "__init__.py", '"""Agent tools service."""\n')
    _write(base / "descriptions" / "__init__.py", "")
    _write(base / "models" / "__init__.py", "")
    _write(base / "helpers" / "__init__.py", "")
    _write(base / "handlers" / "__init__.py", "")

    _write(
        base / "registry.py",
        "\n".join([
            '"""Tool registration decorator for agent tools."""',
            "",
            "from __future__ import annotations",
            "",
            "from collections.abc import Callable",
            "from typing import Any",
            "",
            "TOOL_REGISTRY: dict[str, dict[str, Any]] = {}",
            "",
            "",
            "def register_tool(*, name: str, description: str, read_only: bool = True) -> Callable:",
            "    def decorator(fn: Callable) -> Callable:",
            '        TOOL_REGISTRY[name] = {"fn": fn, "description": description, "read_only": read_only}',
            "        fn._tool_name = name",
            "        return fn",
            "    return decorator",
            "",
        ]),
    )

    for domain in DOMAINS:
        tools = DOMAIN_TOOLS[domain]
        desc_lines = [
            f'"""Tool descriptions for the {domain} domain."""',
            "",
            "TOOL_DESCRIPTIONS: dict[str, str] = {",
        ]
        for name, desc, _ in tools:
            desc_lines.append(f"    {name!r}: (")
            desc_lines.append(f"        {desc!r}")
            if domain == "scheduling":
                desc_lines.extend([
                    '        " Includes detailed scheduling window validation, roster alignment rules, "'
                    '        "multi-location shift matching semantics, and deterministic JSON output fields "'
                    '        "consumed by downstream workforce planning agents."'
                ])
            desc_lines.append("    ),")
        desc_lines.append("}")
        if domain == "scheduling":
            desc_lines.extend(["", "SCHEDULING_SCHEMA_NOTES = ["])
            for idx in range(280):
                desc_lines.append(
                    f'    "scheduling_field_spec_{idx:03d}: ISO-8601 window bounds, location_id scoping, '
                    f'break-rule thresholds, overtime tier mappings, and shift tag constraints for audit batch {idx:03d}",'
                )
            desc_lines.append("]")
        _write(base / "descriptions" / f"{domain}_descriptions.py", "\n".join(desc_lines) + "\n")

        model_lines = [
            f'"""Dataclass models for the {domain} domain."""',
            "",
            "from __future__ import annotations",
            "",
            "from dataclasses import dataclass, field",
            "from datetime import datetime",
            "",
            "",
            "@dataclass",
            f"class {domain.capitalize()}Record:",
            "    record_id: str",
            "    workspace_id: str",
            "    status: str",
            "    created_at: datetime",
            "    tags: list[str] = field(default_factory=list)",
        ]
        if domain == "scheduling":
            for m_idx in range(95):
                model_lines.extend([
                    "",
                    "@dataclass",
                    f"class SchedulingPolicyProfile{m_idx:03d}:",
                    f"    profile_id: str = 'prof_{m_idx:03d}'",
                    f"    location_code: str = 'loc_{m_idx % 12:02d}'",
                    f"    min_rest_minutes: int = {480 + (m_idx % 5) * 30}",
                    f"    max_shift_minutes: int = {720 + (m_idx % 4) * 60}",
                    f"    overtime_threshold_minutes: int = {480 + (m_idx % 3) * 60}",
                    f"    allow_split_shifts: bool = {'True' if m_idx % 2 == 0 else 'False'}",
                    f"    effective_timezone: str = 'UTC'",
                    f"    audit_notes: str = 'Scheduling profile {m_idx:03d} compliance envelope and shift rules'",
                ])
        _write(base / "models" / f"{domain}_models.py", "\n".join(model_lines) + "\n")

        helper_lines = _build_domain_helpers(domain)
        _write(base / "helpers" / f"{domain}_helpers.py", "\n".join(helper_lines) + "\n")

        handler_lines = _build_domain_handlers(domain, tools)
        _write(base / "handlers" / f"{domain}_handlers.py", "\n".join(handler_lines) + "\n")

    scheduling_bytes = sum(
        (base / sub / f"scheduling_{sub}.py").stat().st_size
        for sub in ("descriptions", "models", "helpers", "handlers")
    )
    return {"tool_count": sum(len(v) for v in DOMAIN_TOOLS.values()), "scheduling_suite_bytes": scheduling_bytes}


def _build_domain_helpers(domain: str) -> list[str]:
    lines = [
        f'"""Helper functions for the {domain} domain."""',
        "",
        "from __future__ import annotations",
        "",
        "from datetime import datetime, timezone",
        "from typing import Any",
        "",
    ]
    if domain == "compliance":
        lines.extend([
            "DEFAULT_AUDIT_HARD_CAP = 25",
            "",
            "",
            "def clamp_audit_limit(requested_limit: int) -> int:",
            "    if requested_limit < 1:",
            "        return 1",
            "    return min(requested_limit, DEFAULT_AUDIT_HARD_CAP)",
            "",
        ])
    elif domain == "leave":
        lines.extend([
            "def record_accrual_mutation(ledger: dict[str, float], employee_id: str, delta: float) -> float:",
            "    updated = round(ledger.get(employee_id, 0.0) + delta, 4)",
            "    ledger[employee_id] = updated",
            "    return updated",
            "",
        ])
    elif domain == "scheduling":
        lines.extend([
            "def format_overtime_summary(seconds_total: int, window_end: datetime) -> dict[str, Any]:",
            "    # Converts to local naive wall time and preserves raw seconds.",
            "    naive_end = window_end.replace(tzinfo=None)",
            "    return {",
            '        "overtime_duration": int(seconds_total),',
            '        "window_end_iso": naive_end.isoformat(),',
            '        "tz_aware": naive_end.tzinfo is not None,',
            "    }",
            "",
        ])
    lines.extend([
        f"def normalize_{domain}_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:",
        "    return [dict(item) for item in rows]",
        "",
    ])
    return lines


def _build_domain_handlers(domain: str, tools: tuple[tuple[str, str, str], ...]) -> list[str]:
    lines = [
        f'"""Registered agent tool handlers for the {domain} domain."""',
        "",
        "from __future__ import annotations",
        "",
        "from datetime import date, datetime, timedelta, timezone",
        "import hashlib",
        "import hmac",
        "import math",
        "from typing import Any",
        "",
        f"from services.svc_agent_tools.descriptions.{domain}_descriptions import TOOL_DESCRIPTIONS",
        f"from services.svc_agent_tools.helpers.{domain}_helpers import normalize_{domain}_payload",
        "from services.svc_agent_tools.registry import register_tool",
    ]
    if domain == "compliance":
        lines.append("from services.svc_agent_tools.helpers.compliance_helpers import clamp_audit_limit")
    if domain == "leave":
        lines.append("from services.svc_agent_tools.helpers.leave_helpers import record_accrual_mutation")
    if domain == "scheduling":
        lines.append("from services.svc_agent_tools.helpers.scheduling_helpers import format_overtime_summary")
    lines.append("")

    if domain == "scheduling":
        # Interleave the 5 scheduling tools across 210 matrix validators (~42 between each)
        # so the scheduling suite exceeds 270 KB and handlers span lines ~25..4000+.
        validators_per_tool = 42
        for tool_idx, (name, _desc, _bug) in enumerate(tools):
            lines.extend(_render_single_handler(domain, name))
            start_v = tool_idx * validators_per_tool
            for idx in range(start_v, start_v + validators_per_tool):
                lines.extend([
                    "",
                    f"def _scheduling_matrix_validator_{idx:03d}(shifts: list[dict[str, Any]], rules: dict[str, int]) -> list[dict[str, Any]]:",
                    f'    """Validate scheduling window batch {idx:03d} against rest, overlap, and break rules."""',
                    "    accepted: list[dict[str, Any]] = []",
                    "    min_rest = int(rules.get('min_rest_minutes', 480))",
                    "    max_span = int(rules.get('max_shift_minutes', 720))",
                    "    for pos, shift in enumerate(shifts):",
                    "        duration = int(shift.get('duration_minutes', 0))",
                    "        rest_gap = int(shift.get('rest_gap_minutes', min_rest))",
                    "        if duration <= 0 or duration > max_span:",
                    "            continue",
                    "        if rest_gap < min_rest:",
                    "            continue",
                    "        accepted.append({",
                    f"            'validator_index': {idx},",
                    "            'position': pos,",
                    "            'shift_id': shift.get('shift_id', ''),",
                    "            'duration_minutes': duration,",
                    "            'rest_gap_minutes': rest_gap,",
                    "            'location_id': shift.get('location_id', 'default'),",
                    "            'compliant': True,",
                    "        })",
                    "    return accepted",
                ])
    else:
        validators_per_tool = 4
        for tool_idx, (name, _desc, _bug) in enumerate(tools):
            lines.extend(_render_single_handler(domain, name))
            start_v = tool_idx * validators_per_tool
            for idx in range(start_v, start_v + validators_per_tool):
                lines.extend([
                    "",
                    f"def _{domain}_envelope_check_{idx:02d}(rows: list[dict[str, Any]], workspace_id: str) -> list[dict[str, Any]]:",
                    f'    """Internal envelope pre-check {idx:02d} for {domain} batch processing."""',
                    "    valid: list[dict[str, Any]] = []",
                    "    for pos, row in enumerate(rows):",
                    "        if row.get('workspace_id') not in (None, workspace_id):",
                    "            continue",
                    "        if row.get('deleted_at') is not None:",
                    "            continue",
                    "        valid.append({",
                    f"            'check_index': {idx},",
                    "            'position': pos,",
                    "            'record_id': row.get('record_id', ''),",
                    "            'workspace_id': workspace_id,",
                    "        })",
                    "    return valid",
                ])

    return lines


_CLEAN_TOOL_IMPLEMENTATIONS: dict[str, list[str]] = {
    # --- payroll ---
    "get_payroll_summary": [
        "def get_payroll_summary(runs: dict[str, dict[str, Any]], payroll_run_id: str) -> dict[str, Any]:",
        "    run = runs.get(payroll_run_id)",
        "    if run is None:",
        "        raise KeyError(f'Unknown payroll_run_id: {payroll_run_id}')",
        "    if not run.get('finalized', False):",
        "        raise ValueError(f'Payroll run {payroll_run_id} is not finalized')",
        "    entries = run.get('entries', [])",
        "    gross_pay = sum(int(e.get('gross_cents', 0)) for e in entries)",
        "    tax_withholdings = sum(int(e.get('tax_cents', 0)) for e in entries)",
        "    headcount = len({e.get('employee_id') for e in entries if e.get('employee_id')})",
        "    return {",
        "        'payroll_run_id': payroll_run_id,",
        "        'gross_pay': gross_pay,",
        "        'tax_withholdings': tax_withholdings,",
        "        'headcount': headcount,",
        "    }",
    ],
    "validate_bank_routing": [
        "def validate_bank_routing(routing_number: str) -> bool:",
        "    if len(routing_number) != 9 or not (routing_number.isascii() and routing_number.isdigit()):",
        "        return False",
        "    digits = [int(ch) for ch in routing_number]",
        "    checksum = (",
        "        3 * (digits[0] + digits[3] + digits[6])",
        "        + 7 * (digits[1] + digits[4] + digits[7])",
        "        + (digits[2] + digits[5] + digits[8])",
        "    )",
        "    return checksum % 10 == 0",
    ],
    "list_pay_periods": [
        "def list_pay_periods(periods: list[dict[str, Any]], pay_schedule_id: str, start_date: date, end_date: date) -> list[dict[str, Any]]:",
        "    matched = [",
        "        p for p in periods",
        "        if p.get('pay_schedule_id') == pay_schedule_id",
        "        and p.get('status') in {'open', 'closed'}",
        "        and start_date <= p['period_date'] <= end_date",
        "    ]",
        "    return normalize_payroll_payload(sorted(matched, key=lambda p: p['period_date']))",
    ],
    "calculate_prorated_salary": [
        "def calculate_prorated_salary(annual_base_cents: int, active_days: int, period_days: int, periods_per_year: int = 26) -> int:",
        "    if period_days <= 0 or periods_per_year <= 0:",
        "        raise ValueError('period_days and periods_per_year must be positive')",
        "    clamped_days = max(0, min(active_days, period_days))",
        "    period_base = annual_base_cents / periods_per_year",
        "    return int(round(period_base * (clamped_days / period_days)))",
    ],
    # --- compliance ---
    "get_policy_acknowledgements": [
        "def get_policy_acknowledgements(records: list[dict[str, Any]], policy_id: str, employee_ids: list[str]) -> dict[str, str | None]:",
        "    target_ids = set(employee_ids)",
        "    timestamps: dict[str, str | None] = {emp_id: None for emp_id in employee_ids}",
        "    for rec in records:",
        "        emp_id = rec.get('employee_id')",
        "        if rec.get('policy_id') == policy_id and emp_id in target_ids:",
        "            timestamps[emp_id] = rec.get('acknowledged_at')",
        "    return timestamps",
    ],
    "check_document_retention": [
        "def check_document_retention(artifact: dict[str, Any], retention_days: int) -> dict[str, Any]:",
        "    age_days = int(artifact.get('age_days', 0))",
        "    return {",
        "        'category': artifact.get('category', 'general'),",
        "        'age_days': age_days,",
        "        'required_days': retention_days,",
        "        'compliant': age_days >= retention_days,",
        "    }",
    ],
    "list_expiring_certifications": [
        "def list_expiring_certifications(certs: list[dict[str, Any]], reference_date: date, horizon_days: int = 30) -> list[dict[str, Any]]:",
        "    results: list[dict[str, Any]] = []",
        "    for cert in certs:",
        "        if not cert.get('active', True):",
        "            continue",
        "        days_left = (cert['expires_on'] - reference_date).days",
        "        if 0 <= days_left <= horizon_days:",
        "            results.append(cert)",
        "    return normalize_compliance_payload(sorted(results, key=lambda c: c['expires_on']))",
    ],
    "verify_i9_completeness": [
        "def verify_i9_completeness(packet: dict[str, Any], required_fields: tuple[str, ...] = ('section_1', 'section_2', 'doc_type', 'signed_at')) -> dict[str, Any]:",
        "    missing = [field for field in required_fields if not packet.get(field)]",
        "    return {'complete': len(missing) == 0, 'missing_fields': missing}",
    ],
    # --- directory ---
    "lookup_employee_profile": [
        "def lookup_employee_profile(employees: dict[str, dict[str, Any]], employee_id: str, include_inactive: bool = False) -> dict[str, Any] | None:",
        "    profile = employees.get(employee_id)",
        "    if profile is None:",
        "        return None",
        "    if profile.get('deactivated', False) and not include_inactive:",
        "        return None",
        "    return dict(profile)",
    ],
    "list_reporting_chain": [
        "def list_reporting_chain(employees: dict[str, dict[str, Any]], employee_id: str) -> list[str]:",
        "    chain: list[str] = []",
        "    seen: set[str] = {employee_id}",
        "    curr = employees.get(employee_id)",
        "    while curr is not None:",
        "        manager_id = curr.get('manager_id')",
        "        if not manager_id or manager_id in seen:",
        "            break",
        "        seen.add(manager_id)",
        "        chain.append(manager_id)",
        "        curr = employees.get(manager_id)",
        "    return chain",
    ],
    "find_department_leads": [
        "def find_department_leads(departments: list[dict[str, Any]], department_ids: list[str] | None = None) -> dict[str, str | None]:",
        "    wanted = set(department_ids) if department_ids is not None else None",
        "    leads: dict[str, str | None] = {}",
        "    for dept in departments:",
        "        dept_id = dept.get('department_id')",
        "        if dept_id and (wanted is None or dept_id in wanted):",
        "            leads[dept_id] = dept.get('lead_employee_id')",
        "    return leads",
    ],
    "resolve_work_location": [
        "def resolve_work_location(locations: dict[str, dict[str, Any]], location_code: str) -> dict[str, Any]:",
        "    loc = locations.get(location_code)",
        "    if loc is None:",
        "        raise KeyError(f'Unknown location_code: {location_code}')",
        "    return {",
        "        'location_code': location_code,",
        "        'timezone': loc['timezone'],",
        "        'address': loc['address'],",
        "    }",
    ],
    # --- leave ---
    "list_leave_requests": [
        "def list_leave_requests(requests: list[dict[str, Any]], employee_id: str, status: str | None = None) -> list[dict[str, Any]]:",
        "    matched = [",
        "        req for req in requests",
        "        if req.get('employee_id') == employee_id",
        "        and (status is None or req.get('status') == status)",
        "    ]",
        "    return normalize_leave_payload(matched)",
    ],
    "check_holiday_overlap": [
        "def check_holiday_overlap(start_date: date, end_date: date, company_holidays: set[date]) -> int:",
        "    if end_date < start_date:",
        "        return 0",
        "    working_days = 0",
        "    curr = start_date",
        "    while curr <= end_date:",
        "        if curr.weekday() < 5 and curr not in company_holidays:",
        "            working_days += 1",
        "        curr += timedelta(days=1)",
        "    return working_days",
    ],
    "get_leave_policy_caps": [
        "def get_leave_policy_caps(policies: dict[str, dict[str, float]], leave_type: str) -> dict[str, float | str]:",
        "    policy = policies.get(leave_type)",
        "    if policy is None:",
        "        raise KeyError(f'Unknown leave_type: {leave_type}')",
        "    return {",
        "        'leave_type': leave_type,",
        "        'max_carryover': float(policy['max_carryover']),",
        "        'annual_accrual_cap': float(policy['annual_accrual_cap']),",
        "    }",
    ],
    "summarize_team_absences": [
        "def summarize_team_absences(absences: list[dict[str, Any]], start_date: date, end_date: date) -> dict[str, int]:",
        "    counts: dict[str, int] = {}",
        "    for item in absences:",
        "        if item.get('status') != 'approved':",
        "            continue",
        "        if item['end_date'] < start_date or item['start_date'] > end_date:",
        "            continue",
        "        team_id = str(item.get('team_id', 'unassigned'))",
        "        counts[team_id] = counts.get(team_id, 0) + 1",
        "    return counts",
    ],
    # --- incidents ---
    "get_incident_timeline": [
        "def get_incident_timeline(events: list[dict[str, Any]], incident_id: str) -> list[dict[str, Any]]:",
        "    scoped = [e for e in events if e.get('incident_id') == incident_id]",
        "    return normalize_incidents_payload(sorted(scoped, key=lambda e: e['occurred_at']))",
    ],
    "compute_mttr_stats": [
        "def compute_mttr_stats(incidents: list[dict[str, Any]]) -> dict[str, float]:",
        "    durations_min = sorted(",
        "        (inc['resolved_at'] - inc['created_at']).total_seconds() / 60.0",
        "        for inc in incidents",
        "        if inc.get('resolved_at') is not None and inc.get('created_at') is not None",
        "    )",
        "    if not durations_min:",
        "        return {'mean_minutes': 0.0, 'p95_minutes': 0.0}",
        "    mean_min = sum(durations_min) / len(durations_min)",
        "    p95_idx = min(len(durations_min) - 1, int(math.ceil(0.95 * len(durations_min))) - 1)",
        "    return {'mean_minutes': round(mean_min, 2), 'p95_minutes': round(durations_min[p95_idx], 2)}",
    ],
    "list_oncall_escalations": [
        "def list_oncall_escalations(schedules: list[dict[str, Any]], service_name: str) -> list[dict[str, Any]]:",
        "    scoped = [s for s in schedules if s.get('service_name') == service_name]",
        "    return normalize_incidents_payload(sorted(scoped, key=lambda s: int(s.get('tier', 0))))",
    ],
    "validate_postmortem_fields": [
        "def validate_postmortem_fields(draft: dict[str, Any]) -> dict[str, Any]:",
        "    required = ('root_cause', 'impact', 'action_items')",
        "    missing = [field for field in required if not draft.get(field)]",
        "    return {'valid': len(missing) == 0, 'missing_fields': missing}",
    ],
    # --- billing ---
    "get_invoice_breakdown": [
        "def get_invoice_breakdown(invoices: dict[str, dict[str, Any]], invoice_id: str) -> dict[str, Any]:",
        "    inv = invoices.get(invoice_id)",
        "    if inv is None:",
        "        raise KeyError(f'Unknown invoice_id: {invoice_id}')",
        "    subtotal_cents = sum(int(line.get('amount_cents', 0)) for line in inv.get('line_items', []))",
        "    tax_cents = int(inv.get('tax_cents', 0))",
        "    return {",
        "        'invoice_id': invoice_id,",
        "        'subtotal_cents': subtotal_cents,",
        "        'tax_cents': tax_cents,",
        "        'total_cents': subtotal_cents + tax_cents,",
        "    }",
    ],
    "list_delinquent_accounts": [
        "def list_delinquent_accounts(accounts: list[dict[str, Any]], grace_days: int = 14) -> list[dict[str, Any]]:",
        "    return normalize_billing_payload([a for a in accounts if int(a.get('overdue_days', 0)) > grace_days])",
    ],
    "estimate_seat_proration": [
        "def estimate_seat_proration(seat_price_cents: int, added_seats: int, remaining_days: int, cycle_days: int) -> int:",
        "    if cycle_days <= 0:",
        "        raise ValueError('cycle_days must be positive')",
        "    clamped_days = max(0, min(remaining_days, cycle_days))",
        "    return int(round(seat_price_cents * max(0, added_seats) * (clamped_days / cycle_days)))",
    ],
    "list_credit_memos": [
        "def list_credit_memos(memos: list[dict[str, Any]], account_id: str) -> list[dict[str, Any]]:",
        "    scoped = [m for m in memos if m.get('account_id') == account_id]",
        "    return normalize_billing_payload(sorted(scoped, key=lambda m: m['issued_at'], reverse=True))",
    ],
    "check_entitlement_quota": [
        "def check_entitlement_quota(used_units: int, hard_cap: int) -> dict[str, Any]:",
        "    remaining = max(0, hard_cap - max(0, used_units))",
        "    return {'remaining_units': remaining, 'reached_hard_cap': used_units >= hard_cap}",
    ],
    # --- analytics ---
    "compute_retention_cohort": [
        "def compute_retention_cohort(cohorts: list[dict[str, Any]], start_week: date) -> list[dict[str, Any]]:",
        "    results: list[dict[str, Any]] = []",
        "    for cohort in sorted(cohorts, key=lambda c: c['cohort_week']):",
        "        if cohort['cohort_week'] < start_week:",
        "            continue",
        "        size = max(1, int(cohort.get('initial_users', 0)))",
        "        weekly_pct = [round((int(cnt) / size) * 100.0, 2) for cnt in cohort.get('weekly_retained', [])]",
        "        results.append({'cohort_week': cohort['cohort_week'].isoformat(), 'retention_percentages': weekly_pct})",
        "    return results",
    ],
    "summarize_feature_adoption": [
        "def summarize_feature_adoption(events: list[dict[str, Any]], reference_date: date, window_days: int = 30) -> dict[str, int]:",
        "    cutoff = reference_date - timedelta(days=max(0, window_days))",
        "    by_feature: dict[str, set[str]] = {}",
        "    for ev in events:",
        "        if ev['event_date'] < cutoff or ev['event_date'] > reference_date:",
        "            continue",
        "        fkey = ev.get('feature_key')",
        "        ws = ev.get('workspace_id')",
        "        if fkey and ws:",
        "            by_feature.setdefault(fkey, set()).add(ws)",
        "    return {fkey: len(workspaces) for fkey, workspaces in by_feature.items()}",
    ],
    "rank_slow_endpoints": [
        "def rank_slow_endpoints(metrics: list[dict[str, Any]], n: int = 10) -> list[dict[str, Any]]:",
        "    return normalize_analytics_payload(sorted(metrics, key=lambda m: m['p99_ms'], reverse=True)[:max(1, n)])",
    ],
    "export_funnel_metrics": [
        "def export_funnel_metrics(counts_by_step: dict[str, int], funnel_steps: list[str]) -> list[dict[str, Any]]:",
        "    output: list[dict[str, Any]] = []",
        "    prev_count: int | None = None",
        "    for step in funnel_steps:",
        "        count = int(counts_by_step.get(step, 0))",
        "        rate = 1.0 if prev_count is None else (round(count / prev_count, 4) if prev_count > 0 else 0.0)",
        "        output.append({'step': step, 'count': count, 'conversion_rate': rate})",
        "        prev_count = count",
        "    return output",
    ],
    "detect_usage_anomalies": [
        "def detect_usage_anomalies(series: list[dict[str, Any]], threshold_sigma: float = 2.5) -> list[dict[str, Any]]:",
        "    if len(series) < 2:",
        "        return []",
        "    values = [float(point['value']) for point in series]",
        "    mean = sum(values) / len(values)",
        "    variance = sum((v - mean) ** 2 for v in values) / len(values)",
        "    stddev = math.sqrt(variance)",
        "    if stddev == 0.0:",
        "        return []",
        "    flagged: list[dict[str, Any]] = []",
        "    for point, val in zip(series, values):",
        "        z_score = (val - mean) / stddev",
        "        if abs(z_score) > threshold_sigma:",
        "            flagged.append({**point, 'z_score': round(z_score, 3)})",
        "    return flagged",
    ],
    # --- webhooks ---
    "list_webhook_endpoints": [
        "def list_webhook_endpoints(endpoints: list[dict[str, Any]], workspace_id: str) -> list[dict[str, Any]]:",
        "    redacted: list[dict[str, Any]] = []",
        "    for ep in endpoints:",
        "        if ep.get('workspace_id') != workspace_id:",
        "            continue",
        "        entry = dict(ep)",
        "        if 'signing_secret' in entry:",
        "            entry['signing_secret'] = '***'",
        "        redacted.append(entry)",
        "    return redacted",
    ],
    "inspect_delivery_attempts": [
        "def inspect_delivery_attempts(attempts: list[dict[str, Any]], endpoint_id: str, limit: int = 25) -> list[dict[str, Any]]:",
        "    scoped = [a for a in attempts if a.get('endpoint_id') == endpoint_id]",
        "    ordered = sorted(scoped, key=lambda a: a['attempted_at'], reverse=True)",
        "    return [",
        "        {'attempt_id': a.get('attempt_id'), 'status_code': a.get('status_code'), 'attempted_at': a['attempted_at']}",
        "        for a in ordered[:max(1, limit)]",
        "    ]",
    ],
    "verify_webhook_signature": [
        "def verify_webhook_signature(secret: bytes, payload: bytes, header_sig: str) -> bool:",
        "    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()",
        "    return hmac.compare_digest(expected, header_sig)",
    ],
    "list_dead_letter_events": [
        "def list_dead_letter_events(dlq: list[dict[str, Any]], endpoint_id: str, limit: int = 50) -> list[dict[str, Any]]:",
        "    undelivered = [",
        "        ev for ev in dlq",
        "        if ev.get('endpoint_id') == endpoint_id and not ev.get('delivered', False)",
        "    ]",
        "    return normalize_webhooks_payload(undelivered[:max(1, limit)])",
    ],
    "compute_backoff_schedule": [
        "def compute_backoff_schedule(attempts: int, base_delay_seconds: int = 5, max_delay_seconds: int = 3600) -> list[int]:",
        "    return [",
        "        min(max_delay_seconds, max(0, base_delay_seconds) * (2 ** idx))",
        "        for idx in range(max(0, attempts))",
        "    ]",
    ],
    # --- onboarding ---
    "get_checklist_progress": [
        "def get_checklist_progress(tasks: list[dict[str, Any]], hire_id: str) -> dict[str, list[str]]:",
        "    completed: list[str] = []",
        "    pending: list[str] = []",
        "    for task in tasks:",
        "        if task.get('hire_id') != hire_id:",
        "            continue",
        "        tid = str(task['task_id'])",
        "        if task.get('completed', False):",
        "            completed.append(tid)",
        "        else:",
        "            pending.append(tid)",
        "    return {'completed': completed, 'pending': pending}",
    ],
    "list_pending_equipment": [
        "def list_pending_equipment(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:",
        "    unshipped = [r for r in requests if not r.get('shipped', False)]",
        "    return normalize_onboarding_payload(sorted(unshipped, key=lambda r: r['start_date']))",
    ],
    "validate_ssn_last4": [
        "def validate_ssn_last4(ssn_last4: str) -> bool:",
        "    return len(ssn_last4) == 4 and ssn_last4.isascii() and ssn_last4.isdigit()",
    ],
    "assign_mentor_candidates": [
        "def assign_mentor_candidates(mentors: list[dict[str, Any]]) -> list[dict[str, Any]]:",
        "    eligible = [m for m in mentors if int(m.get('capacity', 0)) > 0]",
        "    return normalize_onboarding_payload(sorted(eligible, key=lambda m: int(m.get('active_mentees', 0))))",
    ],
    "summarize_cohort_readiness": [
        "def summarize_cohort_readiness(checklists: list[dict[str, Any]], start_date: date) -> float:",
        "    cohort = [c for c in checklists if c.get('start_date') == start_date]",
        "    if not cohort:",
        "        return 0.0",
        "    complete_count = sum(1 for c in cohort if c.get('complete', False))",
        "    return round((complete_count / len(cohort)) * 100.0, 2)",
    ],
    # --- scheduling (clean tools) ---
    "find_unassigned_shifts": [
        "def find_unassigned_shifts(shifts: list[dict[str, Any]], location_id: str) -> list[dict[str, Any]]:",
        "    unassigned = [",
        "        s for s in shifts",
        "        if s.get('location_id') == location_id and s.get('employee_id') is None",
        "    ]",
        "    return normalize_scheduling_payload(sorted(unassigned, key=lambda s: s['starts_at']))",
    ],
    "check_rest_period_compliance": [
        "def check_rest_period_compliance(shifts: list[dict[str, Any]], min_rest_hours: float = 8.0) -> list[dict[str, Any]]:",
        "    ordered = sorted(shifts, key=lambda s: s['starts_at'])",
        "    violations: list[dict[str, Any]] = []",
        "    for prev, curr in zip(ordered, ordered[1:]):",
        "        gap_hours = (curr['starts_at'] - prev['ends_at']).total_seconds() / 3600.0",
        "        if gap_hours < min_rest_hours:",
        "            violations.append({'prev_shift_id': prev.get('shift_id'), 'next_shift_id': curr.get('shift_id'), 'rest_hours': round(gap_hours, 2)})",
        "    return violations",
    ],
    "compute_labor_cost_forecast": [
        "def compute_labor_cost_forecast(draft_shifts: list[dict[str, Any]], hourly_rate_cents: int, overtime_multiplier: float = 1.5) -> dict[str, int]:",
        "    regular_cents = 0",
        "    overtime_cents = 0",
        "    for shift in draft_shifts:",
        "        reg_hours = float(shift.get('regular_hours', 0.0))",
        "        ot_hours = float(shift.get('overtime_hours', 0.0))",
        "        regular_cents += int(round(reg_hours * hourly_rate_cents))",
        "        overtime_cents += int(round(ot_hours * hourly_rate_cents * overtime_multiplier))",
        "    return {'regular_cents': regular_cents, 'overtime_cents': overtime_cents, 'total_cents': regular_cents + overtime_cents}",
    ],
}


def _render_single_handler(domain: str, name: str) -> list[str]:
    header = [
        "",
        "@register_tool(",
        f"    name={name!r},",
        f"    description=TOOL_DESCRIPTIONS[{name!r}],",
        ")",
    ]
    if name == "list_payroll_runs":
        return header + [
            "def list_payroll_runs(runs: list[dict[str, Any]], status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:",
            "    # Note: legacy callers expected chronological playback during batch migration.",
            "    filtered = [r for r in runs if status is None or r.get('status') == status]",
            "    ordered = sorted(filtered, key=lambda item: item['created_at'], reverse=False)",
            "    return ordered[:max(1, limit)]",
        ]
    if name == "search_audit_events":
        return header + [
            "def search_audit_events(",
            "    events: list[dict[str, Any]],",
            "    actor_id: str | None = None,",
            "    action: str | None = None,",
            "    limit: int = 50,",
            ") -> list[dict[str, Any]]:",
            "    def _matches(entry: dict[str, Any]) -> bool:",
            "        if actor_id is not None and entry.get('actor_id') != actor_id:",
            "            return False",
            "        if action is not None and entry.get('action') != action:",
            "            return False",
            "        return True",
            "    matched = [e for e in events if _matches(e)]",
            "    bounded = clamp_audit_limit(limit)",
            "    return matched[:bounded]",
        ]
    if name == "export_team_roster":
        return header + [
            "def export_team_roster(employees: list[dict[str, Any]], department_id: str | None = None) -> list[dict[str, Any]]:",
            "    if department_id is None:",
            "        raise ValueError('department_id is required')",
            "    rows = [e for e in employees if e.get('active', True) and e.get('department_id') == department_id]",
            "    return normalize_directory_payload(rows)",
        ]
    if name == "preview_leave_accrual":
        return header + [
            "def preview_leave_accrual(ledger: dict[str, float], employee_id: str, days_worked: int, rate: float = 0.05) -> dict[str, Any]:",
            "    projected_delta = round(max(0, days_worked) * rate, 4)",
            "    new_balance = record_accrual_mutation(ledger, employee_id, projected_delta)",
            "    return {'employee_id': employee_id, 'projected_delta': projected_delta, 'balance': new_balance}",
        ]
    if name == "list_open_incidents":
        return header + [
            "def list_open_incidents(incidents: list[dict[str, Any]], include_archived: bool = False) -> list[dict[str, Any]]:",
            "    # Keep historical entries accessible for correlation lookups.",
            "    selected = [inc for inc in incidents if inc.get('status') != 'closed']",
            "    return normalize_incidents_payload(selected)",
        ]
    if name == "query_shift_coverage":
        return header + [
            "def query_shift_coverage(shifts: list[dict[str, Any]], start_date: date, end_date: date) -> list[dict[str, Any]]:",
            "    rows = [s for s in shifts if start_date <= s['shift_date'] < end_date]",
            "    return sorted(rows, key=lambda item: item['shift_date'])",
        ]
    if name == "summarize_overtime_window":
        return header + [
            "def summarize_overtime_window(entries: list[dict[str, Any]], window_end: datetime | None = None) -> dict[str, Any]:",
            "    end_dt = window_end or datetime.now(timezone.utc)",
            "    total_seconds = sum(int(e.get('overtime_minutes', 0)) * 60 for e in entries)",
            "    return format_overtime_summary(total_seconds, end_dt)",
        ]

    return header + _CLEAN_TOOL_IMPLEMENTATIONS[name]


def _write_monolith_file(root: Path, *, scale: str) -> dict[str, int]:
    """Generate services/svc_monolith/legacy_core.py (~7,400 lines / 150 defs in full mode)."""
    path = root / "services" / "svc_monolith" / "legacy_core.py"
    _write(path.parent / "__init__.py", "")
    target_defs = 150 if scale == "full" else 20
    lines: list[str] = [
        '"""Legacy monolithic engine module for AST outline stress testing."""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "from typing import Any",
        "",
        "",
        "def traced(tag: str):",
        "    def decorator(fn):",
        "        fn._trace_tag = tag",
        "        return fn",
        "    return decorator",
        "",
    ]
    # Header already defines 2 defs (traced, decorator). Remaining defs: target_defs - 2.
    # Structure: classes with methods + nested helpers, and top-level functions.
    remaining = target_defs - 2
    if scale == "full":
        num_classes = 12
        # Each class: 6 methods, where 2 methods each contain 1 nested def => 12 * 8 = 96 defs.
        # Remaining 148 - 96 = 52 module-level defs.
        methods_per_class = 6
        nested_method_indices = {1, 4}
        module_funcs = remaining - num_classes * (methods_per_class + len(nested_method_indices))
        body_pad_lines = 38
    else:
        num_classes = 2
        # Each class: 3 methods, 1 method has 1 nested def => 2 * 4 = 8 defs.
        # Remaining 18 - 8 = 10 module-level defs.
        methods_per_class = 3
        nested_method_indices = {1}
        module_funcs = remaining - num_classes * (methods_per_class + len(nested_method_indices))
        body_pad_lines = 4

    for c_idx in range(num_classes):
        lines.extend([
            "",
            f"class LegacySubsystem{c_idx:02d}:",
            f"    subsystem_id: str = 'sub_{c_idx:02d}'",
            f"    batch_size: int = {50 + c_idx * 10}",
            "",
        ])
        for m_idx in range(methods_per_class):
            is_async = (c_idx + m_idx) % 4 == 0
            kw = "async def" if is_async else "def"
            lines.extend([
                f"    @traced(tag='c{c_idx}_m{m_idx}')",
                f"    {kw} process_step_{m_idx:02d}(",
                "        self,",
                "        payload: dict[str, Any],",
                "        limit: int = 100,",
                "    ) -> dict[str, Any]:",
            ])
            if m_idx in nested_method_indices:
                lines.extend([
                    f"        def _inner_normalize(entry: dict[str, Any]) -> dict[str, Any]:",
                    "            result = dict(entry)",
                    f"            result['step'] = {m_idx}",
                    "            return result",
                    "",
                ])
            for p_idx in range(body_pad_lines):
                lines.append(f"        acc_{p_idx} = int(payload.get('k_{p_idx}', {p_idx})) + {c_idx + m_idx}")
            lines.append("        return {'subsystem': self.subsystem_id, 'limit': limit, 'last': acc_0}")
            lines.append("")

    for f_idx in range(module_funcs):
        kw = "async def" if f_idx % 5 == 0 else "def"
        lines.extend([
            "",
            f"@traced(tag='mod_{f_idx:03d}')",
            f"{kw} legacy_module_handler_{f_idx:03d}(",
            "    records: list[dict[str, Any]],",
            "    strict: bool = False,",
            ") -> list[dict[str, Any]]:",
        ])
        for p_idx in range(body_pad_lines):
            lines.append(f"    metric_{p_idx} = len(records) + {p_idx}")
        lines.append("    return [dict(r, strict=strict, metric=metric_0) for r in records]")

    _write(path, "\n".join(lines) + "\n")
    return {"lines": len(lines), "target_defs": target_defs}


def _write_workers(root: Path, *, scale: str) -> None:
    """Generate worker modules with exact module-level and nested-in-method >=150-line functions."""
    base = root / "services" / "svc_workers"
    _write(base / "__init__.py", "")
    if scale == "full":
        num_files = 292
        long_module_count = 39
        long_nested_count = 5
    else:
        num_files = 20
        long_module_count = 5
        long_nested_count = 2

    for idx in range(num_files):
        lines = [
            f'"""Worker module worker_{idx:03d}."""',
            "",
            "from __future__ import annotations",
            "",
            "from typing import Any",
            "",
        ]
        if idx < long_module_count:
            lines.extend([
                "",
                f"def run_long_worker_{idx:03d}(items: list[dict[str, Any]]) -> dict[str, int]:",
                "    total = 0",
            ])
            for step in range(155):
                lines.append(f"    total += len(items) + {step}")
            lines.extend([
                "    return {'worker': " + str(idx) + ", 'total': total}",
                "",
            ])
        elif idx < long_module_count + long_nested_count:
            nested_idx = idx - long_module_count
            lines.extend([
                "",
                f"class WorkerRunner{idx:03d}:",
                "    def execute(self, items: list[dict[str, Any]]) -> dict[str, int]:",
                f"        def nested_long_step_{nested_idx:02d}(rows: list[dict[str, Any]]) -> int:",
                "            subtotal = 0",
            ])
            for step in range(155):
                lines.append(f"            subtotal += len(rows) + {step}")
            lines.extend([
                "            return subtotal",
                "",
                f"        return {{'nested': nested_long_step_{nested_idx:02d}(items)}}",
                "",
            ])
        else:
            lines.extend([
                "",
                f"def run_short_worker_{idx:03d}(items: list[dict[str, Any]]) -> int:",
                "    return len(items)",
                "",
            ])
        _write(base / f"worker_{idx:03d}.py", "\n".join(lines))


def _write_models(root: Path, *, scale: str) -> None:
    """Generate model files totaling >=347 class fields in full mode."""
    base = root / "services" / "svc_models"
    _write(base / "__init__.py", "")
    num_files = 120 if scale == "full" else 12
    for idx in range(num_files):
        lines = [
            f'"""Domain models module model_{idx:03d}."""',
            "",
            "from __future__ import annotations",
            "",
            "from dataclasses import dataclass",
            "",
            "",
            "@dataclass",
            f"class DomainEntity{idx:03d}:",
            "    entity_id: str",
            "    tenant_id: str",
            "    status: str = 'active'",
            "",
            "    def __init__(self, entity_id: str, tenant_id: str) -> None:",
            "        self.entity_id = entity_id",
            "        self.tenant_id = tenant_id",
            "        self.computed_version = 1",
            "",
        ]
        _write(base / f"model_{idx:03d}.py", "\n".join(lines))


def _write_background_packages(root: Path, *, scale: str) -> None:
    gw = root / "services" / "svc_api_gateway"
    _write(gw / "__init__.py", "")
    _write(gw / "app.py", '"""API gateway entrypoint."""\n\ndef health() -> dict[str, str]:\n    return {"status": "ok"}\n')

    if scale == "full":
        num_pkgs = 20
        mods_per_pkg = 121  # 20 * 122 files (~2,440 files) -> ~2,900 total .py files
    else:
        num_pkgs = 5
        mods_per_pkg = 4

    for pkg_idx in range(num_pkgs):
        pkg_dir = root / "packages" / f"pkg_{pkg_idx:02d}"
        _write(pkg_dir / "__init__.py", f'"""Shared package {pkg_idx:02d}."""\n')
        for mod_idx in range(mods_per_pkg):
            _write(
                pkg_dir / f"mod_{mod_idx:03d}.py",
                "\n".join([
                    f'"""Module mod_{mod_idx:03d} in pkg_{pkg_idx:02d}."""',
                    "",
                    f"MODULE_CODE = 'pkg_{pkg_idx:02d}.mod_{mod_idx:03d}'",
                    "",
                    f"def compute_{mod_idx:03d}(x: int) -> int:",
                    f"    return x + {pkg_idx + mod_idx}",
                    "",
                ]),
            )


def _write_policy_and_sandbox_fixtures(root: Path, *, include_oversized: bool) -> None:
    _write(root / ".github" / "workflows" / "ci.yml", "name: CI\non: [push]\n")
    _write(root / ".claude" / "settings.json", '{"permissions": {"allow": []}}\n')
    _write(root / ".claude" / "settings.local.json", '{"env": {"LOCAL_TOKEN": "secret-local"}}\n')
    _write(root / ".env.example", "APP_ENV=development\nPORT=8000\n")
    _write(root / "services" / "svc_api_gateway" / ".env.example", "GATEWAY_TIMEOUT=30\n")
    _write(root / ".env", "SECRET_API_KEY=do-not-read-me\n")
    _write(root / ".npmrc", "//registry.npmjs.org/:_authToken=secret-npm\n")
    _write(root / ".pypirc", "[pypi]\npassword = secret-pypi\n")
    _write(root / ".mcp.local.json", '{"mcpServers": {"secret": {"token": "abc"}}}\n')
    _write(root / ".gitignore", ".mcp.json\ngenerated_cache/\n")
    _write(root / "services" / "svc_api_gateway" / ".mcp.json", '{"mcpServers": {}}\n')
    _write(root / "generated_cache" / "stale.py", "CACHED = True\n")
    if include_oversized:
        chunk = ("oversized diagnostic log line for scan limit testing\n") * 88_000
        _write(root / "data" / "oversized_fixture.txt", chunk)


def _compute_ast_ground_truth(root: Path) -> dict[str, Any]:
    # 1. legacy_core.py
    monolith_path = root / "services" / "svc_monolith" / "legacy_core.py"
    monolith_tree = ast.parse(monolith_path.read_text(encoding="utf-8"))
    total_defs = 0
    module_level_funcs = 0
    methods = 0
    nested_funcs = 0

    def _walk_monolith(body: list[ast.stmt], scope: list[str], in_class: bool) -> None:
        nonlocal total_defs, module_level_funcs, methods, nested_funcs
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                _walk_monolith(stmt.body, [*scope, stmt.name], True)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_defs += 1
                if not scope:
                    module_level_funcs += 1
                elif in_class:
                    methods += 1
                else:
                    nested_funcs += 1
                _walk_monolith(stmt.body, [*scope, stmt.name], False)

    _walk_monolith(monolith_tree.body, [], False)

    # 2. svc_workers long functions (>= 150 lines)
    long_module_funcs: list[dict[str, Any]] = []
    long_nested_funcs: list[dict[str, Any]] = []
    workers_dir = root / "services" / "svc_workers"
    for py_file in sorted(workers_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        rel = str(py_file.relative_to(root))
        tree = ast.parse(py_file.read_text(encoding="utf-8"))

        def _walk_worker(body: list[ast.stmt], scope: list[str], in_class: bool) -> None:
            for stmt in body:
                if isinstance(stmt, ast.ClassDef):
                    _walk_worker(stmt.body, [*scope, stmt.name], True)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    span = (getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno) - stmt.lineno + 1
                    if span >= 150:
                        rec = {
                            "file": rel,
                            "function": stmt.name,
                            "qualified": ".".join([*scope, stmt.name]),
                            "lines": span,
                        }
                        if not scope:
                            long_module_funcs.append(rec)
                        elif not in_class:
                            long_nested_funcs.append(rec)
                    _walk_worker(stmt.body, [*scope, stmt.name], False)

        _walk_worker(tree.body, [], False)

    long_module_funcs.sort(key=lambda r: (r["file"], r["function"]))
    long_nested_funcs.sort(key=lambda r: (r["file"], r["qualified"]))

    # 3. svc_models class field count
    total_class_fields = 0
    models_dir = root / "services" / "svc_models"
    for py_file in sorted(models_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                seen: set[str] = set()
                for member in stmt.body:
                    if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                        seen.add(member.target.id)
                    elif isinstance(member, ast.Assign):
                        for tgt in member.targets:
                            if isinstance(tgt, ast.Name):
                                seen.add(tgt.id)
                total_class_fields += len(seen)

    return {
        "legacy_core": {
            "total_defs": total_defs,
            "module_level_functions": module_level_funcs,
            "methods": methods,
            "nested_functions": nested_funcs,
        },
        "long_functions_module_level": long_module_funcs,
        "long_functions_nested": long_nested_funcs,
        "total_class_fields": total_class_fields,
    }


def generate_monorepo_corpus(
    root: Path,
    *,
    seed: int = 20261128,
    scale: str = "full",
    include_oversized: bool | None = None,
) -> dict[str, Any]:
    """Generate a deterministic complex Python monorepo corpus and ground-truth manifest."""
    if scale not in {"small", "full"}:
        raise ValueError(f"Unsupported scale {scale!r}; expected 'small' or 'full'")
    resolved_oversized = (scale == "full") if include_oversized is None else include_oversized
    _ = random.Random(seed)

    root.mkdir(parents=True, exist_ok=True)
    _write(root / "services" / "__init__.py", "")
    _write(root / "packages" / "__init__.py", "")

    tool_stats = _write_agent_tools_service(root)
    monolith_stats = _write_monolith_file(root, scale=scale)
    _write_workers(root, scale=scale)
    _write_models(root, scale=scale)
    _write_background_packages(root, scale=scale)
    _write_policy_and_sandbox_fixtures(root, include_oversized=resolved_oversized)

    ast_truth = _compute_ast_ground_truth(root)
    total_py_files = sum(1 for _ in root.rglob("*.py"))

    long_fn_expected = [
        {"file": r["file"], "function": r["function"]}
        for r in ast_truth["long_functions_module_level"]
    ]
    eval_cases = [
        {
            "eval_id": "tool_contract_audit",
            "prompt": (
                "In the codebase at `{target_path}`, audit every registered agent tool under "
                "`services/svc_agent_tools/`. For each tool, compare its declared description "
                "(including any referenced description dictionaries, models, and delegated helper "
                "functions) against its actual implementation behavior. Report any contradictions "
                "between what the description promises and what the code actually does. "
                "Return JSON matching: "
                "`{\"contradictions\": [{\"tool\": \"...\", \"file_line\": \"...\", \"claim\": \"...\", \"actual\": \"...\"}]}`."
            ),
            "notes": (
                "Neutral per-item audit of 50 tools with 7 planted contradictions "
                "(including 2 inside the >270 KB scheduling mega-suite and 2 in delegated helpers)."
            ),
            "assertions": [
                {"id": "json_shape", "kind": "contains_all", "values": ["contradictions"]},
            ],
        },
        {
            "eval_id": "long_functions_module_level",
            "prompt": (
                "In the codebase at `{target_path}`, inspect all Python files under `services/svc_workers/`. "
                "Find every module-level function (`depth == 0`, excluding methods and nested functions) "
                "whose definition spans at least 150 lines. Return a JSON array of "
                "`{\"file\": ..., \"function\": ...}` objects sorted by `file` then `function`."
            ),
            "assertions": [
                {"id": "exact_long_functions", "kind": "json_equals", "values": [json.dumps(long_fn_expected)]},
            ],
        },
        {
            "eval_id": "monolith_outline_counts",
            "prompt": (
                "In the codebase at `{target_path}`, inspect `services/svc_monolith/legacy_core.py` "
                "and return a JSON object with exact counts: "
                "`{\"total_defs\": int, \"module_level_functions\": int, \"methods\": int, \"nested_functions\": int}`."
            ),
            "assertions": [
                {
                    "id": "exact_outline_counts",
                    "kind": "json_equals",
                    "values": [json.dumps(ast_truth["legacy_core"])],
                },
            ],
        },
        {
            "eval_id": "model_field_inventory",
            "prompt": (
                "In the codebase at `{target_path}`, count the total number of class-level annotated/assigned "
                "fields across all classes in `services/svc_models/`. Return JSON: `{\"total_class_fields\": int}`."
            ),
            "assertions": [
                {
                    "id": "exact_field_count",
                    "kind": "json_equals",
                    "values": [json.dumps({"total_class_fields": ast_truth["total_class_fields"]})],
                },
            ],
        },
    ]

    return {
        "scale": scale,
        "seed": seed,
        "target_path": str(root.resolve()),
        "total_py_files": total_py_files,
        "tool_count": tool_stats["tool_count"],
        "scheduling_suite_bytes": tool_stats["scheduling_suite_bytes"],
        "legacy_core_lines": monolith_stats["lines"],
        "ground_truth": ast_truth,
        "planted_bugs": PLANTED_BUG_SPECS,
        "audit_prompt": eval_cases[0]["prompt"],
        "evals": eval_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ~2,900-file Python monorepo benchmark.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for generated monorepo")
    parser.add_argument("--scale", choices=["small", "full"], default="full", help="Corpus scale")
    parser.add_argument("--seed", type=int, default=20261128, help="Deterministic seed")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional path to write JSON manifest")
    args = parser.parse_args()

    manifest = generate_monorepo_corpus(args.output, seed=args.seed, scale=args.scale)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "output": manifest["target_path"],
        "scale": manifest["scale"],
        "total_py_files": manifest["total_py_files"],
        "tool_count": manifest["tool_count"],
        "scheduling_suite_bytes": manifest["scheduling_suite_bytes"],
        "legacy_core_lines": manifest["legacy_core_lines"],
        "legacy_core_defs": manifest["ground_truth"]["legacy_core"]["total_defs"],
        "long_module_functions": len(manifest["ground_truth"]["long_functions_module_level"]),
        "long_nested_functions": len(manifest["ground_truth"]["long_functions_nested"]),
        "total_class_fields": manifest["ground_truth"]["total_class_fields"],
        "planted_bugs": len(manifest["planted_bugs"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
