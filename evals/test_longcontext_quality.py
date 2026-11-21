"""Offline reference traces for the structured long-context benchmark cases."""

import ast
import json
import re
from pathlib import Path

import pytest

from evals.conftest import DEFAULT_LONG_CONTEXT_CORPUS_PATH
from evals.grade_eval import grade_eval_case, load_eval_specs
from rlm_tools.sandbox import Sandbox

SPEC_PATH = Path(__file__).parent / "specs" / "rlm_long_context.json"


@pytest.fixture(scope="module")
def corpus():
    sandbox = Sandbox(base_path=str(DEFAULT_LONG_CONTEXT_CORPUS_PATH), max_output_chars=50_000)
    try:
        yield sandbox
    finally:
        sandbox.close()


def explore(corpus, code):
    """Run discovery and file reads through the sandbox worker and its helpers."""
    result = corpus.execute("import json\n" + code)
    assert result.error is None, result.error
    return json.loads(result.stdout)


def read_sources(corpus, paths):
    return explore(corpus, f"print(json.dumps(read_files({paths!r})))")


def definition(source, name):
    return next(node for node in ast.parse(source).body if getattr(node, "name", None) == name)


def assignment(source, name):
    for node in ast.parse(source).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return node.value
    pytest.fail(f"No assignment to {name}")


def grade_trace(eval_id, observed, wrong_answers):
    spec = next(spec for spec in load_eval_specs(SPEC_PATH) if spec.eval_id == eval_id)
    assert grade_eval_case(spec, json.dumps(observed))["score"] == 1
    for wrong in wrong_answers:
        assert grade_eval_case(spec, json.dumps(wrong))["score"] == 0, wrong


def test_notification_dispatch_routes(corpus):
    dispatcher = "app/notifications/dispatcher.py"
    settings = "app/settings.py"
    config = "settings.yaml"
    channel_files = explore(
        corpus,
        "print(json.dumps(sorted(glob_files('app/notifications/channels/**/*.py'))))",
    )
    sources = read_sources(corpus, [dispatcher, settings, config, *channel_files])
    lookup = assignment(sources[settings], "NOTIFICATION_CHANNEL_MAP")
    assert lookup.func.id == "get_setting"
    config_group, config_key = ast.literal_eval(lookup.args[0]).split(".")
    # Read the two-level mapping from the checked-in YAML without a YAML dependency.
    group = sources[config].split(f"\n{config_group}:\n", 1)[1].split("\n\n", 1)[0]
    mapping = group.split(f"  {config_key}:\n", 1)[1]
    routes = dict(re.findall(r'^    (\w+): "(\w+)"$', mapping, flags=re.MULTILINE))

    loader = definition(sources[dispatcher], "_load_registry")
    registry = {
        ast.literal_eval(node.targets[0].slice): node.value.id
        for node in loader.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
    }
    channel_classes = {
        node.name
        for file in channel_files
        for node in ast.parse(sources[file]).body
        if isinstance(node, ast.ClassDef)
    }
    dispatch = definition(sources[dispatcher], "dispatch")
    dispatch_calls = {ast.unparse(node.func) for node in ast.walk(dispatch) if isinstance(node, ast.Call)}
    assert {"_load_registry", "NOTIFICATION_CHANNEL_MAP.get", "_CHANNEL_REGISTRY.get",
            "channel_cls", "channel_instance.send"} <= dispatch_calls
    route_lookup = next(
        node for node in ast.walk(dispatch)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "NOTIFICATION_CHANNEL_MAP.get"
    )
    assert [ast.unparse(arg) for arg in route_lookup.args] == ["event_type", "DEFAULT_CHANNEL"]
    default = ast.literal_eval(assignment(sources[dispatcher], "DEFAULT_CHANNEL"))
    observed = {
        "event_channels": {event: registry[channel] for event, channel in routes.items()},
        "default_channel": registry[default],
        "unregistered_channel_classes": sorted(channel_classes.difference(registry.values())),
        "dispatcher_file": dispatcher,
        "config_file": config,
    }
    grade_trace("notification_dispatch_routes", observed, [
        {**observed, "event_channels": {"refund_issued": "WebhookChannel"}},
        {**observed, "default_channel": "SlackChannel"},
        {**observed, "unregistered_channel_classes": []},
    ])


def test_middleware_active_stack(corpus):
    hits = explore(corpus, "print(json.dumps(grep('^def build_middleware_chain', 'app')))")
    assert len(hits) == 1
    chain_path = hits[0]["file"]
    chain = read_sources(corpus, [chain_path])[chain_path]
    declared_order = ast.literal_eval(assignment(chain, "MIDDLEWARE_ORDER"))
    registry_node = assignment(chain, "_MIDDLEWARE_REGISTRY")
    registry = {ast.literal_eval(key): value.id for key, value in zip(registry_node.keys, registry_node.values)}
    builder = definition(chain, "build_middleware_chain")
    loop = next(node for node in builder.body if isinstance(node, ast.For))
    assert ast.unparse(loop.iter) == "reversed(MIDDLEWARE_ORDER)"
    assert ast.unparse(loop.body[0]) == "middleware_cls = _MIDDLEWARE_REGISTRY.get(name)"
    condition = next(node for node in loop.body if isinstance(node, ast.If))
    assert ast.unparse(condition.test) == "middleware_cls is not None"
    assert ast.unparse(condition.body[0]) == "wrapped = middleware_cls(wrapped)"
    active = [registry[name] for name in declared_order if name in registry]
    auth_import = next(
        node.module for node in ast.parse(chain).body
        if isinstance(node, ast.ImportFrom) and any(alias.name in active for alias in node.names)
    )
    auth_file = auth_import.replace(".", "/") + ".py"
    auth_source = read_sources(corpus, [auth_file])[auth_file]
    auth_class = definition(auth_source, active[0])
    init = next(node for node in auth_class.body if getattr(node, "name", None) == "__init__")
    exclusion = next(
        node.value for node in init.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "self._exclude_paths"
    )
    observed = {
        "outermost_to_innermost": active,
        "skipped_in_declared_order": [name for name in declared_order if name not in registry],
        "default_excluded_prefixes": ast.literal_eval(exclusion.values[1]),
        "chain_file": chain_path,
        "auth_file": auth_file,
    }
    grade_trace("middleware_active_stack", observed, [
        {**observed, "outermost_to_innermost": declared_order, "skipped_in_declared_order": []},
        {**observed, "default_excluded_prefixes": ["/health"]},
        {**observed, "auth_file": "app/auth/legacy_auth.py"},
    ])


def test_bootstrap_export_outcomes(corpus):
    hits = explore(corpus, "print(json.dumps(grep(r'^def (boot|build_export_pipeline)\\(', 'app')))")
    bootstrap = next(hit["file"] for hit in hits if hit["line"] == "def boot() -> None:")
    source = read_sources(corpus, [bootstrap])[bootstrap]
    boot = definition(source, "boot")
    factory_import = next(
        node.module for node in ast.parse(source).body
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "build_export_pipeline" for alias in node.names)
    )
    factory_path = factory_import.replace(".", "/") + ".py"
    assert factory_path in {hit["file"] for hit in hits}
    assert any(ast.unparse(node) == "pipeline = build_export_pipeline()" for node in boot.body)
    assert any(ast.unparse(node) == "pipeline.run()" for node in boot.body)

    factory_source = read_sources(corpus, [factory_path])[factory_path]
    factory = definition(factory_source, "build_export_pipeline")
    builders_import = next(
        node.module for node in ast.parse(factory_source).body
        if isinstance(node, ast.ImportFrom) and any(alias.name.endswith("Builder") for alias in node.names)
    )
    steps = builders_import.replace(".", "/") + ".py"
    flags = "app/feature_flags.py"
    sources = read_sources(corpus, [steps, flags])
    branch, fallback = factory.body
    assert ast.unparse(branch.test.func) == "feature_flags.is_enabled"
    controlling_flag = branch.test.args[0].attr
    assert isinstance(definition(sources[steps], "DryRunExportBuilder"), ast.ClassDef)
    flag_values = assignment(sources[flags], "_FLAGS")
    defaults = {key.id: ast.literal_eval(value) for key, value in zip(flag_values.keys, flag_values.values)}
    assert "EXPORT_DRY_RUN" in defaults

    def describe(return_statement):
        builder = return_statement.value.func.id
        builder_class = definition(sources[steps], builder)
        run = next(node for node in builder_class.body if getattr(node, "name", None) == "run")
        return {"builder": builder, "run_result": ast.literal_eval(run.body[0].value)}

    observed = {
        "factory_file": factory_path,
        "builders_file": steps,
        "flag": controlling_flag,
        "checked_in_default": defaults[controlling_flag],
        "when_enabled": describe(branch.body[0]),
        "when_disabled": describe(fallback),
        "dry_run_reachable": any(
            isinstance(node, ast.Name) and node.id == "DryRunExportBuilder" for node in ast.walk(factory)
        ),
    }
    grade_trace("bootstrap_export_outcomes", observed, [
        {**observed, "when_enabled": observed["when_disabled"], "when_disabled": observed["when_enabled"]},
        {**observed, "dry_run_reachable": True},
        {**observed, "factory_file": "app/export/legacy_pipeline.py"},
    ])


@pytest.mark.parametrize(
    ("eval_id", "expected", "wrong_fields"),
    [
        (
            "export_pipeline_gate",
            {
                "flag": "EXPORT_PIPELINE_V2",
                "enabled_builder": "DeltaExportBuilder",
                "factory_file": "app/export/pipeline.py",
            },
            {"enabled_builder": "LegacyExportBuilder"},
        ),
        (
            "shipment_retry_budget",
            {
                "workflow": "shipment",
                "budget_constant": "SHIPMENT_RETRY_BUDGET",
                "budget": 5,
                "missing_workflow_budget": 2,
                "budget_file": "app/retry_config.py",
            },
            {"missing_workflow_budget": 12},
        ),
        (
            "reconciliation_success_metrics",
            {
                "helper": "emit_success_metrics",
                "positive_row_metrics": ["reconcile.rows_updated", "reconcile.latency_ms"],
                "nonpositive_row_metrics": [],
            },
            {"nonpositive_row_metrics": ["reconcile.error_count", "reconcile.retry_count"]},
        ),
        (
            "stripe_webhook_secret_chain",
            {
                "config_key": "stripe.webhook_secret",
                "python_constant": "STRIPE_WEBHOOK_SECRET",
                "settings_file": "app/settings.py",
            },
            {"config_key": "STRIPE_WEBHOOK_SECRET", "python_constant": "stripe.webhook_secret"},
        ),
        (
            "auth_token_algorithm",
            {
                "configured_algorithm": "HS512",
                "missing_setting_algorithm": "HS256",
                "config_key": "auth.token_algorithm",
                "config_file": "settings.yaml",
                "algorithm_constant_file": "app/auth/middleware.py",
            },
            {"configured_algorithm": "HS256", "missing_setting_algorithm": "HS512"},
        ),
        (
            "refund_notification_channel",
            {
                "event": "refund_issued",
                "sender_class": "WebhookChannel",
                "unmapped_channel": "email",
            },
            {"sender_class": "EmailChannel", "unmapped_channel": "webhook"},
        ),
        (
            "product_cache_ttl",
            {
                "ttl_seconds": 300,
                "ttl_constant": "PRODUCT_CACHE_TTL",
                "ttl_file": "app/cache/registry.py",
                "default_backend": "RedisBackend",
            },
            {"ttl_seconds": 1300},
        ),
        (
            "order_terminal_states",
            {
                "terminal_states": ["cancelled", "delivered", "refunded"],
                "ledger_file": "app/reconcile/ledger.py",
                "refund_metrics": ["refund.amount", "refund.count"],
            },
            {"terminal_states": ["cancelled", "delivered", "refunded", "returned"]},
        ),
    ],
)
def test_existing_prompt_grading(eval_id, expected, wrong_fields):
    grade_trace(eval_id, expected, [{**expected, **wrong_fields}])
