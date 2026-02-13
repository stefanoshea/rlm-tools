import json

from rlm_tools.setup_cli import (
    POLICY_END,
    POLICY_START,
    build_install_command,
    render_json_mcp_config,
    upsert_policy_block,
)


def test_build_install_command():
    assert build_install_command("claude") == [
        "claude",
        "mcp",
        "add",
        "rlm-tools",
        "--",
        "uvx",
        "rlm-tools",
    ]
    assert build_install_command("codex") == [
        "codex",
        "mcp",
        "add",
        "rlm-tools",
        "--",
        "uvx",
        "rlm-tools",
    ]


def test_render_json_mcp_config():
    data = json.loads(render_json_mcp_config())
    assert data["mcpServers"]["rlm-tools"]["command"] == "uvx"
    assert data["mcpServers"]["rlm-tools"]["args"] == ["rlm-tools"]


def test_upsert_policy_block_appends_when_missing():
    existing = "# Agent Rules\n\nKeep responses concise.\n"
    updated = upsert_policy_block(existing)
    assert POLICY_START in updated
    assert POLICY_END in updated
    assert "prefer `rlm-tools` over standard read/grep/glob tools" in updated


def test_upsert_policy_block_replaces_existing():
    existing = (
        "# Agent Rules\n\n"
        f"{POLICY_START}\n"
        "Old content.\n"
        f"{POLICY_END}\n"
        "\nMore content.\n"
    )
    updated = upsert_policy_block(existing)
    assert "Old content." not in updated
    assert updated.count(POLICY_START) == 1
    assert updated.count(POLICY_END) == 1
    assert "More content." in updated

