import json
import os
import stat
import sys

import pytest

from rlm_tools import setup_cli

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


def test_print_only_does_not_install(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rlm-tools-setup", "--print-json"])
    monkeypatch.setattr(setup_cli, "detect_supported_clients", lambda: pytest.fail("install attempted"))
    setup_cli.main()
    assert '"mcpServers"' in capsys.readouterr().out


def test_partial_install_failure_sets_exit_code_even_with_policy(monkeypatch, tmp_path):
    path = tmp_path / "AGENTS.md"
    monkeypatch.setattr(sys, "argv", ["rlm-tools-setup", "--write-policy", str(path)])
    monkeypatch.setattr(setup_cli, "detect_supported_clients", lambda: ["claude", "codex"])
    monkeypatch.setattr(setup_cli, "run_install", lambda client, dry_run: (client == "claude", client))
    with pytest.raises(SystemExit) as error:
        setup_cli.main()
    assert error.value.code == 1
    assert POLICY_START in path.read_text()


def test_dry_run_preserves_existing_policy(monkeypatch, tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text("Existing rules\n")
    monkeypatch.setattr(sys, "argv", ["rlm-tools-setup", "--client", "codex", "--dry-run", "--write-policy", str(path)])
    setup_cli.main()
    assert path.read_text() == "Existing rules\n"


@pytest.mark.parametrize("markers", [[POLICY_START], [POLICY_END, POLICY_START], [POLICY_START, POLICY_START, POLICY_END]])
def test_malformed_policy_markers_preserve_file(tmp_path, markers):
    path = tmp_path / "AGENTS.md"
    original = "\nExisting instructions\n".join(markers)
    path.write_text(original)
    with pytest.raises(ValueError):
        setup_cli.write_policy(path)
    assert path.read_text() == original


def test_write_policy_rejects_symlink(tmp_path):
    note = tmp_path / "notes.md"
    note.write_text("Original notes\n")
    policy = tmp_path / "AGENTS.md"
    policy.symlink_to(note)
    with pytest.raises(ValueError, match="regular file"):
        setup_cli.write_policy(policy)
    assert policy.is_symlink()
    assert note.read_text() == "Original notes\n"


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX permissions")
def test_write_policy_preserves_file_mode(tmp_path):
    policy = tmp_path / "AGENTS.md"
    policy.write_text("Existing policy\n")
    policy.chmod(0o640)
    setup_cli.write_policy(policy)
    assert stat.S_IMODE(policy.stat().st_mode) == 0o640
    assert POLICY_START in policy.read_text()

