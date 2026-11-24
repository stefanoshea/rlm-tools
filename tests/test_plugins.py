import json
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
KEBAB_CASE_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

MCP_MANIFEST_PATHS = (
    REPO_ROOT / "plugins" / "rlm-tools" / ".mcp.json",
    REPO_ROOT / "plugins" / "rlm-tools" / "mcp.json",
    REPO_ROOT / "gemini-extension.json",
)


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pyproject_version() -> str:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    try:
        import tomllib
    except ModuleNotFoundError:
        match = re.search(
            r'(?ms)^\[project\]\s.*?^version\s*=\s*"([^"]+)"',
            pyproject_path.read_text(encoding="utf-8"),
        )
        assert match is not None, "Could not parse [project].version from pyproject.toml"
        return match.group(1)
    with pyproject_path.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _resolve_within_repo(relative_path: str) -> pathlib.Path:
    candidate = (REPO_ROOT / relative_path).resolve()
    assert candidate.is_relative_to(REPO_ROOT), f"Path escapes repo root: {relative_path}"
    return candidate


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", text, re.DOTALL)
    assert match is not None, "SKILL.md must begin with YAML frontmatter delimited by ---"
    raw_meta, body = match.groups()
    meta: dict[str, str] = {}
    for line in raw_meta.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        assert sep == ":", f"Expected key: value frontmatter line, got: {line!r}"
        meta[key.strip()] = value.strip()
    return meta, body


def test_plugin_and_registry_versions_match_pyproject():
    version = _pyproject_version()
    expected_pkg_spec = f"rlm-tools=={version}"

    server = _load_json(REPO_ROOT / "server.json")
    assert server["version"] == version
    pypi_pkgs = [
        pkg for pkg in server["packages"]
        if pkg.get("registryType") == "pypi" and pkg.get("identifier") == "rlm-tools"
    ]
    assert len(pypi_pkgs) == 1
    assert pypi_pkgs[0]["version"] == version

    claude_marketplace = _load_json(REPO_ROOT / ".claude-plugin" / "marketplace.json")
    assert claude_marketplace["plugins"][0]["version"] == version

    claude_plugin = _load_json(REPO_ROOT / "plugins" / "rlm-tools" / ".claude-plugin" / "plugin.json")
    assert claude_plugin["version"] == version

    agent_plugin = _load_json(REPO_ROOT / "plugins" / "rlm-tools" / "plugin.json")
    assert agent_plugin["version"] == version

    codex_marketplace = _load_json(REPO_ROOT / ".agents" / "plugins" / "marketplace.json")
    if "version" in codex_marketplace["plugins"][0]:
        assert codex_marketplace["plugins"][0]["version"] == version

    gemini_ext = _load_json(REPO_ROOT / "gemini-extension.json")
    assert gemini_ext["version"] == version

    for manifest_path in MCP_MANIFEST_PATHS:
        manifest = _load_json(manifest_path)
        assert manifest["mcpServers"]["rlm-tools"]["args"] == [expected_pkg_spec]


def test_claude_plugin_and_marketplace_structure():
    marketplace = _load_json(REPO_ROOT / ".claude-plugin" / "marketplace.json")
    assert KEBAB_CASE_RE.match(marketplace["name"])
    assert isinstance(marketplace["owner"]["name"], str) and marketplace["owner"]["name"].strip()
    assert len(marketplace["plugins"]) == 1

    entry = marketplace["plugins"][0]
    assert KEBAB_CASE_RE.match(entry["name"])
    plugin_dir = _resolve_within_repo(entry["source"])
    assert plugin_dir == REPO_ROOT / "plugins" / "rlm-tools"
    assert (plugin_dir / ".claude-plugin" / "plugin.json").is_file()
    assert (plugin_dir / ".mcp.json").is_file()
    assert (plugin_dir / "skills" / "rlm-explore" / "SKILL.md").is_file()

    plugin_manifest = _load_json(plugin_dir / ".claude-plugin" / "plugin.json")
    assert KEBAB_CASE_RE.match(plugin_manifest["name"])


def test_agent_plugin_v1_and_codex_marketplace_structure():
    plugin_dir = REPO_ROOT / "plugins" / "rlm-tools"

    plugin_manifest = _load_json(plugin_dir / "plugin.json")
    assert plugin_manifest["$schema"] == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    assert KEBAB_CASE_RE.match(plugin_manifest["name"])
    allowed_plugin_keys = {
        "$schema", "name", "version", "description", "author",
        "homepage", "repository", "license", "keywords", "extensions",
    }
    assert set(plugin_manifest) <= allowed_plugin_keys
    assert set(plugin_manifest["author"]) <= {"name", "email", "url"}

    mcp_manifest = _load_json(plugin_dir / "mcp.json")
    assert mcp_manifest["$schema"] == "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
    server_cfg = mcp_manifest["mcpServers"]["rlm-tools"]
    assert server_cfg["type"] == "stdio"
    assert server_cfg["command"] == "uvx"
    assert set(server_cfg) <= {"type", "command", "args", "env", "cwd"}

    marketplace = _load_json(REPO_ROOT / ".agents" / "plugins" / "marketplace.json")
    assert KEBAB_CASE_RE.match(marketplace["name"])
    assert len(marketplace["plugins"]) == 1
    entry = marketplace["plugins"][0]
    assert entry["source"]["source"] == "local"
    local_path = entry["source"]["path"]
    assert local_path.startswith("./") and local_path != "./"
    assert _resolve_within_repo(local_path) == plugin_dir
    assert entry["policy"]["installation"] in {"AVAILABLE", "INSTALLED_BY_DEFAULT", "NOT_AVAILABLE"}


def test_gemini_discovers_the_bundled_skill_on_demand():
    ext = _load_json(REPO_ROOT / "gemini-extension.json")
    assert KEBAB_CASE_RE.match(ext["name"])
    assert ext["mcpServers"]["rlm-tools"]["command"] == "uvx"
    assert "contextFileName" not in ext
    skill = REPO_ROOT / "skills" / "rlm-explore" / "SKILL.md"
    meta, _ = _parse_frontmatter(skill.read_text(encoding="utf-8"))
    assert meta["name"] == "rlm-explore"
    plugin_skill = REPO_ROOT / "plugins" / "rlm-tools" / "skills" / "rlm-explore" / "SKILL.md"
    assert skill.read_bytes() == plugin_skill.read_bytes(), "Keep both packaged skills in sync"


def test_plugin_mcp_configs_preserve_local_workspace_boundaries():
    forbidden_env_keys = {
        "RLM_ALLOWED_ROOTS",
        "RLM_EXTERNAL_ROOTS",
        "ANTHROPIC_API_KEY",
        "TYPESAFE_API_KEY",
    }
    for manifest_path in MCP_MANIFEST_PATHS:
        server_cfg = _load_json(manifest_path)["mcpServers"]["rlm-tools"]
        assert "cwd" not in server_cfg, f"{manifest_path} must leave cwd selection to the client"
        env = server_cfg.get("env", {})
        assert not (forbidden_env_keys & set(env)), (
            f"{manifest_path} must not pre-set workspace boundary or credential env vars"
        )

    plugin = REPO_ROOT / "plugins" / "rlm-tools"
    portable = _load_json(plugin / "mcp.json")["mcpServers"]
    overlay = _load_json(plugin / ".codex-plugin" / "plugin.json")["mcpServers"]
    assert set(overlay) == set(portable)
    assert overlay["rlm-tools"] == {
        "command": portable["rlm-tools"]["command"],
        "env_vars": ["RLM_ALLOWED_ROOTS"],
    }


def test_skill_frontmatter_and_no_symlinks():
    skill_path = REPO_ROOT / "plugins" / "rlm-tools" / "skills" / "rlm-explore" / "SKILL.md"
    meta, body = _parse_frontmatter(skill_path.read_text(encoding="utf-8"))
    assert meta.get("name") == skill_path.parent.name
    assert meta.get("description")
    assert "disable-model-invocation" not in meta
    for tool_name in ("rlm_start", "rlm_execute", "rlm_end"):
        assert tool_name in body

    assert not (REPO_ROOT / "gemini-extension.json").is_symlink()
    for root_dir in (REPO_ROOT / ".claude-plugin", REPO_ROOT / ".agents", REPO_ROOT / "plugins", REPO_ROOT / "skills"):
        assert root_dir.is_dir() and not root_dir.is_symlink()
        for entry in root_dir.rglob("*"):
            assert not entry.is_symlink(), f"Unexpected symlink in plugin tree: {entry}"


def test_claude_plugin_subagent_rlm_explorer():
    agent_path = REPO_ROOT / "plugins" / "rlm-tools" / "agents" / "rlm-explorer.md"
    assert agent_path.is_file() and not agent_path.is_symlink()
    meta, body = _parse_frontmatter(agent_path.read_text(encoding="utf-8"))
    assert meta.get("name") == "rlm-explorer"
    assert meta.get("description")
    assert meta.get("model") == "haiku"
    assert "tools" not in meta
    disallowed = {item.strip() for item in meta.get("disallowedTools", "").split(",") if item.strip()}
    for tool_name in ("Bash", "Edit", "Write", "Read", "Grep", "Glob"):
        assert tool_name in disallowed
    for mcp_tool in ("rlm_start", "rlm_execute", "rlm_end"):
        assert mcp_tool in body
