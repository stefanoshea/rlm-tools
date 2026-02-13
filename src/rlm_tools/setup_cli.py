import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


POLICY_START = "<!-- rlm-tools-policy:start -->"
POLICY_END = "<!-- rlm-tools-policy:end -->"


def build_install_command(client: str) -> list[str]:
    if client == "claude":
        return ["claude", "mcp", "add", "rlm-tools", "--", "uvx", "rlm-tools"]
    if client == "codex":
        return ["codex", "mcp", "add", "rlm-tools", "--", "uvx", "rlm-tools"]
    raise ValueError(f"Unsupported client: {client}")


def detect_supported_clients() -> list[str]:
    clients = []
    if shutil.which("claude"):
        clients.append("claude")
    if shutil.which("codex"):
        clients.append("codex")
    return clients


def run_install(client: str, dry_run: bool = False) -> tuple[bool, str]:
    command = build_install_command(client)
    command_str = " ".join(shlex.quote(part) for part in command)

    if dry_run:
        return True, f"[dry-run] {command_str}"

    if shutil.which(command[0]) is None:
        return False, f"{command[0]} CLI not found on PATH"

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        output = result.stdout.strip() or "Configured successfully."
        return True, output

    output = result.stderr.strip() or result.stdout.strip() or "Unknown error"
    return False, output


def render_json_mcp_config() -> str:
    config = {
        "mcpServers": {
            "rlm-tools": {
                "command": "uvx",
                "args": ["rlm-tools"],
            }
        }
    }
    return json.dumps(config, indent=2)


def render_policy_block() -> str:
    return (
        f"{POLICY_START}\n"
        "## RLM Tools Policy\n\n"
        "For codebase exploration, prefer `rlm-tools` over standard read/grep/glob tools.\n\n"
        "Workflow:\n"
        "1. Start with `rlm_start(path, query)`.\n"
        "2. Use `rlm_execute(session_id, code)` to batch search/read/analyze steps.\n"
        "3. Print concise summaries rather than raw payloads.\n"
        "4. End with `rlm_end(session_id)` when finished.\n\n"
        "Fallback rule:\n"
        "- Use standard read/grep/glob only if `rlm-tools` is unavailable or errors.\n"
        f"{POLICY_END}\n"
    )


def upsert_policy_block(existing_text: str) -> str:
    block = render_policy_block()
    start_idx = existing_text.find(POLICY_START)
    end_idx = existing_text.find(POLICY_END)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_idx = end_idx + len(POLICY_END)
        replacement = block.rstrip("\n")
        return existing_text[:start_idx] + replacement + existing_text[end_idx:]

    if not existing_text.strip():
        return block
    return existing_text.rstrip() + "\n\n" + block


def write_policy(path: Path) -> None:
    existing_text = path.read_text() if path.exists() else ""
    updated_text = upsert_policy_block(existing_text)
    path.write_text(updated_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rlm-tools-setup",
        description=(
            "Configure rlm-tools for Claude/Codex and optionally write policy guidance "
            "that makes agents prefer rlm-tools for code exploration."
        ),
    )
    parser.add_argument(
        "--client",
        choices=["auto", "claude", "codex", "json"],
        default="auto",
        help="Target client to configure (default: auto-detect claude/codex).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print install commands without executing them.",
    )
    parser.add_argument(
        "--print-policy",
        action="store_true",
        help="Print the recommended policy block for AGENTS.md or system instructions.",
    )
    parser.add_argument(
        "--write-policy",
        metavar="PATH",
        help="Write or update the policy block in a markdown file (e.g. AGENTS.md).",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print JSON MCP config snippet for non-CLI clients.",
    )

    args = parser.parse_args()

    any_success = False
    install_targets: list[str] = []
    if args.client == "auto":
        install_targets = detect_supported_clients()
    elif args.client in {"claude", "codex"}:
        install_targets = [args.client]

    if args.client == "json" or args.print_json:
        print("JSON MCP config snippet:")
        print(render_json_mcp_config())
        print()

    for client in install_targets:
        ok, output = run_install(client, dry_run=args.dry_run)
        status = "ok" if ok else "error"
        print(f"[{status}] {client}: {output}")
        any_success = any_success or ok

    if args.client == "auto" and not install_targets:
        print("No supported client CLI found (expected `claude` and/or `codex` on PATH).")
        print("Use --print-json to configure another MCP client.")

    if args.write_policy:
        policy_path = Path(args.write_policy)
        write_policy(policy_path)
        print(f"Policy block written to {policy_path}")
        any_success = True
    elif args.print_policy:
        print("Recommended policy block:")
        print(render_policy_block())

    if (
        not install_targets
        and args.client != "json"
        and not args.write_policy
        and not args.print_json
        and not args.print_policy
    ):
        print("Nothing to do. Try `rlm-tools-setup --print-policy --print-json`.", file=sys.stderr)
        sys.exit(1)

    if install_targets and not any_success and not args.write_policy:
        sys.exit(1)

