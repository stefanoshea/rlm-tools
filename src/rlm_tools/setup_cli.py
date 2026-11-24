import argparse
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


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
        "Choose the exploration tool based on task shape:\n"
        "- Use `rlm-tools` for broad searches, multi-file audits/aggregations, large files, "
        "cross-file counting/filtering, and multi-step analysis.\n"
        "- Use standard read/grep/glob directly for quick narrow lookups (1-5 known small "
        "files or a single targeted grep).\n\n"
        "Workflow:\n"
        "1. Start once per task with `rlm_start(path, query)` (or pass an initial exploration "
        "`code=...` script).\n"
        "2. Batch multiple search/read/analyze steps into `rlm_execute(session_id, code)`, "
        "building on persistent variables, and pass `end_session=true` on your final "
        "`rlm_execute` (or finish with `rlm_end(session_id)`).\n"
        "3. Follow sandbox rules: use built-in helpers (`read_file`, `read_files`, `grep`, "
        "`grep_summary`, `grep_read`, `glob_files`, `tree`) and f-strings — do not import "
        "`os`, `glob`, `pathlib`, or `ast`, and do not call `open()`, `exec()`, `eval()`, "
        "`getattr()`, `dir()`, `type()`, or `.format()`.\n"
        "4. Aggregate in Python variables and print concise summaries rather than raw file dumps.\n\n"
        "Fallback rule:\n"
        "- Fall back to standard read/grep/glob if `rlm-tools` is unavailable or errors.\n"
        f"{POLICY_END}\n"
    )


def upsert_policy_block(existing_text: str) -> str:
    block = render_policy_block()
    if POLICY_START in existing_text or POLICY_END in existing_text:
        if existing_text.count(POLICY_START) != 1 or existing_text.count(POLICY_END) != 1:
            raise ValueError("Expected exactly one pair of rlm-tools policy markers")
        start_idx = existing_text.index(POLICY_START)
        end_idx = existing_text.index(POLICY_END) + len(POLICY_END)
        if end_idx <= start_idx:
            raise ValueError("The rlm-tools policy markers are out of order")
        replacement = block.rstrip("\n")
        return existing_text[:start_idx] + replacement + existing_text[end_idx:]

    if not existing_text.strip():
        return block
    return existing_text.rstrip() + "\n\n" + block


def write_policy(path: Path) -> None:
    try:
        original = path.lstat()
    except FileNotFoundError:
        original = None

    existing_text = ""
    if original is not None:
        if not stat.S_ISREG(original.st_mode):
            raise ValueError("Policy path must be a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        with os.fdopen(os.open(path, flags), encoding="utf-8", newline="") as source:
            if os.fstat(source.fileno()) != original:
                raise ValueError("Policy file changed while being opened")
            existing_text = source.read()

    updated_text = upsert_policy_block(existing_text)
    if updated_text == existing_text:
        return

    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                delete=False) as destination:
            temporary_path = Path(destination.name)
            destination.write(updated_text)
        if original is not None:
            temporary_path.chmod(original.st_mode & 0o777)
        try:
            current = path.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and original is not None:
            # Reading the file may change its access time.
            unchanged = (
                os.path.samestat(current, original)
                and current.st_mtime_ns == original.st_mtime_ns
                and current.st_ctime_ns == original.st_ctime_ns
            )
        else:
            unchanged = current is original
        if not unchanged:
            raise ValueError("Policy file changed during the update")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rlm-tools-setup",
        description=(
            "Configure rlm-tools for Claude/Codex and optionally write workload-aware policy "
            "guidance for code exploration."
        ),
    )
    parser.add_argument(
        "--client",
        choices=["auto", "claude", "codex", "json"],
        help="Target client to configure (default: auto-detect claude/codex).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview install commands and policy changes.",
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

    install_failed = False
    install_targets: list[str] = []
    print_only = (args.print_policy or args.print_json) and not args.write_policy
    client_mode = args.client or ("print" if print_only else "auto")
    if client_mode == "auto":
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
        install_failed = install_failed or not ok

    if client_mode == "auto" and not install_targets:
        print("No supported client CLI found (expected `claude` and/or `codex` on PATH).")
        print("Use --print-json to configure another MCP client.")

    if args.write_policy:
        policy_path = Path(args.write_policy)
        if args.dry_run:
            print(f"[dry-run] Would write policy block to {policy_path}")
        else:
            try:
                write_policy(policy_path)
            except (OSError, ValueError) as exc:
                parser.exit(1, f"Could not update {policy_path}: {exc}\n")
            print(f"Policy block written to {policy_path}")
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

    if install_failed:
        sys.exit(1)

