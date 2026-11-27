"""Prepare the pinned Prefect OSS realism benchmark directory and manifest."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
from typing import Any


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "prefect_spec.json"


def load_prefect_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _compute_repo_stats(repo_root: Path, scope_files: list[str]) -> dict[str, Any]:
    total_py_files = sum(1 for _ in repo_root.rglob("*.py"))
    scope_details: dict[str, dict[str, int]] = {}
    endpoint_count = 0
    for rel in scope_files:
        path = repo_root / rel
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel)
        routes = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any("router." in ast.unparse(dec) for dec in node.decorator_list):
                    routes += 1
        endpoint_count += routes
        scope_details[rel] = {
            "bytes": path.stat().st_size,
            "lines": len(text.splitlines()),
            "routes": routes,
        }
    return {
        "total_py_files": total_py_files,
        "endpoint_count": endpoint_count,
        "scope_details": scope_details,
    }


def setup_oss_benchmark(dest: Path, *, manifest_path: Path | None = None) -> dict[str, Any]:
    spec = load_prefect_spec()
    commit_sha = spec["commit_sha"]
    repo_url = spec["repo_url"]
    patch_path = HERE / spec["patch_file"]
    if not patch_path.is_file():
        raise FileNotFoundError(f"Missing patch file: {patch_path}")

    dest = dest.resolve()
    if (dest / ".git").is_dir():
        subprocess.run(["git", "-C", str(dest), "reset", "--hard", "HEAD"], check=True)
        subprocess.run(["git", "-C", str(dest), "clean", "-fd"], check=True)
        head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if head != commit_sha:
            subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit_sha], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", commit_sha], check=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", repo_url], check=True)
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit_sha], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"], check=True)

    subprocess.run(["git", "-C", str(dest), "apply", "--check", str(patch_path)], check=True)
    subprocess.run(["git", "-C", str(dest), "apply", str(patch_path)], check=True)

    ast_stats = _compute_repo_stats(dest, spec["scope_files"])
    manifest = {
        "name": "prefect_oss_realism",
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "target_path": str(dest),
        "scope_files": spec["scope_files"],
        "endpoint_count": ast_stats["endpoint_count"],
        "total_py_files": ast_stats["total_py_files"],
        "ast_stats": ast_stats,
        "planted_bugs": spec["planted_bugs"],
        "audit_prompt": spec["audit_prompt"],
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone pinned Prefect commit and apply planted-bug patch.")
    parser.add_argument("--dest", required=True, type=Path, help="Target checkout directory (e.g. /tmp/rlm-oss-prefect)")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional path to write manifest JSON")
    args = parser.parse_args()

    manifest = setup_oss_benchmark(args.dest, manifest_path=args.manifest)
    print(json.dumps({
        "target_path": manifest["target_path"],
        "commit_sha": manifest["commit_sha"],
        "total_py_files": manifest["total_py_files"],
        "endpoint_count": manifest["endpoint_count"],
        "planted_bugs": len(manifest["planted_bugs"]),
    }, indent=2))


if __name__ == "__main__":
    main()
