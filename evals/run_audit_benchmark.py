"""Shared 3-run per-item audit benchmark runner, transcript token counter, and scorer."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from evals.artifacts import ensure_artifact_dir, write_artifact
from evals.monorepo_benchmark import generate_monorepo_corpus


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PROMPT_PATH = REPO_ROOT / "plugins" / "rlm-tools" / "agents" / "rlm-explorer.md"
_FENCED_JSON_RE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL | re.IGNORECASE)


def _strip_frontmatter(text: str) -> str:
    match = re.match(r"^---\r?\n.*?\r?\n---\r?\n(.*)$", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _extract_tool_uses_and_results(event: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    tool_uses: list[dict[str, Any]] = []
    result_texts: list[str] = []

    message = event.get("message")
    blocks: list[Any] = []
    if isinstance(message, dict) and isinstance(message.get("content"), list):
        blocks = message["content"]
    elif isinstance(event.get("content"), list):
        blocks = event["content"]

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "tool_use":
            tool_uses.append(block)
        elif btype == "tool_result":
            content = block.get("content")
            if isinstance(content, str):
                result_texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        result_texts.append(part["text"])

    if event.get("type") == "tool_use" and isinstance(event.get("name"), str):
        tool_uses.append(event)
    if event.get("type") == "tool_result":
        content = event.get("content")
        if isinstance(content, str):
            result_texts.append(content)

    return tool_uses, result_texts


def _unwrap_mcp_payload(text: str) -> dict[str, Any] | None:
    try:
        outer = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(outer, dict):
        return None
    inner = outer.get("result")
    if isinstance(inner, str):
        try:
            parsed_inner = json.loads(inner)
            if isinstance(parsed_inner, dict):
                return parsed_inner
        except json.JSONDecodeError:
            return outer
    return outer


def parse_transcript_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum token usage across all turns in a stream-json transcript and inspect RLM tool behavior."""
    usage_by_message_id: dict[str, dict[str, int]] = {}
    anonymous_usages: list[dict[str, int]] = []
    fallback_result_usage: dict[str, int] | None = None

    rlm_start_count = 0
    rlm_execute_count = 0
    rlm_end_count = 0
    rlm_recurse_batched_calls = 0
    llm_calls_used = 0
    child_budget_exhausted_count = 0
    sub_llm_cost_usd: float | None = None
    llm_cap_hit = False
    child_budget_exhausted_seen = False
    result_text = ""
    total_cost_usd: float | None = None
    duration_ms: float | None = None
    seen_tool_ids: set[str] = set()

    def _norm_usage(raw: Any) -> dict[str, int] | None:
        if not isinstance(raw, dict):
            return None
        fields = (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
        extracted: dict[str, int] = {}
        for key in fields:
            val = raw.get(key, 0)
            if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
                extracted[key] = val
            else:
                extracted[key] = 0
        if not any(key in raw for key in fields):
            return None
        return extracted

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        message = event.get("message")

        if etype == "assistant" or (isinstance(message, dict) and message.get("role") == "assistant"):
            raw_usage = (message.get("usage") if isinstance(message, dict) else None) or event.get("usage")
            norm = _norm_usage(raw_usage)
            if norm is not None:
                msg_id = message.get("id") if isinstance(message, dict) else event.get("id")
                if isinstance(msg_id, str) and msg_id:
                    usage_by_message_id[msg_id] = norm
                else:
                    anonymous_usages.append(norm)

        if etype == "result":
            if isinstance(event.get("result"), str):
                result_text = event["result"]
            if isinstance(event.get("total_cost_usd"), (int, float)) and not isinstance(event.get("total_cost_usd"), bool):
                total_cost_usd = float(event["total_cost_usd"])
            if isinstance(event.get("duration_ms"), (int, float)) and not isinstance(event.get("duration_ms"), bool):
                duration_ms = float(event["duration_ms"])
            if isinstance(event.get("usage"), dict):
                fallback_result_usage = _norm_usage(event["usage"])

        tool_uses, result_texts = _extract_tool_uses_and_results(event)
        for tu in tool_uses:
            tu_id = tu.get("id")
            if isinstance(tu_id, str) and tu_id:
                if tu_id in seen_tool_ids:
                    continue
                seen_tool_ids.add(tu_id)
            name = str(tu.get("name", ""))
            if name.endswith("rlm_start"):
                rlm_start_count += 1
            elif name.endswith("rlm_execute"):
                rlm_execute_count += 1
            elif name.endswith("rlm_end"):
                rlm_end_count += 1
            if name.endswith("rlm_start") or name.endswith("rlm_execute"):
                tu_input = tu.get("input")
                if isinstance(tu_input, dict):
                    code_str = tu_input.get("code")
                    if isinstance(code_str, str) and "rlm_recurse_batched(" in code_str:
                        rlm_recurse_batched_calls += 1

        for text in result_texts:
            if "LLM call limit exceeded" in text or "LLM call budget too small" in text:
                llm_cap_hit = True
            if "[BUDGET_EXHAUSTED]" in text:
                child_budget_exhausted_seen = True
            payload = _unwrap_mcp_payload(text)
            if payload is not None:
                usage_obj = payload.get("usage")
                if isinstance(usage_obj, dict):
                    used = usage_obj.get("llm_calls_used")
                    if isinstance(used, int) and not isinstance(used, bool):
                        llm_calls_used = max(llm_calls_used, used)
                    exhausted = usage_obj.get("child_budget_exhausted")
                    if isinstance(exhausted, int) and not isinstance(exhausted, bool):
                        child_budget_exhausted_count = max(child_budget_exhausted_count, exhausted)
                    sub_usage = usage_obj.get("sub_llm_usage")
                    if isinstance(sub_usage, dict):
                        sc = sub_usage.get("total_cost_usd")
                        if isinstance(sc, (int, float)) and not isinstance(sc, bool):
                            sub_llm_cost_usd = max(sub_llm_cost_usd or 0.0, float(sc))

    all_usages = list(usage_by_message_id.values()) + anonymous_usages
    if not all_usages and fallback_result_usage is not None:
        all_usages = [fallback_result_usage]

    totals = {
        "input_tokens": sum(u["input_tokens"] for u in all_usages),
        "output_tokens": sum(u["output_tokens"] for u in all_usages),
        "cache_creation_input_tokens": sum(u["cache_creation_input_tokens"] for u in all_usages),
        "cache_read_input_tokens": sum(u["cache_read_input_tokens"] for u in all_usages),
    }
    totals["total_tokens"] = (
        totals["input_tokens"]
        + totals["output_tokens"]
        + totals["cache_creation_input_tokens"]
        + totals["cache_read_input_tokens"]
    )

    return {
        **totals,
        "turns_counted": len(all_usages),
        "rlm_start_count": rlm_start_count,
        "rlm_execute_count": rlm_execute_count,
        "rlm_end_count": rlm_end_count,
        "rlm_recurse_batched_calls": rlm_recurse_batched_calls,
        "llm_calls_used": llm_calls_used,
        "child_budget_exhausted_count": child_budget_exhausted_count,
        "sub_llm_cost_usd": sub_llm_cost_usd,
        "second_session_opened": rlm_start_count > 1,
        "llm_cap_hit": llm_cap_hit,
        "child_budget_exhausted_seen": child_budget_exhausted_seen,
        "total_cost_usd": total_cost_usd,
        "duration_ms": duration_ms,
        "result_text": result_text,
    }


def _extract_reported_items(output_text: str) -> list[str] | None:
    candidates = [output_text.strip()]
    for match in _FENCED_JSON_RE.finditer(output_text):
        candidates.append(match.group(1).strip())
    brace_start = output_text.find("{")
    brace_end = output_text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(output_text[brace_start:brace_end + 1])

    for raw in candidates:
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items_list: list[Any] | None = None
        if isinstance(parsed, dict) and isinstance(parsed.get("contradictions"), list):
            items_list = parsed["contradictions"]
        elif isinstance(parsed, list):
            items_list = parsed
        if items_list is not None:
            names: list[str] = []
            for entry in items_list:
                if isinstance(entry, str):
                    names.append(entry)
                elif isinstance(entry, dict):
                    label = entry.get("tool") or entry.get("item") or entry.get("name") or entry.get("endpoint")
                    if isinstance(label, str) and label.strip():
                        names.append(label.strip())
            return names
    return None


def score_audit_findings(output_text: str, planted_bugs: list[dict[str, Any]]) -> dict[str, Any]:
    """Score recalled planted bugs and false positives from the agent's audit output."""
    expected_items = [bug["item"] for bug in planted_bugs]
    reported_items = _extract_reported_items(output_text)

    found: list[str] = []
    missed: list[str] = []
    if reported_items is not None:
        reported_lower = [item.lower() for item in reported_items]
        for expected in expected_items:
            if any(expected.lower() in rep for rep in reported_lower):
                found.append(expected)
            else:
                missed.append(expected)
        false_positives = [
            rep for rep in reported_items
            if not any(expected.lower() in rep.lower() for expected in expected_items)
        ]
    else:
        lowered = output_text.lower()
        for expected in expected_items:
            if expected.lower() in lowered:
                found.append(expected)
            else:
                missed.append(expected)
        false_positives = []

    total_bugs = len(expected_items)
    recall_count = len(found)
    target_min = max(1, total_bugs - 1) if total_bugs else 0
    return {
        "found": found,
        "missed": missed,
        "false_positives": false_positives,
        "recall_count": recall_count,
        "total_bugs": total_bugs,
        "passed_target": recall_count >= target_min,
    }


def _run_single_audit(
    *,
    prompt: str,
    target: Path,
    system_prompt: str,
    model: str,
    sub_provider: str,
    sub_model: str,
    timeout_seconds: int,
    transcript_path: Path,
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="rlm-audit-") as workdir_str:
        workdir = Path(workdir_str)
        mcp_config = {
            "mcpServers": {
                "rlm-tools": {
                    "command": "uv",
                    "args": ["--directory", str(REPO_ROOT), "run", "--no-sync", "rlm-tools"],
                    "env": {
                        "RLM_ALLOWED_ROOTS": str(target),
                        "RLM_EXTERNAL_ROOTS": str(target),
                        "RLM_SUB_PROVIDER": sub_provider,
                        "RLM_SUB_MODEL": sub_model,
                    },
                }
            }
        }
        config_path = workdir / ".mcp.json"
        config_path.write_text(json.dumps(mcp_config), encoding="utf-8")

        command = [
            "claude",
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config", str(config_path),
            "--tools", "",
            "--allowedTools", "mcp__rlm-tools__rlm_start,mcp__rlm-tools__rlm_execute,mcp__rlm-tools__rlm_end",
            "--add-dir", str(target),
            "--append-system-prompt", system_prompt,
            "--disable-slash-commands",
            "--setting-sources", "",
            "--model", model,
        ]
        child_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        started = perf_counter()
        proc = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            cwd=workdir,
            env=child_env,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        transcript_path.write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude exited with {proc.returncode}: {(proc.stderr or '').strip()[:500]}"
            )

        events: list[dict[str, Any]] = []
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    events.append(parsed)
            except json.JSONDecodeError:
                continue
        metrics = parse_transcript_metrics(events)
        if metrics["duration_ms"] is None:
            metrics["duration_ms"] = elapsed_ms
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-trial per-item audit acceptance benchmark.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="Generate and audit the synthetic monorepo")
    mode.add_argument("--target", type=Path, help="Existing target directory (requires --manifest)")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to benchmark manifest JSON")
    parser.add_argument("--scale", choices=["small", "full"], default="full", help="Synthetic corpus scale")
    parser.add_argument("--seed", type=int, default=20261128, help="Synthetic corpus seed")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--model", default="haiku", help="Parent Claude model")
    parser.add_argument("--sub-provider", default="claude-cli", help="RLM_SUB_PROVIDER for rlm-tools")
    parser.add_argument("--sub-model", default="haiku", help="RLM_SUB_MODEL for rlm-tools")
    parser.add_argument("--timeout", type=int, default=900, help="Per-run timeout in seconds")
    args = parser.parse_args()

    if args.target is not None and args.manifest is None:
        parser.error("--target requires --manifest")

    results_root = ensure_artifact_dir(REPO_ROOT / "evals" / "ab_results")
    dest = ensure_artifact_dir(
        results_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f_audit_benchmark"),
        exist_ok=False,
    )
    system_prompt = _strip_frontmatter(AGENT_PROMPT_PATH.read_text(encoding="utf-8"))

    with TemporaryDirectory(prefix="rlm-synth-") as tmpdir_str:
        if args.synthetic:
            target = Path(tmpdir_str) / "monorepo"
            manifest = generate_monorepo_corpus(target, seed=args.seed, scale=args.scale)
            write_artifact(dest / "manifest.json", json.dumps(manifest, indent=2))
        else:
            target = args.target.resolve()
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            write_artifact(dest / "manifest.json", json.dumps(manifest, indent=2))

        prompt = manifest["audit_prompt"].replace("{target_path}", str(target))
        planted_bugs = manifest["planted_bugs"]
        run_reports: list[dict[str, Any]] = []

        for idx in range(1, args.runs + 1):
            print(f"Running audit trial {idx}/{args.runs} against {target}...", flush=True)
            transcript_path = dest / f"run_{idx:02d}_transcript.jsonl"
            metrics = _run_single_audit(
                prompt=prompt,
                target=target,
                system_prompt=system_prompt,
                model=args.model,
                sub_provider=args.sub_provider,
                sub_model=args.sub_model,
                timeout_seconds=args.timeout,
                transcript_path=transcript_path,
            )
            score = score_audit_findings(metrics["result_text"], planted_bugs)
            passed = (
                score["passed_target"]
                and not metrics["llm_cap_hit"]
                and not metrics["second_session_opened"]
            )
            report = {
                "run": idx,
                "passed": passed,
                "score": score,
                "metrics": {k: v for k, v in metrics.items() if k != "result_text"},
            }
            run_reports.append(report)
            print(
                f"Run {idx}: {score['recall_count']}/{score['total_bugs']} bugs, "
                f"FP={len(score['false_positives'])}, "
                f"recurse_batched={metrics['rlm_recurse_batched_calls']}, "
                f"llm_calls={metrics['llm_calls_used']}, "
                f"child_exhausted={metrics['child_budget_exhausted_count']}, "
                f"cap_hit={metrics['llm_cap_hit']}, "
                f"2nd_session={metrics['second_session_opened']}, "
                f"tokens={metrics['total_tokens']:,}",
                flush=True,
            )

    summary = {"target": str(target), "runs": run_reports}
    write_artifact(dest / "summary.json", json.dumps(summary, indent=2))

    print("\nRun | Bugs | FP | Batches | LLMCalls | ChildExh | CapHit | 2ndSess | Tokens (transcript) | ParentCost | SubCost | Status")
    print("----+------+----+---------+----------+----------+--------+---------+---------------------+------------+---------+-------")
    for rep in run_reports:
        sc = rep["score"]
        m = rep["metrics"]
        cost = f"${m['total_cost_usd']:.4f}" if m["total_cost_usd"] is not None else "n/a"
        sub_cost = f"${m['sub_llm_cost_usd']:.4f}" if m["sub_llm_cost_usd"] is not None else "n/a"
        status = "PASS" if rep["passed"] else "FAIL"
        print(
            f"{rep['run']:>3} | {sc['recall_count']}/{sc['total_bugs']}  | {len(sc['false_positives']):>2} | "
            f"{m['rlm_recurse_batched_calls']:>7} | {m['llm_calls_used']:>8} | {m['child_budget_exhausted_count']:>8} | "
            f"{str(m['llm_cap_hit']):<6} | {str(m['second_session_opened']):<7} | "
            f"{m['total_tokens']:>19,} | {cost:>10} | {sub_cost:>7} | {status}"
        )
    print(f"\nArtifacts saved to {dest}")


if __name__ == "__main__":
    main()
