#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve caller-relative paths before entering the repository.
export RLM_TOOLS_PATH="${RLM_TOOLS_PATH:-$REPO_ROOT}"
export RLM_AB_TARGET_PATH="${RLM_AB_TARGET_PATH:-$REPO_ROOT/tests/fixtures/rlm_corpus_longcontext}"
export RLM_AB_SPEC_PATH="${1:-${RLM_AB_SPEC_PATH:-$REPO_ROOT/evals/specs/rlm_long_context.json}}"
for name in RLM_TOOLS_PATH RLM_AB_TARGET_PATH RLM_AB_SPEC_PATH; do
  if [[ "${!name}" != /* ]]; then
    export "$name=$PWD/${!name}"
  fi
done

cd "$REPO_ROOT"
exec uv run python -m evals.run_ab_eval
