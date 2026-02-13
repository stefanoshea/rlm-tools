#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/ab_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for cmd in claude jq bc; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: '$cmd' is required but not installed."
    exit 1
  fi
done

APPLE_ROOT="${RLM_EVAL_PROJECT_PATH:-}"
if [[ -z "$APPLE_ROOT" ]]; then
  echo "ERROR: Set RLM_EVAL_PROJECT_PATH to your local your-iOS-project directory."
  exit 1
fi
if [[ ! -d "$APPLE_ROOT" ]]; then
  echo "ERROR: RLM_EVAL_PROJECT_PATH does not exist: $APPLE_ROOT"
  exit 1
fi

TARGET_PROJECT_PATH="${RLM_AB_TARGET_PATH:-$APPLE_ROOT/app}"
if [[ ! -d "$TARGET_PROJECT_PATH" ]]; then
  echo "ERROR: Target project path does not exist: $TARGET_PROJECT_PATH"
  echo "Set RLM_AB_TARGET_PATH to a codebase directory to explore."
  exit 1
fi

RLM_SERVER_PATH="${RLM_TOOLS_PATH:-$REPO_ROOT}"

DEFAULT_TASK="Explore the iOS project at $TARGET_PROJECT_PATH. Find all TCA Reducers in the Dashboard module, identify which ones use @Dependency, and list the dependency properties they access. Do not make any changes."

TASK_PROMPT="${1:-}"
if [[ -z "$TASK_PROMPT" ]]; then
  TASK_PROMPT="$DEFAULT_TASK"
elif [[ -f "$TASK_PROMPT" ]]; then
  TASK_PROMPT="$(cat "$TASK_PROMPT")"
fi

mkdir -p "$RESULTS_DIR"

TMPDIR_BASE=$(mktemp -d)
DIR_WITH_RLM="$TMPDIR_BASE/with_rlm"
DIR_WITHOUT_RLM="$TMPDIR_BASE/without_rlm"
mkdir -p "$DIR_WITH_RLM" "$DIR_WITHOUT_RLM"

trap 'rm -rf "$TMPDIR_BASE"' EXIT

# --- "With RLM" setup ---
cat > "$DIR_WITH_RLM/.mcp.json" <<MCP_EOF
{
  "mcpServers": {
    "rlm-tools": {
      "command": "uv",
      "args": ["--directory", "$RLM_SERVER_PATH", "run", "rlm-tools"]
    }
  }
}
MCP_EOF

cat > "$DIR_WITH_RLM/CLAUDE.md" <<'CLAUDE_EOF'
# RLM Tools Preferred

When exploring a codebase, prefer using rlm-tools MCP tools over built-in tools:

1. Use `rlm_start` to begin an exploration session for a directory before reading files.
2. Use `rlm_execute` with exploration commands (list, search, read, tree, etc.) instead of Read, Grep, and Glob.
3. Use `rlm_end` when you are done exploring.

These tools provide structured codebase exploration with better context management.
Only fall back to Read/Grep/Glob if rlm-tools cannot handle the specific operation.
CLAUDE_EOF

# --- "Without RLM" setup ---
cat > "$DIR_WITHOUT_RLM/.mcp.json" <<'MCP_EOF'
{
  "mcpServers": {}
}
MCP_EOF

RESULT_WITH="$RESULTS_DIR/${TIMESTAMP}_with_rlm.json"
RESULT_WITHOUT="$RESULTS_DIR/${TIMESTAMP}_without_rlm.json"

echo "============================================"
echo "  A/B Eval: Claude CLI ± rlm-tools"
echo "============================================"
echo ""
echo "Task prompt:"
echo "  ${TASK_PROMPT:0:120}..."
echo ""
echo "Temp dirs:"
echo "  With RLM:    $DIR_WITH_RLM"
echo "  Without RLM: $DIR_WITHOUT_RLM"
echo ""
echo "Running both agents in parallel..."
echo ""

START_TIME=$SECONDS

claude -p "$TASK_PROMPT" \
  --output-format json \
  --no-session-persistence \
  --mcp-config "$DIR_WITH_RLM/.mcp.json" \
  --strict-mcp-config \
  --append-system-prompt "$(cat "$DIR_WITH_RLM/CLAUDE.md")" \
  > "$RESULT_WITH" 2>/dev/null &
PID_WITH=$!

claude -p "$TASK_PROMPT" \
  --output-format json \
  --no-session-persistence \
  --mcp-config "$DIR_WITHOUT_RLM/.mcp.json" \
  --strict-mcp-config \
  > "$RESULT_WITHOUT" 2>/dev/null &
PID_WITHOUT=$!

# Wait for both, capture exit codes
WAIT_WITH=0
WAIT_WITHOUT=0
wait $PID_WITH || WAIT_WITH=$?
wait $PID_WITHOUT || WAIT_WITHOUT=$?

WALL_TIME=$(( SECONDS - START_TIME ))

echo "Both agents finished (wall time: ${WALL_TIME}s)"
echo ""

if [[ $WAIT_WITH -ne 0 ]]; then
  echo "WARNING: 'With RLM' agent exited with code $WAIT_WITH"
fi
if [[ $WAIT_WITHOUT -ne 0 ]]; then
  echo "WARNING: 'Without RLM' agent exited with code $WAIT_WITHOUT"
fi

# --- Parse results ---
extract() {
  local file="$1"
  local paths="$2"
  # Try each jq path in order, return first non-null
  for p in $paths; do
    val=$(jq -r "$p // empty" "$file" 2>/dev/null || true)
    if [[ -n "$val" ]]; then
      echo "$val"
      return
    fi
  done
  echo "n/a"
}

get_input_tokens()  { extract "$1" ".usage.input_tokens .input_tokens .total_input_tokens"; }
get_output_tokens() { extract "$1" ".usage.output_tokens .output_tokens .total_output_tokens"; }
get_cost()          { extract "$1" ".cost_usd .total_cost_usd .cost"; }
get_duration()      { extract "$1" ".duration_ms .total_duration_ms .duration"; }
get_duration_api()  { extract "$1" ".duration_api_ms .total_api_duration_ms .api_duration_ms"; }
get_num_turns()     { extract "$1" ".num_turns .turns"; }
get_is_error()      { extract "$1" ".is_error .error"; }

# Extract raw values for math
raw_input_with=$(get_input_tokens "$RESULT_WITH")
raw_output_with=$(get_output_tokens "$RESULT_WITH")
raw_input_without=$(get_input_tokens "$RESULT_WITHOUT")
raw_output_without=$(get_output_tokens "$RESULT_WITHOUT")

total_with="n/a"
total_without="n/a"
if [[ "$raw_input_with" != "n/a" && "$raw_output_with" != "n/a" ]]; then
  total_with=$(( raw_input_with + raw_output_with ))
fi
if [[ "$raw_input_without" != "n/a" && "$raw_output_without" != "n/a" ]]; then
  total_without=$(( raw_input_without + raw_output_without ))
fi

fmt() {
  if [[ "$1" == "n/a" ]]; then echo "n/a"; else printf "%'d" "$1" 2>/dev/null || echo "$1"; fi
}

fmt_cost() {
  if [[ "$1" == "n/a" ]]; then echo "n/a"; else printf "\$%.4f" "$1" 2>/dev/null || echo "$1"; fi
}

fmt_dur() {
  if [[ "$1" == "n/a" ]]; then
    echo "n/a"
  else
    local secs=$(( $1 / 1000 ))
    local mins=$(( secs / 60 ))
    local rem=$(( secs % 60 ))
    if [[ $mins -gt 0 ]]; then
      echo "${mins}m ${rem}s"
    else
      echo "${secs}s"
    fi
  fi
}

cost_with=$(get_cost "$RESULT_WITH")
cost_without=$(get_cost "$RESULT_WITHOUT")
dur_with=$(get_duration "$RESULT_WITH")
dur_without=$(get_duration "$RESULT_WITHOUT")
dur_api_with=$(get_duration_api "$RESULT_WITH")
dur_api_without=$(get_duration_api "$RESULT_WITHOUT")
turns_with=$(get_num_turns "$RESULT_WITH")
turns_without=$(get_num_turns "$RESULT_WITHOUT")
err_with=$(get_is_error "$RESULT_WITH")
err_without=$(get_is_error "$RESULT_WITHOUT")

# --- Print comparison table ---
printf "\n"
printf "%-20s  %15s  %15s\n" "Metric" "With RLM" "Without RLM"
printf "%-20s  %15s  %15s\n" "--------------------" "---------------" "---------------"
printf "%-20s  %15s  %15s\n" "Input tokens"    "$(fmt "$raw_input_with")"   "$(fmt "$raw_input_without")"
printf "%-20s  %15s  %15s\n" "Output tokens"   "$(fmt "$raw_output_with")"  "$(fmt "$raw_output_without")"
printf "%-20s  %15s  %15s\n" "Total tokens"    "$(fmt "$total_with")"       "$(fmt "$total_without")"
printf "%-20s  %15s  %15s\n" "Cost"            "$(fmt_cost "$cost_with")"   "$(fmt_cost "$cost_without")"
printf "%-20s  %15s  %15s\n" "Duration"        "$(fmt_dur "$dur_with")"     "$(fmt_dur "$dur_without")"
printf "%-20s  %15s  %15s\n" "API duration"    "$(fmt_dur "$dur_api_with")" "$(fmt_dur "$dur_api_without")"
printf "%-20s  %15s  %15s\n" "Turns"           "$turns_with"                "$turns_without"
printf "%-20s  %15s  %15s\n" "Error"           "$err_with"                  "$err_without"
printf "\n"

# --- Compute deltas ---
if [[ "$total_with" != "n/a" && "$total_without" != "n/a" && "$total_without" -gt 0 ]]; then
  pct=$(echo "scale=1; ($total_with - $total_without) * 100 / $total_without" | bc)
  echo "Token delta: ${pct}% (with RLM vs without)"
fi
if [[ "$cost_with" != "n/a" && "$cost_without" != "n/a" ]]; then
  cost_delta=$(echo "scale=4; $cost_with - $cost_without" | bc)
  echo "Cost delta: \$${cost_delta}"
fi
if [[ "$dur_with" != "n/a" && "$dur_without" != "n/a" && "$dur_without" -gt 0 ]]; then
  dur_pct=$(echo "scale=1; ($dur_with - $dur_without) * 100 / $dur_without" | bc)
  echo "Duration delta: ${dur_pct}%"
fi

echo ""
echo "Full results saved to:"
echo "  $RESULT_WITH"
echo "  $RESULT_WITHOUT"
echo ""

# Dump top-level keys for debugging if fields were missing
if [[ "$(get_input_tokens "$RESULT_WITH")" == "n/a" ]]; then
  echo "WARNING: Could not parse expected fields from 'With RLM' output."
  echo "Top-level keys: $(jq -r 'keys | join(", ")' "$RESULT_WITH" 2>/dev/null || echo 'not valid JSON')"
  echo "Run 'jq . $RESULT_WITH' to inspect."
  echo ""
fi
if [[ "$(get_input_tokens "$RESULT_WITHOUT")" == "n/a" ]]; then
  echo "WARNING: Could not parse expected fields from 'Without RLM' output."
  echo "Top-level keys: $(jq -r 'keys | join(", ")' "$RESULT_WITHOUT" 2>/dev/null || echo 'not valid JSON')"
  echo "Run 'jq . $RESULT_WITHOUT' to inspect."
  echo ""
fi
