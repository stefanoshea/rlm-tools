# Option 2: Pinned Open-Source Realism Benchmark (Prefect)

Uses a pinned commit of `PrefectHQ/prefect` (`5f7d879aa1636e42ca6697548c5c0065dafbff7a`, ~1,960 Python files) plus a local 7-bug patch ([`prefect_planted_bugs.patch`](prefect_planted_bugs.patch)) spanning 50 FastAPI route handlers across:

- `src/prefect/server/api/deployments.py` (20 routes, ~59 KB)
- `src/prefect/server/api/flow_runs.py` (21 routes, ~44 KB)
- `src/prefect/server/api/variables.py` (9 routes, ~6 KB)

## Usage

1. Prepare the patched checkout and manifest in `/tmp`:
   ```bash
   uv run python -m evals.oss_realism.setup_oss_benchmark \
     --dest /tmp/rlm-oss-prefect \
     --manifest /tmp/rlm-oss-prefect-manifest.json
   ```

2. Run the 3-run `rlm-explorer` audit benchmark against it:
   ```bash
   uv run python -m evals.run_audit_benchmark \
     --target /tmp/rlm-oss-prefect \
     --manifest /tmp/rlm-oss-prefect-manifest.json \
     --runs 3
   ```

3. Remove `/tmp/rlm-oss-prefect` and `/tmp/rlm-oss-prefect-manifest.json` when finished if desired.
