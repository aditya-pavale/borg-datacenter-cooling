#!/usr/bin/env bash
# Runs the clusterdata workload extraction for every cell that doesn't
# already have an output file, dry-running each query first and refusing
# to proceed if it would scan more than MAX_BYTES (default 200 GB per
# cell -- observed cost is ~145-185 GB per cell with the lite query).
#
# This project (borg-datacenter-cooling-v2) intentionally has NO billing
# account linked (user decision: stay on the free tier rather than pay).
# BigQuery Sandbox's free monthly scan quota applies. Cells a-d were
# extracted on 2026-09-19 with the heavier (extract_clusterdata_workload.sql)
# query (~724 GB cumulative, includes a n_machines column); cell e then
# failed on quota with that same heavier query. Cells e-h use the lighter
# extract_clusterdata_workload_lite.sql (no n_machines column, ~627 GB for
# all four combined) and should be run once the monthly free allowance
# resets -- re-run this script then; it skips cells already extracted.
set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data/raw/official_google_2019"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
MAX_BYTES="${MAX_BYTES:-200000000000}"  # 200 GB safety cap per cell

for cell in a b c d e f g h; do
  out="${OUT_DIR}/clusterdata_workload_5min_cell_${cell}.csv"
  if [[ -s "$out" ]] && ! grep -q "^BigQuery error" "$out"; then
    echo "[skip] cell $cell already extracted ($out)"
    continue
  fi

  # a-d already exist from the heavier query; any cell extracted fresh
  # by this script uses the lite query (no n_machines column).
  template="${SCRIPT_DIR}/extract_clusterdata_workload_lite.sql"
  sql="$(sed "s/{CELL}/${cell}/g" "$template")"

  bytes=$(bq query --nouse_legacy_sql --dry_run --format=prettyjson <<<"$sql" \
    | grep -m1 totalBytesProcessed | grep -o '[0-9]*')
  echo "[cell $cell] dry-run estimate: ${bytes} bytes"

  if (( bytes > MAX_BYTES )); then
    echo "[cell $cell] ABORT: estimate exceeds MAX_BYTES=${MAX_BYTES}, refusing to run"
    exit 1
  fi

  echo "[cell $cell] running..."
  bq query --nouse_legacy_sql --format=csv --max_rows=20000 <<<"$sql" > "$out"
  echo "[cell $cell] done -> $out ($(wc -l < "$out") lines)"
done
