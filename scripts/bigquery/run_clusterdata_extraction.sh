#!/usr/bin/env bash
# Runs extract_clusterdata_workload.sql for every cell that doesn't already
# have an output file, dry-running each query first and refusing to proceed
# if it would scan more than MAX_BYTES (default 200 GB per cell -- observed
# cost for one cell's instance_usage table is ~181 GB with this query).
#
# This project (borg-datacenter-cooling-v2) has no billing account linked,
# so BigQuery Sandbox's free monthly scan quota applies. Cells a-d were
# extracted on 2026-09-19 (~724 GB cumulative); cell e failed on quota.
# Re-run this script after enabling billing to pick up the remaining cells.
set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data/raw/official_google_2019"
TEMPLATE="$(dirname "${BASH_SOURCE[0]}")/extract_clusterdata_workload.sql"
MAX_BYTES="${MAX_BYTES:-200000000000}"  # 200 GB safety cap per cell

for cell in a b c d e f g h; do
  out="${OUT_DIR}/clusterdata_workload_5min_cell_${cell}.csv"
  if [[ -s "$out" ]] && ! grep -q "^BigQuery error" "$out"; then
    echo "[skip] cell $cell already extracted ($out)"
    continue
  fi

  sql="$(sed "s/{CELL}/${cell}/g" "$TEMPLATE")"

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
