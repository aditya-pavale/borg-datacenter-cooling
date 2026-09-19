-- Official ClusterData2019 workload extraction, aggregated to 5-minute buckets
-- per Borg cell. This is the finest time resolution documented for
-- instance_usage records and matches PowerData2019's native cadence
-- (verified empirically: powerdata tables have exactly 8928 rows each,
-- i.e. one row per 5 minutes across the ~31-day May 2019 trace).
--
-- Column pruning matters here: instance_usage is ~2.0-2.4 TB per cell
-- (8 cells total). Selecting only start_time and the two average_usage
-- subfields reduces the dry-run cost to ~181 GB scanned per cell
-- (verified via `bq query --dry_run` on 2026-09-19).
--
-- Usage: replace {CELL} with a/b/c/d/e/f/g/h and run through
-- `bq query --nouse_legacy_sql`. ALWAYS dry-run first:
--   bq query --nouse_legacy_sql --dry_run < this_file (with {CELL} substituted)
--
-- Cost: this project (borg-datacenter-cooling-v2) has NO billing account
-- linked (`gcloud beta billing projects describe` -> billingEnabled: false),
-- so it runs in BigQuery Sandbox mode with a free monthly bytes-scanned cap.
-- Cells a, b, c, d (2026-09-19) succeeded (~181 GB each, ~724 GB cumulative).
-- Cell e failed with "Quota exceeded: ... free query bytes scanned" --
-- the sandbox cap was exhausted. Cells e-h require billing to be enabled
-- (see docs/official_data_alignment_audit.md) before they can be extracted.
SELECT
  '{CELL}' AS cell,
  CAST(FLOOR(start_time / 300000000) AS INT64) AS bucket_5min,
  SUM(average_usage.cpus) AS sum_cpu,
  SUM(average_usage.memory) AS sum_mem,
  COUNT(DISTINCT machine_id) AS n_machines,
  COUNT(*) AS n_records
FROM `google.com:google-cluster-data.clusterdata_2019_{CELL}.instance_usage`
GROUP BY bucket_5min
ORDER BY bucket_5min
