-- Lighter variant of extract_clusterdata_workload.sql, used for cells
-- e-h once the BigQuery Sandbox free monthly bytes-scanned allowance
-- resets. Drops COUNT(DISTINCT machine_id) (a-d have this column;
-- e-h will not -- documented, not hidden, in
-- docs/official_data_alignment_audit.md and docs/storage_manifest.md).
--
-- Verified via dry-run (2026-09-19), user has chosen to stay on the
-- free tier rather than enable billing:
--   cell e: 155,765,859,864 bytes (~145 GiB) -- was ~181 GB with machine_id
--   cell f: 184,987,408,800 bytes (~172 GiB)
--   cell g: 161,392,125,864 bytes (~150 GiB)
--   cell h: 171,203,376,648 bytes (~159 GiB)
--   TOTAL for e+f+g+h: 673,348,771,176 bytes (~627 GiB)
-- This must fit inside one month's free allowance (cells a-d, with the
-- heavier query, consumed ~724 GB and cell e's 5th attempt at that
-- heavier query pushed the cumulative total over the cap -- so the
-- cap is somewhere between ~724 GB and ~905 GB). ~627 GiB for all four
-- remaining cells in one reset cycle has real margin under that range.
SELECT
  '{CELL}' AS cell,
  CAST(FLOOR(start_time / 300000000) AS INT64) AS bucket_5min,
  SUM(average_usage.cpus) AS sum_cpu,
  SUM(average_usage.memory) AS sum_mem,
  COUNT(*) AS n_records
FROM `google.com:google-cluster-data.clusterdata_2019_{CELL}.instance_usage`
GROUP BY bucket_5min
ORDER BY bucket_5min
