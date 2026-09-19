-- Official ClusterData2019 machine_events, all 8 cells, unioned.
-- Cheap: ~16 MB total scanned (verified via dry-run, 2026-09-19) because
-- machine_events is a low-cardinality table (machine additions/removals/
-- updates) rather than the high-frequency instance_usage table.
-- Used for per-cell machine capacity (capacity.cpus/memory), not for
-- any machine-to-power-domain mapping -- no such mapping is documented
-- (see docs/official_data_alignment_audit.md).
SELECT 'a' AS cell, time, machine_id, type, capacity.cpus AS cap_cpus, capacity.memory AS cap_mem FROM `google.com:google-cluster-data.clusterdata_2019_a.machine_events`
UNION ALL
SELECT 'b', time, machine_id, type, capacity.cpus, capacity.memory FROM `google.com:google-cluster-data.clusterdata_2019_b.machine_events`
UNION ALL
SELECT 'c', time, machine_id, type, capacity.cpus, capacity.memory FROM `google.com:google-cluster-data.clusterdata_2019_c.machine_events`
UNION ALL
SELECT 'd', time, machine_id, type, capacity.cpus, capacity.memory FROM `google.com:google-cluster-data.clusterdata_2019_d.machine_events`
UNION ALL
SELECT 'e', time, machine_id, type, capacity.cpus, capacity.memory FROM `google.com:google-cluster-data.clusterdata_2019_e.machine_events`
UNION ALL
SELECT 'f', time, machine_id, type, capacity.cpus, capacity.memory FROM `google.com:google-cluster-data.clusterdata_2019_f.machine_events`
UNION ALL
SELECT 'g', time, machine_id, type, capacity.cpus, capacity.memory FROM `google.com:google-cluster-data.clusterdata_2019_g.machine_events`
UNION ALL
SELECT 'h', time, machine_id, type, capacity.cpus, capacity.memory FROM `google.com:google-cluster-data.clusterdata_2019_h.machine_events`
