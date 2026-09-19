# Storage Manifest

## Disk state observed this session (root partition `/dev/nvme0n1p5`, 79 GB total)

| When | Free space |
|---|---|
| Start of session (before any V2 work) | 1.5 GB free (99% full) |
| After user-performed cleanup, before BigQuery extraction | 16 GB free (79% full) |
| After extracting cells a–d, all `powerdata_2019`, all `machine_events` | 16 GB free (80% full) |

The cleanup between the first two rows was performed by the user outside
this session; this document records the observed before/after state, not
the specific files removed (unknown to this session).

## New files added this session

All under `data/raw/official_google_2019/` (gitignored — raw extracts are
not committed, matching V1's convention for `data/raw/`):

| File | Rows | Bytes | SHA-256 |
|---|---|---|---|
| `clusterdata_workload_5min_cell_a.csv` | 8,929 + header | 495,023 | `d7ca391cab5f34527f4037711260219e19a45fac5ad2dbf1493639fc54342b37` |
| `clusterdata_workload_5min_cell_b.csv` | 8,929 + header | 489,923 | `ea13c59e34fa6951b971e4050ae24bfec5e46ff3b7cb70dbc77051ace6a36fd7` |
| `clusterdata_workload_5min_cell_c.csv` | 8,929 + header | 504,138 | `536cf398cf2deb158256e3dce90747da1b5ce2554340dac8e474b26b827af69c` |
| `clusterdata_workload_5min_cell_d.csv` | 8,929 + header | 504,495 | `56eb1613bb6d5c4633a4db805c5418bc773aa5aab0e27d168640b403a8204816` |
| `machine_events_all_cells.csv` | 405,895 + header | 19,064,020 | `743b41a3d213f99c947221715a36afe055b4a72f73dd125159bc54204205beae` |
| `powerdata_all_pdus.csv` | 446,400 + header | 20,061,328 | `adb67b1e6bf49cc7cbe44012d636f3157615cc4e393821caa2ebe018ba7098f2` |

Total: ~40 MB. Cells e–h of the workload file are **not present**
(blocked on BigQuery billing — see
`docs/official_data_alignment_audit.md` §7) and must not be fabricated
or interpolated from the 4 available cells.

## Large-data policy (per the master plan)

The source tables themselves (`instance_usage`: ~2–2.4 TB per cell, ~18
TB across all 8 cells) are never downloaded in full and never will be —
they stay in BigQuery, queried only through the aggregating,
column-pruned SQL in `scripts/bigquery/`. Reproducibility is preserved
through the committed query text and this manifest's checksums, not
through committing raw multi-terabyte data.

## Safe cleanup performed by this session
None yet — no caches or temp files needed removal at this stage.
`__pycache__`/`.pytest_cache` from V1 already existed and are already
covered by `.gitignore` per V1's final handoff audit.
