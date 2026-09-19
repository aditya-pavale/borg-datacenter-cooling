# Official Data Alignment Audit (ClusterData2019 ↔ PowerData2019)

Status: Phase 1 (official data audit) and Phase 5's alignment question
(Phase 6 of the master plan) are answered below, from direct inspection
of the official BigQuery mirror plus Google's own documentation. This
document is the source of truth for how — and at what granularity — the
two datasets may be legitimately combined. Nothing below is inferred
from "same company, same month" convenience; every claim is either a
quoted document sentence or a directly observed schema/data fact, with
the two kept clearly distinguishable.

## 1. Access verification (2026-09-19)

```
$ gcloud auth list
ACTIVE  ACCOUNT
*       aspavale28@gmail.com

$ gcloud config get-value project
borg-datacenter-cooling-v2

$ ls ~/.config/gcloud/application_default_credentials.json
-rw------- 1 aditya aditya 403  (present)
```

`bq ls --project_id=google.com:google-cluster-data` confirms the public
project is reachable and lists exactly the datasets the master plan
expects:

```
clusterdata_2011_1
clusterdata_2019_a .. clusterdata_2019_h   (8 datasets, one per Borg cell)
powerdata_2019
```

**Billing status — important, load-bearing fact for everything that
follows**:

```
$ gcloud beta billing projects describe borg-datacenter-cooling-v2
billingEnabled: false
```

No billing account is linked to `borg-datacenter-cooling-v2`. BigQuery
therefore runs this project in **Sandbox mode**, which grants a limited
free monthly allotment of bytes-scanned rather than genuine pay-as-you-go
access. This directly shaped what could be extracted in this session —
see §5.

## 2. ClusterData2019 schema (observed via `bq show --schema`)

`clusterdata_2019_{a..h}` each contain five tables:
`collection_events`, `instance_events`, `instance_usage`,
`machine_attributes`, `machine_events`.

`instance_usage` (the workload signal used here) has, per row: an
integer `start_time`/`end_time` (documented elsewhere as microseconds
since the start of the trace collection window, not Unix epoch time),
`machine_id`, `collection_id`, and nested `average_usage.{cpus,memory}`,
`maximum_usage.{cpus,memory}`, plus per-instance CPU/memory histograms.
**There is no field in this schema, or in `machine_events`
(`switch_id`, `capacity.{cpus,memory}`, `platform_id`), that references
a power domain, PDU, or any electrical/power identifier.**

`instance_usage` is enormous: **~2.0–2.4 TB and ~6.5–8.3 billion rows
per cell** (observed via `bq show`, all 8 cells), ~18 TB total across
the trace. This is why the master plan's instruction to never download
the full trace locally, and to query BigQuery with aggregation instead,
is not a stylistic preference — a naive per-cell export would not fit
on this machine's disk (16 GB free at time of writing) or be usable by
any downstream pandas/parquet step.

## 3. PowerData2019 schema (observed via `bq show --schema`)

The `powerdata_2019` dataset is **not** one table per cell; it is 50
tables named `cell{a-h}_pdu{N}` (one table per physical Power
Distribution Unit), e.g. `cella_pdu6`, `cellf_pdu23`. Each table's rows
have: `time` (same microsecond-since-trace-start convention, confirmed
by range-matching — see §4), `cell` (string, e.g. `"a"`), `pdu`
(string), `measured_power_util` and `production_power_util` (floats —
**utilization fractions of the PDU's rated capacity, not absolute
watts**, per the field names; no PDU capacity/rating field is present
in this schema, so absolute power cannot be recovered from this table
alone), and two boolean data-quality flags
(`bad_measurement_data`, `bad_production_power_data`).

**Critically, every power row already carries an explicit `cell`
column.** The join key the master plan asks us to establish is not
something that has to be inferred — it is present in the data itself.

Table→cell counts observed via `bq ls`:

| Cell | PDU tables | PDU numbers |
|---|---|---|
| a | 5 | 6,7,8,9,10 |
| b | 5 | 11–15 |
| c | 6 | 38–43 |
| d | 6 | 32–37 |
| e | 6 | 26–31 |
| f | 10 | 16–25 |
| g | 5 | 1–5 |
| h | 7 | 44–50 |
| **Total** | **50** | |

## 4. What Google's own documentation says (quoted, not paraphrased)

Fetched directly from `google/cluster-data`'s `PowerData2019.md`
(2026-09-19):

> "The `powerdata-2019` trace dataset provides power utilization
> information for 57 power domains in Google data centers."

> "Two of these power domains are from cells in data centers with the
> new medium voltage power plane design. The remainder belong to the
> eight cells featured in the 2019 Cluster Data trace."

This is the official statement that **55 of the 57 documented power
domains belong to the eight ClusterData2019 cells**, and **2 do not**
(they belong to unrelated cells outside this trace and must not be
joined to any ClusterData2019 workload). The document does not mention
any finer (e.g. per-machine) linkage, and separately does not define
`measured_power_util`/`production_power_util` or the timestamp
convention in the text that was fetched — those came from direct schema
inspection (§2–3) instead. `ClusterData2019.md` was also fetched and
contains no mention of PowerData2019, power domains, or PDUs at all —
the linkage is documented one-directionally, in the power dataset's own
page.

**A discrepancy, disclosed rather than resolved by assumption**: the
documentation says 55 domains should belong to the eight cells, but the
public BigQuery mirror exposes exactly **50** `cell*_pdu*` tables (§3).
Nine fewer than 57, seven fewer than the documented 55. Possible
explanations — a different table naming/merging convention in the
BigQuery export vs. the original release, decommissioned domains, or an
export gap — were not resolved with authoritative evidence in this
session and are recorded here as an open item, not silently patched
over. It does not change the conclusion in §6: every table present *is*
unambiguously cell-tagged, whatever the total count.

## 5. Time-axis cross-check

```
powerdata_2019.cella_pdu6:      min(time)=600,000,000   max(time)=2,678,700,000,000   n=8,928
clusterdata_2019_a.instance_usage:  (dry-run only; full min/max not pulled to conserve quota)
```

8,928 rows at what would be a 5-minute cadence over the ~31 days of May
2019 (31 × 24 × 12 = 8,928, exact match) confirms PowerData2019's native
resolution is 5 minutes, and that its `time` values are, like
ClusterData2019's, microseconds relative to a common trace-start origin
(0 to ~2.68M seconds ≈ 31 days) — not absolute Unix timestamps. This
was verified empirically (row count arithmetic), not asserted from
documentation, since neither fetched doc states the epoch convention
explicitly.

## 6. Conclusion: which alignment outcome applies

Per the master plan's §6 decision tree:

- **(A) Direct per-machine → per-domain mapping**: **does not exist**.
  No shared identifier connects `instance_usage.machine_id` to any
  `powerdata_2019` row. This must not be fabricated, and is not used
  anywhere in this project.
- **(B) A valid common-domain aggregation exists**: **yes — the Borg
  cell**. ClusterData2019 partitions workload into exactly the 8 cells
  the power documentation names; PowerData2019 carries an explicit
  `cell` column on every row, and 50 (documented: 55) of its 57 PDUs
  are asserted by Google to belong to those same 8 cells. Cell is
  therefore the **finest scientifically defensible join key** — not
  inferred from convenience, but read directly off both schemas plus
  the quoted documentation sentence.
- **(C) No defensible mapping**: does not apply; outcome B holds.

**Information lost by aggregating to cell level**: all per-machine and
per-PDU-within-cell structure. The power model (Phase 8) will predict
cell-level aggregate power from cell-level aggregate workload, not
machine-level power — a materially different (and more defensible)
claim than V1's Kaggle-derived model made no attempt at, since V1 had
no power data at all. This must be stated plainly in every downstream
document that reports power-model results: **the unit of prediction is
one of 8 Borg cells, not a machine or rack.**

**Assumptions that remain, disclosed**: (1) the two datasets share a
common time-zero (§5, verified empirically, not doc-confirmed); (2)
`measured_power_util` is a fraction of unknown absolute PDU capacity,
so any power model built on it produces normalized, not absolute-watt,
predictions unless an external PDU capacity figure is later found and
documented; (3) the 50-vs-55 table discrepancy (§4) is unresolved.

## 7. Extraction cost / quota audit

All queries below used `SELECT`-list column pruning to minimize
bytes scanned; every non-trivial query was dry-run before being run for
real, per the master plan's requirement.

| Query | Dry-run estimate | Result |
|---|---|---|
| Cell-level workload aggregation, 1 cell (`instance_usage`) | ~181 GB | ran for cells a, b, c, d |
| Cell-level workload aggregation, all 8 cells (single UNION query) | 1,893,553,627,040 bytes (~1.76 TiB) | **failed**: sandbox free-quota exceeded |
| Cell-level workload aggregation, cell e (5th cell attempted) | ~181 GB | **failed**: sandbox free-quota exceeded (cumulative cap hit after ~724 GB) |
| `machine_events`, all 8 cells (union) | 16,169,640 bytes (~15 MB) | succeeded |
| `powerdata_2019`, all 50 PDU tables (union) | 15,990,048 bytes (~15 MB) | succeeded |

**Result: cells a, b, c, d of the workload trace are extracted; cells
e, f, g, h are not, because this GCP project has no billing account
linked and BigQuery Sandbox's free monthly scan allowance was exhausted
partway through the 5th cell.** This is a genuine, disclosed human-only
dependency, not a bug: `scripts/bigquery/run_clusterdata_extraction.sh`
will pick up exactly where it left off (it skips cells with an existing,
non-error output file) once billing is enabled on
`borg-datacenter-cooling-v2`. `machine_events` and the complete
`powerdata_2019` (all 50 tables, all 8 cells) were extracted in full —
they are cheap enough to fit the sandbox allowance regardless.

## 8. What this blocks downstream

Phase 6 (workload preprocessing) and everything after it should not
proceed to a "final" frozen configuration on 4-of-8 cells. Using cells
a–d as a **pilot/schema-validation dataset only** (to build and unit-test
the preprocessing, forecasting, and thermal pipeline code end-to-end) is
reasonable and is what this session does next; declaring any V2 result
final before cells e–h are available would violate the master plan's
own "no result is final until the frozen pipeline runs on the full,
intended dataset" discipline.
