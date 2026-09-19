# Dataset Audit — Borg Traces (Kaggle) — PHASE 1

Status: **Phase 1 complete, pending review.** No downstream pipeline
(workload extraction, forecasting, thermal modeling, RL) has been built.
The raw file in `data/raw/` has not been modified.

Evidence classification used throughout this document:

- **A — Directly observed from data**: computed from the actual file in this repo.
- **B — Documented by dataset provider**: stated by the Kaggle dataset page/metadata.
- **C — Inferred from data**: a reasoned interpretation of A-type evidence, not asserted by the provider.
- **D — Simulation/modeling assumption**: something we would have to assume later, not present in the data.
- **E — Not determinable**: cannot currently be established.

---

## 1. Dataset Identity and Provenance

| Field | Value | Status |
|---|---|---|
| Kaggle URL | `kaggle.com/datasets/muzairbair/borg-traces-data` | user-supplied |
| Kaggle dataset ID | `6297131` | B (`kaggle datasets metadata`) |
| Owner | `muzairbair` | B |
| Title | "borg traces data" | B |
| Subtitle | *(empty)* | B |
| Description | *(empty — the dataset page has no written description)* | B |
| License | `apache-2.0` | B |
| Usability rating | 0.235 / 1.0 | B |
| Total votes / downloads (at audit time) | 1 vote / 68 downloads | B |
| Files listed by Kaggle | `borg_traces_data.csv`, 328,310,714 bytes, created 2024-12-13 | B |

**Important finding:** the Kaggle dataset provides **no description, no column
documentation, and no stated relationship to the original Google Borg
cluster-trace release**. Everything below about what the columns *mean*
is therefore inferred from the data itself (category C) and, where noted,
cross-checked against general public knowledge of the Google
cluster-usage-trace v3 schema (which is *not* something this Kaggle
uploader documented, so it is still marked C, not B). Column names
(`instance_events_type`, `collections_events_type`, `resource_request`,
`average_usage`, `cpu_usage_distribution`, etc.) match the field names
used in Google's publicly released 2019 Borg cluster trace schema
(tables `instance_events`, `collection_events`, `instance_usage`), which
strongly suggests this CSV is a derivative/flattened export of that
trace, but **this is an inference, not a documented fact**, since the
uploader did not state it.

**Raw file integrity:** the local file size (328,310,714 bytes) exactly
matches the size reported by the Kaggle API for `borg_traces_data.csv`.
`data/raw/` was not altered by this audit or any subsequent phase
(read-only access throughout, enforced by `src/data/verify_raw_integrity.py`
— see §19). **Caveat:** no cryptographic hash of the file was captured
at download time, before the first inspection. A baseline hash
(`b88966b4394c54206282a6b4a0c85b57`) was recorded once processing began
(§19) and has been reverified unchanged at every subsequent audit and
build step. This establishes "unmodified since we started working with
it," not "unmodified since the moment it was downloaded" — the latter
cannot be established from evidence available to this project.

---

## 2. File Inventory

| Filename | Size (bytes) | Size (MB) | Format |
|---|---|---|---|
| `borg_traces_data.csv` | 328,310,714 | 328.31 | CSV, comma-delimited, quoted fields, header row present |

This is the **only** file in `data/raw/` and the only file in the
Kaggle dataset. No compression, no companion schema/README file, no
separate metadata file was provided by the uploader.

---

## 3. Shape — a critical, non-obvious finding

| Measurement | Value | Status |
|---|---|---|
| Physical newline-terminated lines (incl. header) | 1,324,695 | A |
| **Actual logical CSV records** (correctly parsed) | **405,894** | A |
| Columns | 34 | A |
| In-memory size when loaded as a pandas DataFrame | ≈600 MB | A |

**A naive `wc -l`-based row count is wrong by a factor of ~3.3x.** Two
columns (`cpu_usage_distribution`, `tail_cpu_usage_distribution`) store
the **`str()`/`repr()` output of a numpy array**, and numpy inserts
literal newline characters when it line-wraps a printed array. Those
newlines land *inside a quoted CSV field*, so a single logical record
can span several physical lines. A correct CSV parser (pandas' C
parser, used here) resolves this correctly; a line-counting approach
would not. This was verified directly: reading the file with
`pandas.read_csv` (which respects quoting) yields exactly 405,894 rows,
confirmed stable across chunked and full reads.

**Consequence for later phases:** any teammate or tool that estimates
row count via `wc -l`, or that tries to hand-parse this CSV line by
line, will silently get row/record boundaries wrong. All future
processing must go through a real CSV parser, never naive line
splitting.

---

## 4. Schema

34 columns (the first is an unlabeled pandas row index left over from
how the uploader saved the file):

| # | Column | dtype (as parsed) | Null % | Unique | Notes |
|---|---|---|---|---|---|
| 0 | `Unnamed: 0` | int64 | 0 | 405,894 | Exactly the row index 0..405893. Not semantic data — an artifact of `DataFrame.to_csv()` with `index=True`. Should be dropped. |
| 1 | `time` | int64 | 0 | 334,354 | Independent event timestamp; see §6. |
| 2 | `instance_events_type` | int64 (0–10, code 7 absent) | 0 | 10 | Byte-identical to `collections_events_type` (100% of rows) and a perfect bijection with `event` (see §5). |
| 3 | `collection_id` | int64 | 0 | 4,057 | Job/collection identifier. |
| 4 | `scheduling_class` | int64 (0–3) | 0 | 4 | Matches Borg's known 0–3 scheduling-class enum. |
| 5 | `collection_type` | int64 (0/1) | 0 | 2 | 91.2% = 0. |
| 6 | `priority` | int64 (0–450) | 0 | 22 | Discrete priority levels. |
| 7 | `alloc_collection_id` | int64 | 0 | 448 | 77.8% = 0 (sentinel: "no alloc set"); nonzero values start at 2,449, a separate ID namespace from `collection_id` (min 6,824). |
| 8 | `instance_index` | int64 | 0 | 30,298 | Index of a task instance within its collection. |
| 9 | `machine_id` | int64 | 0 | 96,174 | 19 rows = −1 (sentinel: "no machine assigned/unscheduled"). Almost cluster-exclusive (see §7). |
| 10 | `resource_request` | object (dict-string) | 0.19% | 21,900 | `{'cpus': float, 'memory': float}`, normalized units (values ≪ 1). |
| 11 | `constraint` | object (numpy-array-string) | 0 | 78 | List of `{'name','value','relation'}` dicts, printed via numpy repr (not directly `ast.literal_eval`-safe when >1 element — see §8). |
| 12 | `collections_events_type` | int64 | 0 | 10 | Duplicate of `instance_events_type`; see §5. |
| 13 | `user` | object (hashed string) | 0 | 898 | Anonymized/hashed user identifier. |
| 14 | `collection_name` | object (hashed string) | 0 | 3,248 | Anonymized job name. |
| 15 | `collection_logical_name` | object (hashed string) | 0 | 2,261 | Anonymized logical job name (job family). |
| 16 | `start_after_collection_ids` | object (numpy-array-string) | 0 | 34 (raw string form) | List of collection IDs this collection depends on; `[]` for 400,705 rows (98.7%). |
| 17 | `vertical_scaling` | float64 (1–3) | 0.24% | 3 | |
| 18 | `scheduler` | float64 (0/1) | 0.24% | 2 | |
| 19 | `start_time` | int64 | 0 | 4,182 | Usage-measurement window start (microseconds, inferred — see §6). |
| 20 | `end_time` | int64 | 0 | 4,135 | Usage-measurement window end. Always ≥ `start_time` (0 violations). |
| 21 | `average_usage` | object (dict-string) | 0 | 4,716 | `{'cpus': float, 'memory': float}`, normalized units. **Primary workload-signal candidate.** |
| 22 | `maximum_usage` | object (dict-string) | 0 | 4,717 | Same shape; peak usage in the interval. |
| 23 | `random_sample_usage` | object (dict-string) | 0 | 2,398 | `'memory'` key is `None` in 100% of a 5,000-row sample — effectively CPU-only. |
| 24 | `assigned_memory` | float64 | 0 | 1,534 | Normalized units, max 0.286. |
| 25 | `page_cache_memory` | float64 | 0 | 1,207 | Normalized units. |
| 26 | `cycles_per_instruction` | float64 | 30.72% | 3,417 | Hardware performance-counter field; not available for all machine platforms — this is the expected/known reason for high nulls in the public trace, not a data-corruption signal. |
| 27 | `memory_accesses_per_instruction` | float64 | 30.72% | 3,418 | Same null pattern/rows as `cycles_per_instruction`. |
| 28 | `sample_rate` | float64 (0.11–1.0) | 0 | 94 | Fraction of the interval actually sampled; 99.2% of rows ≥ 0.985. |
| 29 | `cpu_usage_distribution` | object (numpy-array-string) | 0 | 4,552 | ~11-point CPU usage percentile distribution; `[]` for 15,675 rows (3.9%). Not directly `ast.literal_eval`-parseable — see §8. |
| 30 | `tail_cpu_usage_distribution` | object (numpy-array-string) | 0 | 4,519 | ~9-point tail-percentile distribution; same parsing caveat. |
| 31 | `cluster` | int64 (1–8) | 0 | 8 | Independent Borg cell/cluster identifier (see §7). |
| 32 | `event` | object (string enum) | 0 | 10 | Human-readable mirror of `instance_events_type`/`collections_events_type`. |
| 33 | `failed` | int64 (0/1) | 0 | 2 | 22.8% = 1. Correlates with `event == 'FAIL'` (both count 92,678 — consistent, not yet cross-verified row-by-row beyond count match). |

---

## 5. A Major Data-Construction Finding: Duplicated / Derived Columns

Verified directly on the full 405,894-row dataset:

- `instance_events_type == collections_events_type` for **100%** of rows.
  These two columns are byte-identical. This is almost certainly a
  construction artifact of however the uploader merged source tables
  (e.g., a single underlying `type` column was copied into both an
  "instance" and a "collection" column name), **not** evidence that two
  independently-sourced tables happen to agree perfectly on every one
  of 405,894 rows.
- `event` (string) is a **perfect bijection** of `instance_events_type`
  (int), confirmed via a full cross-tabulation with zero off-diagonal
  entries:

  | code | event string | count |
  |---|---|---|
  | 0 | ENABLE | 75,907 |
  | 1 | EVICT | 14,756 |
  | 2 | FAIL | 92,678 |
  | 3 | FINISH | 92,867 |
  | 4 | KILL | 951 |
  | 5 | LOST | 59,515 |
  | 6 | SCHEDULE | 69,104 |
  | 8 | QUEUE | 4 |
  | 9 | UPDATE_PENDING | 111 |
  | 10 | UPDATE_RUNNING | 1 |

  Code 7 (which would be `SUBMIT` in the public Borg enum ordering)
  **never appears** in this file. This numeric-code-to-string mapping
  does not match the commonly cited public Borg enum ordering exactly
  (e.g. `QUEUE` here is 8, not an early code), so it should be treated
  as **this file's own internal encoding**, not assumed to match any
  external reference table.

**Practical implication for Phase 2+:** treat `event` as the single
source of truth for event semantics; drop `instance_events_type`
and/or `collections_events_type` as redundant, or keep exactly one and
document that the other was dropped as a duplicate.

---

## 6. Timestamps

Three integer time-like columns: `time`, `start_time`, `end_time`.

| Column | Min | Max | Distinct values |
|---|---|---|---|
| `start_time` | 300,000,000 | 2,678,400,000,000 | 4,182 |
| `end_time` | 600,000,000 | 2,678,700,000,000 | 4,135 |
| `time` | 0 | 9,223,372,036,854,775,807 | 334,354 |

**Units (C — inferred, not documented by the uploader):** treating
`start_time`/`end_time` as **microseconds** gives a span of exactly
**31.0 days** (`(max(end_time) - min(start_time)) / 1e6 / 86400 =
31.0`), which matches the well-known ~1-month duration of the public
Google 2019 Borg cluster trace. This is strong circumstantial evidence
for microsecond units and for this being derived from that public
trace, but it is an inference, since nothing in the Kaggle metadata
states units.

**`start_time`/`end_time` sampling granularity:** the single most
common gap between consecutive distinct values is exactly
300,000,000 µs (5 minutes), and 71.6% of `start_time` values (71.6% of
`end_time` values) are exact multiples of 300,000,000 µs. This is
**not** a perfectly regular 5-minute grid — the remaining ~28% of
values fall off-grid — so the measurement-interval timestamps must be
treated as **irregularly sampled, with a dominant 5-minute cadence**,
not resampled/binned data. `end_time >= start_time` holds for 100% of
rows (0 violations), which is a good internal-consistency signal.

**`time` is a materially different, independent field.** It is
byte-different from both `start_time` and `end_time` for every
sampled row, and has no consistent ordering relationship to the
`[start_time, end_time]` interval of the same row (in a random sample,
`time` falls before `start_time` in some rows and after `end_time` in
others; overall only 50.5% of rows have `time <= end_time` and only
57.9% of rows with `time > 0` have `time >= start_time`). Combined
with 13.9% of rows having `time == 0` and exactly 3 rows pinned at the
int64 maximum value (9,223,372,036,854,775,807 — a classic
"unset/never" sentinel), the most defensible interpretation (C) is
that `time` is the **native event timestamp of a separate original
record** (e.g., an `instance_events`/`collection_events` row's own
`time` field) that was joined onto a *different* table's usage-interval
fields (`start_time`, `end_time`) during however this CSV was
constructed — i.e. two different timelines from two different source
tables, merged into one row, not necessarily temporally aligned.
**This is inferred, not documented**, and should not be treated as
established fact.

**Duplicate timestamps:** not applicable as a standalone check — there
is no single-column primary key expectation here; see §9 for row-level
duplicate analysis (zero exact duplicate rows found).

---

## 7. Clusters and Machines (relevant to the "3-zone" design question)

| Finding | Value | Status |
|---|---|---|
| Distinct `cluster` values | 8 (values 1–8) | A |
| Rows per cluster | 42,713 – 58,783 (roughly balanced) | A |
| Time span per cluster | Each cluster individually spans ~30.7–31.0 days, **concurrently** (all 8 clusters' min/max `start_time` overlap the same ~31-day window) | A |
| `machine_id` uniqueness across clusters | Only 42 of 96,174 distinct `machine_id` values appear in more than one cluster (0.04%) | A |

**Interpretation (C):** `cluster` behaves like an identifier for 8
independent, concurrently-running Borg cells/datacenters, each with its
own effectively-private machine namespace. This is directly useful for
Phase 15 (three-zone mapping): selecting **workload from real,
distinct clusters** (e.g., 3 of the 8 `cluster` values) is a more
defensible zone-partitioning strategy than any within-cluster
synthetic split, because clusters are real, independently-operating
units in the data rather than an invented partition. This does **not**
give physical rack/room location — see §10.

---

## 8. Nested / Encoded Columns — a data-quality finding requiring two different parsers

Two distinct serialization styles are mixed in this file, and they are
**not interchangeable**:

1. **Proper Python dict/list literals** (parseable with `ast.literal_eval`):
   `resource_request`, `average_usage`, `maximum_usage`,
   `random_sample_usage`. **Verified on the full 405,894-row population**
   (upgraded from an initial 5,000-row sample check during independent
   review): 0 parse errors, keys are always exactly `{'cpus', 'memory'}`
   for every one of the four columns, no exceptions found.
   `random_sample_usage['memory']` is `None` in 100% of rows — this
   sub-field carries no information in this file.

2. **`numpy.ndarray.__repr__()` strings** (NOT safely
   `ast.literal_eval`-parseable once an array has more than one
   element, because numpy prints array elements **space-separated with
   no commas**, and inserts a literal `\n` when the line wraps):
   `constraint`, `start_after_collection_ids`, `cpu_usage_distribution`,
   `tail_cpu_usage_distribution`. Example raw value observed in the
   file for `cpu_usage_distribution`:
   ```
   [0.00314331 0.00381088 0.00401306 0.00415039 0.00432587 0.00449371
    0.00463104 0.00478363 0.00498962 0.00530243 0.01194763]
   ```
   356,745 of 405,894 non-null values in this column (87.9%) contain an
   embedded newline. Extracting the numeric values requires a
   regex/`np.fromstring`-style parser, not JSON or `literal_eval`.
   **Corrected, full-population finding (was originally reported from a
   200-row sample as "min 0, max 11, mean 10.45 points," which
   understated the true regularity by conflating empty arrays into a
   length distribution):** `cpu_usage_distribution` has **exactly 11**
   elements whenever non-empty (390,219/390,219 non-empty rows checked)
   and is `[]` (empty) for exactly 15,675 rows (3.9%);
   `tail_cpu_usage_distribution` has **exactly 9** elements whenever
   non-empty and is `[]` for exactly the same 15,675 rows — verified as
   a 100% row-index overlap between the two columns' empty sets, not
   merely an equal count.

**Practical implication:** Phase 2 preprocessing code needs a
dedicated parser for the numpy-repr columns; the dict columns can use
`ast.literal_eval` directly.

---

## 9. Data Quality Summary

| Check | Result | Status |
|---|---|---|
| Exact duplicate rows (all 34 columns, incl. index) | 0 | A |
| Exact duplicate rows (excluding the `Unnamed: 0` index column) | 0 | A |
| Rows with any null value | `resource_request` is the only column with nulls below 1% (0.19%, 774 rows); `vertical_scaling`/`scheduler` at 0.24% (959 rows); `cycles_per_instruction`/`memory_accesses_per_instruction` at 30.72% (124,688 rows, same rows for both) | A |
| Null-pattern structure (**corrected**, full 34-column check — see §20 correction log) | **6** distinct null patterns across all 34 columns. **69.2%** of rows (280,877/405,894) have zero nulls anywhere; the remaining 30.8% are fully explained by two already-documented, independent missingness mechanisms — `cycles_per_instruction`/`memory_accesses_per_instruction` missing together (124,688 rows total across 3 of the 6 patterns, matching the separately-reported 30.72%) and `vertical_scaling`/`scheduler` missing together (959 rows total across 2 patterns) — plus `resource_request` missing alone or in combination (774 rows total). No pattern suggests a sparse block-diagonal union structure. | A |
| `start_time <= end_time` violations | 0 / 405,894 | A |
| `machine_id == -1` (sentinel: unscheduled) | 19 rows | A |
| `alloc_collection_id == 0` (sentinel: no alloc set) | 77.8% of rows | A |
| `time` at int64-max sentinel | 3 rows | A |
| `time == 0` | 56,452 rows (13.9%) | A |

**Interpretation of the null-pattern finding (C) — corrected:** an
earlier draft of this audit checked null co-occurrence across only 8
hand-picked columns and reported "99.81% fully populated, 1 alternate
pattern." A full 34-column check (§20) found 6 patterns and a lower
69.2% fully-non-null rate. The corrected picture still supports the
original qualitative conclusion, but on stronger, fuller evidence: all
6 patterns decompose into two independent, already-documented
missingness mechanisms (hardware-counter fields, and
scaling/scheduler metadata) rather than into a pattern consistent with
a block-diagonal union of heterogeneous source-table row types (which
would show many more, larger, mutually exclusive sparse blocks). It
still looks like the uploader performed a **join** across source
tables (most likely keyed on collection/instance/machine identifiers)
before exporting, producing one wide, mostly-dense table. Combined
with the `time` vs. `start_time`/`end_time` mismatch in §6, the most
defensible working hypothesis remains: **each row = one instance-level
scheduling/lifecycle event, joined to one resource-usage measurement
interval for the same instance, where the two may come from different
moments in that instance's life.** This is inferred, not documented,
and is flagged as a modeling risk in §14.

**Corroborating check added in review (A):** no evidence of duplicate-
join fan-out was found. `df.duplicated(subset=['collection_id',
'instance_index','machine_id','start_time','end_time'])` = **0** across
the full 405,894-row population — every row sharing an instance/machine
key has a distinct usage-measurement window. This directly supports
treating rows as independent, non-redundant measurements safe to sum
when building a workload aggregate (§16), and rules out the specific
risk that summing `average_usage.cpus` across rows in a time bin could
double-count a single underlying measurement duplicated by the join.

---

## 10. Record / Entity Semantics

**What does one row represent?** Evidence-based answer: a row is **not**
one machine-second of telemetry, and **not** one job.

**Corrected grain (see §20 correction log — an earlier draft of this
audit used `(collection_id, instance_index)` alone as "the natural
task-instance key," which is wrong):** `(collection_id, instance_index)`
is **not** a stable single-instance identity — a single such pair can
correspond to thousands of physically different task instances running
on different machines. Verified: grouping by `(collection_id,
instance_index)` alone gives 242,946 pairs, mean 1.67 rows/pair, max
**7,704** — but that maximum traces to exactly one pair
(`collection_id=116646838687, instance_index=17`) that spans **6,640
distinct `machine_id` values** while sharing only 2 distinct
`(start_time, end_time)` windows across all of them. This is most
plausibly a job that was repeatedly evicted/rescheduled (or spread
across many machines) over the trace — a real phenomenon, not a data
defect — but it means `instance_index` alone does not identify one
physical task instance.

The correct, stable task-instance-on-a-machine key is `(collection_id,
instance_index, machine_id)`: grouping on all three gives **393,573**
distinct keys, mean **1.03** rows/key, median 1.0, **max 3**. 15.5%
(37,680/242,946) of `(collection_id, instance_index)` pairs touch more
than one machine, but this is concentrated — the 5 most extreme cases
account for only ~3.1% of all rows. Combined with the join hypothesis
in §9 and the zero-fan-out finding above, the best-supported
description (C) is:

> One row = one (instance lifecycle event ∪ resource-usage-interval
> report) record for a specific `(collection_id, instance_index,
> machine_id)` task instance, within one of 8 Borg clusters, at some
> point during a ~31-day trace.

This is **not** a machine-level record, and **not** a cluster-level
aggregate — those would need to be constructed (§12).

---

## 11. Workload / CPU Statistics (exact, full population — not sampled)

Computed by parsing every row's `average_usage` dict (405,894/405,894
parsed successfully, 0 errors):

| Field | count | mean | std | min | 25% | 50% | 75% | max | % exactly zero |
|---|---|---|---|---|---|---|---|---|---|
| `average_usage.cpus` | 405,894 | 0.00745 | 0.01857 | 0.0 | 0.00020 | 0.00105 | 0.00729 | 0.53809 | 11.75% |
| `average_usage.memory` | 405,894 | 0.00564 | 0.01656 | 0.0 | 0.00024 | 0.00128 | 0.00426 | 0.22388 | 2.16% |
| `resource_request.cpus` | 405,120 | 0.01534 | 0.02865 | 0.0 | 0.00405 | 0.00810 | 0.01590 | 0.58301 | n/a |

Units (C — inferred): all values are far below 1.0 and are consistent
with the publicly-known Google Borg convention of **normalizing
CPU/memory to fractions of the largest machine size in the cell**, not
absolute cores or bytes. This is not stated by the Kaggle uploader.

**`sample_rate`** (fraction of the `[start_time, end_time]` interval
actually covered by the usage measurement): min 0.110, 99% of rows ≥
0.985, median 1.0 — the large majority of usage records represent
(near-)complete coverage of their stated interval.

---

## 12. Other Important Fields

- **`scheduling_class`**: 4 values (0–3), roughly balanced (48,584 –
  131,567 rows each) — matches the public Borg convention of a small
  discrete latency-sensitivity class.
- **`priority`**: 22 discrete values, heavily concentrated at a few
  levels (103, 200, 0, 360 alone account for ~80% of rows).
- **`failed`**: 22.8% of rows = 1. **Verified row-wise** (upgraded from
  an initial count-only check during independent review): `failed==1`
  if and only if `event=='FAIL'` for all 405,894 rows — 0 mismatches in
  either direction. `failed` is conclusively a derived flag from the
  event type, not merely correlated with it.
- **Entity cardinalities**: `machine_id` 96,174 unique; `collection_id`
  4,057 unique; `alloc_collection_id` 448 unique (mostly the 0
  sentinel); `user` 898 unique (hashed); `collection_name` 3,248
  unique (hashed); `collection_logical_name` 2,261 unique (hashed);
  `instance_index` 30,298 unique.
- **No failure-reason or error-message field exists** beyond the
  binary `failed` flag and the `event` categorical (`FAIL`, `EVICT`,
  `LOST`, `KILL` are the failure/termination-adjacent event types).

---

## 13. Temporal Suitability for Time-Series Construction

**Feasible, with caveats:**

- `start_time`/`end_time` give each usage record a well-defined time
  interval within a consistent ~31-day timeline (once interpreted as
  microseconds — an inference, §6).
- The dominant native granularity is ~5 minutes, but sampling is
  **irregular** (only 71.6% of timestamps land on an exact 5-minute
  boundary), so building a regular time series requires an explicit
  **binning/aggregation** step (e.g., assign each `[start_time,
  end_time)` interval to the 5-minute bin(s) it overlaps), not a
  direct reindex.
- 8 clusters run concurrently over the same window, so a per-cluster
  (or per-group-of-clusters) time series is directly constructible
  without needing to align separate trace periods.
- The `time` column, given its inconsistent relationship to
  `start_time`/`end_time` (§6), is **not recommended** as the primary
  timeline for workload aggregation; `start_time` (interval start) is
  the more defensible choice.

---

## 14. Cooling-Project Suitability and Limitations

**What this dataset can support:**
- A real, non-synthetic **workload signal** (`average_usage.cpus`,
  optionally `.memory`) with genuine burstiness, at ~5-minute native
  granularity, over a real 31-day multi-cluster trace — suitable for
  the Borg → workload → forecasting pipeline (Phases 2–4).
- A defensible basis for **3 logical zones** using real, distinct
  Borg clusters (§7) rather than an arbitrary synthetic split.

**What this dataset cannot support, and must be treated as simulation
assumptions later (category D, not to be presented as data-derived):**
- **No physical location, rack, room, or cooling-zone information of
  any kind.** `cluster` is an operational/administrative grouping, not
  a physical layout — mapping cluster(s) to "thermal zones" is a
  simulation choice, not something the data proves.
- **No server power, temperature, humidity, airflow, or cooling
  equipment measurements exist anywhere in this file.** CPU→power and
  power→heat conversions in later phases will necessarily be simulation
  assumptions (as Phase 17/18 of the master plan already anticipates),
  not calibrated against this dataset.
- **No timezone is documented** for any timestamp field; only the
  relative 31-day span is inferable.
- **The exact meaning of `time` could not be established** (§6) —
  documented here as "Not determinable from the available data" beyond
  the circumstantial evidence presented.
- **The join/construction method used by the uploader to build this
  CSV from (presumably) multiple original Borg tables is undocumented**
  and had to be reverse-engineered from null patterns and duplicate
  columns (§5, §9). There is residual uncertainty in the record
  semantics in §10.
- **`cycles_per_instruction`/`memory_accesses_per_instruction` are
  missing for 30.7% of rows** — expected to be a hardware-platform
  limitation (not all machines expose these counters) per general
  knowledge of the source trace, but this Kaggle file does not confirm
  that explanation; treat the reason as inferred (C), not confirmed.

---

## 15. Required Status Table

| Item | Status | Evidence |
|---|---|---|
| Dataset has 1 CSV file, 328,310,714 bytes | Directly observed | Local `ls -la` + Kaggle API file listing match exactly |
| License is Apache-2.0 | Documented by dataset source | `kaggle datasets metadata` |
| Dataset has no written description | Documented by dataset source | `kaggle datasets metadata` (`description: ""`) |
| File contains 405,894 logical CSV records, 34 columns | Directly observed | `pandas.read_csv` full parse |
| `wc -l`-style physical line count (1,324,695) is not the row count | Directly observed | Embedded newlines inside quoted numpy-repr fields, confirmed by parser comparison |
| `instance_events_type` ≡ `collections_events_type` | Directly observed | 100% row-wise equality on full data |
| `event` is a bijection of `instance_events_type` | Directly observed | Full cross-tabulation, zero off-diagonal entries |
| Timestamp units are microseconds | Inferred from data | 31.0-day span matches known public Borg-trace duration when so interpreted; not stated by uploader |
| `start_time`/`end_time` dominant granularity is 5 minutes, irregular | Directly observed | 71.6% exact 5-min-multiple alignment; mode of consecutive-value gaps = 300,000,000 |
| `cluster` = 8 independent, concurrent Borg cells | Inferred from data | Per-cluster time spans overlap; machine IDs 99.96% cluster-exclusive |
| CPU/memory usage values are normalized fractions, not absolute units | Inferred from data | All values ≪ 1; consistent with publicly known Borg convention, not stated by uploader |
| `average_usage`, `maximum_usage`, `random_sample_usage`, `resource_request` are `ast.literal_eval`-safe dict strings | Directly observed | 0/5,000 parse errors per column |
| `constraint`, `start_after_collection_ids`, `cpu_usage_distribution`, `tail_cpu_usage_distribution` are numpy-repr strings, not literal_eval-safe | Directly observed | Embedded newlines, no commas, verified on samples |
| No exact duplicate rows | Directly observed | Full-row hash comparison, 0 collisions |
| Physical rack/room/zone location of any kind | Not determinable | No such field exists in the schema |
| Server power/temperature/cooling measurements | Not determinable | No such field exists in the schema |
| Exact semantics of the `time` column | Not determinable | Inconsistent ordering vs. `start_time`/`end_time`; best guess documented as inference only |
| Whether this CSV is a full or partial export of the original Google trace | Not determinable | Uploader provided no provenance statement |

---

## 16. Required Decision Analysis (Section 51)

**What is the most defensible workload signal we can extract?**
`average_usage.cpus` from the usage-interval records, aggregated over
`start_time`. It is a real, directly-observed, continuously-valued
utilization signal with genuine variance (std ≈ 2.5× mean, heavy right
tail up to 0.54), as opposed to `resource_request` (a requested/allocated
ceiling, not actual usage) or `maximum_usage` (a peak, not representative
of sustained load). `.memory` is a secondary candidate for a
multi-resource extension.

**What is the appropriate temporal aggregation?**
Bin by `start_time` (not `time`, per §6/§13) into fixed 5-minute windows
— matching the dominant native granularity — using an overlap-aware
assignment (an interval can contribute to more than one bin if it
crosses a bin boundary) rather than a naive floor/round, because
sampling is irregular (§13).

**What entity level should be aggregated?**
Instance-level (`collection_id` + `instance_index` + `machine_id`)
records summed/averaged up to cluster level per time bin. Task-instance
level is the finest level the data actually supports (§10); machine-
level or sub-instance-level aggregation is not directly supported
without further assumptions.

**What information should be retained vs. discarded?**
Retain: `start_time`, `end_time`, `cluster`, `machine_id`,
`collection_id`, `instance_index`, `average_usage` (cpus, memory),
`resource_request` (cpus, memory), `sample_rate`, `failed`, `event`.
Discard as redundant/non-informative: `Unnamed: 0` (pandas index
artifact), `collections_events_type` (exact duplicate of
`instance_events_type`), `random_sample_usage.memory` (always `None`),
`instance_events_type` (kept only if `event` string is not used
instead — keep exactly one of the two).

**Can the data support three logical workload zones?**
Yes, with the caveat that "zone" is a logical/administrative grouping,
not a physical one (§14). The most defensible option is to use 3 of
the 8 real `cluster` values as 3 zones — these are genuinely
independent, concurrently-operating groupings in the data, not an
invented split.

**If physical zone information is absent, what mapping strategies are
possible?**
(a) Direct use of 3 real `cluster` IDs as 3 zones (simplest, most
defensible, but zone "coupling" would then be a pure simulation
assumption since real clusters don't thermally interact). (b) Split one
cluster's machines into 3 groups by a deterministic rule (e.g. hash of
`machine_id` mod 3) to get correlated-but-distinct workload for 3
notionally co-located zones. (c) Cluster machines within one Borg
cluster by their workload time-series profile (e.g. k-means on
per-machine load curves) into 3 profile groups.

**Which mapping strategy is most defensible for simulation and why?**
Option (b) or (c) applied **within a single real cluster** is more
defensible for the cooling use case than option (a): the project's
premise requires zones that plausibly sit in the *same* physical
facility and thermally couple, and 8 independent Borg cells almost
certainly do not share a building. Using 3 real, independent clusters
as "3 zones" would misrepresent independent datacenters as one coupled
facility. This choice must be documented explicitly as a **simulation
assumption (D)** in Phase 2, with sensitivity analysis over the
grouping rule as the master plan already requires (§15 of the master
prompt).

**What information is missing for physical thermal calibration?**
All of it: no temperature, power, airflow, humidity, PUE, chiller, or
CRAC/CRAH data exists anywhere in this file. Every thermal/power
parameter in Phases 17–19 will be a sourced-from-literature or assumed
value, never a calibrated one, and must be labeled as such.

**What forecast target should be used?**
Per-zone aggregated `average_usage.cpus` (and optionally `.memory`),
next 30 minutes (6 steps of 5 minutes), consistent with the master
plan's forecast horizon.

**What forecast horizon is feasible?**
30 minutes (6 × 5-minute steps) is reasonable given the ~5-minute
native granularity; going materially shorter than 5 minutes is not
supported by the data's own sampling cadence without interpolation.

**What are the major risks of using this dataset?**
(1) Uploader-side provenance is undocumented — record semantics were
reverse-engineered (§5, §9, §10) and could be wrong in ways not yet
visible. (2) The `time` column's true meaning is unresolved (§6). (3)
Irregular sampling means naive resampling will introduce artifacts if
not handled carefully. (4) No physical zone or thermal data exists at
all, so the entire thermal side of the project is, by necessity,
simulation, and must never be described as validated against real
measurements. (5) 30.7% missingness in the CPI/MAPI fields means those
two columns are unsuitable as reliable primary features without
imputation or exclusion.

**What should Phase 2 do?**
Build the workload-extraction pipeline: parse `average_usage` (and
`resource_request`) via `ast.literal_eval`; parse
`cpu_usage_distribution`/`tail_cpu_usage_distribution` with a
numpy-repr-aware parser (or drop them, if not needed for the workload
signal); decide and implement the 5-minute binning rule for
`start_time`; decide and implement the zone-mapping rule (recommend
option b/c above, within one cluster, as a documented simulation
assumption); and produce a clean, chronologically-ordered
`(timestamp, zone, cpu_utilization, memory_utilization)` table as the
input to Phase 3 (chronological train/val/test split). Do not begin
forecasting model code until that table exists and has passed basic
sanity checks (monotonic time, no leakage, expected value ranges).

---

## 19. Raw Data Integrity Verification (Ongoing)

A baseline manifest was recorded once processing began and is checked
before every subsequent phase that reads `data/raw/`:

| Field | Value |
|---|---|
| File | `data/raw/borg_traces_data.csv` |
| Size | 328,310,714 bytes |
| MD5 | `b88966b4394c54206282a6b4a0c85b57` |
| Baseline recorded | during Phase 1 (first inspection), not at download time — see the caveat in §1 |

The manifest is stored at `docs/raw_data_manifest.json` and is checked
by `src/data/verify_raw_integrity.py`, which every downstream
processing script (Phase 2 onward) calls before touching
`data/raw/`, aborting if the hash no longer matches. As of the most
recent independent audit review, the hash and file size were
reverified identical across two separate sessions.

---

## 20. Independent Review — Correction Log

An independent adversarial review of this audit (re-deriving every
material statistic via different code paths: the stdlib `csv` module
instead of pandas, regex parsing instead of `ast.literal_eval`,
`.duplicated()` instead of `hash_pandas_object`, and full-population
checks instead of sampled ones) found that all core numbers reproduced
exactly, with two corrections required to claims that were imprecise
or too narrowly scoped as originally written. Both are now folded into
the sections above; they are logged here for traceability:

1. **Null-pattern scope** (§9): the original draft checked only 8
   hand-picked columns and reported "99.81% fully populated, 1
   alternate pattern." The full 34-column check found 6 patterns and a
   69.2% fully-non-null rate. The qualitative conclusion (evidence for
   a joined table, not a union of heterogeneous record types) held up
   under the fuller check, but the narrower framing was corrected.
2. **Task-instance grain** (§10): the original draft used
   `(collection_id, instance_index)` as "the natural task-instance
   key," which is not a stable single-instance identity (one such pair
   can span thousands of machines). The corrected key is
   `(collection_id, instance_index, machine_id)`, and a corroborating
   zero-duplication check was added confirming no join fan-out exists
   at that grain — which positively supports, rather than undermines,
   the sum-based workload-aggregation plan in §16.

No other claim in this document was found to be factually incorrect;
several (dict-column parsing, `failed`/`event` correspondence,
array-length regularity) were upgraded from sampled to full-population
verification during the same review, and are marked as such inline.

---

## 21. Files Created by This Audit

- `docs/dataset_audit.md` — this file.
- `docs/dataset_summary.json` — machine-readable statistics backing this report (per-column stats, timestamp gap histograms, null-pattern analysis, dict/array-column parsing diagnostics, categorical distributions, entity cardinalities).
- `docs/raw_data_manifest.json` — the baseline integrity manifest (§19).
- `src/data/inspect_borg.py` — the reproducible audit script (read-only w.r.t. `data/raw/`).
- `src/data/verify_raw_integrity.py` — the ongoing raw-data integrity check used by all downstream phases.

## 22. Exact Reproduction Command

```bash
cd "borg-datacenter-cooling"
.venv/bin/python src/data/inspect_borg.py
```

This regenerates `docs/dataset_summary.json` from
`data/raw/borg_traces_data.csv`. It performs a full (non-chunked) load
of the CSV — verified in this audit to require only ~600 MB of RAM and
~3 seconds, well within this machine's available memory (~8 GB free at
audit time) — so no chunking was needed for correctness; this is
recorded explicitly rather than left as a silent assumption, per the
large-data-safety requirement in the master plan. Several additional
one-off verification checks used during this audit (row-wise column
equality, event/type cross-tabulation, `time` vs. `start_time`/`end_time`
ordering, exact `average_usage` statistics via full-population
`ast.literal_eval` parsing, per-cluster time spans, machine/cluster
exclusivity) were run interactively and are described inline in this
document with their exact results, but were not folded into
`inspect_borg.py` to keep that script focused on the core audit; they
are simple pandas one-liners reproducible from the descriptions above
if needed.
