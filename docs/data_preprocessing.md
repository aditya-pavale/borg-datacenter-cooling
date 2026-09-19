# Data Preprocessing — Phases 2 & 3

Status: complete. Raw data untouched throughout (verified by
`src/data/verify_raw_integrity.py` at the start of every script below).

## 1. Pipeline

```
data/raw/borg_traces_data.csv
        │  src/data/clean_borg.py
        ▼
data/processed/borg_cleaned.parquet      (405,894 rows, typed, dict-columns parsed)
        │  src/data/aggregate_workload.py
        ▼
data/processed/workload_timeseries.parquet  (2,977 bins x 3 zones = 8,931 rows)
        │  src/data/split_workload.py
        ▼
data/processed/{train,val,test}.parquet   (chronological 60/20/20)
```

Reproduce with:
```bash
.venv/bin/python src/data/clean_borg.py
.venv/bin/python src/data/aggregate_workload.py
.venv/bin/python src/data/split_workload.py
```

## 2. Cleaning (`clean_borg.py`)

Parses `average_usage` and `resource_request` (Python-dict-literal
strings, verified 0 parse errors on the full population in the Phase-1
audit) into flat `average_usage_cpus`, `average_usage_memory`,
`resource_request_cpus`, `resource_request_memory` columns via a regex
extractor (faster than `ast.literal_eval` at this row count and
equally exact — cross-validated against `ast.literal_eval` during the
Phase-1 independent review). Drops columns identified in
`docs/dataset_audit.md` as redundant (`Unnamed: 0`,
`instance_events_type`/`collections_events_type` duplicate pair, the
numpy-repr columns not needed for a workload signal, hashed identifier
columns, and secondary usage/scheduling metadata not used downstream).
Full column-by-column rationale is in the module docstring of
`clean_borg.py` itself.

## 3. Aggregation — what we tried first, what broke, and the corrected design

### 3.1 Original design (attempted, measured, rejected)

The originally planned design (matching the master project brief's
default expectation) was: pick **one real Borg `cluster`** as the
workload source for a 3-zone facility (physically cleaner — one real
cluster is at least one real operational unit, whereas 8 independent
clusters are almost certainly not co-located), split its machines into
3 zones via `hash(machine_id) % 3`, and bin usage records into
**5-minute** intervals (matching the dataset's dominant native
granularity, per `docs/dataset_audit.md` §13).

This was implemented and run. Result, using `cluster == 3` (the
largest cluster, 13,206 machines, 58,783 usage records over the
31-day trace): **94.36%** of the resulting (zone, 5-minute-bin) cells
had **zero** usage records and were filled with `U = 0` (idle).
Cluster-wide (before the 3-way zone split), **89.4%** of all 5-minute
bins across the entire 31-day span had zero usage-report rows anywhere
in the whole cluster.

**Root cause (evidence-based):** this is not a coding bug. A follow-up
check on non-empty bins showed they contain a *burst* of ~60 records on
average, not a steady trickle — i.e. usage rows arrive in clusters tied
to specific moments, not on a fixed per-instance cadence. This is
consistent with the Phase-1 audit's join hypothesis (docs/dataset_audit.md
§9/§20): each row appears to be tied to a **lifecycle event**
(SCHEDULE, ENABLE, FINISH, FAIL, EVICT, KILL, LOST — the values of the
`event` column, present on every row), not a periodic telemetry
snapshot. A real Borg cluster's full `instance_usage` table would
contain roughly one row **per running instance per fixed sampling
interval**, which for a cluster of thousands of machines over 31 days
would produce many more than 58,783 rows. The small total row count of
this Kaggle export (405,894 rows for an entire 8-cluster,
96,174-machine, 31-day trace) is itself evidence that **this file is a
small subsample or event-filtered extract of a much larger original
trace**, not a complete periodic-usage stream. This is an
**INFERRED**, not documented, conclusion — the Kaggle page states
nothing about sampling methodology — and is recorded here as a
dataset limitation, not asserted as certain.

A near-constant-zero workload signal would make the forecasting
problem trivial and the thermal/RL simulation nearly static, defeating
the project's purpose. Per the master plan's explicit instruction
("do not impose 5-minute bins blindly... determine the actual temporal
characteristics"), this was treated as license to revise the
discretization rather than silently proceed with a degenerate signal.

### 3.2 Corrected design (implemented, used downstream)

Two changes, both evidence-driven and both documented as trade-offs
(not free improvements):

1. **Pool all 8 Borg clusters** as the workload source, instead of one
   cluster. This raises the number of contributing machines to 96,174
   and roughly triples the record density per unit time. **Trade-off,
   stated plainly:** this weakens the "one physical facility"
   narrative for the 3 zones even further than the single-cluster
   design already did — the simulated "datacenter" now draws its
   workload from a demand sample spanning 8 real, independent, almost
   certainly non-co-located Google cells. This is recorded as a
   **simulation assumption (category D)**, made for numerical
   necessity, not as a claim that these machines share a building.
2. **Widen the bin from 5 to 15 minutes.** Every usage-interval
   duration is ≤ 300s (verified — see §4 below), so a 15-minute bin
   still cannot be split across more than one interval-boundary issue
   in a way that causes double counting; it simply aggregates more
   (still real, still non-fabricated) records per cell. The 30-minute
   forecast objective from the master plan is preserved by setting the
   GRU's forecast horizon to **2 steps of 15 minutes** rather than 6
   steps of 5 minutes (`configs/config.yaml: forecasting.horizon_steps: 2`).

**Result after the correction:** 2,977 fifteen-minute bins × 3 zones =
8,931 (zone, bin) cells; **30.3%** are still exactly zero (idle), the
rest have a real, parsed, non-fabricated mean CPU utilization value.
The residual 30% zero-inflation is disclosed as a genuine
characteristic of this dataset (see §3.1), not hidden or interpolated
away.

### 3.3 Zone assignment

`zone = (machine_id * 2654435761) mod 2^32) mod n_zones` — a fixed,
seed-free integer hash, deterministic and reproducible without relying
on Python's per-process salted `hash()`. Verified: each machine maps to
exactly one zone (no machine appears in two zones), zone sizes are
balanced (32,145 / 31,975 / 32,054 machines), and the sum of per-zone
`average_usage_cpus` over all bins equals the un-split cluster-wide
total to floating-point precision (`3022.1837` both ways) — i.e. **no
workload is duplicated or lost by the zone split.**

### 3.4 Aggregation operator: mean, not sum

For each (zone, 15-minute bin), the workload value is the **mean** of
`average_usage_cpus` across all usage records whose zone-assigned
machine reported activity in that bin. Mean (not sum) was chosen
because `average_usage.cpus` is a normalized-utilization-like quantity
(values ≪ 1, consistent with Borg's convention of normalizing to
fraction-of-largest-machine — inferred, §11 of the audit); summing it
across a time-varying, arbitrary number of reporting instances would
produce a signal whose scale depends on how many instances happened to
report in that bin rather than on load intensity, which does not map
cleanly onto the downstream IT-power model's utilization input
`U ∈ [0,1]`. Empty bins are filled with `U = 0`, treated as genuinely
idle rather than "unknown," per the module's docstring and the
master plan's prohibition on fabricating workload during gaps — we do
not interpolate or forward-fill; an unmeasured bin is recorded as zero
load, which is the most conservative, least-fabricated choice
available.

**Information lost:** per-machine/per-instance granularity within a
zone; sub-bin timing of an interval (an interval is credited entirely
to the bin containing its `start_time`, not fractionally split, though
this cannot cause more than one bin's worth of misattribution since
durations never exceed the bin width — see §4). **Information
retained:** `n_active_instances` and `n_distinct_machines` per
(zone, bin) are stored alongside `cpu_util`/`mem_util` specifically so
that later analysis can distinguish "genuinely near-zero load" from
"very few instances reported."

## 4. Interval-duration check (binning validity)

`end_time - start_time` was checked across the full population:
minimum 1,000,000 µs (1s), maximum exactly 300,000,000 µs (300s = 5
minutes), for every one of 405,894 rows. Because no interval is ever
longer than the original 5-minute native granularity, and the chosen
bin (15 minutes) is 3x that, no interval can span more than one
15-minute bin's worth of ambiguity — `aggregate_workload.py` asserts
this bound (`max_duration <= bin_us`) at runtime rather than assuming
it silently.

## 5. Chronological split

`src/data/split_workload.py` splits the 2,977-bin timeline
chronologically (not randomly) into 60% train / 20% validation / 20%
test **by bin index**, so the test set is the final ~20% of the trace
in time (never shuffled in). No scaler, normalizer, or model parameter
is fit on validation or test data anywhere downstream; this is
enforced structurally (the forecasting scaler is fit only on the
`train` split — see `docs/forecasting_design.md`).

## 6. Known limitations carried forward

- The residual 30% zero-inflation in the workload signal (§3.2) is a
  property of this specific Kaggle export and is not corrected or
  hidden; forecasting and RL results must be interpreted with this in
  mind (a model that predicts "mostly idle, occasional bursts" is
  learning a real property of this data, not failing).
- Pooling all 8 clusters (§3.2, point 1) is a stronger-than-ideal
  simulation assumption; it is necessary given this dataset's sparsity
  and is disclosed as such throughout the documentation, per
  `docs/assumptions_and_limitations.md`.
- The zone mapping is a logical/simulation abstraction with no claim
  to physical rack/room location, consistent with `docs/dataset_audit.md`.
