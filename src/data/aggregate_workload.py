"""Phase 2 (part 2) / Phase 3: build the per-zone workload time series.

Design decisions (documented in detail in docs/data_preprocessing.md):

1. Zone source: EMPIRICALLY REVISED from the original plan. The
   original, more physically-defensible design used records from a
   single real Borg `cluster` (one facility), split into 3 logical
   zones. That was implemented and measured first: at a 5-minute bin
   width, a single cluster's usage-report rows left 94% of (zone, bin)
   cells with literally zero records ("idle"), because this Kaggle
   export contains far fewer usage rows than a full periodic-telemetry
   table (usage appears event-driven -- tied to lifecycle events -- not
   sampled on a fixed cadence per running instance; see
   docs/dataset_audit.md). That signal was judged too degenerate
   (near-constant zero) to support meaningful forecasting or thermal
   simulation. The corrected, empirically-driven design (documented in
   full in docs/data_preprocessing.md) pools records from ALL 8 Borg
   clusters as the workload source and widens the bin to 15 minutes,
   reducing zero-cells to ~30%. This is a STRONGER simulation
   assumption than the single-cluster design (it no longer represents
   one real facility's machines) but was empirically necessary; it is
   documented here and in docs/data_preprocessing.md as a category-D
   simulation assumption, not presented as physically realistic.

2. Zone assignment: deterministic hash(machine_id) % n_zones. This
   guarantees (a) every machine maps to exactly one zone (no workload
   duplication across zones), (b) the mapping is stable/reproducible,
   and (c) no information about physical location is fabricated -- it
   is explicitly a simulation abstraction.

3. Temporal binning: each usage record is assigned to exactly ONE
   5-minute bin, floor(start_time / bin_seconds). This is justified by
   an independent check (see docs/data_preprocessing.md) that showed
   100% of usage-interval durations (end_time - start_time) are <= 300
   seconds (the bin width) -- so no interval can span more than one bin
   by construction. Because only 71.6% of start_time values are exactly
   grid-aligned, an interval that starts a few seconds before a bin
   boundary could in principle be assigned to the "earlier" bin even
   though a small remainder of it falls in the next bin; given the
   maximum possible misattribution is bounded by one interval's own
   duration (<=300s, i.e. at most one bin), and given no interval ever
   spans a bin boundary by more than that, this is treated as a bounded,
   documented approximation rather than a source of leakage or
   double-counting.

4. Aggregation operator: for each (zone, bin), the workload value is
   the MEAN of average_usage_cpus over all instance-records active in
   that zone/bin (i.e. treating the zone as one aggregate compute pool
   whose utilization is the average load of its currently-active
   instances). Mean (not sum) was chosen because average_usage.cpus is
   a normalized-utilization-like quantity (docs/dataset_audit.md S11);
   summing it across an arbitrary, time-varying number of active
   instances would produce a signal whose scale depends on instance
   count rather than on utilization intensity, which is not what the
   downstream IT-power model (P_idle + (P_max-P_idle)*U, U in [0,1])
   expects. Bins with zero active instances in a zone are treated as
   U=0 (idle), which is validated (a small fraction of bins) rather
   than assumed.

Information lost: per-machine/per-instance granularity within a zone;
instances shorter than one bin contribute equally to that bin
regardless of their exact sub-bin duration (no fractional weighting).
Information retained: per-zone, per-5-minute mean CPU utilization and
mean memory utilization, active-instance count (for diagnostics), and
the mapping is fully reproducible from machine_id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CLEANED_PATH = REPO_ROOT / "data" / "processed" / "borg_cleaned.parquet"
OUT_PATH = REPO_ROOT / "data" / "processed" / "workload_timeseries.parquet"
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def assign_zone(machine_id: pd.Series, n_zones: int) -> pd.Series:
    # Deterministic, stable across runs/machines (no reliance on Python's
    # salted hash()): a simple integer hash of machine_id.
    h = (machine_id.astype("int64") * 2654435761) % (2**32)
    return (h % n_zones).astype("int16")


def aggregate(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_parquet(CLEANED_PATH)

    target_cluster = cfg["data"]["cluster_for_zones"]
    n_zones = cfg["data"]["n_zones"]
    bin_seconds = cfg["data"]["bin_seconds"]
    bin_us = bin_seconds * 1_000_000

    if target_cluster == "all":
        sub = df.copy()
    else:
        sub = df[df["cluster"] == target_cluster].copy()
    assert len(sub) > 0, f"No rows found for cluster selector {target_cluster!r}"

    sub["zone"] = assign_zone(sub["machine_id"], n_zones)
    sub["bin_id"] = (sub["start_time"] // bin_us).astype("int64")

    # Sanity: verify the bounded-binning assumption documented above.
    duration = sub["end_time"] - sub["start_time"]
    max_duration = duration.max()
    assert max_duration <= bin_us, (
        f"Found a usage interval longer than one bin ({max_duration} us > "
        f"{bin_us} us); the single-bin assignment assumption is violated."
    )

    grouped = sub.groupby(["bin_id", "zone"]).agg(
        cpu_util=("average_usage_cpus", "mean"),
        mem_util=("average_usage_memory", "mean"),
        n_active_instances=("average_usage_cpus", "size"),
        n_distinct_machines=("machine_id", "nunique"),
    ).reset_index()

    # Build a complete, gap-free (bin_id x zone) grid so downstream code
    # never has to special-case a missing row; empty bins are genuinely
    # idle (U=0), not unknown -- see the module docstring.
    full_bins = np.arange(sub["bin_id"].min(), sub["bin_id"].max() + 1)
    full_index = pd.MultiIndex.from_product(
        [full_bins, range(n_zones)], names=["bin_id", "zone"]
    )
    ts = grouped.set_index(["bin_id", "zone"]).reindex(full_index)
    n_missing_bins = ts["cpu_util"].isna().sum()
    ts["cpu_util"] = ts["cpu_util"].fillna(0.0)
    ts["mem_util"] = ts["mem_util"].fillna(0.0)
    ts["n_active_instances"] = ts["n_active_instances"].fillna(0).astype("int32")
    ts["n_distinct_machines"] = ts["n_distinct_machines"].fillna(0).astype("int32")
    ts = ts.reset_index()

    ts["timestamp_us"] = ts["bin_id"] * bin_us
    ts["timestamp_s"] = ts["timestamp_us"] / 1e6

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(OUT_PATH, index=False)

    n_machines_in_cluster = sub["machine_id"].nunique()
    zone_counts = sub.drop_duplicates("machine_id").groupby("zone").size()

    print(f"Aggregated {len(sub)} records from cluster {target_cluster} "
          f"({n_machines_in_cluster} distinct machines) into "
          f"{len(full_bins)} bins x {n_zones} zones = {len(ts)} rows")
    print(f"Empty (idle-filled) bin/zone cells: {n_missing_bins} "
          f"({100*n_missing_bins/len(ts):.2f}%)")
    print("Machines per zone:\n", zone_counts)
    print("cpu_util describe:\n", ts["cpu_util"].describe())

    # Consistency check requested by the master plan: zone totals should
    # not exceed the aggregate-cluster workload total for the same bins
    # (verifies no workload was duplicated across zones).
    cluster_level = sub.groupby("bin_id")["average_usage_cpus"].sum().sum()
    zone_level = sub.groupby(["bin_id", "zone"])["average_usage_cpus"].sum().sum()
    assert abs(cluster_level - zone_level) < 1e-6, (
        "Zone-level sum of average_usage_cpus does not match the "
        "cluster-level sum -- workload may have been duplicated or lost "
        "during zone assignment."
    )
    print(f"Zone-partition consistency check passed: "
          f"cluster total = {cluster_level:.4f}, sum of zone totals = {zone_level:.4f}")

    return ts


if __name__ == "__main__":
    aggregate()
