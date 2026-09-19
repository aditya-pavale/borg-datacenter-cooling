"""Phase 2 (part 1): clean the raw Borg CSV into a compact, typed table.

Reads data/raw/borg_traces_data.csv (read-only, integrity-checked),
parses the nested usage/request dict-string columns, drops columns
identified as redundant or uninformative in docs/dataset_audit.md, and
writes a typed parquet file to data/processed/borg_cleaned.parquet.

Columns kept and why (see docs/dataset_audit.md for evidence):
  - start_time, end_time: usage-measurement interval bounds (microseconds,
    inferred unit). Used for binning (see aggregate_workload.py).
  - cluster: real, independent Borg cell id (8 values). Used to select
    the workload source for the 3-zone simulation.
  - machine_id: used for the deterministic zone-assignment hash.
  - collection_id, instance_index: retained for traceability/debugging,
    not used directly in aggregation (the stable task-instance grain is
    (collection_id, instance_index, machine_id), not instance_index alone
    -- see docs/dataset_audit.md S10/S20).
  - average_usage_cpus, average_usage_memory: the primary/secondary
    workload signal, parsed out of the average_usage dict-string.
  - resource_request_cpus, resource_request_memory: requested/allocated
    ceiling, kept as a secondary reference signal (NOT used as the
    primary workload signal -- it measures demand ceiling, not actual
    usage).
  - sample_rate: fraction of the interval actually measured; retained
    for optional quality-weighting in aggregation.
  - event, failed: retained for diagnostics (e.g. excluding FAIL/LOST
    intervals from the workload signal is investigated, not assumed).

Columns dropped and why:
  - Unnamed: 0 -- pandas row-index artifact, not data.
  - instance_events_type -- byte-identical duplicate of
    collections_events_type; event (string) is kept as the single
    source of truth for event semantics instead of either numeric code.
  - collections_events_type -- same as above.
  - time -- inconsistent relationship to start_time/end_time, unresolved
    semantics (docs/dataset_audit.md S6); not used anywhere downstream.
  - random_sample_usage -- memory sub-field is always None; cpus
    sub-field is a lower-quality duplicate of average_usage.cpus.
  - constraint, start_after_collection_ids, cpu_usage_distribution,
    tail_cpu_usage_distribution -- scheduling/percentile metadata not
    needed for a workload time series; also the numpy-repr parsing
    quirk (docs/dataset_audit.md S8) makes them costly to parse for no
    benefit here.
  - user, collection_name, collection_logical_name -- hashed identifiers
    not needed for workload aggregation.
  - scheduling_class, collection_type, priority, alloc_collection_id,
    vertical_scaling, scheduler -- scheduling metadata not needed for
    the workload signal.
  - assigned_memory, page_cache_memory, cycles_per_instruction,
    memory_accesses_per_instruction -- secondary usage metrics not
    needed for the CPU-driven workload signal used in this project.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.verify_raw_integrity import assert_raw_data_unchanged, RAW_DIR  # noqa: E402

RAW_CSV = RAW_DIR / "borg_traces_data.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "borg_cleaned.parquet"

_CPUS_RE = re.compile(r"'cpus':\s*([-\d.eE]+)")
_MEM_RE = re.compile(r"'memory':\s*([-\d.eEnN]+)")


def _extract_field(series: pd.Series, pattern: re.Pattern) -> pd.Series:
    def parse(s):
        if not isinstance(s, str):
            return float("nan")
        m = pattern.search(s)
        if m is None:
            return float("nan")
        val = m.group(1)
        return float("nan") if val in ("None", "nan") else float(val)

    return series.map(parse)


def clean() -> pd.DataFrame:
    assert_raw_data_unchanged()

    usecols = [
        "start_time",
        "end_time",
        "cluster",
        "machine_id",
        "collection_id",
        "instance_index",
        "average_usage",
        "resource_request",
        "sample_rate",
        "event",
        "failed",
    ]
    df = pd.read_csv(RAW_CSV, usecols=usecols, low_memory=False)

    df["average_usage_cpus"] = _extract_field(df["average_usage"], _CPUS_RE)
    df["average_usage_memory"] = _extract_field(df["average_usage"], _MEM_RE)
    df["resource_request_cpus"] = _extract_field(df["resource_request"], _CPUS_RE)
    df["resource_request_memory"] = _extract_field(df["resource_request"], _MEM_RE)

    df = df.drop(columns=["average_usage", "resource_request"])

    # Dtype tightening for memory efficiency (safe ranges verified in the audit).
    df["cluster"] = df["cluster"].astype("int16")
    df["failed"] = df["failed"].astype("int8")
    df["event"] = df["event"].astype("category")
    df["start_time"] = df["start_time"].astype("int64")
    df["end_time"] = df["end_time"].astype("int64")
    df["machine_id"] = df["machine_id"].astype("int64")
    df["collection_id"] = df["collection_id"].astype("int64")
    df["instance_index"] = df["instance_index"].astype("int64")

    n_before = len(df)
    n_usage_nan = df["average_usage_cpus"].isna().sum()
    assert n_usage_nan == 0, (
        f"{n_usage_nan} rows failed to parse average_usage.cpus; "
        "this column had 0 parse errors in the Phase-1 audit, so any "
        "failure here indicates a code bug, not a data issue."
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows (from {n_before} raw rows) to {OUT_PATH}")
    print(df.dtypes)
    return df


if __name__ == "__main__":
    clean()
