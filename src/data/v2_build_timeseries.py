"""V2 Phase 6: build the per-cell workload+power time series from the
official Google 2019 BigQuery extracts (data/raw/official_google_2019/,
produced by scripts/bigquery/).

Each Borg cell IS a zone in V2 (see configs/data_config.yaml for the
rationale) -- unlike V1, which hashed machine_id into 3 synthetic zones
because the Kaggle export carried no real machine partition.

Join: workload (per cell, per 5-minute bucket: sum_cpu, sum_mem,
n_records[, n_machines]) INNER JOINed with power (per cell, per
5-minute bucket: mean measured_power_util across that cell's PDUs,
excluding rows flagged bad_measurement_data) on (cell, bucket_5min).
This is the cell-level join established as scientifically defensible in
docs/official_data_alignment_audit.md -- no machine-to-PDU mapping is
used or assumed anywhere in this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "data_config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _active_cells(cfg: dict) -> list[str]:
    return cfg["cells"][cfg["active_cell_set"]]


def load_workload(cfg: dict) -> pd.DataFrame:
    raw_dir = REPO_ROOT / cfg["paths"]["raw_dir"]
    cells = _active_cells(cfg)
    frames = []
    for cell in cells:
        path = raw_dir / f"clusterdata_workload_5min_cell_{cell}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing workload extract for cell {cell!r}: {path}. "
                f"Run scripts/bigquery/run_clusterdata_extraction.sh."
            )
        df = pd.read_csv(path)
        first_line = path.read_text().splitlines()[0]
        if "BigQuery error" in first_line or df.empty:
            raise ValueError(f"{path} looks like a failed query result, not data.")
        frames.append(df)
    workload = pd.concat(frames, ignore_index=True)
    workload["cell"] = workload["cell"].astype(str)
    return workload


def load_power(cfg: dict) -> pd.DataFrame:
    path = REPO_ROOT / cfg["paths"]["raw_dir"] / cfg["paths"]["powerdata_file"]
    df = pd.read_csv(path)
    bucket_s = cfg["resolution"]["native_bucket_seconds"]
    df["bucket_5min"] = (df["time"] // (bucket_s * 1_000_000)).astype("int64")

    # Exclude rows Google itself flags as bad measurements -- do not
    # silently include or "repair" them.
    n_before = len(df)
    df = df[~df["bad_measurement_data"]].copy()
    n_dropped = n_before - len(df)

    agg = (
        df.groupby(["cell", "bucket_5min"])
        .agg(
            mean_measured_power_util=("measured_power_util", "mean"),
            n_pdus_reporting=("pdu", "nunique"),
        )
        .reset_index()
    )
    print(f"Power: dropped {n_dropped}/{n_before} rows flagged bad_measurement_data "
          f"({100*n_dropped/n_before:.2f}%)")
    return agg


def build(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    cells = _active_cells(cfg)

    workload = load_workload(cfg)
    power = load_power(cfg)
    power = power[power["cell"].isin(cells)]

    merged = workload.merge(power, on=["cell", "bucket_5min"], how="inner")

    for cell in cells:
        w_n = (workload["cell"] == cell).sum()
        m_n = (merged["cell"] == cell).sum()
        coverage = 100 * m_n / w_n if w_n else 0.0
        print(f"cell {cell}: {w_n} workload buckets, {m_n} matched to power "
              f"({coverage:.1f}% coverage)")
        if coverage < 95.0:
            raise AssertionError(
                f"cell {cell}: only {coverage:.1f}% of workload buckets found a "
                f"matching power bucket -- investigate before proceeding "
                f"(expected near-100% given both sources cover the same "
                f"~31-day trace at 5-minute resolution)."
            )

    bucket_s = cfg["resolution"]["native_bucket_seconds"]
    merged["timestamp_s"] = merged["bucket_5min"] * bucket_s
    merged = merged.sort_values(["cell", "bucket_5min"]).reset_index(drop=True)

    # Aliases so V1's zone-indexed forecasting/environment code
    # (src/forecasting/dataset.py, src/environment/cooling_core.py) can
    # be reused verbatim for V2: "zone" = deterministic integer code for
    # cell (alphabetical order, stable across pilot/final cell sets),
    # "bin_id" = bucket_5min under V1's column name.
    cell_order = sorted(cells)
    cell_to_zone = {c: i for i, c in enumerate(cell_order)}
    merged["zone"] = merged["cell"].map(cell_to_zone).astype("int16")
    merged["bin_id"] = merged["bucket_5min"]

    # Leakage/ordering sanity: no duplicate (cell, bucket) rows.
    dup = merged.duplicated(subset=["cell", "bucket_5min"]).sum()
    assert dup == 0, f"{dup} duplicate (cell, bucket_5min) rows found after join"

    out_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"workload_power_5min_{cfg['active_cell_set']}.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"Wrote {len(merged)} rows -> {out_path}")
    print(merged[["sum_cpu", "sum_mem", "mean_measured_power_util"]].describe())
    return merged


if __name__ == "__main__":
    build()
