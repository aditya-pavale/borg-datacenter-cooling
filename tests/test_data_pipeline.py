"""Tests for Phase 1-3: raw integrity, cleaning, aggregation, splitting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.verify_raw_integrity import assert_raw_data_unchanged  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())


@pytest.fixture(scope="module")
def cleaned():
    path = REPO_ROOT / "data" / "processed" / "borg_cleaned.parquet"
    assert path.exists(), "run src/data/clean_borg.py first"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def ts(cfg):
    path = REPO_ROOT / "data" / "processed" / "workload_timeseries.parquet"
    assert path.exists(), "run src/data/aggregate_workload.py first"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def splits():
    d = {}
    for name in ["train", "val", "test"]:
        p = REPO_ROOT / "data" / "processed" / f"{name}.parquet"
        assert p.exists(), f"run src/data/split_workload.py first (missing {name})"
        d[name] = pd.read_parquet(p)
    return d


# ---------- raw data integrity ----------

def test_raw_data_unchanged():
    assert_raw_data_unchanged()  # raises on mismatch


# ---------- cleaned data ----------

def test_cleaned_schema(cleaned):
    expected_cols = {
        "collection_id", "instance_index", "machine_id", "start_time", "end_time",
        "sample_rate", "cluster", "event", "failed",
        "average_usage_cpus", "average_usage_memory",
        "resource_request_cpus", "resource_request_memory",
    }
    assert expected_cols.issubset(set(cleaned.columns))


def test_cleaned_row_count_matches_audit(cleaned):
    assert len(cleaned) == 405_894


def test_cleaned_no_nan_in_primary_workload_signal(cleaned):
    assert cleaned["average_usage_cpus"].isna().sum() == 0


def test_cleaned_time_ordering_within_instance(cleaned):
    assert (cleaned["end_time"] >= cleaned["start_time"]).all()


def test_cleaned_duplicate_check(cleaned):
    key = ["collection_id", "instance_index", "machine_id", "start_time", "end_time"]
    assert cleaned.duplicated(subset=key).sum() == 0


def test_cleaned_value_ranges(cleaned):
    assert cleaned["average_usage_cpus"].min() >= 0
    assert cleaned["cluster"].between(1, 8).all()
    assert cleaned["failed"].isin([0, 1]).all()


# ---------- aggregation ----------

def test_workload_timeseries_schema(ts):
    expected_cols = {"bin_id", "zone", "cpu_util", "mem_util",
                      "n_active_instances", "n_distinct_machines", "timestamp_s"}
    assert expected_cols.issubset(set(ts.columns))


def test_workload_timeseries_complete_grid(ts, cfg):
    n_zones = cfg["data"]["n_zones"]
    n_bins = ts["bin_id"].nunique()
    assert len(ts) == n_bins * n_zones, "bin x zone grid must be complete (no gaps)"


def test_workload_no_negative_or_nan(ts):
    assert ts["cpu_util"].isna().sum() == 0
    assert (ts["cpu_util"] >= 0).all()
    assert (ts["mem_util"] >= 0).all()


def test_zone_partition_no_double_counting(cleaned, cfg):
    """Sum of per-zone average_usage_cpus over all rows must equal the
    un-partitioned total -- verifies zone assignment neither drops nor
    duplicates any workload."""
    from data.aggregate_workload import assign_zone

    n_zones = cfg["data"]["n_zones"]
    df = cleaned.copy()
    df["zone"] = assign_zone(df["machine_id"], n_zones)

    total = df["average_usage_cpus"].sum()
    by_zone = df.groupby("zone")["average_usage_cpus"].sum().sum()
    assert abs(total - by_zone) < 1e-6

    # every machine maps to exactly one zone
    machine_zone_counts = df.groupby("machine_id")["zone"].nunique()
    assert (machine_zone_counts == 1).all()


def test_bin_width_bounds_interval_duration(cleaned, cfg):
    bin_us = cfg["data"]["bin_seconds"] * 1_000_000
    duration = cleaned["end_time"] - cleaned["start_time"]
    assert duration.max() <= bin_us


# ---------- split ----------

def test_split_disjoint_and_complete(splits, ts):
    train_bins = set(splits["train"]["bin_id"])
    val_bins = set(splits["val"]["bin_id"])
    test_bins = set(splits["test"]["bin_id"])
    assert train_bins.isdisjoint(val_bins)
    assert train_bins.isdisjoint(test_bins)
    assert val_bins.isdisjoint(test_bins)
    assert train_bins | val_bins | test_bins == set(ts["bin_id"].unique())


def test_split_is_chronological(splits):
    assert splits["train"]["bin_id"].max() < splits["val"]["bin_id"].min()
    assert splits["val"]["bin_id"].max() < splits["test"]["bin_id"].min()


def test_split_fractions_approximately_60_20_20(splits, ts):
    n_total = ts["bin_id"].nunique()
    for name, expected_frac in [("train", 0.6), ("val", 0.2), ("test", 0.2)]:
        n = splits[name]["bin_id"].nunique()
        assert abs(n / n_total - expected_frac) < 0.02
