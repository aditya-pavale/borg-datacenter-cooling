"""Phase 3: chronological train/validation/test split of the workload
time series. Never shuffles; the test set is strictly the final segment
in time. See docs/data_preprocessing.md S5.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TS_PATH = REPO_ROOT / "data" / "processed" / "workload_timeseries.parquet"
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
OUT_DIR = REPO_ROOT / "data" / "processed"


def split(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or yaml.safe_load(CONFIG_PATH.read_text())
    ts = pd.read_parquet(TS_PATH)

    bin_ids = sorted(ts["bin_id"].unique())
    n_bins = len(bin_ids)
    train_frac = cfg["split"]["train_frac"]
    val_frac = cfg["split"]["val_frac"]

    n_train = int(round(n_bins * train_frac))
    n_val = int(round(n_bins * val_frac))

    train_bins = set(bin_ids[:n_train])
    val_bins = set(bin_ids[n_train:n_train + n_val])
    test_bins = set(bin_ids[n_train + n_val:])

    assert train_bins.isdisjoint(val_bins)
    assert train_bins.isdisjoint(test_bins)
    assert val_bins.isdisjoint(test_bins)
    assert train_bins | val_bins | test_bins == set(bin_ids)

    # Chronology check: every train bin_id < every val bin_id < every test bin_id.
    assert max(train_bins) < min(val_bins), "train/val split is not chronological"
    assert max(val_bins) < min(test_bins), "val/test split is not chronological"

    splits = {}
    for name, bins in [("train", train_bins), ("val", val_bins), ("test", test_bins)]:
        part = ts[ts["bin_id"].isin(bins)].sort_values(["bin_id", "zone"]).reset_index(drop=True)
        out_path = OUT_DIR / f"{name}.parquet"
        part.to_parquet(out_path, index=False)
        splits[name] = part
        print(f"{name}: {len(bins)} bins ({100*len(bins)/n_bins:.1f}%), "
              f"{len(part)} rows, bin_id range [{min(bins)}, {max(bins)}] -> {out_path}")

    return splits


if __name__ == "__main__":
    split()
