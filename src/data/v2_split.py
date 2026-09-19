"""V2 Phase 6 (part 2): chronological train/val/test split of the
combined workload+power time series. Same discipline as V1's
src/data/split_workload.py: never shuffles, one shared bucket-index
cutoff applied identically across every cell (all cells share the same
May-2019 trace-relative time axis), test is strictly the final segment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "data_config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def split(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    in_path = (
        REPO_ROOT / cfg["paths"]["processed_dir"]
        / f"workload_power_5min_{cfg['active_cell_set']}.parquet"
    )
    ts = pd.read_parquet(in_path)

    buckets = sorted(ts["bucket_5min"].unique())
    n = len(buckets)
    n_train = int(round(n * cfg["split"]["train_frac"]))
    n_val = int(round(n * cfg["split"]["val_frac"]))

    train_b = set(buckets[:n_train])
    val_b = set(buckets[n_train:n_train + n_val])
    test_b = set(buckets[n_train + n_val:])

    assert train_b.isdisjoint(val_b) and train_b.isdisjoint(test_b) and val_b.isdisjoint(test_b)
    assert train_b | val_b | test_b == set(buckets)
    assert max(train_b) < min(val_b), "train/val split is not chronological"
    assert max(val_b) < min(test_b), "val/test split is not chronological"

    out_dir = REPO_ROOT / cfg["paths"]["processed_dir"]
    splits = {}
    for name, bset in [("train", train_b), ("val", val_b), ("test", test_b)]:
        part = ts[ts["bucket_5min"].isin(bset)].sort_values(["cell", "bucket_5min"]).reset_index(drop=True)
        out_path = out_dir / f"{name}_{cfg['active_cell_set']}.parquet"
        part.to_parquet(out_path, index=False)
        splits[name] = part
        print(f"{name}: {len(bset)} buckets ({100*len(bset)/n:.1f}%), "
              f"{len(part)} rows across {part['cell'].nunique()} cells -> {out_path}")

    return splits


if __name__ == "__main__":
    split()
