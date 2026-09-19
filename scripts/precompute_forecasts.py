"""Precompute causal GRU forecasts for every bin in each split, using the
FROZEN model trained in Phase 4. For bin t, forecast[t] = predicted
values for [t+1, ..., t+horizon], produced from the lookback window
[t-lookback+1, ..., t] -- i.e. only data at or before t. Bins before
`lookback_steps` (per split) get an all-zero forecast (no false future
data injected) and are marked unusable in `valid_from`; the RL
environment only ever samples episode start indices >= lookback_steps
to avoid exposing those degenerate forecasts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecasting.gru_model import GRUForecaster  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "forecasting"


def main():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())
    fc_cfg = cfg["forecasting"]
    lookback = fc_cfg["lookback_steps"]
    horizon = fc_cfg["horizon_steps"]
    n_zones = cfg["data"]["n_zones"]

    scaler = json.loads((MODELS_DIR / "scaler.json").read_text())
    train_min, train_max = scaler["min"], scaler["max"]
    scale = max(train_max - train_min, 1e-8)

    model = GRUForecaster(hidden_size=fc_cfg["gru_hidden_size"],
                           num_layers=fc_cfg["gru_num_layers"],
                           horizon=horizon)
    model.load_state_dict(torch.load(MODELS_DIR / "gru_forecaster.pt"))
    model.eval()

    for split in ["train", "val", "test"]:
        df = pd.read_parquet(DATA_DIR / f"{split}.parquet")
        pivot = df.pivot(index="bin_id", columns="zone", values="cpu_util").sort_index()
        n_bins = len(pivot)
        workload = pivot.to_numpy(dtype=np.float32)  # (n_bins, n_zones)

        forecasts = np.zeros((n_bins, n_zones, horizon), dtype=np.float32)
        with torch.no_grad():
            for z in range(n_zones):
                series = workload[:, z]
                for t in range(lookback - 1, n_bins):
                    window = series[t - lookback + 1: t + 1]
                    x = torch.tensor((window - train_min) / scale, dtype=torch.float32).unsqueeze(0)
                    pred_n = model(x).numpy()[0]
                    forecasts[t, z, :] = pred_n * scale + train_min

        out_path = DATA_DIR / f"forecast_{split}.npy"
        np.save(out_path, forecasts)
        print(f"{split}: forecasts shape {forecasts.shape}, valid from bin index {lookback-1}, saved to {out_path}")


if __name__ == "__main__":
    main()
