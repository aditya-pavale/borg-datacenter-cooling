"""V2 Phase 7 (pilot): precompute causal GRU forecasts for every bin in
each split, using the frozen model trained by v2_train_forecaster.py.
Same causality/degenerate-bin discipline as V1's precompute_forecasts.py
(see its docstring) -- forecast[t] only ever uses data at or before t.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecasting.gru_model import GRUForecaster  # noqa: E402
from environment.v2_cooling_core import load_v2_config  # noqa: E402
from evaluation.v2_common import models_dir  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "processed" / "v2"


def main():
    cfg = load_v2_config()
    fc_cfg = cfg["forecasting"]
    lookback = fc_cfg["lookback_steps"]
    horizon = fc_cfg["horizon_steps"]
    value_col = fc_cfg["value_col"]
    cell_set = cfg["data"]["active_cell_set"]
    m_dir = models_dir(cfg, "forecasting")

    scaler = json.loads((m_dir / "scaler.json").read_text())
    train_min, train_max = scaler["min"], scaler["max"]
    scale = max(train_max - train_min, 1e-8)

    model = GRUForecaster(hidden_size=fc_cfg["gru_hidden_size"],
                           num_layers=fc_cfg["gru_num_layers"],
                           horizon=horizon)
    model.load_state_dict(torch.load(m_dir / "gru_forecaster.pt"))
    model.eval()

    for split in ["train", "val", "test"]:
        df = pd.read_parquet(DATA_DIR / f"{split}_{cell_set}.parquet")
        pivot = df.pivot(index="bin_id", columns="zone", values=value_col).sort_index()
        n_bins = len(pivot)
        n_zones = pivot.shape[1]  # derived, not config-trusted
        workload = pivot.to_numpy(dtype=np.float32)

        forecasts = np.zeros((n_bins, n_zones, horizon), dtype=np.float32)
        with torch.no_grad():
            for z in range(n_zones):
                series = workload[:, z]
                for t in range(lookback - 1, n_bins):
                    window = series[t - lookback + 1: t + 1]
                    x = torch.tensor((window - train_min) / scale, dtype=torch.float32).unsqueeze(0)
                    pred_n = model(x).numpy()[0]
                    forecasts[t, z, :] = pred_n * scale + train_min

        out_path = DATA_DIR / f"forecast_{split}_{cell_set}.npy"
        np.save(out_path, forecasts)
        print(f"{split}: forecasts shape {forecasts.shape}, valid from bin index {lookback-1} -> {out_path}")


if __name__ == "__main__":
    main()
