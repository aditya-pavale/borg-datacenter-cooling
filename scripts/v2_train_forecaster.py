"""V2 Phase 7 (pilot): GRU workload forecaster on official Google
ClusterData2019, cells a-d. Reuses V1's windowing (src/forecasting/dataset.py),
model (src/forecasting/gru_model.py), baselines, and metric helpers
(src/forecasting/train_forecaster.py) unchanged -- only the data source,
value column, and n_zones differ, per configs/v2_config.yaml.

Causality, train-only normalization, and test-set discipline are
identical to V1's train_forecaster.py (see its docstring); this pilot
run treats cells a-d as PILOT DATA (results/pilot/README.md) --
forecasting will be re-run and re-selected on the full 8-cell dataset
once cells e-h are extracted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecasting.dataset import make_windows, per_zone_series  # noqa: E402
from forecasting.baselines import persistence_forecast, moving_average_forecast  # noqa: E402
from forecasting.gru_model import GRUForecaster  # noqa: E402
from forecasting.train_forecaster import mae, rmse, r2, segment_metrics  # noqa: E402
from environment.v2_cooling_core import load_v2_config  # noqa: E402
from evaluation.v2_common import results_dir, models_dir, disclaimer  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "processed" / "v2"


def build_all_windows(df: pd.DataFrame, n_zones: int, lookback: int, horizon: int, value_col: str):
    Xs, ys, zones = [], [], []
    for z in range(n_zones):
        series = per_zone_series(df, z, value_col=value_col)
        X, y = make_windows(series, lookback, horizon)
        Xs.append(X)
        ys.append(y)
        zones.append(np.full(len(X), z))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(zones)


def main():
    cfg = load_v2_config()
    fc_cfg = cfg["forecasting"]
    lookback = fc_cfg["lookback_steps"]
    horizon = fc_cfg["horizon_steps"]
    value_col = fc_cfg["value_col"]
    cell_set = cfg["data"]["active_cell_set"]

    train = pd.read_parquet(DATA_DIR / f"train_{cell_set}.parquet")
    val = pd.read_parquet(DATA_DIR / f"val_{cell_set}.parquet")
    test = pd.read_parquet(DATA_DIR / f"test_{cell_set}.parquet")
    n_zones = train["zone"].nunique()  # derived, not config-trusted (see v2_cooling_core.load_v2_config)
    assert n_zones == len(cfg["data"]["active_cells"]), (
        f"train_{cell_set}.parquet has {n_zones} zones but "
        f"active_cell_set={cell_set!r} expects {len(cfg['data']['active_cells'])}"
    )

    X_train, y_train, _ = build_all_windows(train, n_zones, lookback, horizon, value_col)
    X_val, y_val, _ = build_all_windows(val, n_zones, lookback, horizon, value_col)
    X_test, y_test, z_test = build_all_windows(test, n_zones, lookback, horizon, value_col)

    train_min = float(X_train.min())
    train_max = float(X_train.max())
    scale = max(train_max - train_min, 1e-8)

    def normalize(a):
        return (a - train_min) / scale

    def denormalize(a):
        return a * scale + train_min

    Xtr_n, ytr_n = normalize(X_train), normalize(y_train)
    Xval_n, yval_n = normalize(X_val), normalize(y_val)
    Xtest_n = normalize(X_test)

    torch.manual_seed(cfg["seed"])
    model = GRUForecaster(hidden_size=fc_cfg["gru_hidden_size"],
                           num_layers=fc_cfg["gru_num_layers"],
                           horizon=horizon)
    opt = torch.optim.Adam(model.parameters(), lr=fc_cfg["learning_rate"])
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr_n, dtype=torch.float32)
    ytr_t = torch.tensor(ytr_n, dtype=torch.float32)
    Xval_t = torch.tensor(Xval_n, dtype=torch.float32)
    yval_t = torch.tensor(yval_n, dtype=torch.float32)

    n = len(Xtr_t)
    batch_size = fc_cfg["batch_size"]
    best_val_loss = float("inf")
    best_state = None
    patience = fc_cfg["early_stopping_patience"]
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(fc_cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), yval_t).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch+1}/{fc_cfg['epochs']} train_loss={epoch_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        Xtest_t = torch.tensor(Xtest_n, dtype=torch.float32)
        gru_pred_n = model(Xtest_t).numpy()
    gru_pred = denormalize(gru_pred_n)

    persist_pred = persistence_forecast(X_test, horizon)
    ma_pred = moving_average_forecast(X_test, horizon, window=4)

    results = {
        "disclaimer": disclaimer(cfg),
        "config": {"lookback": lookback, "horizon": horizon, "n_zones": n_zones,
                    "bin_seconds": cfg["data"]["bin_seconds"], "value_col": value_col,
                    "cell_set": cell_set},
        "overall": {
            "gru": segment_metrics(y_test, gru_pred, "gru_overall"),
            "persistence": segment_metrics(y_test, persist_pred, "persistence_overall"),
            "moving_average": segment_metrics(y_test, ma_pred, "moving_average_overall"),
        },
        "by_regime": {},
    }

    last_obs = X_test[:, -1]
    median = np.median(last_obs)
    p90 = np.percentile(last_obs, 90)
    for name, mask in [("low_load", last_obs <= median), ("high_load", last_obs > median),
                        ("burst", last_obs >= p90)]:
        if mask.sum() == 0:
            continue
        results["by_regime"][name] = {
            "gru": segment_metrics(y_test[mask], gru_pred[mask], f"gru_{name}"),
            "persistence": segment_metrics(y_test[mask], persist_pred[mask], f"persistence_{name}"),
            "moving_average": segment_metrics(y_test[mask], ma_pred[mask], f"moving_average_{name}"),
        }

    m_dir = models_dir(cfg, "forecasting")
    r_dir = results_dir(cfg, "forecasting")
    m_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), m_dir / "gru_forecaster.pt")
    (m_dir / "scaler.json").write_text(json.dumps({"min": train_min, "max": train_max}))
    (m_dir / "config.json").write_text(json.dumps(fc_cfg))
    (r_dir / "training_history.json").write_text(json.dumps(history))
    (r_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    print(json.dumps(results["overall"], indent=2))
    return results


if __name__ == "__main__":
    main()
