"""Phase 4: train and evaluate the GRU workload forecaster against
persistence and moving-average baselines.

Causality / no-leakage guarantees:
  - Windows are built independently per split (train/val/test) using
    src.forecasting.dataset.make_windows, so no window's target ever
    comes from a different split than its input.
  - The scaler (a simple min-max computed from TRAIN data only) is
    fit once on pooled training-split values and applied unchanged to
    val/test.
  - Model selection (best epoch by val loss) uses the validation split
    only; test metrics are computed once, at the end, and are not used
    to pick hyperparameters.

MAPE is intentionally omitted from the primary metrics: ~30% of the
workload signal is exactly zero (docs/data_preprocessing.md), and MAPE
is undefined/explodes when the denominator is zero, so it is not
mathematically appropriate for this series (master plan S16).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecasting.dataset import make_windows, per_zone_series  # noqa: E402
from forecasting.baselines import persistence_forecast, moving_average_forecast  # noqa: E402
from forecasting.gru_model import GRUForecaster  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "forecasting"
RESULTS_DIR = REPO_ROOT / "results" / "forecasting"
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def build_all_windows(df: pd.DataFrame, n_zones: int, lookback: int, horizon: int):
    Xs, ys, zones = [], [], []
    for z in range(n_zones):
        series = per_zone_series(df, z)
        X, y = make_windows(series, lookback, horizon)
        Xs.append(X)
        ys.append(y)
        zones.append(np.full(len(X), z))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(zones)


def segment_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    return {
        "label": label,
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def main():
    cfg = load_config()
    fc_cfg = cfg["forecasting"]
    lookback = fc_cfg["lookback_steps"]
    horizon = fc_cfg["horizon_steps"]
    n_zones = cfg["data"]["n_zones"]

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    X_train, y_train, _ = build_all_windows(train, n_zones, lookback, horizon)
    X_val, y_val, _ = build_all_windows(val, n_zones, lookback, horizon)
    X_test, y_test, z_test = build_all_windows(test, n_zones, lookback, horizon)

    # Scaler: min-max fit on TRAIN ONLY.
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
            val_pred = model(Xval_t)
            val_loss = loss_fn(val_pred, yval_t).item()

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

    # ---- Final evaluation on TEST (untouched until now) ----
    model.eval()
    with torch.no_grad():
        Xtest_t = torch.tensor(Xtest_n, dtype=torch.float32)
        gru_pred_n = model(Xtest_t).numpy()
    gru_pred = denormalize(gru_pred_n)

    persist_pred = persistence_forecast(X_test, horizon)
    ma_pred = moving_average_forecast(X_test, horizon, window=4)

    results = {
        "config": {"lookback": lookback, "horizon": horizon, "n_zones": n_zones,
                    "bin_seconds": cfg["data"]["bin_seconds"]},
        "overall": {
            "gru": segment_metrics(y_test, gru_pred, "gru_overall"),
            "persistence": segment_metrics(y_test, persist_pred, "persistence_overall"),
            "moving_average": segment_metrics(y_test, ma_pred, "moving_average_overall"),
        },
        "by_regime": {},
    }

    # Evaluate across regimes: low load, high load, bursts (per master plan S16).
    last_obs = X_test[:, -1]
    median = np.median(last_obs)
    p90 = np.percentile(last_obs, 90)
    low_mask = last_obs <= median
    high_mask = last_obs > median
    burst_mask = last_obs >= p90

    for name, mask in [("low_load", low_mask), ("high_load", high_mask), ("burst", burst_mask)]:
        if mask.sum() == 0:
            continue
        results["by_regime"][name] = {
            "gru": segment_metrics(y_test[mask], gru_pred[mask], f"gru_{name}"),
            "persistence": segment_metrics(y_test[mask], persist_pred[mask], f"persistence_{name}"),
            "moving_average": segment_metrics(y_test[mask], ma_pred[mask], f"moving_average_{name}"),
        }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODELS_DIR / "gru_forecaster.pt")
    (MODELS_DIR / "scaler.json").write_text(json.dumps({"min": train_min, "max": train_max}))
    (MODELS_DIR / "config.json").write_text(json.dumps(fc_cfg))
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(history))
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    # A few example forecasts for qualitative inspection.
    example_idx = np.linspace(0, len(y_test) - 1, min(20, len(y_test)), dtype=int)
    examples = {
        "true": y_test[example_idx].tolist(),
        "gru": gru_pred[example_idx].tolist(),
        "persistence": persist_pred[example_idx].tolist(),
        "zone": z_test[example_idx].tolist(),
    }
    (RESULTS_DIR / "forecast_examples.json").write_text(json.dumps(examples, indent=2))

    print(json.dumps(results["overall"], indent=2))
    return results


if __name__ == "__main__":
    main()
