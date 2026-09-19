"""V2 Phase 9 (pilot): empirical workload -> power model.

Unlike V1 (which had NO power measurements and used an ASSUMED linear
CPU->power formula, docs/thermal_model.md), V2 has real PowerData2019
measurements aligned at cell granularity
(docs/official_data_alignment_audit.md). This script fits candidate
regressors on TRAIN, selects on VALIDATION only, and reports residuals
by operating regime. TEST is not touched here -- this is pilot
(cells a-d) work; final model selection happens again on the full
8-cell dataset (results/pilot/README.md).

Target: mean_measured_power_util (fraction of PDU rated capacity, per
cell per 5-minute bucket) -- NOT absolute watts (no PDU capacity figure
is in the public schema; see docs/official_data_alignment_audit.md S3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from environment.v2_cooling_core import load_v2_config  # noqa: E402
from evaluation.v2_common import results_dir, disclaimer  # noqa: E402

PROC_DIR = REPO_ROOT / "data" / "processed" / "v2"

FEATURES = ["sum_cpu", "sum_mem"]
TARGET = "mean_measured_power_util"


def _design(df: pd.DataFrame, encoder: OneHotEncoder | None, fit: bool):
    X_num = df[FEATURES].to_numpy()
    if encoder is None:
        return X_num, None
    cell_col = df[["cell"]]
    if fit:
        X_cell = encoder.fit_transform(cell_col)
    else:
        X_cell = encoder.transform(cell_col)
    return np.hstack([X_num, X_cell]), encoder


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    cfg = load_v2_config()
    cell_set = cfg["data"]["active_cell_set"]
    train = pd.read_parquet(PROC_DIR / f"train_{cell_set}.parquet")
    val = pd.read_parquet(PROC_DIR / f"val_{cell_set}.parquet")

    results = {}

    # 1. Constant baseline (train mean)
    dummy = DummyRegressor(strategy="mean").fit(train[FEATURES], train[TARGET])
    results["constant_baseline"] = evaluate(val[TARGET], dummy.predict(val[FEATURES]))

    # 2. Linear, pooled across cells (no cell identity)
    lin = LinearRegression().fit(train[FEATURES], train[TARGET])
    results["linear_pooled"] = evaluate(val[TARGET], lin.predict(val[FEATURES]))
    results["linear_pooled"]["coef"] = dict(zip(FEATURES, lin.coef_.tolist()))
    results["linear_pooled"]["intercept"] = float(lin.intercept_)

    # 3. Linear + per-cell fixed effect (cells likely differ in idle/base
    #    power draw -- different PDU counts per cell, see docs/official_data_alignment_audit.md)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train, enc = _design(train, enc, fit=True)
    X_val, _ = _design(val, enc, fit=False)
    lin_fe = LinearRegression().fit(X_train, train[TARGET])
    results["linear_plus_cell_fixed_effect"] = evaluate(val[TARGET], lin_fe.predict(X_val))

    # 4. Random forest (nonlinear, handles cell as categorical directly)
    train_cat = train.copy()
    val_cat = val.copy()
    train_cat["cell_code"] = train_cat["cell"].astype("category").cat.codes
    val_cat["cell_code"] = val_cat["cell"].astype("category").cat.codes
    rf_features = FEATURES + ["cell_code"]
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0, n_jobs=-1)
    rf.fit(train_cat[rf_features], train_cat[TARGET])
    results["random_forest"] = evaluate(val_cat[TARGET], rf.predict(val_cat[rf_features]))
    results["random_forest"]["feature_importances"] = dict(
        zip(rf_features, rf.feature_importances_.tolist())
    )

    # Residual analysis for the best-on-val model by RMSE (selection rule
    # fixed BEFORE looking at test; test is never touched in this script).
    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"Best on validation (by RMSE): {best_name}")

    if best_name == "random_forest":
        val_pred = rf.predict(val_cat[rf_features])
    elif best_name == "linear_plus_cell_fixed_effect":
        val_pred = lin_fe.predict(X_val)
    elif best_name == "linear_pooled":
        val_pred = lin.predict(val[FEATURES])
    else:
        val_pred = dummy.predict(val[FEATURES])

    residuals = val[TARGET].to_numpy() - val_pred
    load_tercile = pd.qcut(val["sum_cpu"], 3, labels=["low", "mid", "high"])
    by_regime = (
        pd.DataFrame({"residual": residuals, "regime": load_tercile})
        .groupby("regime", observed=True)["residual"]
        .agg(["mean", "std", "count"])
    )
    print("Residuals by workload-tercile (validation, best model):")
    print(by_regime)

    out_dir = results_dir(cfg, "power_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "selected_on": "validation RMSE",
                "best_model": best_name,
                "results": results,
                "residuals_by_load_tercile": by_regime.reset_index().to_dict("records"),
                "disclaimer": disclaimer(cfg) + " Target is a PDU-capacity "
                        "utilization fraction, not absolute watts.",
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_dir / 'metrics.json'}")

    for name, m in results.items():
        print(name, {k: v for k, v in m.items() if k in ("mae", "rmse", "r2")})


if __name__ == "__main__":
    main()
