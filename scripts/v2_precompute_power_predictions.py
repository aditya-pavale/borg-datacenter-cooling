"""V2 Phase 9 (pilot): fit the selected empirical power model (random
forest, chosen on validation RMSE in scripts/v2_fit_power_model.py) on
TRAIN only, then attach predicted_power_util to every row of
train/val/test so the environment (v2_cooling_core.py) can read heat
input directly rather than loading sklearn at simulation runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from environment.v2_cooling_core import load_v2_config  # noqa: E402
from evaluation.v2_common import models_dir  # noqa: E402


def main():
    cfg = load_v2_config()
    cell_set = cfg["data"]["active_cell_set"]
    pcfg = cfg["power_model"]
    data_dir = REPO_ROOT / cfg["paths"]["processed_dir"]

    train = pd.read_parquet(data_dir / f"train_{cell_set}.parquet")

    features = ["sum_cpu", "sum_mem", "zone"]  # zone == cell_code, see v2_build_timeseries.py
    target = pcfg["target"]

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0, n_jobs=-1)
    model.fit(train[features], train[target])

    m_dir = models_dir(cfg, "power_model")
    m_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, m_dir / "random_forest.joblib")

    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}_{cell_set}.parquet"
        df = pd.read_parquet(path)
        df["predicted_power_util"] = model.predict(df[features])
        df.to_parquet(path, index=False)
        print(f"{split}: attached predicted_power_util to {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
