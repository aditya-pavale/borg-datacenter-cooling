"""V2-specific tests: data alignment, causality/leakage, and the
shield-confound regression caught during thermal calibration
(docs/version2_research_design.md S3). V1's tests/ (46 tests) already
cover the shared, dataset-independent modules (thermal ODE, safety
shield math, PPO/MAPPO structural correctness) and are unaffected by
V2 -- these tests cover what's actually new: the cell-level join, the
V2 data pipeline, and V2CoolingCore's data-driven heat generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.v2_cooling_core import V2CoolingCore, load_v2_config  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "processed" / "v2"
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "train_pilot.parquet").exists(),
    reason="V2 pilot data not present (run src/data/v2_build_timeseries.py first)",
)


@pytest.fixture(scope="module")
def cfg():
    return load_v2_config()


def test_no_duplicate_cell_bucket_rows():
    df = pd.read_parquet(DATA_DIR / "workload_power_5min_pilot.parquet")
    assert df.duplicated(subset=["cell", "bucket_5min"]).sum() == 0


def test_chronological_split_disjoint_and_ordered():
    train = pd.read_parquet(DATA_DIR / "train_pilot.parquet")
    val = pd.read_parquet(DATA_DIR / "val_pilot.parquet")
    test = pd.read_parquet(DATA_DIR / "test_pilot.parquet")
    train_b, val_b, test_b = set(train.bucket_5min), set(val.bucket_5min), set(test.bucket_5min)
    assert train_b.isdisjoint(val_b)
    assert train_b.isdisjoint(test_b)
    assert val_b.isdisjoint(test_b)
    assert max(train_b) < min(val_b) < max(val_b) < min(test_b)


def test_zone_is_deterministic_alphabetical_cell_code():
    df = pd.read_parquet(DATA_DIR / "train_pilot.parquet")
    mapping = df[["cell", "zone"]].drop_duplicates().set_index("cell")["zone"].to_dict()
    expected = {c: i for i, c in enumerate(sorted(mapping))}
    assert mapping == expected


def test_power_util_is_within_pdu_capacity_fraction_bounds():
    # measured_power_util is a fraction of PDU rated capacity per
    # docs/official_data_alignment_audit.md S3 -- should never be
    # negative and should not wildly exceed 1.0 (a small margin above 1
    # is possible for genuine peak overrun, but not e.g. 10x).
    df = pd.read_parquet(DATA_DIR / "train_pilot.parquet")
    assert (df["predicted_power_util"] >= -0.01).all()
    assert (df["predicted_power_util"] <= 1.5).all()


def test_forecast_never_leaks_future_bins(cfg):
    forecast = np.load(DATA_DIR / "forecast_train_pilot.npy")
    lookback = cfg["forecasting"]["lookback_steps"]
    # bins before lookback-1 are degenerate (all-zero) by construction --
    # same discipline as V1's precompute_forecasts.py.
    assert np.allclose(forecast[: lookback - 1], 0.0)


def test_heat_generation_matches_fitted_power_model(cfg):
    core = V2CoolingCore(split="train", cfg=cfg, use_safety_shield=False)
    core.reset(start_idx=cfg["forecasting"]["lookback_steps"] - 1, seed=0)
    expected = core.predicted_power_util[core.start_idx] * cfg["power_model"]["heat_scale_kw"]
    np.testing.assert_allclose(core._current_heat_kw(), expected)


def test_open_loop_cooling_off_is_invariant_to_cooling_max_kw(cfg):
    """Regression test for the shield-confound bug found in
    docs/version2_research_design.md S3: with the safety shield
    disabled, a cooling-off (action=0) trajectory must not depend on
    cooling_max_kw at all, since cooling_kw = action * cooling_max_kw
    and action=0 zeroes that term regardless."""
    results = []
    for cooling_max in [0.5, 0.8]:
        core = V2CoolingCore(split="train", cfg=cfg, use_safety_shield=False)
        core.thermal_params.cooling_max_kw = cooling_max
        core.reset(start_idx=100, seed=0)
        max_temp = -np.inf
        for _ in range(50):
            obs, r, done, info = core.step(np.zeros(core.n_zones))
            max_temp = max(max_temp, info["temps"].max())
        results.append(max_temp)
    assert abs(results[0] - results[1]) < 1e-9, (
        f"cooling-off trajectory changed with cooling_max_kw ({results}) -- "
        f"the shield-confound bug has regressed"
    )


def test_open_loop_cooling_off_DOES_depend_on_shield_state(cfg):
    """Complementary check: with the shield ENABLED, action=0 CAN be
    overridden -- confirms the previous test isn't trivially passing
    because the shield code path is dead."""
    core_shielded = V2CoolingCore(split="train", cfg=cfg, use_safety_shield=True)
    core_shielded.reset(start_idx=100, seed=0)
    core_open = V2CoolingCore(split="train", cfg=cfg, use_safety_shield=False)
    core_open.reset(start_idx=100, seed=0)

    shielded_final, open_final = None, None
    for _ in range(250):
        _, _, _, info_s = core_shielded.step(np.zeros(core_shielded.n_zones))
        _, _, _, info_o = core_open.step(np.zeros(core_open.n_zones))
    shielded_final = info_s["temps"].max()
    open_final = info_o["temps"].max()
    assert shielded_final < open_final - 1.0, (
        "expected the shield to measurably suppress the open-loop "
        f"cooling-off trajectory (shielded={shielded_final}, open={open_final})"
    )


def test_n_zones_equals_active_cell_count(cfg):
    core = V2CoolingCore(split="train", cfg=cfg)
    df = pd.read_parquet(DATA_DIR / "train_pilot.parquet")
    assert core.n_zones == cfg["data"]["n_zones"]
    assert core.n_zones == df["cell"].nunique()
