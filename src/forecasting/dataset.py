"""Windowing utilities shared by baselines and the GRU forecaster.

Causality guarantee: at time index t, a sample's input window covers
bins [t - lookback, t) and its target covers [t, t + horizon). No
target value ever appears in its own input window. This is checked by
tests/test_forecasting.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    series: np.ndarray, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """series: 1D array, chronologically ordered, single zone.

    Returns X of shape (n_samples, lookback), y of shape (n_samples, horizon).
    """
    n = len(series)
    n_samples = n - lookback - horizon + 1
    if n_samples <= 0:
        return np.empty((0, lookback)), np.empty((0, horizon))
    X = np.stack([series[i:i + lookback] for i in range(n_samples)])
    y = np.stack([series[i + lookback:i + lookback + horizon] for i in range(n_samples)])
    return X, y


def per_zone_series(df: pd.DataFrame, zone: int, value_col: str = "cpu_util") -> np.ndarray:
    sub = df[df["zone"] == zone].sort_values("bin_id")
    assert sub["bin_id"].is_monotonic_increasing
    assert (sub["bin_id"].diff().dropna() == 1).all(), "bin_id must be contiguous (no gaps) within a split"
    return sub[value_col].to_numpy(dtype=np.float32)
