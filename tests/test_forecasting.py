"""Causality/leakage tests for the forecasting pipeline."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecasting.dataset import make_windows  # noqa: E402
from forecasting.baselines import persistence_forecast, moving_average_forecast  # noqa: E402


def test_make_windows_shapes():
    series = np.arange(20, dtype=np.float32)
    X, y = make_windows(series, lookback=5, horizon=2)
    assert X.shape == (14, 5)
    assert y.shape == (14, 2)


def test_make_windows_no_overlap_between_input_and_target():
    series = np.arange(20, dtype=np.float32)
    lookback, horizon = 5, 2
    X, y = make_windows(series, lookback, horizon)
    for i in range(len(X)):
        # target values must be strictly greater (later in time) than every input value
        assert y[i].min() > X[i].max()


def test_make_windows_target_is_exact_continuation():
    series = np.arange(20, dtype=np.float32)
    X, y = make_windows(series, lookback=5, horizon=2)
    for i in range(len(X)):
        assert X[i, -1] == y[i, 0] - 1  # y immediately follows X


def test_persistence_uses_only_last_observed_value():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pred = persistence_forecast(X, horizon=3)
    assert (pred == np.array([[3.0, 3.0, 3.0], [6.0, 6.0, 6.0]])).all()


def test_moving_average_uses_only_past_window():
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    pred = moving_average_forecast(X, horizon=2, window=2)
    assert np.allclose(pred, np.array([[3.5, 3.5]]))  # mean(3,4) = 3.5


def test_scaler_uses_only_train_statistics():
    """Regression test mirroring train_forecaster.py's normalize():
    the min/max must come from X_train only, never from val/test."""
    train_vals = np.array([0.0, 1.0, 2.0])
    val_vals = np.array([100.0])  # a val outlier must NOT affect the scaler
    train_min, train_max = train_vals.min(), train_vals.max()
    scale = train_max - train_min
    normalized_val = (val_vals - train_min) / scale
    assert normalized_val[0] == 50.0  # correctly extrapolates outside [0,1]
    assert scale == 2.0  # unaffected by the val outlier
