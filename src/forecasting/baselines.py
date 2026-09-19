"""Persistence and moving-average forecasting baselines.

Both are causal by construction: a prediction for [t, t+horizon) uses
only values at or before t.
"""

from __future__ import annotations

import numpy as np


def persistence_forecast(X: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every horizon step."""
    last = X[:, -1:]
    return np.repeat(last, horizon, axis=1)


def moving_average_forecast(X: np.ndarray, horizon: int, window: int = 4) -> np.ndarray:
    """Repeat the mean of the last `window` observed values."""
    w = min(window, X.shape[1])
    avg = X[:, -w:].mean(axis=1, keepdims=True)
    return np.repeat(avg, horizon, axis=1)
