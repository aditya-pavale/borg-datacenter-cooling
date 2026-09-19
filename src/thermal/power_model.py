"""Phase 5: CPU utilization -> IT power -> heat generation.

P_IT(U) = P_idle + (P_max - P_idle) * U,  U = normalized CPU utilization in [0, 1]
Q_heat  = P_IT * heat_fraction

Parameter provenance (see configs/config.yaml `power_model`):
  - p_idle_kw, p_max_kw: ASSUMED / externally-typical values for a
    small server-cluster-equivalent load per zone, illustrative of
    published server power curves (e.g. Barroso & Holzle-style linear
    idle+dynamic model). NOT calibrated against any measurement -- the
    Borg dataset contains no power measurements of any kind
    (docs/dataset_audit.md S14).
  - heat_fraction: ASSUMED = 1.0 (first-order approximation: all
    electrical IT power is eventually rejected as heat). This is a
    standard simplification, not a measured value.

This module performs NO calibration; it is a pure, documented,
parameterized function. Sensitivity analysis is in
scripts/power_model_sensitivity.py.
"""

from __future__ import annotations

import numpy as np


def it_power_kw(utilization: np.ndarray, p_idle_kw: float, p_max_kw: float) -> np.ndarray:
    u = np.clip(utilization, 0.0, 1.0)
    return p_idle_kw + (p_max_kw - p_idle_kw) * u


def heat_kw(utilization: np.ndarray, p_idle_kw: float, p_max_kw: float,
            heat_fraction: float = 1.0) -> np.ndarray:
    return it_power_kw(utilization, p_idle_kw, p_max_kw) * heat_fraction
