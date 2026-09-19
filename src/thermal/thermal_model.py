"""Phase 6: three-zone coupled lumped-parameter thermal model.

Conceptual continuous-time formulation (per zone i):

    C_i * dT_i/dt = Q_i - Q_cooling,i + sum_j K_ij (T_j - T_i) + K_amb (T_amb - T_i)

Discretized with explicit (forward) Euler at dt = thermal.dt_seconds:

    T_i[t+1] = T_i[t] + (dt_hours / C_i) * (
        Q_i[t] - Q_cooling_i[t] + sum_j K_ij (T_j[t]-T_i[t]) + K_amb (T_amb - T_i[t])
    )

Zone topology: a line graph 0 -- 1 -- 2 (zone 1 couples to both 0 and 2;
zones 0 and 2 do not couple directly), matching the master plan's
"Zone1 affects Zone2, Zone2 affects Zone1 and Zone3, Zone3 affects
Zone2."

All parameters (C_i, K_ij, K_amb, cooling_max_kw) are SIMULATION
ASSUMPTIONS (category D) -- there are no thermal measurements anywhere
in the Borg dataset to calibrate against (docs/dataset_audit.md S14).
This module implements the equations correctly and stably; it does not
claim physical validation. See docs/thermal_model.md for the required
stability/behavior validation experiments.

Forward Euler stability: stable provided dt < 2*tau where
tau = C_i / (K_amb + sum_j K_ij) is the zone's thermal time constant.
With the default config (C=2.0 kWh/C, K_amb=0.02, K_ij=0.05 per
neighbor), tau ~ 2.0/(0.02+0.1) ~= 16.7 hours for a middle zone, vs.
dt = 0.25 hours -- comfortably stable (checked at runtime, see
`max_stable_dt_hours`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThermalParams:
    dt_seconds: float = 900.0
    thermal_mass_kwh_per_c: float = 2.0
    coupling_kw_per_c: float = 0.05
    ambient_coupling_kw_per_c: float = 0.02
    ambient_temp_c: float = 22.0
    cooling_max_kw: float = 0.5
    cooling_efficiency: float = 1.0
    initial_temp_c: float = 24.0
    safety_limit_c: float = 27.0
    n_zones: int = 3

    @property
    def dt_hours(self) -> float:
        return self.dt_seconds / 3600.0

    def adjacency(self) -> np.ndarray:
        """Line-graph adjacency (1 if adjacent, 0 otherwise)."""
        n = self.n_zones
        adj = np.zeros((n, n))
        for i in range(n - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        return adj

    def max_stable_dt_hours(self) -> float:
        n_neighbors = self.adjacency().sum(axis=1).max()
        k_total = self.ambient_coupling_kw_per_c + n_neighbors * self.coupling_kw_per_c
        tau = self.thermal_mass_kwh_per_c / k_total
        return 2 * tau


class ThreeZoneThermalModel:
    def __init__(self, params: ThermalParams):
        self.p = params
        self.adj = params.adjacency()
        assert params.dt_hours < params.max_stable_dt_hours(), (
            f"dt_hours={params.dt_hours} exceeds the forward-Euler stability "
            f"bound {params.max_stable_dt_hours()} for these thermal parameters"
        )
        self.temps: np.ndarray = np.full(params.n_zones, params.initial_temp_c)

    def reset(self, initial_temps: np.ndarray | None = None) -> np.ndarray:
        self.temps = (
            np.full(self.p.n_zones, self.p.initial_temp_c)
            if initial_temps is None
            else np.array(initial_temps, dtype=float)
        )
        return self.temps.copy()

    def step(self, heat_kw: np.ndarray, cooling_actions: np.ndarray) -> np.ndarray:
        """heat_kw: (n_zones,) heat input per zone (from the power model).
        cooling_actions: (n_zones,) in [0,1]. Returns new temps (n_zones,)."""
        heat_kw = np.asarray(heat_kw, dtype=float)
        cooling_actions = np.clip(np.asarray(cooling_actions, dtype=float), 0.0, 1.0)

        cooling_kw = cooling_actions * self.p.cooling_max_kw * self.p.cooling_efficiency

        T = self.temps
        coupling_term = self.p.coupling_kw_per_c * (self.adj @ T - self.adj.sum(axis=1) * T)
        ambient_term = self.p.ambient_coupling_kw_per_c * (self.p.ambient_temp_c - T)

        dT = (self.p.dt_hours / self.p.thermal_mass_kwh_per_c) * (
            heat_kw - cooling_kw + coupling_term + ambient_term
        )
        self.temps = T + dT
        return self.temps.copy()
