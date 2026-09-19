"""Phase 8/25: classical / model-based baseline controllers. All operate
on the SAME CoolingCore observation dict as the RL policies, so the
evaluation harness (src/evaluation/run_controller.py) can drive any of
them identically (Phase 28 fairness).
"""

from __future__ import annotations

import numpy as np


class FixedController:
    def __init__(self, action: float, n_zones: int = 3):
        self.a = action
        self.n_zones = n_zones

    def reset(self):
        pass

    def act(self, obs: dict) -> np.ndarray:
        return np.full(self.n_zones, self.a)


class ThresholdController:
    def __init__(self, on_temp_c: float, off_temp_c: float, on_action: float,
                 off_action: float, n_zones: int = 3):
        self.on_temp = on_temp_c
        self.off_temp = off_temp_c
        self.on_action = on_action
        self.off_action = off_action
        self.n_zones = n_zones
        self._state = np.zeros(n_zones, dtype=bool)  # True = cooling "on"

    def reset(self):
        self._state = np.zeros(self.n_zones, dtype=bool)

    def act(self, obs: dict) -> np.ndarray:
        temps = obs["temps"]
        self._state = np.where(temps >= self.on_temp, True,
                                np.where(temps <= self.off_temp, False, self._state))
        return np.where(self._state, self.on_action, self.off_action)


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, setpoint_c: float, n_zones: int = 3):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint_c
        self.n_zones = n_zones
        self._integral = np.zeros(n_zones)
        self._prev_error = np.zeros(n_zones)

    def reset(self):
        self._integral = np.zeros(self.n_zones)
        self._prev_error = np.zeros(self.n_zones)

    def act(self, obs: dict) -> np.ndarray:
        error = obs["temps"] - self.setpoint  # positive error = too hot -> more cooling
        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error
        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        return np.clip(u, 0.0, 1.0)


class MPCController:
    """Random-shooting MPC over the thermal model (Phase 8/25). At every
    step, samples `n_candidates` candidate action sequences of length
    `horizon_steps`, rolls each forward through a COPY of the thermal
    model using the (already-causal) forecast for future heat, and
    picks the first action of whichever sequence minimizes cumulative
    predicted cost (energy + safety violation). This is a simple,
    correct, non-fabricated MPC -- not a full nonlinear-program solver
    (documented departure: random shooting instead of gradient/CEM
    optimization, chosen for CPU-only compute budget and implementation
    correctness/simplicity; see docs/experiment_log.md)."""

    def __init__(self, thermal_params, power_cfg, reward_cfg, horizon_steps: int,
                 n_candidates: int, n_zones: int = 3, seed: int = 0):
        self.params = thermal_params
        self.power_cfg = power_cfg
        self.reward_cfg = reward_cfg
        self.horizon = horizon_steps
        self.n_candidates = n_candidates
        self.n_zones = n_zones
        self.rng = np.random.default_rng(seed)
        self.adj = thermal_params.adjacency()

    def reset(self):
        pass

    def _rollout_cost(self, temps0, heat_sequence, action_sequence, adj):
        from thermal.power_model import it_power_kw  # local import to avoid cycle
        temps = temps0.copy()
        cost = 0.0
        for t in range(len(action_sequence)):
            a = action_sequence[t]
            heat = heat_sequence[t]
            cooling_kw = a * self.params.cooling_max_kw * self.params.cooling_efficiency
            coupling = self.params.coupling_kw_per_c * (adj @ temps - adj.sum(axis=1) * temps)
            ambient = self.params.ambient_coupling_kw_per_c * (self.params.ambient_temp_c - temps)
            dT = (self.params.dt_hours / self.params.thermal_mass_kwh_per_c) * (
                heat - cooling_kw + coupling + ambient
            )
            temps = temps + dT
            violation = np.maximum(0.0, temps - self.params.safety_limit_c)
            cost += (self.reward_cfg["energy_weight"] * cooling_kw.sum()
                     + self.reward_cfg["safety_penalty_weight"] * violation.sum())
        return cost

    def act(self, obs: dict) -> np.ndarray:
        from thermal.power_model import heat_kw as heat_kw_fn

        adj = self.adj
        temps0 = obs["temps"]
        forecast = obs["forecast"]  # (n_zones, horizon) -- causal forecast, not ground truth
        h = min(self.horizon, forecast.shape[1]) if forecast.shape[1] > 0 else 1
        heat_sequence = []
        for t in range(h):
            u = forecast[:, t] if forecast.shape[1] > 0 else obs["workload"]
            heat_sequence.append(heat_kw_fn(u, self.power_cfg["p_idle_kw"], self.power_cfg["p_max_kw"])
                                  * self.power_cfg["heat_fraction"])

        best_cost = np.inf
        best_first_action = np.full(self.n_zones, 0.5)
        for _ in range(self.n_candidates):
            candidate = self.rng.uniform(0.0, 1.0, size=(h, self.n_zones))
            cost = self._rollout_cost(temps0, heat_sequence, candidate, adj)
            if cost < best_cost:
                best_cost = cost
                best_first_action = candidate[0]
        return best_first_action
