"""V2 (pilot): random-shooting MPC adapted for V2CoolingCore. Same
random-shooting approach and documented departure from a gradient/CEM
solver as V1's MPCController (src/controllers/classical.py), but the
heat-rollout step differs because V1's heat came from an assumed
closed-form CPU->power formula (so future heat could be computed
directly from the forecasted workload), while V2's heat comes from a
FITTED sklearn model that needs (sum_cpu, sum_mem, zone) as features --
sum_mem has no forecast.

**Documented simplification**: heat is held CONSTANT at the current
step's `predicted_power_util * heat_scale_kw` for the entire rollout
horizon, rather than re-querying the fitted power model with forecasted
sum_cpu. This is justified by the forecasting result
(results/pilot/forecasting/metrics.json): the workload signal changes
slowly relative to the 30-minute (6-step) horizon used here, so a
constant-heat approximation over that short a horizon is a reasonable,
disclosed simplification -- not a fabricated one. A future improvement
would re-run the fitted power model per rollout step using the GRU's
forecasted sum_cpu (paired with the current sum_mem, since memory usage
changes slower than CPU) instead of holding heat constant.
"""

from __future__ import annotations

import numpy as np


class V2MPCController:
    def __init__(self, core, reward_cfg: dict, horizon_steps: int,
                 n_candidates: int, n_zones: int, seed: int = 0):
        # `core`: a live reference to the V2CoolingCore instance being
        # stepped, so this controller can read `core._current_heat_kw()`
        # directly. This is a documented, deliberate exception to the
        # "controller only sees obs" convention (unlike V1's MPC, which
        # could reconstruct heat from `obs["forecast"]` via the assumed
        # formula) -- see this module's docstring for why V2's fitted
        # power model makes that reconstruction impractical.
        self.core = core
        self.params = core.thermal_params
        self.reward_cfg = reward_cfg
        self.horizon = horizon_steps
        self.n_candidates = n_candidates
        self.n_zones = n_zones
        self.rng = np.random.default_rng(seed)
        self.adj = core.thermal_params.adjacency()

    def reset(self):
        pass

    def _rollout_cost(self, temps0, heat_const, action_sequence):
        temps = temps0.copy()
        cost = 0.0
        for a in action_sequence:
            cooling_kw = a * self.params.cooling_max_kw * self.params.cooling_efficiency
            coupling = self.params.coupling_kw_per_c * (self.adj @ temps - self.adj.sum(axis=1) * temps)
            ambient = self.params.ambient_coupling_kw_per_c * (self.params.ambient_temp_c - temps)
            dT = (self.params.dt_hours / self.params.thermal_mass_kwh_per_c) * (
                heat_const - cooling_kw + coupling + ambient
            )
            temps = temps + dT
            violation = np.maximum(0.0, temps - self.params.safety_limit_c)
            cost += (self.reward_cfg["energy_weight"] * cooling_kw.sum()
                     + self.reward_cfg["safety_penalty_weight"] * violation.sum())
        return cost

    def act(self, obs: dict) -> np.ndarray:
        heat_const = self.core._current_heat_kw()
        temps0 = obs["temps"]
        best_cost = np.inf
        best_first_action = np.full(self.n_zones, 0.5)
        for _ in range(self.n_candidates):
            candidate = self.rng.uniform(0.0, 1.0, size=(self.horizon, self.n_zones))
            cost = self._rollout_cost(temps0, heat_const, candidate)
            if cost < best_cost:
                best_cost = cost
                best_first_action = candidate[0]
        return best_first_action
