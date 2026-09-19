"""V2 Phase 10: simulation core over official Google 2019 data, built to
the same interface as V1's CoolingCore (src/environment/cooling_core.py)
so it can be injected into the unchanged SingleAgentCoolingEnv /
MultiAgentCoolingEnv wrappers (`core=` parameter) -- PPO, MAPPO, and the
classical controllers all run against this without modification.

Two things differ from V1's core, both load-bearing:
  1. Zones ARE Borg cells (n_zones == len(active cells)), not a
     synthetic hash-based partition -- see
     docs/version2_research_design.md.
  2. Heat is computed from the FITTED empirical power model's
     precomputed predictions (scripts/v2_precompute_power_predictions.py),
     not an assumed linear CPU->power formula. `heat_scale_kw`
     (configs/v2_config.yaml `power_model.heat_scale_kw`) remains an
     ASSUMED simulation constant, since no PDU capacity in kW exists in
     the public PowerData2019 schema -- see
     docs/official_data_alignment_audit.md S3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thermal.thermal_model import ThermalParams, ThreeZoneThermalModel  # noqa: E402
from environment.safety_shield import shield_action  # noqa: E402


def load_v2_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "v2_config.yaml").read_text())


class V2CoolingCore:
    def __init__(self, split: str = "train", use_forecast: bool = True,
                 use_safety_shield: bool | None = None, cfg: dict | None = None):
        self.cfg = cfg or load_v2_config()
        self.use_safety_shield = (
            self.cfg["safety_shield"]["enabled"] if use_safety_shield is None else use_safety_shield
        )
        self.n_zones = self.cfg["data"]["n_zones"]
        self.bin_seconds = self.cfg["data"]["bin_seconds"]
        self.steps_per_day = int(86400 / self.bin_seconds)
        self.use_forecast = use_forecast
        self.horizon = self.cfg["forecasting"]["horizon_steps"]
        self.value_col = self.cfg["forecasting"]["value_col"]
        cell_set = self.cfg["data"]["active_cell_set"]

        proc_dir = REPO_ROOT / self.cfg["paths"]["processed_dir"]
        df = pd.read_parquet(proc_dir / f"{split}_{cell_set}.parquet")

        workload_pivot = df.pivot(index="bin_id", columns="zone", values=self.value_col).sort_index()
        power_pivot = df.pivot(index="bin_id", columns="zone", values="predicted_power_util").sort_index()
        self.bin_ids = workload_pivot.index.to_numpy()
        self.n_bins = len(self.bin_ids)

        # Normalize the raw workload signal (sum_cpu, not already in
        # [0,1] like V1's cpu_util) for the RL observation only -- heat
        # generation uses the unnormalized predicted_power_util, not
        # this normalized value, so normalization choice here cannot
        # affect the physics, only observation scale.
        self._workload_raw = workload_pivot.to_numpy(dtype=np.float32)
        self._workload_scale = max(float(self._workload_raw.max()), 1e-8)
        self.workload = self._workload_raw / self._workload_scale
        self.predicted_power_util = power_pivot.to_numpy(dtype=np.float32)

        forecast_path = proc_dir / f"forecast_{split}_{cell_set}.npy"
        if use_forecast and forecast_path.exists():
            raw_forecast = np.load(forecast_path)  # unnormalized value_col units
            self.forecast = raw_forecast / self._workload_scale
        else:
            self.forecast = np.zeros((self.n_bins, self.n_zones, self.horizon), dtype=np.float32)

        tcfg = self.cfg["thermal"]
        self.thermal_params = ThermalParams(
            dt_seconds=tcfg["dt_seconds"],
            thermal_mass_kwh_per_c=tcfg["thermal_mass_kwh_per_c"],
            coupling_kw_per_c=tcfg["coupling_kw_per_c"],
            ambient_coupling_kw_per_c=tcfg["ambient_coupling_kw_per_c"],
            ambient_temp_c=tcfg["ambient_temp_c"],
            cooling_max_kw=tcfg["cooling_max_kw"],
            cooling_efficiency=tcfg["cooling_efficiency"],
            initial_temp_c=tcfg["initial_temp_c"],
            safety_limit_c=tcfg["safety_limit_c"],
            n_zones=self.n_zones,
        )
        self.thermal = ThreeZoneThermalModel(self.thermal_params)

        self.pcfg = self.cfg["power_model"]
        self.rcfg = self.cfg["reward"]
        self.acfg = self.cfg["action"]
        self.episode_length = self.cfg["rl"]["episode_length_steps"]

        self.rng = np.random.default_rng(self.cfg["seed"])
        self.t = 0
        self.start_idx = 0
        self.prev_action = np.zeros(self.n_zones)
        self.temps = None

    def reset(self, start_idx: int | None = None, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        lookback = self.cfg["forecasting"]["lookback_steps"]
        min_start = lookback - 1
        max_start = self.n_bins - self.episode_length - self.horizon - 1
        if start_idx is None:
            start_idx = int(self.rng.integers(min_start, max(max_start, min_start + 1)))
        self.start_idx = start_idx
        self.t = 0
        self.prev_action = np.zeros(self.n_zones)
        self.temps = self.thermal.reset(np.full(self.n_zones, self.thermal_params.initial_temp_c))
        return self._get_obs()

    def _current_workload(self) -> np.ndarray:
        return self.workload[self.start_idx + self.t]

    def _current_forecast(self) -> np.ndarray:
        return self.forecast[self.start_idx + self.t]

    def _current_heat_kw(self) -> np.ndarray:
        util = self.predicted_power_util[self.start_idx + self.t]
        return util * self.pcfg["heat_scale_kw"]

    def _time_encoding(self) -> tuple[float, float]:
        phase = 2 * np.pi * ((self.start_idx + self.t) % self.steps_per_day) / self.steps_per_day
        return float(np.sin(phase)), float(np.cos(phase))

    def _get_obs(self) -> dict:
        U = self._current_workload()
        F = self._current_forecast() if self.use_forecast else np.zeros((self.n_zones, self.horizon))
        margin = self.thermal_params.safety_limit_c - self.temps
        sin_t, cos_t = self._time_encoding()
        return {
            "temps": self.temps.copy(),
            "workload": U.copy(),
            "forecast": F.copy(),
            "margin": margin.copy(),
            "prev_action": self.prev_action.copy(),
            "ambient": self.thermal_params.ambient_temp_c,
            "time_sin": sin_t,
            "time_cos": cos_t,
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, np.ndarray, bool, dict]:
        action = np.clip(action, self.acfg["min"], self.acfg["max"])
        max_rate = self.acfg["max_rate_per_step"]
        action = np.clip(action, self.prev_action - max_rate, self.prev_action + max_rate)
        action = np.clip(action, self.acfg["min"], self.acfg["max"])

        heat = self._current_heat_kw()

        shield_info = {"intervened": False, "n_interventions": 0}
        if self.use_safety_shield:
            action, _, shield_info = shield_action(
                self.thermal_params, self.thermal.adj, self.temps, heat, action,
                self.cfg["safety_shield"]["margin_c"],
            )

        self.temps = self.thermal.step(heat, action)

        cooling_kw = action * self.thermal_params.cooling_max_kw * self.thermal_params.cooling_efficiency
        energy = cooling_kw.sum()
        violation = np.maximum(0.0, self.temps - self.thermal_params.safety_limit_c)
        proximity_start_c = self.rcfg.get("proximity_start_below_limit_c", 2.0)
        proximity = np.maximum(0.0, self.temps - (self.thermal_params.safety_limit_c - proximity_start_c))
        smoothness = np.abs(action - self.prev_action)

        shield_penalty_weight = self.rcfg.get("shield_intervention_penalty_weight", 0.0)
        reward = -(
            self.rcfg["energy_weight"] * energy
            + self.rcfg["safety_penalty_weight"] * violation.sum()
            + self.rcfg["proximity_weight"] * proximity.sum()
            + self.rcfg["smoothness_weight"] * smoothness.sum()
            + shield_penalty_weight * shield_info["n_interventions"]
        )

        info = {
            "temps": self.temps.copy(),
            "energy_kw": energy,
            "violation": violation.copy(),
            "n_violations": int((violation > 0).sum()),
            "action": action.copy(),
            "smoothness": smoothness.copy(),
            "shield_intervened": shield_info["intervened"],
            "shield_n_interventions": shield_info["n_interventions"],
        }

        self.prev_action = action
        self.t += 1
        done = self.t >= self.episode_length

        return self._get_obs(), float(reward), done, info
