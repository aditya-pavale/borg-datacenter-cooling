"""Shared simulation core used by every controller (classical, PPO,
MAPPO) so that all comparisons in Phase 27/28 run against IDENTICAL
physics, workload windows, and reward logic. Only the policy interface
differs between wrappers (src/environment/single_agent_env.py,
src/environment/multi_agent_env.py).

No future ground truth ever enters the observation: `forecast` for
decision step t is read from a file that was produced by a FROZEN GRU
model trained only on the training split, using only data available at
or before t in the underlying series (src/forecasting/dataset.py
enforces this at forecast-generation time, see
scripts/precompute_forecasts.py). Actual future workload (`U[t+1..]`)
is never exposed to any controller through the observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from thermal.thermal_model import ThermalParams, ThreeZoneThermalModel  # noqa: E402
from thermal.power_model import heat_kw  # noqa: E402
from environment.safety_shield import shield_action  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())


class CoolingCore:
    """Stateful simulation of n_zones over a windowed real-Borg workload
    trace. `use_forecast` controls whether forecast features are exposed
    (used for the "no forecast vs forecast" ablation, Experiment D)."""

    def __init__(self, split: str = "train", use_forecast: bool = True,
                 use_safety_shield: bool | None = None, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.use_safety_shield = (
            self.cfg["safety_shield"]["enabled"] if use_safety_shield is None else use_safety_shield
        )
        self.n_zones = self.cfg["data"]["n_zones"]
        self.bin_seconds = self.cfg["data"]["bin_seconds"]
        self.steps_per_day = int(86400 / self.bin_seconds)
        self.use_forecast = use_forecast
        self.horizon = self.cfg["forecasting"]["horizon_steps"]

        ts_path = REPO_ROOT / "data" / "processed" / f"{split}.parquet"
        df = pd.read_parquet(ts_path)
        # pivot to (bin_id, zone) -> cpu_util matrix, shape (n_bins, n_zones)
        pivot = df.pivot(index="bin_id", columns="zone", values="cpu_util").sort_index()
        self.bin_ids = pivot.index.to_numpy()
        self.workload = pivot.to_numpy(dtype=np.float32)  # (n_bins, n_zones)
        self.n_bins = len(self.bin_ids)

        forecast_path = REPO_ROOT / "data" / "processed" / f"forecast_{split}.npy"
        if use_forecast and forecast_path.exists():
            self.forecast = np.load(forecast_path)  # (n_bins, n_zones, horizon)
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
        min_start = lookback - 1  # forecast is only valid from this bin onward
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
        return self.forecast[self.start_idx + self.t]  # (n_zones, horizon)

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
        """action: (n_zones,) in [0,1]. Returns (obs, global_reward,
        per_zone_reward_components, done, info)."""
        action = np.clip(action, self.acfg["min"], self.acfg["max"])
        max_rate = self.acfg["max_rate_per_step"]
        action = np.clip(action, self.prev_action - max_rate, self.prev_action + max_rate)
        action = np.clip(action, self.acfg["min"], self.acfg["max"])

        U = self._current_workload()
        heat = heat_kw(U, self.pcfg["p_idle_kw"], self.pcfg["p_max_kw"]) * self.pcfg["heat_fraction"]

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
