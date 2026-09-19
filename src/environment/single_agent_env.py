"""Single-agent Gymnasium wrapper around CoolingCore, for the PPO
baseline (Phase 9/24). Controls all 3 zones with one policy and one
flat action/observation vector -- built on the SAME CoolingCore as the
multi-agent env, so physics/workload/reward are identical between the
single-agent PPO and MAPPO comparisons (fairness requirement, Phase 28).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.cooling_core import CoolingCore


def flatten_obs(obs: dict, n_zones: int, horizon: int) -> np.ndarray:
    return np.concatenate([
        obs["temps"],
        obs["margin"],
        obs["workload"],
        obs["forecast"].reshape(-1),
        obs["prev_action"],
        [obs["ambient"], obs["time_sin"], obs["time_cos"]],
    ]).astype(np.float32)


class SingleAgentCoolingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, split: str = "train", use_forecast: bool = True,
                 use_safety_shield: bool | None = None, cfg: dict | None = None):
        super().__init__()
        self.core = CoolingCore(split=split, use_forecast=use_forecast,
                                 use_safety_shield=use_safety_shield, cfg=cfg)
        n = self.core.n_zones
        h = self.core.horizon
        obs_dim = n + n + n + n * h + n + 3
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        start_idx = None if options is None else options.get("start_idx")
        obs = self.core.reset(start_idx=start_idx, seed=seed)
        return flatten_obs(obs, self.core.n_zones, self.core.horizon), {}

    def step(self, action):
        obs, reward, done, info = self.core.step(np.asarray(action))
        flat = flatten_obs(obs, self.core.n_zones, self.core.horizon)
        return flat, reward, done, False, info
