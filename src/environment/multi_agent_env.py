"""Multi-agent environment for MAPPO (Phase 20/23), built on the SAME
CoolingCore used by the single-agent PPO baseline (Phase 28 fairness).

Follows the PettingZoo ParallelEnv API shape (reset/step return
per-agent dicts), used directly by the custom MAPPO trainer
(src/marl/mappo.py) rather than requiring the full PettingZoo test
suite, since MAPPO here is a from-scratch implementation, not a
third-party library integration. Local (per-zone) observations are
what each agent's ACTOR sees at execution time; the centralized CRITIC
(only used during training) additionally receives the concatenated
global state -- see `get_global_state()`. No agent's local observation
or the global critic state ever contains a future ground-truth value;
both are built from the same causal CoolingCore observation as the
single-agent env.
"""

from __future__ import annotations

import numpy as np

from environment.cooling_core import CoolingCore

AGENT_NAMES = [f"zone_{i}" for i in range(8)]  # extended from 3 to cover V2's
                                                 # up-to-8-cell-as-zones design
                                                 # (docs/version2_research_design.md);
                                                 # a slice of the first 3 names is
                                                 # unchanged for V1 callers.


class MultiAgentCoolingEnv:
    def __init__(self, split: str = "train", use_forecast: bool = True,
                 use_safety_shield: bool | None = None, cfg: dict | None = None,
                 core=None):
        # `core`: see single_agent_env.py -- same injection point for V2.
        self.core = core if core is not None else CoolingCore(
            split=split, use_forecast=use_forecast,
            use_safety_shield=use_safety_shield, cfg=cfg,
        )
        self.n_zones = self.core.n_zones
        self.agents = AGENT_NAMES[: self.n_zones]
        self.horizon = self.core.horizon
        self.local_obs_dim = 1 + 1 + 1 + self.horizon + 1 + 3  # temp, margin, workload, forecast, prev_action, [ambient,sin,cos]
        self.global_state_dim = self.local_obs_dim * self.n_zones

    def _local_obs(self, obs: dict, i: int) -> np.ndarray:
        return np.concatenate([
            [obs["temps"][i]],
            [obs["margin"][i]],
            [obs["workload"][i]],
            obs["forecast"][i],
            [obs["prev_action"][i]],
            [obs["ambient"], obs["time_sin"], obs["time_cos"]],
        ]).astype(np.float32)

    def _to_agent_dict(self, obs: dict) -> dict[str, np.ndarray]:
        return {a: self._local_obs(obs, i) for i, a in enumerate(self.agents)}

    def get_global_state(self, obs: dict) -> np.ndarray:
        return np.concatenate([self._local_obs(obs, i) for i in range(self.n_zones)]).astype(np.float32)

    def reset(self, start_idx=None, seed=None):
        obs = self.core.reset(start_idx=start_idx, seed=seed)
        self._last_obs = obs
        return self._to_agent_dict(obs), {a: {} for a in self.agents}

    def step(self, action_dict: dict[str, float]):
        action = np.array([action_dict[a] for a in self.agents], dtype=np.float32)
        obs, global_reward, done, info = self.core.step(action)
        self._last_obs = obs
        obs_dict = self._to_agent_dict(obs)
        # Shared/global reward for every agent -- see docs/marl_design.md
        # for the credit-assignment justification.
        rewards = {a: global_reward for a in self.agents}
        terminations = {a: done for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: info for a in self.agents}
        return obs_dict, rewards, terminations, truncations, infos
