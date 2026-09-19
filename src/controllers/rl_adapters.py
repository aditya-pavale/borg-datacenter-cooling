"""Adapters exposing trained RL policies through the same
`.reset()`/`.act(obs)` interface used by classical controllers, so
`src/evaluation/run_controller.py` can evaluate every controller type
identically (Phase 28 fairness)."""

from __future__ import annotations

import numpy as np
import torch

from environment.single_agent_env import flatten_obs
from marl.mappo import MAPPOTrainer


class PPOControllerAdapter:
    def __init__(self, model, n_zones: int, horizon: int):
        self.model = model
        self.n_zones = n_zones
        self.horizon = horizon

    def reset(self):
        pass

    def act(self, obs: dict) -> np.ndarray:
        flat = flatten_obs(obs, self.n_zones, self.horizon)
        action, _ = self.model.predict(flat, deterministic=True)
        return np.asarray(action)


class MAPPOControllerAdapter:
    def __init__(self, trainer: MAPPOTrainer):
        self.trainer = trainer
        self.agents = trainer.env.agents

    def reset(self):
        pass

    def act(self, obs: dict) -> np.ndarray:
        env = self.trainer.env
        local = np.stack([env._local_obs(obs, i) for i in range(len(self.agents))])
        with torch.no_grad():
            dist = self.trainer.actor.forward(torch.tensor(local, dtype=torch.float32))
            action = torch.clamp(dist.mean, 0.0, 1.0)
        return action.numpy()[:, 0]
