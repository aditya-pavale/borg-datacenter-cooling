"""Actor (decentralized, parameter-shared) and centralized critic
networks for MAPPO."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1) - 0.5)

    def forward(self, obs: torch.Tensor) -> Normal:
        # NOTE: the distribution's mean is left UNBOUNDED (raw linear
        # output), matching the standard PPO convention for continuous
        # control (e.g. SB3's default continuous MlpPolicy): the
        # environment clips the sampled action to [0,1] at execution
        # time (GaussianActor.act / CoolingCore.step), and the boundary
        # distortion this introduces to the log-probability is a
        # well-known, accepted PPO simplification.
        #
        # An earlier version squashed the mean through sigmoid() here,
        # which was found (during autonomous validation, see
        # docs/experiment_log.md) to create a vanishing-gradient trap:
        # once training pushed the pre-sigmoid value to a large negative
        # number (e.g. -8.3, observed directly), sigmoid's local
        # gradient there is ~2e-4, making it effectively impossible for
        # further gradient steps to recover a higher action even when
        # the reward clearly penalized the resulting behavior. Removing
        # the sigmoid fixed this.
        h = self.net(obs)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor):
        """Returns (env_action, raw_action, logp). `env_action` is clamped
        to [0,1] for the environment; `raw_action` is the untouched sample
        from the (unbounded) Gaussian and MUST be what gets stored and
        reused for the PPO importance-ratio log-probability at update
        time (see mappo.py) -- reusing the CLAMPED action there would
        evaluate log_prob() at the wrong point whenever clamping was
        active, silently corrupting the ratio even when the policy has
        not changed. This was found and fixed during autonomous
        validation (docs/experiment_log.md)."""
        dist = self.forward(obs)
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        action = torch.clamp(raw, 0.0, 1.0)
        return action, raw, logp

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor):
        """raw_action must be the UNCLAMPED action originally sampled by
        act(), not the environment-clamped action -- see act()'s docstring."""
        dist = self.forward(obs)
        logp = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy


class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)
