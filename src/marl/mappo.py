"""MAPPO training loop: centralized-training / decentralized-execution,
shared global reward, parameter-shared actor across the 3 zone agents,
centralized critic over the concatenated global state. See
docs/marl_design.md for the full design justification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.multi_agent_env import MultiAgentCoolingEnv  # noqa: E402
from marl.networks import GaussianActor, CentralizedCritic  # noqa: E402


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values[:T])
    return advantages, returns


class MAPPOTrainer:
    def __init__(self, env: MultiAgentCoolingEnv, cfg: dict, seed: int = 0):
        self.env = env
        self.cfg = cfg["rl"]["mappo"]
        self.n_agents = len(env.agents)
        torch.manual_seed(seed)
        self.actor = GaussianActor(env.local_obs_dim)
        self.critic = CentralizedCritic(env.global_state_dim)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg["learning_rate"],
        )

    def collect_rollout(self, rollout_length: int, seed: int | None = None):
        obs_dict, _ = self.env.reset(seed=seed)
        buf = {k: [] for k in ["local_obs", "global_state", "raw_actions", "logp", "values", "rewards", "dones"]}
        for _ in range(rollout_length):
            local_obs = np.stack([obs_dict[a] for a in self.env.agents])  # (n_agents, obs_dim)
            global_state = self.env.get_global_state(self.env._last_obs)

            with torch.no_grad():
                obs_t = torch.tensor(local_obs, dtype=torch.float32)
                env_action_t, raw_action_t, logp_t = self.actor.act(obs_t)
                value = self.critic(torch.tensor(global_state, dtype=torch.float32).unsqueeze(0)).item()

            action_dict = {a: float(env_action_t[i, 0]) for i, a in enumerate(self.env.agents)}
            next_obs_dict, rewards, terms, truncs, infos = self.env.step(action_dict)
            done = any(terms.values()) or any(truncs.values())
            # reward_scale is a TRAINER-side stabilization trick (see
            # configs/config.yaml comment) -- it never touches the
            # environment's reward, so evaluation/reporting always uses
            # the true, unscaled reward via src/evaluation/metrics.py.
            global_reward = rewards[self.env.agents[0]] * self.cfg.get("reward_scale", 1.0)

            buf["local_obs"].append(local_obs)
            buf["global_state"].append(global_state)
            buf["raw_actions"].append(raw_action_t.numpy()[:, 0])
            buf["logp"].append(logp_t.numpy())
            buf["values"].append(value)
            buf["rewards"].append(global_reward)
            buf["dones"].append(float(done))

            obs_dict = next_obs_dict
            if done:
                obs_dict, _ = self.env.reset()

        with torch.no_grad():
            bootstrap_state = self.env.get_global_state(self.env._last_obs)
            bootstrap_value = self.critic(torch.tensor(bootstrap_state, dtype=torch.float32).unsqueeze(0)).item()
        buf["values"].append(bootstrap_value)
        return buf

    def update(self, buf, n_epochs: int, minibatch_size: int):
        local_obs = torch.tensor(np.array(buf["local_obs"]), dtype=torch.float32)  # (T, n_agents, obs_dim)
        global_state = torch.tensor(np.array(buf["global_state"]), dtype=torch.float32)  # (T, global_dim)
        raw_actions = torch.tensor(np.array(buf["raw_actions"]), dtype=torch.float32).unsqueeze(-1)  # (T, n_agents, 1)
        old_logp = torch.tensor(np.array(buf["logp"]), dtype=torch.float32)  # (T, n_agents)

        advantages, returns = compute_gae(
            buf["rewards"], buf["values"], buf["dones"], self.cfg["gamma"], self.cfg["gae_lambda"]
        )
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = local_obs.shape[0]
        idx = np.arange(T)
        stats = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(n_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, minibatch_size):
                mb = idx[start:start + minibatch_size]
                mb_obs = local_obs[mb].reshape(-1, local_obs.shape[-1])
                mb_raw_actions = raw_actions[mb].reshape(-1, 1)
                mb_old_logp = old_logp[mb].reshape(-1)
                mb_adv = advantages[mb].unsqueeze(-1).expand(-1, self.n_agents).reshape(-1)
                mb_returns = returns[mb]
                mb_global = global_state[mb]

                new_logp, entropy = self.actor.evaluate(mb_obs, mb_raw_actions)
                ratio = torch.exp(new_logp - mb_old_logp)
                clip = self.cfg["clip_ratio"]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean() - self.cfg["entropy_coef"] * entropy.mean()

                values = self.critic(mb_global)
                value_loss = F.mse_loss(values, mb_returns)

                loss = policy_loss + self.cfg["value_coef"] * value_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def act_deterministic(self, obs_dict: dict) -> dict:
        with torch.no_grad():
            local_obs = torch.tensor(
                np.stack([obs_dict[a] for a in self.env.agents]), dtype=torch.float32
            )
            dist = self.actor.forward(local_obs)
            action = torch.clamp(dist.mean, 0.0, 1.0)
        return {a: float(action[i, 0]) for i, a in enumerate(self.env.agents)}
