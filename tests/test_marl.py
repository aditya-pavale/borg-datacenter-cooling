"""Phase 33 MARL-specific tests."""

from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.multi_agent_env import MultiAgentCoolingEnv  # noqa: E402
from environment.cooling_core import load_config  # noqa: E402
from marl.mappo import MAPPOTrainer, compute_gae  # noqa: E402


def test_correct_agent_count_and_observation_shape():
    env = MultiAgentCoolingEnv(split="train")
    obs, _ = env.reset(start_idx=50)
    assert len(env.agents) == 3
    for a in env.agents:
        assert obs[a].ndim == 1


def test_joint_action_applied_correctly():
    """Actions within the configured max_rate_per_step (0.3 from the
    zero-initialized prev_action) should be applied unchanged and
    correctly routed to the corresponding zone."""
    env = MultiAgentCoolingEnv(split="train")
    env.reset(start_idx=50)
    actions = {"zone_0": 0.1, "zone_1": 0.25, "zone_2": 0.05}
    obs, rewards, terms, truncs, infos = env.step(actions)
    applied = infos["zone_0"]["action"]
    assert np.allclose(applied, [0.1, 0.25, 0.05], atol=1e-6)


def test_reward_consistency_shared_across_agents():
    env = MultiAgentCoolingEnv(split="train")
    env.reset(start_idx=50)
    _, rewards, _, _, _ = env.step({a: 0.5 for a in env.agents})
    vals = list(rewards.values())
    assert all(v == vals[0] for v in vals)


def test_centralized_critic_input_is_concat_of_actor_inputs():
    env = MultiAgentCoolingEnv(split="train")
    obs_dict, _ = env.reset(start_idx=50)
    global_state = env.get_global_state(env._last_obs)
    reconstructed = np.concatenate([obs_dict[a] for a in env.agents])
    assert np.allclose(global_state, reconstructed)


def test_actor_never_receives_other_agents_observations():
    """Structural check: the actor network's input dimension equals a
    SINGLE agent's local_obs_dim, not the global state dim -- i.e. it is
    architecturally impossible for the decentralized actor to consume
    another agent's or the global observation."""
    env = MultiAgentCoolingEnv(split="train")
    cfg = load_config()
    trainer = MAPPOTrainer(env, cfg)
    assert trainer.actor.net[0].in_features == env.local_obs_dim
    assert trainer.actor.net[0].in_features != env.global_state_dim
    assert trainer.critic.net[0].in_features == env.global_state_dim


def test_gae_computation_basic_sanity():
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5, 0.0]
    dones = [0.0, 0.0, 1.0]
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert len(adv) == 3
    assert len(ret) == 3
    assert np.isfinite(adv).all()


def test_mappo_rollout_and_update_run_without_error():
    env = MultiAgentCoolingEnv(split="train")
    cfg = load_config()
    trainer = MAPPOTrainer(env, cfg, seed=0)
    buf = trainer.collect_rollout(rollout_length=16, seed=0)
    assert len(buf["rewards"]) == 16
    stats = trainer.update(buf, n_epochs=2, minibatch_size=8)
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])


def test_mappo_deterministic_action_in_bounds():
    env = MultiAgentCoolingEnv(split="train")
    cfg = load_config()
    trainer = MAPPOTrainer(env, cfg, seed=0)
    obs, _ = env.reset(start_idx=50)
    actions = trainer.act_deterministic(obs)
    for a, v in actions.items():
        assert 0.0 <= v <= 1.0
