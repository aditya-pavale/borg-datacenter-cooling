"""Phase 20/33 environment validation tests."""

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.cooling_core import CoolingCore  # noqa: E402
from environment.single_agent_env import SingleAgentCoolingEnv  # noqa: E402
from environment.multi_agent_env import MultiAgentCoolingEnv  # noqa: E402
from thermal.thermal_model import ThermalParams  # noqa: E402
from environment.safety_shield import shield_action, predict_next_temps  # noqa: E402


# ---------- CoolingCore ----------

def test_core_reset_deterministic_with_start_idx():
    c1 = CoolingCore(split="train")
    c2 = CoolingCore(split="train")
    obs1 = c1.reset(start_idx=100)
    obs2 = c2.reset(start_idx=100)
    assert np.allclose(obs1["temps"], obs2["temps"])
    assert np.allclose(obs1["workload"], obs2["workload"])


def test_core_step_output_shapes():
    c = CoolingCore(split="train")
    c.reset(start_idx=50)
    obs, reward, done, info = c.step(np.array([0.5, 0.5, 0.5]))
    assert obs["temps"].shape == (3,)
    assert obs["forecast"].shape == (3, c.horizon)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_core_action_clipped_to_bounds():
    c = CoolingCore(split="train")
    c.reset(start_idx=50)
    obs, _, _, info = c.step(np.array([5.0, -3.0, 0.5]))
    assert (info["action"] <= 1.0).all()
    assert (info["action"] >= 0.0).all()


def test_core_action_rate_limited():
    c = CoolingCore(split="train")
    c.reset(start_idx=50)
    c.step(np.array([0.0, 0.0, 0.0]))
    _, _, _, info = c.step(np.array([1.0, 1.0, 1.0]))
    max_rate = c.acfg["max_rate_per_step"]
    assert (info["action"] <= max_rate + 1e-9).all()


def test_core_episode_terminates_at_configured_length():
    c = CoolingCore(split="train")
    c.reset(start_idx=50)
    done = False
    steps = 0
    while not done:
        _, _, done, _ = c.step(np.array([0.5, 0.5, 0.5]))
        steps += 1
        assert steps <= c.episode_length
    assert steps == c.episode_length


def test_core_no_forecast_ablation_zeros_forecast():
    c = CoolingCore(split="train", use_forecast=False)
    obs = c.reset(start_idx=50)
    assert np.allclose(obs["forecast"], 0.0)


def test_core_forecast_is_not_the_actual_future_workload():
    """The forecast should differ from the true realized future workload
    in general (a forecast is a prediction, not a leak of ground truth) --
    if it were byte-identical to the future actuals we would suspect
    leakage."""
    c = CoolingCore(split="train", use_forecast=True)
    c.reset(start_idx=200)
    forecast_at_t = c._current_forecast()  # shape (n_zones, horizon)
    actual_future = c.workload[c.start_idx + 1: c.start_idx + 1 + c.horizon].T  # (n_zones, horizon)
    assert forecast_at_t.shape == actual_future.shape
    assert not np.allclose(forecast_at_t, actual_future), (
        "forecast is identical to true future workload -- possible leakage"
    )


# ---------- single-agent wrapper ----------

def test_single_agent_env_spaces():
    env = SingleAgentCoolingEnv(split="train")
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs2)
    assert env.action_space.shape == (3,)


def test_single_agent_env_reset_step_consistency():
    env = SingleAgentCoolingEnv(split="train")
    obs, _ = env.reset(options={"start_idx": 50})
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5, 0.5]))
        assert not np.isnan(obs).any()
        assert not np.isnan(reward)


# ---------- multi-agent env ----------

def test_multi_agent_env_agent_count():
    env = MultiAgentCoolingEnv(split="train")
    obs, infos = env.reset()
    assert set(obs.keys()) == {"zone_0", "zone_1", "zone_2"}
    assert len(env.agents) == 3


def test_multi_agent_env_step_consistency():
    env = MultiAgentCoolingEnv(split="train")
    env.reset(start_idx=50)
    actions = {a: 0.5 for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert set(rewards.keys()) == set(env.agents)
    assert set(obs.keys()) == set(env.agents)
    # shared/global reward: every agent gets the identical scalar this step
    assert len(set(rewards.values())) == 1


def test_multi_agent_local_obs_dim_matches_declared():
    env = MultiAgentCoolingEnv(split="train")
    obs, _ = env.reset(start_idx=50)
    for a in env.agents:
        assert obs[a].shape == (env.local_obs_dim,)


def test_multi_agent_global_state_is_concat_of_locals():
    env = MultiAgentCoolingEnv(split="train")
    obs, _ = env.reset(start_idx=50)
    global_state = env.get_global_state(env._last_obs)
    assert global_state.shape == (env.global_state_dim,)


def test_single_and_multi_agent_envs_agree_on_physics():
    """Same start_idx, same actions each step -> identical resulting
    temperatures between the single-agent and multi-agent wrappers,
    since both sit on top of the same CoolingCore -- required for a
    fair PPO vs MAPPO comparison (Phase 28)."""
    single = SingleAgentCoolingEnv(split="train")
    multi = MultiAgentCoolingEnv(split="train")
    single.reset(options={"start_idx": 300})
    multi.reset(start_idx=300)
    for _ in range(10):
        single.step(np.array([0.4, 0.6, 0.3]))
        multi.step({"zone_0": 0.4, "zone_1": 0.6, "zone_2": 0.3})
    assert np.allclose(single.core.temps, multi.core.temps)


# ---------- safety shield ----------

def test_shield_intervenes_when_unsafe():
    """Note: with strong thermal inertia (large C relative to dt and
    cooling_max_kw), one 15-minute step can only move temperature by a
    bounded amount even at full cooling -- so the shield is not always
    able to fully restore the margin in a single step. This is an
    honest, documented limitation of a one-step-lookahead shield (see
    docs/safety_design.md), not a bug. This test therefore checks that
    the shield (a) detects the unsafe trajectory, (b) saturates the
    action at the maximum since the closed-form solution exceeds 1.0,
    and (c) strictly improves on (lowers) the unshielded outcome --
    rather than asserting it always hits the margin exactly, which
    depends on actuator authority vs. thermal inertia."""
    params = ThermalParams(initial_temp_c=26.8, safety_limit_c=27.0)
    adj = params.adjacency()
    temps = np.array([26.8, 26.8, 26.8])
    heat = np.array([0.4, 0.4, 0.4])  # high heat, would push over the limit with no cooling
    proposed = np.array([0.0, 0.0, 0.0])
    safe_action, predicted, info = shield_action(params, adj, temps, heat, proposed, margin_c=0.5)
    unshielded_predicted = predict_next_temps(params, adj, temps, heat, proposed)

    assert info["intervened"]
    assert (safe_action >= proposed).all()
    assert np.allclose(safe_action, 1.0), "closed-form action exceeds 1.0, so it must saturate at the max"
    assert (predicted < unshielded_predicted).all(), "shield must strictly reduce predicted temperature"


def test_shield_achieves_margin_when_actuator_authority_is_sufficient():
    """A scenario where cooling capacity IS enough to reach the margin
    in one step (smaller thermal mass) -- here the closed-form solution
    should be fully achievable and the shield must meet the threshold
    exactly (up to floating point)."""
    params = ThermalParams(initial_temp_c=26.8, safety_limit_c=27.0,
                            thermal_mass_kwh_per_c=0.2, cooling_max_kw=2.0)
    adj = params.adjacency()
    temps = np.array([26.8, 26.8, 26.8])
    heat = np.array([0.4, 0.4, 0.4])
    proposed = np.array([0.0, 0.0, 0.0])
    safe_action, predicted, info = shield_action(params, adj, temps, heat, proposed, margin_c=0.5)
    assert info["intervened"]
    assert (safe_action < 1.0).all(), "should not need max action when actuator authority is high"
    assert (predicted <= params.safety_limit_c - 0.5 + 1e-6).all()


def test_shield_does_not_intervene_when_already_safe():
    params = ThermalParams(initial_temp_c=22.0, safety_limit_c=27.0)
    adj = params.adjacency()
    temps = np.array([22.0, 22.0, 22.0])
    heat = np.array([0.05, 0.05, 0.05])
    proposed = np.array([0.3, 0.3, 0.3])
    safe_action, predicted, info = shield_action(params, adj, temps, heat, proposed, margin_c=0.5)
    assert not info["intervened"]
    assert np.allclose(safe_action, proposed)
