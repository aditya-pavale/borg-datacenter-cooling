"""Phase 15/27/30: robustness experiments B and C.

Experiment B: sustained workload spike. SYNTHETIC perturbation --
clearly labeled as such, not presented as real Borg data: the test
split's real workload is scaled up by a fixed multiplier for the whole
episode, simulating a sustained demand surge.

Experiment C: forecast-error robustness. SYNTHETIC perturbation --
independent Gaussian noise is added to the (already-real, causal) GRU
forecast at evaluation time only, simulating degraded forecast quality,
to see how forecast-reliant controllers degrade.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.cooling_core import CoolingCore, load_config  # noqa: E402
from environment.multi_agent_env import MultiAgentCoolingEnv  # noqa: E402
from controllers.classical import PIDController  # noqa: E402
from controllers.rl_adapters import PPOControllerAdapter, MAPPOControllerAdapter  # noqa: E402
from marl.mappo import MAPPOTrainer  # noqa: E402
from evaluation.metrics import compute_episode_metrics, aggregate_across_episodes  # noqa: E402
from evaluation.run_controller import fixed_test_start_indices  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_with_workload_scale(controller, split, start_indices, scale, cfg):
    dt_hours = cfg["thermal"]["dt_seconds"] / 3600.0
    safety_limit = cfg["thermal"]["safety_limit_c"]
    core = CoolingCore(split=split, cfg=cfg)
    core.workload = np.clip(core.workload * scale, 0.0, None)
    core.forecast = np.clip(core.forecast * scale, 0.0, None)

    episode_metrics = []
    for start_idx in start_indices:
        controller.reset()
        obs = core.reset(start_idx=start_idx)
        trajectory = []
        done = False
        while not done:
            action = controller.act(obs)
            obs, reward, done, info = core.step(action)
            trajectory.append(info)
        episode_metrics.append(compute_episode_metrics(trajectory, safety_limit, dt_hours))
    return aggregate_across_episodes(episode_metrics)


def run_with_forecast_noise(controller, split, start_indices, noise_std, cfg):
    dt_hours = cfg["thermal"]["dt_seconds"] / 3600.0
    safety_limit = cfg["thermal"]["safety_limit_c"]
    rng = np.random.default_rng(123)
    core = CoolingCore(split=split, cfg=cfg)
    core.forecast = np.clip(core.forecast + rng.normal(0, noise_std, core.forecast.shape), 0.0, None)

    episode_metrics = []
    for start_idx in start_indices:
        controller.reset()
        obs = core.reset(start_idx=start_idx)
        trajectory = []
        done = False
        while not done:
            action = controller.act(obs)
            obs, reward, done, info = core.step(action)
            trajectory.append(info)
        episode_metrics.append(compute_episode_metrics(trajectory, safety_limit, dt_hours))
    return aggregate_across_episodes(episode_metrics)


def main():
    cfg = load_config()
    starts = fixed_test_start_indices("test", n_episodes=15, cfg=cfg)

    pid = PIDController(**cfg["controllers"]["pid"])

    env = MultiAgentCoolingEnv(split="test")
    trainer = MAPPOTrainer(env, cfg, seed=0)
    trainer.actor.load_state_dict(torch.load(REPO_ROOT / "models" / "mappo" / "actor_seed0.pt"))
    mappo = MAPPOControllerAdapter(trainer)

    ppo_model = PPO.load(REPO_ROOT / "models" / "ppo" / "ppo_seed2")  # best-performing PPO seed
    ppo = PPOControllerAdapter(ppo_model, cfg["data"]["n_zones"], cfg["forecasting"]["horizon_steps"])

    controllers = {"pid": pid, "mappo_seed0": mappo, "ppo_seed2": ppo}

    results = {"experiment_B_workload_spike": {}, "experiment_C_forecast_error": {}}

    print("=== Experiment B: SYNTHETIC sustained workload spike ===")
    for scale in [1.0, 2.0, 5.0, 10.0]:
        results["experiment_B_workload_spike"][f"scale_{scale}"] = {}
        for name, ctrl in controllers.items():
            agg = run_with_workload_scale(ctrl, "test", starts, scale, cfg)
            results["experiment_B_workload_spike"][f"scale_{scale}"][name] = {
                "energy": agg["energy.total_kwh"]["mean"],
                "max_temp": agg["thermal.max_temp_c"]["mean"],
                "viol_pct": agg["thermal.violation_pct"]["mean"],
            }
            print(f"scale={scale} {name}: energy={agg['energy.total_kwh']['mean']:.2f} "
                  f"viol_pct={agg['thermal.violation_pct']['mean']:.2f} "
                  f"max_temp={agg['thermal.max_temp_c']['mean']:.2f}")

    print("\n=== Experiment C: SYNTHETIC forecast-error robustness ===")
    for noise_std in [0.0, 0.01, 0.05, 0.1]:
        results["experiment_C_forecast_error"][f"noise_{noise_std}"] = {}
        for name, ctrl in controllers.items():
            agg = run_with_forecast_noise(ctrl, "test", starts, noise_std, cfg)
            results["experiment_C_forecast_error"][f"noise_{noise_std}"][name] = {
                "energy": agg["energy.total_kwh"]["mean"],
                "max_temp": agg["thermal.max_temp_c"]["mean"],
                "viol_pct": agg["thermal.violation_pct"]["mean"],
            }
            print(f"noise_std={noise_std} {name}: energy={agg['energy.total_kwh']['mean']:.2f} "
                  f"viol_pct={agg['thermal.violation_pct']['mean']:.2f}")

    (RESULTS_DIR / "robustness_results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
