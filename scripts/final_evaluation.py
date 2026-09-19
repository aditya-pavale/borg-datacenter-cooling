"""Phase 14: final evaluation. Runs every controller on the TEST split
(untouched until now) under IDENTICAL episode start indices, thermal
model, safety threshold, and metrics (Phase 28 fairness). Also runs
Experiments D (forecast ablation), F (safety shield ablation), and
reports seed variance for PPO/MAPPO (Experiment K).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.cooling_core import load_config  # noqa: E402
from environment.multi_agent_env import MultiAgentCoolingEnv  # noqa: E402
from controllers.classical import FixedController, ThresholdController, PIDController, MPCController  # noqa: E402
from controllers.rl_adapters import PPOControllerAdapter, MAPPOControllerAdapter  # noqa: E402
from marl.mappo import MAPPOTrainer  # noqa: E402
from marl.networks import GaussianActor  # noqa: E402
from thermal.thermal_model import ThermalParams  # noqa: E402
from evaluation.run_controller import run_controller_episodes, fixed_test_start_indices  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "final_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_TEST_EPISODES = 20
SEEDS = [0, 1, 2]


def load_mappo(seed: int, cfg: dict) -> MAPPOControllerAdapter:
    env = MultiAgentCoolingEnv(split="test")  # env only used for its shapes/agents list here
    trainer = MAPPOTrainer(env, cfg, seed=seed)
    trainer.actor.load_state_dict(torch.load(REPO_ROOT / "models" / "mappo" / f"actor_seed{seed}.pt"))
    trainer.actor.eval()
    return MAPPOControllerAdapter(trainer)


def load_ppo(seed: int, cfg: dict) -> PPOControllerAdapter:
    model = PPO.load(REPO_ROOT / "models" / "ppo" / f"ppo_seed{seed}")
    return PPOControllerAdapter(model, cfg["data"]["n_zones"], cfg["forecasting"]["horizon_steps"])


def summarize(agg: dict, keys: list[str]) -> dict:
    return {k: agg[k] for k in keys}


def main():
    cfg = load_config()
    tp = ThermalParams(**cfg["thermal"])
    starts = fixed_test_start_indices("test", n_episodes=N_TEST_EPISODES, cfg=cfg)
    print(f"Using {len(starts)} fixed test episode start indices: {starts}")

    metric_keys = [
        "energy.total_kwh", "thermal.mean_temp_c", "thermal.max_temp_c",
        "thermal.violation_pct", "thermal.cumulative_violation_severity",
        "control.mean_action_change", "control.max_action_change",
        "safety.n_interventions",
    ]

    all_results = {}

    # ---- Classical controllers ----
    classical = {
        "fixed": FixedController(cfg["controllers"]["fixed"]["action"]),
        "threshold": ThresholdController(**cfg["controllers"]["threshold"]),
        "pid": PIDController(**cfg["controllers"]["pid"]),
        "mpc": MPCController(tp, cfg["power_model"], cfg["reward"],
                              cfg["controllers"]["mpc"]["horizon_steps"],
                              cfg["controllers"]["mpc"]["n_candidates"]),
    }
    for name, ctrl in classical.items():
        res = run_controller_episodes(ctrl, "test", starts, cfg=cfg)
        all_results[name] = summarize(res["aggregated"], metric_keys)
        print(f"{name}: {all_results[name]['energy.total_kwh']}")

    # ---- PPO (3 seeds) ----
    for seed in SEEDS:
        ctrl = load_ppo(seed, cfg)
        res = run_controller_episodes(ctrl, "test", starts, cfg=cfg)
        all_results[f"ppo_seed{seed}"] = summarize(res["aggregated"], metric_keys)

    # ---- MAPPO (3 seeds) ----
    for seed in SEEDS:
        ctrl = load_mappo(seed, cfg)
        res = run_controller_episodes(ctrl, "test", starts, cfg=cfg)
        all_results[f"mappo_seed{seed}"] = summarize(res["aggregated"], metric_keys)

    # ---- Experiment D: forecast ablation (MAPPO seed 0, with vs without forecast at EXECUTION) ----
    # Note: models were trained WITH forecast features in the observation;
    # this ablation zeroes the forecast at evaluation time to test reliance
    # on it (a "sensitivity to forecast availability" probe), documented
    # in docs/experiment_plan.md.
    ctrl0 = load_mappo(0, cfg)
    res_with = run_controller_episodes(ctrl0, "test", starts, use_forecast=True, cfg=cfg)
    res_without = run_controller_episodes(ctrl0, "test", starts, use_forecast=False, cfg=cfg)
    all_results["mappo_seed0_forecast_ablation_with"] = summarize(res_with["aggregated"], metric_keys)
    all_results["mappo_seed0_forecast_ablation_without"] = summarize(res_without["aggregated"], metric_keys)

    # ---- Experiment F: safety shield ablation (best MAPPO seed) ----
    best_mappo_seed = min(SEEDS, key=lambda s: all_results[f"mappo_seed{s}"]["energy.total_kwh"]["mean"])
    ctrl_best = load_mappo(best_mappo_seed, cfg)
    res_shield_on = run_controller_episodes(ctrl_best, "test", starts, use_safety_shield=True, cfg=cfg)
    res_shield_off = run_controller_episodes(ctrl_best, "test", starts, use_safety_shield=False, cfg=cfg)
    all_results[f"mappo_seed{best_mappo_seed}_shield_on"] = summarize(res_shield_on["aggregated"], metric_keys)
    all_results[f"mappo_seed{best_mappo_seed}_shield_off"] = summarize(res_shield_off["aggregated"], metric_keys)

    (RESULTS_DIR / "controller_comparison.json").write_text(json.dumps(all_results, indent=2))

    print("\n=== SUMMARY (energy.total_kwh mean, thermal.violation_pct mean) ===")
    for name, m in all_results.items():
        print(f"{name}: energy={m['energy.total_kwh']['mean']:.3f} "
              f"viol_pct={m['thermal.violation_pct']['mean']:.2f} "
              f"max_temp={m['thermal.max_temp_c']['mean']:.2f}")

    return all_results


if __name__ == "__main__":
    main()
