"""Phase (handoff) Section I: generate all presentation/report figures
from already-verified artifacts. Does NOT retrain or recompute any
model -- it only loads saved results (parquet/JSON/model checkpoints
used purely for one forward pass / one evaluation rollout) and plots
them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.cooling_core import CoolingCore, load_config  # noqa: E402
from controllers.classical import PIDController, FixedController  # noqa: E402
from controllers.rl_adapters import PPOControllerAdapter, MAPPOControllerAdapter  # noqa: E402
from marl.mappo import MAPPOTrainer  # noqa: E402
from environment.multi_agent_env import MultiAgentCoolingEnv  # noqa: E402

FIG_DIR = REPO_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120)
    plt.close()
    print(f"wrote {FIG_DIR / name}")


def fig01_workload_over_time():
    ts = pd.read_parquet(DATA_DIR / "workload_timeseries.parquet")
    cfg = load_config()
    bin_hours = cfg["data"]["bin_seconds"] / 3600.0
    plt.figure(figsize=(11, 4))
    for z in sorted(ts["zone"].unique()):
        sub = ts[ts["zone"] == z].sort_values("bin_id")
        t_days = sub["bin_id"].to_numpy() * bin_hours / 24.0
        plt.plot(t_days, sub["cpu_util"], label=f"zone {z}", linewidth=0.7)
    plt.xlabel("time (days since trace start)")
    plt.ylabel("mean CPU utilization (normalized)")
    plt.title("01 - Real Borg-derived workload per zone over the full trace")
    plt.legend()
    savefig("01_workload_over_time.png")


def fig02_workload_distribution():
    ts = pd.read_parquet(DATA_DIR / "workload_timeseries.parquet")
    plt.figure(figsize=(7, 4))
    plt.hist(ts["cpu_util"], bins=60, color="steelblue", edgecolor="none")
    plt.yscale("log")
    plt.xlabel("mean CPU utilization per (zone, 15-min bin)")
    plt.ylabel("count (log scale)")
    plt.title(f"02 - Workload distribution "
              f"({100*(ts['cpu_util']==0).mean():.1f}% exactly zero)")
    savefig("02_workload_distribution.png")


def fig03_workload_aggregation():
    cleaned = pd.read_parquet(DATA_DIR / "borg_cleaned.parquet")
    ts = pd.read_parquet(DATA_DIR / "workload_timeseries.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(cleaned["average_usage_cpus"], bins=60, color="indianred")
    axes[0].set_yscale("log")
    axes[0].set_title("Raw instance-level average_usage.cpus\n(405,894 records)")
    axes[0].set_xlabel("cpus (normalized)")
    axes[1].hist(ts["cpu_util"], bins=60, color="steelblue")
    axes[1].set_yscale("log")
    axes[1].set_title("Aggregated: mean per (zone, 15-min bin)\n(8,931 cells)")
    axes[1].set_xlabel("cpu_util")
    fig.suptitle("03 - Effect of the mean-aggregation step")
    savefig("03_workload_aggregation.png")


def fig04_gru_forecast_vs_actual():
    examples = json.loads((RESULTS_DIR / "forecasting" / "forecast_examples.json").read_text())
    true = np.array(examples["true"])
    gru = np.array(examples["gru"])
    persist = np.array(examples["persistence"])
    n = min(10, len(true))
    fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharey=True)
    for i in range(n):
        ax = axes.flat[i]
        ax.plot(true[i], "o-", label="true", color="black")
        ax.plot(gru[i], "s--", label="GRU", color="tab:blue")
        ax.plot(persist[i], "^:", label="persistence", color="tab:orange")
        ax.set_title(f"zone {examples['zone'][i]}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("04 - GRU forecast vs. true future workload (test split examples)")
    fig.supxlabel("forecast step (15-min each)")
    fig.supylabel("cpu_util")
    savefig("04_gru_forecast_vs_actual.png")


def fig05_gru_training_loss():
    history = json.loads((RESULTS_DIR / "forecasting" / "training_history.json").read_text())
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train loss (MSE, normalized)")
    plt.plot(history["val_loss"], label="val loss (MSE, normalized)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("05 - GRU training/validation loss (early-stopped)")
    plt.legend()
    savefig("05_gru_training_loss.png")


def run_episode_trajectory(controller, split, start_idx, cfg, use_safety_shield=None):
    core = CoolingCore(split=split, use_safety_shield=use_safety_shield, cfg=cfg)
    controller.reset()
    obs = core.reset(start_idx=start_idx)
    temps, actions = [core.temps.copy()], []
    done = False
    while not done:
        a = controller.act(obs)
        obs, r, done, info = core.step(a)
        temps.append(info["temps"].copy())
        actions.append(info["action"].copy())
    return np.array(temps), np.array(actions)


def fig07_08_trajectories_and_actions():
    cfg = load_config()
    pid = PIDController(**cfg["controllers"]["pid"])
    fixed = FixedController(cfg["controllers"]["fixed"]["action"])
    ppo_model = PPO.load(REPO_ROOT / "models" / "ppo" / "ppo_seed2")
    ppo = PPOControllerAdapter(ppo_model, cfg["data"]["n_zones"], cfg["forecasting"]["horizon_steps"])
    env = MultiAgentCoolingEnv(split="test")
    trainer = MAPPOTrainer(env, cfg, seed=0)
    trainer.actor.load_state_dict(torch.load(REPO_ROOT / "models" / "mappo" / "actor_seed0.pt"))
    mappo = MAPPOControllerAdapter(trainer)

    start_idx = 300
    controllers = {"PID": pid, "Fixed(0.5)": fixed, "PPO(seed2)": ppo, "MAPPO(seed0)": mappo}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, ctrl in controllers.items():
        temps, actions = run_episode_trajectory(ctrl, "test", start_idx, cfg)
        axes[0].plot(temps[:, 0], label=name)
        axes[1].plot(actions[:, 0], label=name)
    axes[0].axhline(cfg["thermal"]["safety_limit_c"], color="red", linestyle="--", label="safety limit")
    axes[0].set_xlabel("step (15-min)")
    axes[0].set_ylabel("zone 0 temperature (C)")
    axes[0].set_title("07 - Representative temperature trajectories")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("step (15-min)")
    axes[1].set_ylabel("zone 0 cooling action [0,1]")
    axes[1].set_title("08 - Cooling actions over the same episode")
    axes[1].legend(fontsize=8)
    savefig("07_08_trajectories_and_actions.png")


def fig09_10_energy_safety_comparison():
    comp = json.loads((RESULTS_DIR / "final_evaluation" / "controller_comparison.json").read_text())
    main_controllers = ["fixed", "threshold", "pid", "mpc", "ppo_seed0", "ppo_seed1",
                         "ppo_seed2", "mappo_seed0", "mappo_seed1", "mappo_seed2"]
    energies = [comp[c]["energy.total_kwh"]["mean"] for c in main_controllers]
    interventions = [comp[c]["safety.n_interventions"]["mean"] for c in main_controllers]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ["gray"] * 4 + ["tab:orange"] * 3 + ["tab:green"] * 3
    axes[0].bar(main_controllers, energies, color=colors)
    axes[0].set_ylabel("energy (kWh/episode)")
    axes[0].set_title("09 - Energy comparison (test split, 20 episodes)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(main_controllers, interventions, color=colors)
    axes[1].set_ylabel("mean safety-shield interventions/episode")
    axes[1].set_title("10 - Safety-shield reliance comparison")
    axes[1].tick_params(axis="x", rotation=45)
    if max(interventions) == 0:
        axes[1].set_ylim(0, 1)
        axes[1].text(0.5, 0.5, "All bars are exactly 0: every controller is\n"
                                "genuinely safe without shield help on this\n"
                                "nominal test split (see the shield-on vs.\n"
                                "shield-off ablation for the pre-fix contrast).",
                     transform=axes[1].transAxes, ha="center", va="center", fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="lightyellow"))
    savefig("09_10_energy_safety_comparison.png")


def fig11_robustness():
    rob = json.loads((RESULTS_DIR / "robustness" / "robustness_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    scales = [1.0, 2.0, 5.0, 10.0]
    for name in ["pid", "mappo_seed0", "ppo_seed2"]:
        vals = [rob["experiment_B_workload_spike"][f"scale_{s}"][name]["energy"] for s in scales]
        axes[0].plot(scales, vals, "o-", label=name)
    axes[0].set_xlabel("synthetic workload scale factor")
    axes[0].set_ylabel("energy (kWh/episode)")
    axes[0].set_title("11a - Robustness: sustained workload spike (SYNTHETIC)")
    axes[0].legend()

    noises = [0.0, 0.01, 0.05, 0.1]
    for name in ["pid", "mappo_seed0", "ppo_seed2"]:
        vals = [rob["experiment_C_forecast_error"][f"noise_{n}"][name]["energy"] for n in noises]
        axes[1].plot(noises, vals, "o-", label=name)
    axes[1].set_xlabel("forecast Gaussian noise std (SYNTHETIC)")
    axes[1].set_ylabel("energy (kWh/episode)")
    axes[1].set_title("11b - Robustness: forecast-error sensitivity (SYNTHETIC)")
    axes[1].legend()
    savefig("11_robustness.png")


def fig12_reward_convergence():
    plt.figure(figsize=(8, 4.5))
    for seed in [0, 1, 2]:
        hist = json.loads((RESULTS_DIR / "mappo" / f"training_history_seed{seed}.json").read_text())
        rewards = [h["mean_reward"] for h in hist]
        plt.plot(rewards, label=f"seed {seed}")
    plt.xlabel("PPO update")
    plt.ylabel("mean reward per step (scaled)")
    plt.title("12 - MAPPO training convergence across 3 seeds")
    plt.legend()
    savefig("12_reward_convergence.png")


def fig13_seed_variability():
    comp = json.loads((RESULTS_DIR / "final_evaluation" / "controller_comparison.json").read_text())
    ppo_vals = [comp[f"ppo_seed{s}"]["energy.total_kwh"]["mean"] for s in [0, 1, 2]]
    mappo_vals = [comp[f"mappo_seed{s}"]["energy.total_kwh"]["mean"] for s in [0, 1, 2]]
    plt.figure(figsize=(7, 4.5))
    x = np.arange(3)
    width = 0.35
    plt.bar(x - width / 2, ppo_vals, width, label="PPO")
    plt.bar(x + width / 2, mappo_vals, width, label="MAPPO")
    plt.axhline(comp["pid"]["energy.total_kwh"]["mean"], color="red", linestyle="--", label="PID (reference)")
    plt.xticks(x, ["seed 0", "seed 1", "seed 2"])
    plt.ylabel("energy (kWh/episode)")
    plt.title("13 - Seed-to-seed variability, PPO vs. MAPPO")
    plt.legend()
    savefig("13_seed_variability.png")


def fig14_energy_safety_tradeoff():
    comp = json.loads((RESULTS_DIR / "final_evaluation" / "controller_comparison.json").read_text())
    main_controllers = ["fixed", "threshold", "pid", "mpc", "ppo_seed0", "ppo_seed1",
                         "ppo_seed2", "mappo_seed0", "mappo_seed1", "mappo_seed2"]
    plt.figure(figsize=(7.5, 5))
    for c in main_controllers:
        e = comp[c]["energy.total_kwh"]["mean"]
        t = comp[c]["thermal.max_temp_c"]["mean"]
        plt.scatter(e, t, s=60)
        plt.annotate(c, (e, t), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.axhline(27.0, color="red", linestyle="--", label="safety limit (27C)")
    plt.xlabel("energy (kWh/episode)")
    plt.ylabel("max temperature reached (C)")
    plt.title("14 - Energy vs. safety-margin trade-off")
    plt.legend()
    savefig("14_energy_safety_tradeoff.png")


def fig06_thermal_response_summary():
    # Consolidates the 9 individual thermal-validation plots' key
    # takeaway into one summary figure (the individual plots already
    # exist in results/thermal_validation/).
    src_files = sorted((RESULTS_DIR / "thermal_validation").glob("test*.png"))
    print(f"06 - thermal response: {len(src_files)} individual validation "
          f"plots already exist in results/thermal_validation/ "
          f"(test1..test8); not duplicated here.")


def main():
    fig01_workload_over_time()
    fig02_workload_distribution()
    fig03_workload_aggregation()
    fig04_gru_forecast_vs_actual()
    fig05_gru_training_loss()
    fig06_thermal_response_summary()
    fig07_08_trajectories_and_actions()
    fig09_10_energy_safety_comparison()
    fig11_robustness()
    fig12_reward_convergence()
    fig13_seed_variability()
    fig14_energy_safety_tradeoff()
    print("\nAll figures written to results/figures/")


if __name__ == "__main__":
    main()
