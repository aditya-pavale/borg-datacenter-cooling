"""V2 (pilot): robustness experiments B (synthetic workload spike) and C
(synthetic forecast noise), mirroring V1's robustness_experiments.py
pattern exactly (mutate core.workload/core.forecast post-construction).
Classical controllers only for now (PID, threshold) -- PPO/MAPPO are
still smoke-test-scale (results/pilot/README.md) and not meaningful
robustness subjects yet; re-run this with real-budget RL policies once
those exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.v2_cooling_core import V2CoolingCore, load_v2_config  # noqa: E402
from controllers.classical import PIDController, ThresholdController  # noqa: E402
from evaluation.metrics import compute_episode_metrics, aggregate_across_episodes  # noqa: E402
from evaluation.run_controller import fixed_test_start_indices  # noqa: E402
from evaluation.v2_common import results_dir, disclaimer  # noqa: E402


def run_with_workload_scale(controller, cfg, start_indices, scale):
    dt_hours = cfg["thermal"]["dt_seconds"] / 3600.0
    safety_limit = cfg["thermal"]["safety_limit_c"]
    core = V2CoolingCore(split="test", cfg=cfg)
    core.workload = np.clip(core.workload * scale, 0.0, None)
    # NOTE: unlike V1, scaling "workload" here does not change heat --
    # heat comes from the precomputed predicted_power_util column
    # (the fitted power model's output), not from core.workload
    # directly. So this experiment is scaled consistently for both:
    core.predicted_power_util = np.clip(core.predicted_power_util * scale, 0.0, 1.5)
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


def run_with_forecast_noise(controller, cfg, start_indices, noise_std):
    dt_hours = cfg["thermal"]["dt_seconds"] / 3600.0
    safety_limit = cfg["thermal"]["safety_limit_c"]
    rng = np.random.default_rng(123)
    core = V2CoolingCore(split="test", cfg=cfg)
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
    cfg = load_v2_config()
    core = V2CoolingCore(split="test", cfg=cfg)
    n_zones = core.n_zones
    starts = [int(i) for i in fixed_test_start_indices("test", n_episodes=15, cfg=cfg, core=core)]

    controllers = {
        "pid": PIDController(**cfg["controllers"]["pid"], n_zones=n_zones),
        "threshold": ThresholdController(**cfg["controllers"]["threshold"], n_zones=n_zones),
    }

    results = {"experiment_B_workload_spike": {}, "experiment_C_forecast_error": {}}

    print("=== V2 pilot: Experiment B: SYNTHETIC sustained workload spike ===")
    for scale in [1.0, 2.0, 5.0, 10.0]:
        results["experiment_B_workload_spike"][f"scale_{scale}"] = {}
        for name, ctrl in controllers.items():
            agg = run_with_workload_scale(ctrl, cfg, starts, scale)
            results["experiment_B_workload_spike"][f"scale_{scale}"][name] = {
                "energy": agg["energy.total_kwh"]["mean"],
                "max_temp": agg["thermal.max_temp_c"]["mean"],
                "viol_pct": agg["thermal.violation_pct"]["mean"],
            }
            print(f"scale={scale} {name}: energy={agg['energy.total_kwh']['mean']:.2f} "
                  f"viol_pct={agg['thermal.violation_pct']['mean']:.2f} "
                  f"max_temp={agg['thermal.max_temp_c']['mean']:.2f}")

    print("\n=== V2 pilot: Experiment C: SYNTHETIC forecast-error robustness ===")
    for noise_std in [0.0, 0.01, 0.05, 0.1]:
        results["experiment_C_forecast_error"][f"noise_{noise_std}"] = {}
        for name, ctrl in controllers.items():
            agg = run_with_forecast_noise(ctrl, cfg, starts, noise_std)
            results["experiment_C_forecast_error"][f"noise_{noise_std}"][name] = {
                "energy": agg["energy.total_kwh"]["mean"],
                "viol_pct": agg["thermal.violation_pct"]["mean"],
            }
            print(f"noise_std={noise_std} {name}: energy={agg['energy.total_kwh']['mean']:.2f} "
                  f"viol_pct={agg['thermal.violation_pct']['mean']:.2f}")

    out_dir = results_dir(cfg, "robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "disclaimer": disclaimer(cfg) + " Classical controllers only (RL still smoke-test).",
        **results,
    }
    (out_dir / "robustness_results.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_dir / 'robustness_results.json'}")
    return results


if __name__ == "__main__":
    main()
