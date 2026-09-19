"""V2 (pilot): evaluate every controller -- Fixed, Threshold, PID,
PPO(smoke-test), MAPPO(smoke-test) -- on the SAME 20 fixed TEST-split
episodes, via V2CoolingCore. This mirrors V1's final_evaluation.py
structure but is explicitly a PILOT/SMOKE-TEST run (results/pilot/README.md):
the RL policies here were trained for a tiny fraction of V1's already-
reduced budget, purely to prove the V2 pipeline (official Google data ->
GRU forecast -> fitted power model -> thermal sim -> safety shield ->
controller) executes correctly end-to-end. Do not compare these RL
numbers against the classical numbers as a research conclusion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.v2_cooling_core import V2CoolingCore, load_v2_config  # noqa: E402
from environment.multi_agent_env import MultiAgentCoolingEnv  # noqa: E402
from controllers.classical import FixedController, ThresholdController, PIDController  # noqa: E402
from controllers.rl_adapters import PPOControllerAdapter, MAPPOControllerAdapter  # noqa: E402
from marl.mappo import MAPPOTrainer  # noqa: E402
from evaluation.run_controller import run_controller_episodes, fixed_test_start_indices  # noqa: E402
from evaluation.v2_common import results_dir, models_dir, disclaimer  # noqa: E402

N_EPISODES = 20


def main():
    cfg = load_v2_config()
    assert cfg["data"]["active_cell_set"] == "pilot", (
        "v2_final_evaluation_pilot.py is a smoke-test script by design and "
        "refuses to run against active_cell_set='final' -- write/extend a "
        "proper final-evaluation script for that (see "
        "docs/final_pipeline_readiness_audit.md)."
    )
    ccfg = cfg["controllers"]

    ref_core = V2CoolingCore(split="test", cfg=cfg)
    n_zones = ref_core.n_zones
    start_indices = [int(i) for i in fixed_test_start_indices("test", N_EPISODES, cfg=cfg, core=ref_core)]

    all_results = {}

    classical = {
        "fixed": FixedController(ccfg["fixed"]["action"], n_zones=n_zones),
        "threshold": ThresholdController(**ccfg["threshold"], n_zones=n_zones),
        "pid": PIDController(**ccfg["pid"], n_zones=n_zones),
    }
    for name, controller in classical.items():
        core = V2CoolingCore(split="test", cfg=cfg)
        result = run_controller_episodes(controller, "test", start_indices, cfg=cfg, core=core)
        all_results[name] = result["aggregated"]

    ppo_path = models_dir(cfg, "ppo", "ppo_seed0.zip")
    if ppo_path.exists():
        model = PPO.load(ppo_path)
        core = V2CoolingCore(split="test", cfg=cfg)
        adapter = PPOControllerAdapter(model, n_zones=n_zones, horizon=core.horizon)
        result = run_controller_episodes(adapter, "test", start_indices, cfg=cfg, core=core)
        all_results["ppo_pilot_smoketest"] = result["aggregated"]

    mappo_actor_path = models_dir(cfg, "mappo", "actor_seed0.pt")
    if mappo_actor_path.exists():
        core = V2CoolingCore(split="test", cfg=cfg)
        env = MultiAgentCoolingEnv(core=core)
        trainer = MAPPOTrainer(env, cfg, seed=0)
        trainer.actor.load_state_dict(torch.load(mappo_actor_path))
        adapter = MAPPOControllerAdapter(trainer)
        core2 = V2CoolingCore(split="test", cfg=cfg)
        env2 = MultiAgentCoolingEnv(core=core2)
        trainer.env = env2
        result = run_controller_episodes(adapter, "test", start_indices, cfg=cfg, core=core2)
        all_results["mappo_pilot_smoketest"] = result["aggregated"]

    print(f"{'controller':<24} {'energy_kwh':>12} {'violation_%':>12} {'max_temp_c':>12}")
    for name, agg in all_results.items():
        print(f"{name:<24} {agg['energy.total_kwh']['mean']:>12.2f} "
              f"{agg['thermal.violation_pct']['mean']:>12.2f} "
              f"{agg['thermal.max_temp_c']['mean']:>12.2f}")

    out_dir = results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pilot_controller_comparison_all.json", "w") as f:
        json.dump(
            {
                "disclaimer": (
                    disclaimer(cfg) + " PPO/MAPPO trained for a smoke-test-scale "
                    "budget (8640 timesteps / 30 updates vs V1's already-reduced "
                    "100000 timesteps / 300 updates) purely to validate the V2 "
                    "pipeline executes correctly end-to-end. NOT a real "
                    "controller comparison."
                ),
                "n_episodes": N_EPISODES,
                "start_indices": start_indices,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_dir / 'pilot_controller_comparison_all.json'}")


if __name__ == "__main__":
    main()
