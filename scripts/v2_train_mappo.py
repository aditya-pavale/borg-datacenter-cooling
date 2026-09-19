"""V2 Phase 12 (pilot): CTDE MAPPO on V2CoolingCore (official Google
data, cells a-d, 4 agents = 4 cells). Reuses MultiAgentCoolingEnv and
MAPPOTrainer completely unchanged.

PILOT-SCALE BUDGET (30 updates, vs. V1's already-reduced 300) --
smoke-test for pipeline correctness on real official-Google data, NOT a
real controller comparison. See scripts/v2_train_ppo.py's docstring for
the same caveat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.multi_agent_env import MultiAgentCoolingEnv  # noqa: E402
from environment.v2_cooling_core import V2CoolingCore, load_v2_config  # noqa: E402
from marl.mappo import MAPPOTrainer  # noqa: E402
from evaluation.run_controller import fixed_test_start_indices, run_controller_episodes  # noqa: E402
from controllers.rl_adapters import MAPPOControllerAdapter  # noqa: E402

PILOT_UPDATES = 30


def evaluate_on_val(trainer, cfg, n_episodes=3):
    core = V2CoolingCore(split="val", cfg=cfg)
    starts = [int(i) for i in fixed_test_start_indices("val", n_episodes=n_episodes, cfg=cfg, core=core)]
    adapter = MAPPOControllerAdapter(trainer)
    core = V2CoolingCore(split="val", cfg=cfg)
    res = run_controller_episodes(adapter, "val", starts, cfg=cfg, core=core)
    return res["aggregated"]


def main(seed: int = 0, total_updates: int | None = None):
    cfg = load_v2_config()
    mcfg = cfg["rl"]["mappo"]
    total_updates = total_updates or PILOT_UPDATES

    core = V2CoolingCore(split="train", cfg=cfg)
    env = MultiAgentCoolingEnv(core=core)
    trainer = MAPPOTrainer(env, cfg, seed=seed)

    history = []
    for update in range(total_updates):
        buf = trainer.collect_rollout(mcfg["rollout_length"], seed=seed * 10000 + update)
        stats = trainer.update(buf, mcfg["n_epochs"], mcfg["minibatch_size"])
        mean_reward = float(np.mean(buf["rewards"]))
        history.append({"update": update, "mean_reward": mean_reward, **stats})
        if update % 10 == 0 or update == total_updates - 1:
            val_metrics = evaluate_on_val(trainer, cfg, n_episodes=3)
            print(f"[PILOT] update {update}/{total_updates} train_mean_reward={mean_reward:.4f} "
                  f"val_energy={val_metrics['energy.total_kwh']['mean']:.3f} "
                  f"val_viol_pct={val_metrics['thermal.violation_pct']['mean']:.2f}")

    models_dir = REPO_ROOT / "models" / "v2" / "mappo"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.actor.state_dict(), models_dir / f"actor_seed{seed}.pt")
    torch.save(trainer.critic.state_dict(), models_dir / f"critic_seed{seed}.pt")

    results_dir = REPO_ROOT / "results" / "pilot" / "mappo"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"training_history_seed{seed}.json").write_text(json.dumps(history))

    return trainer, history


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed=seed)
