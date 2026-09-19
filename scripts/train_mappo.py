"""Phase 11: train MAPPO on the training split; track validation
performance for model selection (never touches test data)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.multi_agent_env import MultiAgentCoolingEnv  # noqa: E402
from environment.cooling_core import load_config  # noqa: E402
from marl.mappo import MAPPOTrainer  # noqa: E402
from evaluation.run_controller import fixed_test_start_indices  # noqa: E402
from controllers.rl_adapters import MAPPOControllerAdapter  # noqa: E402


def evaluate_on_val(trainer, cfg, n_episodes=5):
    from evaluation.run_controller import run_controller_episodes
    starts = fixed_test_start_indices("val", n_episodes=n_episodes, cfg=cfg)
    adapter = MAPPOControllerAdapter(trainer)
    res = run_controller_episodes(adapter, "val", starts, cfg=cfg)
    return res["aggregated"]


def main(seed: int = 0, total_updates: int | None = None):
    cfg = load_config()
    mcfg = cfg["rl"]["mappo"]
    total_updates = total_updates or mcfg["total_updates"]

    env = MultiAgentCoolingEnv(split="train")
    trainer = MAPPOTrainer(env, cfg, seed=seed)

    history = []
    for update in range(total_updates):
        buf = trainer.collect_rollout(mcfg["rollout_length"], seed=seed * 10000 + update)
        stats = trainer.update(buf, mcfg["n_epochs"], mcfg["minibatch_size"])
        mean_reward = float(np.mean(buf["rewards"]))
        history.append({"update": update, "mean_reward": mean_reward, **stats})
        if update % 20 == 0 or update == total_updates - 1:
            val_metrics = evaluate_on_val(trainer, cfg, n_episodes=3)
            print(f"update {update}/{total_updates} train_mean_reward={mean_reward:.4f} "
                  f"val_energy={val_metrics['energy.total_kwh']['mean']:.3f} "
                  f"val_viol_pct={val_metrics['thermal.violation_pct']['mean']:.2f}")

    models_dir = REPO_ROOT / "models" / "mappo"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.actor.state_dict(), models_dir / f"actor_seed{seed}.pt")
    torch.save(trainer.critic.state_dict(), models_dir / f"critic_seed{seed}.pt")

    results_dir = REPO_ROOT / "results" / "mappo"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"training_history_seed{seed}.json").write_text(json.dumps(history))

    return trainer, history


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed=seed)
