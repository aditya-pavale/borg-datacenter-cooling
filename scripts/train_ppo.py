"""Phase 9/24: single-agent PPO baseline (Stable-Baselines3), on the
SAME CoolingCore physics as MAPPO, for a direct, fair comparison."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from environment.single_agent_env import SingleAgentCoolingEnv  # noqa: E402
from environment.cooling_core import load_config  # noqa: E402


def main(seed: int = 0, total_timesteps: int | None = None):
    cfg = load_config()
    pcfg = cfg["rl"]["ppo"]
    total_timesteps = total_timesteps or pcfg["total_timesteps"]

    env = Monitor(SingleAgentCoolingEnv(split="train"))

    model = PPO(
        "MlpPolicy", env, verbose=0, seed=seed,
        n_steps=pcfg["n_steps"], batch_size=pcfg["batch_size"],
        n_epochs=pcfg["n_epochs"], learning_rate=pcfg["learning_rate"],
        gamma=pcfg["gamma"],
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    models_dir = REPO_ROOT / "models" / "ppo"
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / f"ppo_seed{seed}")
    print(f"seed {seed}: saved PPO model to {models_dir / f'ppo_seed{seed}'}.zip")
    return model


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed=seed)
