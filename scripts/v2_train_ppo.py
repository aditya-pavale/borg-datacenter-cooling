"""V2 Phase 11 (pilot): single-agent PPO on V2CoolingCore (official
Google data, cells a-d). Reuses SingleAgentCoolingEnv unchanged via its
`core=` injection point.

PILOT-SCALE BUDGET, reduced further than V1's already CPU-reduced
budget (V1: 100k timesteps) to fit this session's time budget purely
for END-TO-END PIPELINE VALIDATION -- this is a smoke test that the
V2 PPO training loop runs correctly against real official-Google-derived
data, NOT a real controller comparison. A real comparison needs V1's
full budget (or larger) and 5 seeds, run once the full 8-cell dataset
and denser thermal recalibration (docs/version2_research_design.md) are
in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from environment.single_agent_env import SingleAgentCoolingEnv  # noqa: E402
from environment.v2_cooling_core import V2CoolingCore, load_v2_config  # noqa: E402
from evaluation.v2_common import models_dir as get_models_dir  # noqa: E402

PILOT_TIMESTEPS = 8640  # 30 episodes worth (288 steps/ep) -- smoke-test scale


def main(seed: int = 0, total_timesteps: int | None = None):
    cfg = load_v2_config()
    pcfg = cfg["rl"]["ppo"]
    total_timesteps = total_timesteps or PILOT_TIMESTEPS

    core = V2CoolingCore(split="train", cfg=cfg)
    env = Monitor(SingleAgentCoolingEnv(core=core))

    model = PPO(
        "MlpPolicy", env, verbose=0, seed=seed,
        n_steps=pcfg["n_steps"], batch_size=pcfg["batch_size"],
        n_epochs=pcfg["n_epochs"], learning_rate=pcfg["learning_rate"],
        gamma=pcfg["gamma"],
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    m_dir = get_models_dir(cfg, "ppo")
    m_dir.mkdir(parents=True, exist_ok=True)
    model.save(m_dir / f"ppo_seed{seed}")
    print(f"[{cfg['data']['active_cell_set'].upper()}] seed {seed}: saved PPO model to "
          f"{m_dir / f'ppo_seed{seed}'}.zip ({total_timesteps} timesteps)")
    return model


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed=seed)
