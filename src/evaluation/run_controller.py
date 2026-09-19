"""Runs a controller for N episodes on a given split and returns
per-episode + aggregated metrics. Used identically for classical
controllers, single-agent PPO, and MAPPO (Phase 28 fairness): same
CoolingCore, same episode length, same safety threshold, same metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.cooling_core import CoolingCore, load_config  # noqa: E402
from evaluation.metrics import compute_episode_metrics, aggregate_across_episodes  # noqa: E402


def run_controller_episodes(controller, split: str, start_indices: list[int],
                             use_forecast: bool = True, use_safety_shield: bool | None = None,
                             cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    dt_hours = cfg["thermal"]["dt_seconds"] / 3600.0
    safety_limit = cfg["thermal"]["safety_limit_c"]

    core = CoolingCore(split=split, use_forecast=use_forecast,
                        use_safety_shield=use_safety_shield, cfg=cfg)

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

    return {
        "per_episode": episode_metrics,
        "aggregated": aggregate_across_episodes(episode_metrics),
    }


def fixed_test_start_indices(split: str, n_episodes: int, cfg: dict | None = None) -> list[int]:
    """Deterministic, evenly-spaced episode start indices for a split --
    identical for every controller evaluated on that split (fairness)."""
    cfg = cfg or load_config()
    core = CoolingCore(split=split, cfg=cfg)
    lookback = cfg["forecasting"]["lookback_steps"]
    min_start = lookback - 1
    max_start = core.n_bins - core.episode_length - core.horizon - 1
    return list(np.linspace(min_start, max_start, n_episodes, dtype=int))
