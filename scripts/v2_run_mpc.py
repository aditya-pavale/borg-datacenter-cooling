"""V2 (pilot): evaluate the random-shooting MPC controller
(src/controllers/v2_mpc.py) on the same 20 fixed test episodes as the
other classical controllers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.v2_cooling_core import V2CoolingCore, load_v2_config  # noqa: E402
from controllers.v2_mpc import V2MPCController  # noqa: E402
from evaluation.run_controller import run_controller_episodes, fixed_test_start_indices  # noqa: E402
from evaluation.v2_common import results_dir, disclaimer  # noqa: E402

N_EPISODES = 20


def main():
    cfg = load_v2_config()
    mcfg = cfg["controllers"]["mpc"]

    core = V2CoolingCore(split="test", cfg=cfg)
    n_zones = core.n_zones
    start_indices = [int(i) for i in fixed_test_start_indices("test", N_EPISODES, cfg=cfg, core=core)]

    controller = V2MPCController(
        core=core, reward_cfg=cfg["reward"], horizon_steps=mcfg["horizon_steps"],
        n_candidates=mcfg["n_candidates"], n_zones=n_zones, seed=0,
    )
    result = run_controller_episodes(controller, "test", start_indices, cfg=cfg, core=core)
    agg = result["aggregated"]
    print(f"mpc: energy={agg['energy.total_kwh']['mean']:.2f} kWh/ep, "
          f"violation={agg['thermal.violation_pct']['mean']:.2f}%, "
          f"max_temp={agg['thermal.max_temp_c']['mean']:.2f}C")

    out_dir = results_dir(cfg, "controllers")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mpc_result.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "disclaimer": disclaimer(cfg),
                "n_episodes": N_EPISODES,
                "result": result,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
