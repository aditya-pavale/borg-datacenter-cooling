"""V2 Phase 11 (pilot): evaluate Fixed, Threshold, and PID controllers
against V2CoolingCore (official Google data, cells a-d, 20 fixed
episodes on the TEST split). MPC is handled separately
(scripts/v2_run_mpc.py) since its forecast-driven heat rollout needs a
V2-specific adaptation (see that script's docstring).

Same fairness discipline as V1 (src/evaluation/run_controller.py):
identical core, identical episode start indices, identical metrics for
every controller.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.v2_cooling_core import V2CoolingCore, load_v2_config  # noqa: E402
from controllers.classical import FixedController, ThresholdController, PIDController  # noqa: E402
from evaluation.run_controller import run_controller_episodes, fixed_test_start_indices  # noqa: E402

N_EPISODES = 20
RESULTS_DIR = REPO_ROOT / "results" / "pilot" / "controllers"


def main():
    cfg = load_v2_config()
    n_zones = cfg["data"]["n_zones"]
    ccfg = cfg["controllers"]

    core = V2CoolingCore(split="test", cfg=cfg)
    start_indices = [int(i) for i in fixed_test_start_indices("test", N_EPISODES, cfg=cfg, core=core)]

    controllers = {
        "fixed": FixedController(ccfg["fixed"]["action"], n_zones=n_zones),
        "threshold": ThresholdController(**ccfg["threshold"], n_zones=n_zones),
        "pid": PIDController(**ccfg["pid"], n_zones=n_zones),
    }

    all_results = {}
    for name, controller in controllers.items():
        core = V2CoolingCore(split="test", cfg=cfg)  # fresh core per controller (no state leakage)
        result = run_controller_episodes(controller, "test", start_indices, cfg=cfg, core=core)
        agg = result["aggregated"]
        print(f"{name}: energy={agg['energy.total_kwh']['mean']:.2f} kWh/ep, "
              f"violation={agg['thermal.violation_pct']['mean']:.2f}%, "
              f"max_temp={agg['thermal.max_temp_c']['mean']:.2f}C")
        all_results[name] = result

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "classical_controller_comparison.json", "w") as f:
        json.dump(
            {
                "pilot_disclaimer": "Cells a-d only -- see results/pilot/README.md. NOT final.",
                "n_episodes": N_EPISODES,
                "start_indices": start_indices,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {RESULTS_DIR / 'classical_controller_comparison.json'}")


if __name__ == "__main__":
    main()
