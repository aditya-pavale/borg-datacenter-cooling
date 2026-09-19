"""V2 (pilot): safety shield ON vs OFF ablation, for every classical
controller, on the same 20 fixed test episodes. Mirrors V1's ablation F
(docs/experiment_plan.md) but is the first time this comparison is run
on official-Google-derived data. Reports intervention counts alongside
energy/violation so a policy's reliance on the shield (vs. genuine
proactive safety) is visible, per docs/safety_design.md's discipline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.v2_cooling_core import V2CoolingCore, load_v2_config  # noqa: E402
from controllers.classical import FixedController, ThresholdController, PIDController  # noqa: E402
from evaluation.run_controller import run_controller_episodes, fixed_test_start_indices  # noqa: E402

N_EPISODES = 20
RESULTS_DIR = REPO_ROOT / "results" / "pilot" / "safety_ablation"


def make_controllers(cfg, n_zones):
    ccfg = cfg["controllers"]
    return {
        "fixed": FixedController(ccfg["fixed"]["action"], n_zones=n_zones),
        "threshold": ThresholdController(**ccfg["threshold"], n_zones=n_zones),
        "pid": PIDController(**ccfg["pid"], n_zones=n_zones),
    }


def main():
    cfg = load_v2_config()
    n_zones = cfg["data"]["n_zones"]

    ref_core = V2CoolingCore(split="test", cfg=cfg)
    start_indices = [int(i) for i in fixed_test_start_indices("test", N_EPISODES, cfg=cfg, core=ref_core)]

    all_results = {}
    for shield_state in [True, False]:
        controllers = make_controllers(cfg, n_zones)
        for name, controller in controllers.items():
            core = V2CoolingCore(split="test", cfg=cfg, use_safety_shield=shield_state)
            result = run_controller_episodes(controller, "test", start_indices, cfg=cfg, core=core)
            key = f"{name}__shield_{'on' if shield_state else 'off'}"
            all_results[key] = result["aggregated"]
            agg = result["aggregated"]
            print(f"{key:<28} energy={agg['energy.total_kwh']['mean']:.2f} kWh  "
                  f"violation={agg['thermal.violation_pct']['mean']:.2f}%  "
                  f"max_temp={agg['thermal.max_temp_c']['mean']:.2f}C  "
                  f"shield_interventions={agg['safety.n_interventions']['mean']:.1f}")

    # Reliance check (docs/safety_design.md discipline): for each
    # controller, does removing the shield make violations dramatically
    # worse? A controller that is "safe" only because of constant shield
    # correction is flagged, not hidden.
    reliance = {}
    for name in ["fixed", "threshold", "pid"]:
        on = all_results[f"{name}__shield_on"]["thermal.violation_pct"]["mean"]
        off = all_results[f"{name}__shield_off"]["thermal.violation_pct"]["mean"]
        reliance[name] = {
            "violation_pct_shield_on": on,
            "violation_pct_shield_off": off,
            "delta": off - on,
            "shield_dependent": bool(off - on > 10.0),  # >10pp worse without shield
        }
        print(f"{name}: shield_on={on:.2f}% shield_off={off:.2f}% delta={off-on:+.2f}pp "
              f"shield_dependent={reliance[name]['shield_dependent']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "shield_ablation.json", "w") as f:
        json.dump(
            {
                "pilot_disclaimer": "Cells a-d only -- see results/pilot/README.md. NOT final.",
                "n_episodes": N_EPISODES,
                "results": all_results,
                "reliance_analysis": reliance,
            },
            f,
            indent=2,
        )
    print(f"Wrote {RESULTS_DIR / 'shield_ablation.json'}")


if __name__ == "__main__":
    main()
