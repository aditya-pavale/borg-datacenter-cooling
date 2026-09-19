# Experiment Plan — Phase 27

| # | Experiment | Script | Status |
|---|---|---|---|
| A | Unseen normal Borg test workload | `scripts/final_evaluation.py` | Done — all controllers on the 20 fixed test episodes |
| B | Sustained workload spike (SYNTHETIC, labeled) | `scripts/robustness_experiments.py` | Done — 1x/2x/5x/10x workload scaling |
| C | Forecast-error robustness (SYNTHETIC, labeled) | `scripts/robustness_experiments.py` | Done — Gaussian noise (std 0/0.01/0.05/0.1) added to the forecast at eval time |
| D | No forecast vs. forecast | `scripts/final_evaluation.py` (`mappo_seed0_forecast_ablation_*`) | Done |
| E | Single-agent PPO vs. MAPPO | `scripts/final_evaluation.py` | Done |
| F | No safety shield vs. safety shield | `scripts/final_evaluation.py` (`mappo_seedX_shield_*`) | Done |
| G | Reward sensitivity | — | **Partial**: the shield-intervention-penalty and proximity-weight changes documented in `docs/experiment_log.md` constitute an ad hoc reward sensitivity study (their effect on trained-policy behavior is directly reported); a systematic grid sweep was not run, given the compute budget already spent diagnosing and fixing two correctness bugs (see below) |
| H | Thermal parameter sensitivity | `scripts/validate_thermal_model.py` (indirectly) + the `thermal_mass_kwh_per_c` sweep in `docs/experiment_log.md` | **Partial**: the recalibration search over `{2.0, 0.5, 0.3, 0.15}` is a real, reported sensitivity result (it directly determines whether the safety constraint is ever binding); a full multi-parameter sweep was not run |
| I | Workload amplitude scaling | `scripts/robustness_experiments.py` (same mechanism as B) | Done |
| J | Ambient-temperature perturbation | `scripts/validate_thermal_model.py` Test 8 | Done (as part of thermal validation; not repeated for every controller, given time budget) |
| K | Multiple RL seeds | `scripts/train_mappo.py` / `scripts/train_ppo.py`, 3 seeds each | Done (3, not 5 — documented compute-budget reduction) |

## Honest scope note
Given the CPU-only, single-session compute budget for this autonomous
build, two experiments (G, H) were completed as targeted,
evidence-driven investigations rather than exhaustive grid sweeps —
this is disclosed here explicitly rather than presented as a full
ablation matrix. Everything else in this table ran to completion
against the real, held-out test split.

## Parameter-sharing vs. separate policies (optional ablation, Section 31G)
Not run as a separate ablation given time budget; the parameter-sharing
design choice is justified analytically in `docs/marl_design.md` rather
than empirically compared against separate per-zone policies. This is
a known gap, listed in `docs/assumptions_and_limitations.md`.
