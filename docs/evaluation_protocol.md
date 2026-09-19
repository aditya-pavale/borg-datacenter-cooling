# Evaluation Protocol — Phase 28/29

## Fairness (Phase 28)
Every controller (fixed, threshold, PID, MPC, PPO, MAPPO) is evaluated
by `src/evaluation/run_controller.py` on:
- the **same** `CoolingCore` physics (`src/environment/cooling_core.py`)
- the **same** 20 fixed, evenly-spaced test-split episode start
  indices (`fixed_test_start_indices`, deterministic, not re-sampled
  per controller)
- the **same** episode length (96 steps = 1 day at 15-minute
  resolution), safety threshold (27°C), and reward/metric definitions
- the **same** safety-shield configuration, unless the shield itself
  is the experimental variable (Experiment F)

**Documented information asymmetry**: PID, Threshold, and Fixed use
only the current temperature/state; MPC and the RL policies
additionally receive the causal GRU forecast. This asymmetry is the
explicit subject of Experiment D (forecast ablation), not hidden.

## Metrics (Phase 29)
Computed identically for every controller by
`src/evaluation/metrics.py::compute_episode_metrics`, aggregated
(mean ± std across episodes) by `aggregate_across_episodes`:
- **Energy**: total kWh/episode, mean kW.
- **Thermal**: mean/max/min temp, variance, violation steps/%,
  cumulative violation severity, max overshoot.
- **Control**: mean/max action change (smoothness/chattering).
- **Safety**: shield intervention count and rate.
- **Forecasting** (separate pipeline): MAE, RMSE, R² —
  `docs/forecasting_design.md`.

## Statistics (Phase 30)
3 random seeds for PPO and MAPPO (reduced from the "prefer 5" guidance
— CPU-only compute budget, documented in `docs/experiment_log.md`);
mean ± std reported per seed and across seeds. No inferential
statistical test (e.g. t-test) is applied given the small number of
seeds — reporting mean/std/range only, avoiding overstated
significance claims per the master plan.

## What is frozen before final evaluation
- The GRU forecaster (`models/forecasting/gru_forecaster.pt`) and its
  scaler are frozen after Phase 4; the same frozen forecasts
  (`data/processed/forecast_test.npy`) are used by every controller
  evaluated on the test split.
- Classical controller parameters (`configs/config.yaml:controllers`)
  are fixed values chosen by inspection/engineering judgement on
  train/val behavior, not tuned against the test split.
- PPO/MAPPO model weights are frozen after training (`models/ppo/`,
  `models/mappo/`) before `scripts/final_evaluation.py` runs.

## Reproduce
```bash
.venv/bin/python scripts/final_evaluation.py
.venv/bin/python scripts/robustness_experiments.py
```
