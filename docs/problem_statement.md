# Problem Statement

## Research question
Can a coordinated multi-agent deep reinforcement learning controller
(MAPPO), using short-term deep-learning workload forecasts and a
coupled multi-zone thermal model, provide effective energy/safety
trade-offs for datacenter cooling under realistic and previously
unseen workload conditions — and how does it compare with single-agent
PPO, classical control (fixed, threshold, PID), and MPC?

This project does not assume any controller wins in advance; the
answer comes from the experiments (`docs/final_validation_report.md`).

## Why this problem matters
Datacenter cooling is a major energy cost. A purely reactive
controller (e.g. threshold, PID) responds only to the current
temperature; a forecast-aware, coordinated multi-agent controller can
in principle anticipate load changes and coordinate across thermally
coupled zones, at the cost of significant implementation and training
complexity. Whether that complexity is justified, under a real
(though sparse) workload trace and a documented simulation thermal
model, is the empirical question this project answers.

## System boundary
- **Real, external data**: the Borg workload trace (`docs/dataset_audit.md`).
- **Derived from real data, causal**: the per-zone workload time series
  (`docs/data_preprocessing.md`) and the GRU forecast
  (`docs/forecasting_design.md`).
- **Simulation, not real**: the CPU→power→heat model
  (`src/thermal/power_model.py`), the 3-zone thermal model
  (`docs/thermal_model.md`), and the zone-to-machine mapping. None of
  these are calibrated against any real measurement — none exist in
  this dataset.
- **Not modeled**: any LLM/agentic component in the real-time control
  loop (explicitly out of scope per the project brief).

## Non-goals
- Claiming physical validation of the thermal model.
- Claiming the Kaggle CSV is the complete, unmodified original Google
  Borg trace (it is not — see `docs/dataset_audit.md`).
- Achieving publication-grade RL results; this is a CPU-only, reduced-
  compute-budget research prototype, and every reduction is documented
  in `docs/experiment_log.md`.
