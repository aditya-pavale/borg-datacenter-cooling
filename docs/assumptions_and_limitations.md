# Assumptions and Limitations

## Real vs. simulated, summarized

| Component | Real / Derived / Simulated | Evidence |
|---|---|---|
| Raw workload trace | Real (Kaggle export, provenance uncertain — `docs/dataset_audit.md`) | `data/raw/borg_traces_data.csv` |
| Per-zone workload signal | Derived from real data, with a documented aggregation/binning/zone-mapping design | `docs/data_preprocessing.md` |
| GRU workload forecast | Derived (trained on real data, causal) | `docs/forecasting_design.md` |
| CPU→IT power model | Simulation assumption (illustrative parameters, not measured) | `src/thermal/power_model.py` |
| 3-zone thermal model | Simulation assumption (lumped-parameter, not calibrated; one parameter empirically recalibrated for non-vacuous safety, still an assumption) | `docs/thermal_model.md` |
| Zone-to-machine mapping | Simulation assumption (deterministic hash, logical not physical) | `docs/data_preprocessing.md` |
| Safety shield | Real algorithm (exact closed-form one-step correction), operating on the simulated thermal model | `docs/safety_design.md` |

## Absence of real cooling/thermal measurements
No temperature, power, humidity, airflow, or cooling-equipment
measurement exists anywhere in the Borg dataset. Every thermal and
power parameter in this project is an assumption, and no claim of
physical validation is made anywhere in this repository.

## Whether the 3 zones correspond to physical locations
No. `docs/dataset_audit.md` and `docs/data_preprocessing.md` establish
this explicitly: no rack/room/location field exists in the source
data. Zones are a logical simulation abstraction (deterministic hash
of `machine_id`).

## Dataset limitations carried through the whole project
- The Kaggle export's provenance and construction method are
  undocumented by its uploader (`docs/dataset_audit.md`).
- The workload signal is sparser than a full periodic-telemetry table
  would be, forcing a coarser-than-originally-planned (15-minute, not
  5-minute) control interval and an all-8-clusters pooling design
  (`docs/data_preprocessing.md`) — both documented departures from the
  originally preferred design, not silent simplifications.
- ~30% of (zone, bin) workload values are exactly zero (idle),
  reflecting genuine sparsity in the underlying export, not a
  processing artifact.

## Forecast uncertainty
The GRU forecaster's test-set R² is close to zero (slightly negative)
— it beats simple baselines on MAE/RMSE but does not explain much
variance in this specific dataset (`docs/forecasting_design.md`).
Downstream forecast-dependent components (MPC, the forecast-aware RL
observation) inherit this uncertainty; Experiment C
(`docs/experiment_plan.md`) probes sensitivity to added forecast noise
directly.

## Simulation-to-reality gap
This entire project is a numerically-validated (`docs/thermal_model.md`)
but physically-unvalidated simulation. Results describe behavior
within this simulation, not predictions about a real datacenter.

## Computational limitations
CPU-only, single machine, single session. Documented reductions
relative to the master plan's defaults:
- 3 RL seeds (not 5).
- MAPPO: 300 on-policy updates (~29k environment steps); PPO: 100k
  timesteps. Both are far below typical published RL training budgets.
- Reward sensitivity (G) and thermal sensitivity (H) experiments were
  targeted investigations, not full grid sweeps (`docs/experiment_plan.md`).
- Parameter-sharing vs. separate MAPPO policies was not empirically
  ablated (analytical justification only, `docs/marl_design.md`).
- MPC uses random-shooting optimization, not a gradient/CEM-based
  solver (`src/controllers/classical.py`), for implementation
  simplicity and compute budget.
- `notebooks/` (populated during the final handoff audit) contains 6
  notebooks that load and visualize already-verified artifacts; they
  do not retrain any model. See `docs/repository_inventory.md` for
  what each one covers.

## Known result, stated plainly
Under this project's specific configuration and compute budget, the
classical PID controller achieved the lowest energy use among all
evaluated controllers while remaining fully safe; both PPO and MAPPO,
correctly implemented and debugged (`docs/experiment_log.md`),
converged to safe but energy-inefficient policies given the available
training budget. This is reported as the genuine finding, not adjusted
to favor any particular method (master plan §39/§41).
