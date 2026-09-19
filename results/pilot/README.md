# Pilot / Development Results — NOT FINAL

Everything under `results/pilot/` was produced using **Borg cells a-d
only** (4 of the 8 official ClusterData2019/PowerData2019 cells).
Cells e-h are extracted from the same, already-validated BigQuery
pipeline once the free-tier monthly quota resets
(`docs/gcp_billing_blocker.md`) — this project intentionally stays on
BigQuery's free Sandbox tier rather than enabling billing.

## What pilot results are for
- Validating that the V2 data pipeline, forecasting, power model,
  thermal model, and control/RL code are implemented correctly.
- Finding implementation bugs, bad assumptions, and numerical issues
  early, on real (if partial) official data.
- Establishing reasonable hyperparameter ranges and architecture
  choices to carry into the final run.

## What pilot results are NOT
- **Not** final Version 2 performance numbers.
- **Not** a valid basis for "MAPPO wins" / "PID wins" / any comparative
  claim about the complete 8-cell research question.
- **Not** to be cited in `docs/final/` deliverables as final evidence.

## Required re-run after cells e-h land
Per the master plan's own discipline: final model selection, final
hyperparameters, final reward/thermal parameters, and final
train/validation/test evaluation must all be re-run on the complete
8-cell dataset before anything in `results/pilot/` is treated as
superseded-but-informative rather than authoritative. See
`results/model_selection_history.csv` for the full experiment log
carried from pilot into final.
