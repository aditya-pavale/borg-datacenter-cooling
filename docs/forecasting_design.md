# Forecasting Design — Phase 4

## Task
Predict the next `horizon_steps=2` bins (30 minutes, at the empirically
chosen 15-minute resolution — see `docs/data_preprocessing.md`) of
per-zone `cpu_util` from the previous `lookback_steps=16` bins (4
hours).

## Models
- **Persistence**: repeats the last observed value.
- **Moving average**: mean of the last 4 observed bins, repeated.
- **GRU**: single-layer GRU (hidden size 32) + linear head producing
  all `horizon_steps` outputs at once (`src/forecasting/gru_model.py`).

## Causality / leakage prevention
- Windows are built separately per chronological split
  (`src/forecasting/dataset.py::make_windows`); a window's target
  indices are always strictly later than its input indices (tested in
  `tests/test_forecasting.py`).
- The min-max scaler is fit on **pooled training-split values only**
  (`train_forecaster.py`); validation/test data never influence it.
- Model selection (best epoch, i.e. early stopping) uses **validation**
  loss only. Test metrics are computed exactly once, after the model
  is frozen.

## Results (test split, untouched until final evaluation)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| GRU | 0.00742 | 0.01124 | -0.014 |
| Persistence | 0.00859 | 0.01578 | -1.000 |
| Moving average (w=4) | 0.00746 | 0.01230 | -0.214 |

Per-regime breakdown (low/high load split at the median of the last
observed value, "burst" = top decile) is saved in
`results/forecasting/metrics.json`.

**Honest interpretation:** the GRU beats both baselines on MAE and
RMSE, but its R² is close to zero (slightly negative), meaning it is
only marginally better than predicting the training-set mean. This is
consistent with the data-quality finding in
`docs/data_preprocessing.md`: ~30% of bins are exactly zero and the
non-zero values are driven by sporadic, event-triggered usage reports
rather than a smooth underlying process, so a large share of the
series' variance is close to irreducible at this level of temporal
aggregation with this specific (sparse, likely sub-sampled) Kaggle
export. This is reported as-is, per the project's research-integrity
requirement, rather than tuned to look better.

MAPE was not computed: with ~30% of true values exactly zero, MAPE is
undefined/explodes and is not a mathematically appropriate metric here
(master plan §16).

## Artifacts
- `models/forecasting/gru_forecaster.pt` — trained weights
- `models/forecasting/scaler.json` — train-only min/max
- `models/forecasting/config.json` — hyperparameters used
- `results/forecasting/training_history.json` — per-epoch train/val loss
- `results/forecasting/metrics.json` — overall + per-regime metrics
- `results/forecasting/forecast_examples.json` — sample true vs. predicted sequences

## Reproduce
```bash
.venv/bin/python src/forecasting/train_forecaster.py
```
