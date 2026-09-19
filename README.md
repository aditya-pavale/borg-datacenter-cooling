# Borg-MARL Datacenter Cooling

Deep-learning workload forecasting and multi-agent reinforcement
learning for energy-efficient datacenter cooling, using the Google
Borg cluster-trace export on Kaggle
(`muzairbair/borg-traces-data`).

**Full result, up front**: under this project's CPU-only compute
budget, a classical PID controller outperformed both single-agent PPO
and a from-scratch MAPPO implementation on energy efficiency, while
all controllers remained safe. This is the genuine, unmanipulated
finding — see `docs/final_validation_report.md` §12 for the numbers
and §20 for the full discussion. The project's value is in the
rigorously-built, tested, and debugged pipeline and the honest
comparison, not in a predetermined "MARL wins" narrative.

## Problem
See `docs/problem_statement.md`. In short: can coordinated MARL
(MAPPO), with causal short-term workload forecasting and a coupled
3-zone thermal model, out-perform classical/model-based control on the
energy/safety trade-off for datacenter cooling, using a real (though
sparse and under-documented) Borg workload trace?

## Dataset
`docs/dataset_audit.md` (+ an independent adversarial re-verification,
logged in its own §20). Key finding: the Kaggle export has **no
uploader-provided documentation**, and is far sparser than a full
periodic-telemetry table would be — both facts materially shaped the
project's design (`docs/data_preprocessing.md`).

## Pipeline
```
data/raw/borg_traces_data.csv (immutable, integrity-checked)
  -> src/data/clean_borg.py          -> data/processed/borg_cleaned.parquet
  -> src/data/aggregate_workload.py  -> data/processed/workload_timeseries.parquet
  -> src/data/split_workload.py      -> data/processed/{train,val,test}.parquet
  -> src/forecasting/train_forecaster.py -> models/forecasting/
  -> scripts/precompute_forecasts.py -> data/processed/forecast_{split}.npy
  -> scripts/validate_thermal_model.py -> results/thermal_validation/
  -> scripts/train_mappo.py / train_ppo.py (x3 seeds each) -> models/{mappo,ppo}/
  -> scripts/final_evaluation.py     -> results/final_evaluation/
  -> scripts/robustness_experiments.py -> results/robustness/
  -> scripts/leakage_audit.py        -> docs/leakage_audit.md
```

## Forecasting
GRU vs. persistence/moving-average baselines, 30-minute horizon (2 x
15-minute steps) from 4 hours of history. `docs/forecasting_design.md`.

## Thermal model
3-zone, coupled, lumped-parameter, numerically validated (9/9 tests
pass), never claimed to be physically validated (no real thermal data
exists anywhere in the source dataset). `docs/thermal_model.md`.

## Multi-agent RL
Custom MAPPO (CTDE, parameter-shared actor, centralized critic),
implemented from scratch (not SB3-PPO relabeled). A genuine PPO
correctness bug was found and fixed during development — see
`docs/experiment_log.md`. `docs/marl_design.md`.

## Safety
A model-based, closed-form, one-step-ahead safety shield, with
documented, tested limitations (it cannot always fully restore the
margin in one step under strong thermal inertia). `docs/safety_design.md`.

## Results
`docs/final_validation_report.md` is the single entry point for all
final numbers, ablations, robustness results, and conclusions.

## Reproduce everything

`data/raw/` is gitignored (raw data is never committed). Before
anything else, download the dataset:
```bash
kaggle datasets download -d muzairbair/borg-traces-data -p data/raw --unzip
```
This requires a Kaggle API token (`~/.kaggle/kaggle.json`); see the
[Kaggle API docs](https://github.com/Kaggle/kaggle-api) if you don't
have one. Then:
```bash
cd borg-datacenter-cooling
.venv/bin/pip install -r requirements.txt

.venv/bin/python src/data/verify_raw_integrity.py    # confirms your download matches docs/raw_data_manifest.json
                                                       # (that manifest is already committed; pass --init instead,
                                                       # only if it does not exist, e.g. a from-scratch setup)
.venv/bin/python src/data/inspect_borg.py            # regenerates docs/dataset_summary.json

.venv/bin/python src/data/clean_borg.py
.venv/bin/python src/data/aggregate_workload.py
.venv/bin/python src/data/split_workload.py

.venv/bin/python src/forecasting/train_forecaster.py
.venv/bin/python scripts/precompute_forecasts.py

.venv/bin/python scripts/validate_thermal_model.py

for s in 0 1 2; do .venv/bin/python scripts/train_mappo.py $s; done
for s in 0 1 2; do .venv/bin/python scripts/train_ppo.py $s; done

.venv/bin/python scripts/final_evaluation.py
.venv/bin/python scripts/robustness_experiments.py
.venv/bin/python scripts/leakage_audit.py

.venv/bin/python -m pytest tests/ -v
```

**Reproducibility caveat** (disclosed, not hidden): exact RL numbers
vary slightly run-to-run despite fixed seeds, a known consequence of
PyTorch CPU non-bit-exactness. The qualitative finding (classical PID
beats compute-budget-limited MAPPO/PPO on energy; all controllers stay
safe) was stable across two independent full pipeline runs — see
`docs/final_validation_report.md` §16.

## Limitations
`docs/assumptions_and_limitations.md` — read this before citing any
result. No real thermal/power measurements exist in this project; the
3 zones are a logical, not physical, abstraction; reward- and
thermal-sensitivity experiments were targeted rather than exhaustive;
this project used 3 RL seeds (not 5) and reduced training budgets
(300 MAPPO updates, 100k PPO timesteps), all on CPU only.

## Project structure
```
data/{raw,processed}/   docs/                     src/{data,forecasting,thermal,
models/{forecasting,      README.md                 environment,controllers,marl,
  mappo,ppo}/              requirements.txt          rl,evaluation}/
results/                 pyproject.toml            scripts/
tests/                    configs/config.yaml
```
`notebooks/` contains 6 executed, error-free notebooks
(01_dataset_analysis, 02_workload_forecasting, 03_thermal_validation,
04_controller_comparison, 05_marl_analysis, 06_final_results) that
load and visualize the artifacts in `results/` and `docs/` — none of
them retrain a model. Regenerate with:
```bash
.venv/bin/python scripts/generate_figures.py
.venv/bin/python scripts/generate_tables.py
.venv/bin/python scripts/build_notebooks.py
```
