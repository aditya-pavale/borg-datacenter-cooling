# Borg-MARL Datacenter Cooling

Deep-learning workload forecasting and multi-agent reinforcement
learning for energy-efficient datacenter cooling.

## Version 2 — in progress (this branch: `v2-official-google-data`)

V2 rebuilds the entire pipeline below on **official Google 2019 traces**
(ClusterData2019 + PowerData2019, via BigQuery) instead of the
undocumented Kaggle re-export V1 used. It is **not finished**: cells
A–D are fully piloted, cells E–H are extracted for power/machine data
but blocked on workload extraction by BigQuery Sandbox's free-tier
monthly scan quota (a deliberate decision to stay off billing, not a
data problem — see `docs/gcp_billing_blocker.md`). Nothing below in
this section is a final result.

| Area | Pilot (cells A–D) finding |
|---|---|
| Data alignment | Borg **cell** is the finest scientifically defensible workload↔power join key — verified against Google's own docs and schema, no machine-level mapping invented. `docs/official_data_alignment_audit.md` |
| Forecasting | GRU test R² ≈ 0.824 vs. persistence 0.772 / moving-average 0.787 — a genuinely learnable signal, unlike V1's near-zero R² on sparse Kaggle data. `results/pilot/forecasting/metrics.json` |
| Power model | First time this project has REAL power data: Random Forest validation R² ≈ 0.780 against official PowerData2019 measurements (V1 had none). `results/pilot/power_model/metrics.json` |
| Thermal model | Recalibrated (`heat_scale_kw = 0.85`) on a dense 251-window sweep after finding and fixing a real calibration bug (the safety shield was silently confounding the first sweep). `docs/version2_research_design.md` |
| Safety | Shield ablation shows PID's shield-on/off results are identical — not because PID is proactively safe, but because its cooling is already saturated. `results/pilot/safety_ablation/shield_ablation.json` |
| RL | PPO/MAPPO pipelines run correctly end-to-end (smoke-test budgets only, far below a real training budget) — **not** a final controller comparison. |
| Engineering | 62/62 tests passing, including a config-consistency suite added after a readiness audit caught a real pilot/final artifact-isolation bug. `docs/final_pipeline_readiness_audit.md` |

Full documentation: `docs/official_data_alignment_audit.md`,
`docs/version2_research_design.md`, `docs/gcp_billing_blocker.md`,
`docs/storage_manifest.md`, `docs/final_pipeline_readiness_audit.md`,
`results/model_selection_history.csv`. A faculty-facing progress deck
(39 slides, generated from these same result files, every pilot slide
tagged accordingly) is at
`docs/presentations/BORG_Datacenter_Cooling_V2_Pilot_Presentation.pptx`.

Once cells E–H are extracted, the full 8-cell dataset goes through
final model selection, a frozen configuration, and one untouched final
test evaluation — pilot numbers above will be re-measured, not
carried forward as final.

---

## Version 1 (complete baseline, preserved at tag `v1-kaggle-baseline`)

Uses the Google Borg cluster-trace export on Kaggle
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
