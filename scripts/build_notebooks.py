"""Builds the 6 presentation/analysis notebooks required for handoff.
Every notebook LOADS already-verified artifacts (parquet/JSON/PNG) and
visualizes/explains them; none retrains a model. Also executes each
notebook (nbclient) to confirm it runs top-to-bottom without error.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)


PREAMBLE = """\
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import Image, display, Markdown

REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
RESULTS = REPO_ROOT / "results"
DOCS = REPO_ROOT / "docs"
DATA = REPO_ROOT / "data" / "processed"
"""


def build_01():
    cells = [
        md("# 01 - Dataset Analysis\n\n"
           "Loads the already-computed Phase-1 audit artifacts "
           "(`docs/dataset_summary.json`) and the aggregated workload "
           "time series. No retraining, no re-parsing of the raw CSV "
           "(that is `src/data/inspect_borg.py`'s job, run separately). "
           "See `docs/dataset_audit.md` for the full narrative audit."),
        code(PREAMBLE),
        md("## Raw dataset shape and integrity"),
        code("summary = json.loads((DOCS / 'dataset_summary.json').read_text())\n"
             "print('file:', summary['file']['path'])\n"
             "print('size (bytes):', summary['file']['size_bytes'])\n"
             "print('physical lines (misleading, see docs/dataset_audit.md S3):', "
             "summary['file']['physical_line_count_including_header'])\n"
             "print('actual logical records:', summary['file']['logical_csv_record_count'])\n"
             "print('columns:', summary['shape']['n_cols'])"),
        md("## Dataset summary table"),
        code("pd.read_csv(RESULTS / 'tables' / '01_dataset_summary.csv')"),
        md("## Workload over time (real Borg-derived, per zone)\n\n"
           "See `docs/data_preprocessing.md` for why an all-8-cluster, "
           "15-minute aggregation was used instead of the originally "
           "planned single-cluster, 5-minute design (the original design "
           "left 94% of cells at exactly zero)."),
        code("display(Image(filename=str(RESULTS / 'figures' / '01_workload_over_time.png')))"),
        md("## Workload distribution and the effect of aggregation"),
        code("display(Image(filename=str(RESULTS / 'figures' / '02_workload_distribution.png')))\n"
             "display(Image(filename=str(RESULTS / 'figures' / '03_workload_aggregation.png')))"),
        md("## Key, evidence-based findings (full detail in `docs/dataset_audit.md`)\n\n"
           "- The Kaggle export has **no uploader-provided documentation**.\n"
           "- 405,894 real logical records (not the 1,324,695 a naive line "
           "count suggests -- embedded newlines in numpy-repr columns).\n"
           "- `instance_events_type` is a byte-identical duplicate of "
           "`collections_events_type`; `event` is a perfect bijection of both.\n"
           "- The stable task-instance grain is "
           "`(collection_id, instance_index, machine_id)`, not `instance_index` alone.\n"
           "- ~30% of the final (zone, 15-min-bin) workload cells are exactly "
           "zero -- a genuine dataset characteristic, not a processing bug."),
    ]
    return cells


def build_02():
    cells = [
        md("# 02 - Workload Forecasting\n\n"
           "Loads the frozen GRU forecaster's saved metrics and training "
           "history (`results/forecasting/`). Does not retrain the model "
           "-- see `src/forecasting/train_forecaster.py` for that. Full "
           "methodology in `docs/forecasting_design.md`."),
        code(PREAMBLE),
        md("## Forecast metrics: GRU vs. baselines (test split)"),
        code("pd.read_csv(RESULTS / 'tables' / '02_forecast_metrics.csv')"),
        md("**Honest interpretation** (see `docs/forecasting_design.md`): "
           "the GRU beats persistence and moving-average on MAE/RMSE, but "
           "its R² is close to zero. Better MAE/RMSE does **not** imply "
           "strong explanatory power when R² is near zero -- both facts "
           "are reported together deliberately."),
        md("## Training/validation loss curve"),
        code("display(Image(filename=str(RESULTS / 'figures' / '05_gru_training_loss.png')))"),
        md("## Forecast vs. true future workload, example windows"),
        code("display(Image(filename=str(RESULTS / 'figures' / '04_gru_forecast_vs_actual.png')))"),
        md("## Causality check\n\n"
           "`docs/leakage_audit.md` programmatically verifies the forecast "
           "is never a copy of the true future and the scaler is fit on "
           "the training split only."),
        code("print((DOCS / 'leakage_audit.md').read_text()[:600])"),
    ]
    return cells


def build_03():
    cells = [
        md("# 03 - Thermal Model Validation\n\n"
           "Loads the 9 mandatory thermal-validation experiment results "
           "(`results/thermal_validation/`). See `docs/thermal_model.md` "
           "for the equations and parameter provenance -- all thermal "
           "parameters are SIMULATION ASSUMPTIONS, never claimed to be "
           "physically validated."),
        code(PREAMBLE),
        md("## Pass/fail summary"),
        code("pd.read_csv(RESULTS / 'tables' / '03_thermal_validation.csv')"),
        md("## Individual experiment plots"),
        code("import glob\n"
             "for p in sorted(glob.glob(str(RESULTS / 'thermal_validation' / 'test*.png'))):\n"
             "    print(p.split('/')[-1])\n"
             "    display(Image(filename=p))"),
        md("## Note on one corrected test (see `docs/experiment_log.md`)\n\n"
           "Test 2 (constant workload, no cooling) originally failed with "
           "a too-short, too-tightly-toleranced window; the model's "
           "dynamics were correct throughout, only the test's horizon was "
           "miscalibrated relative to the system's actual thermal time "
           "constant. Fixed by deriving the test window from `tau = C/K_amb` "
           "and checking convergence to the analytical equilibrium."),
    ]
    return cells


def build_04():
    cells = [
        md("# 04 - Controller Comparison\n\n"
           "Loads the final, frozen evaluation results "
           "(`results/final_evaluation/controller_comparison.json`), "
           "produced by `scripts/final_evaluation.py` on the untouched "
           "test split. No controller is retrained here."),
        code(PREAMBLE),
        md("## Nominal comparison table (20 held-out test episodes)"),
        code("pd.read_csv(RESULTS / 'tables' / '04_controller_comparison.csv')"),
        md("## Energy and safety-shield-reliance comparison"),
        code("display(Image(filename=str(RESULTS / 'figures' / '09_10_energy_safety_comparison.png')))"),
        md("**Headline, reported honestly** (see `docs/final_validation_report.md` "
           "S12): classical PID achieves the lowest energy of any controller "
           "while remaining fully safe. This is not adjusted to favor RL."),
        md("## Representative temperature trajectories and cooling actions"),
        code("display(Image(filename=str(RESULTS / 'figures' / '07_08_trajectories_and_actions.png')))"),
        md("## Energy vs. safety-margin trade-off"),
        code("display(Image(filename=str(RESULTS / 'figures' / '14_energy_safety_tradeoff.png')))"),
    ]
    return cells


def build_05():
    cells = [
        md("# 05 - MARL (MAPPO) Analysis\n\n"
           "Loads the MAPPO training histories "
           "(`results/mappo/training_history_seed*.json`) saved during "
           "`scripts/train_mappo.py`. Architecture and credit-assignment "
           "justification: `docs/marl_design.md`. A real PPO "
           "log-probability bug was found and fixed during development "
           "-- see `docs/experiment_log.md` for the full diagnosis."),
        code(PREAMBLE),
        md("## Training convergence, all 3 seeds"),
        code("display(Image(filename=str(RESULTS / 'figures' / '12_reward_convergence.png')))"),
        md("## Seed-to-seed variability, PPO vs. MAPPO"),
        code("display(Image(filename=str(RESULTS / 'figures' / '13_seed_variability.png')))"),
        md("## Ablations: forecast usage and safety-shield reliance"),
        code("pd.read_csv(RESULTS / 'tables' / '07_ablation_results.csv')"),
        md("**Shield ablation, post-fix**: shield-on and shield-off produce "
           "*identical* results for the (bug-fixed) MAPPO seed-0 policy "
           "(19.15 kWh, 0% violations both ways) -- confirming the policy "
           "is genuinely safe on its own, not shield-dependent. This was "
           "explicitly **not** true before the PPO log-probability bug fix "
           "(`docs/experiment_log.md`)."),
        md("## Seed-level numeric results"),
        code("pd.read_csv(RESULTS / 'tables' / '06_seed_level_rl_results.csv')"),
    ]
    return cells


def build_06():
    cells = [
        md("# 06 - Final Results\n\n"
           "Summary notebook: loads the final comparison, robustness, and "
           "configuration tables. The authoritative narrative document is "
           "`docs/final_validation_report.md`; this notebook is a visual "
           "companion to it, not a replacement."),
        code(PREAMBLE),
        md("## Final controller comparison"),
        code("pd.read_csv(RESULTS / 'tables' / '04_controller_comparison.csv')"),
        md("## Robustness (synthetic workload spike and forecast-noise experiments)"),
        code("display(Image(filename=str(RESULTS / 'figures' / '11_robustness.png')))"),
        code("pd.read_csv(RESULTS / 'tables' / '05_robustness_comparison.csv')"),
        md("## Experiment configuration used for all reported results"),
        code("pd.read_csv(RESULTS / 'tables' / '08_experiment_configuration.csv')"),
        md("## Conclusions, strictly from the evidence\n\n"
           "See `docs/final_validation_report.md` S20 for the full "
           "research-question-by-research-question discussion. Headline: "
           "under this project's CPU-only compute budget, **classical PID "
           "outperformed both single-agent PPO and a from-scratch, "
           "bug-fixed MAPPO implementation on energy efficiency, while all "
           "controllers remained safe.** This is reported as the genuine "
           "finding, not adjusted to favor any method."),
    ]
    return cells


NOTEBOOKS = {
    "01_dataset_analysis.ipynb": build_01,
    "02_workload_forecasting.ipynb": build_02,
    "03_thermal_validation.ipynb": build_03,
    "04_controller_comparison.ipynb": build_04,
    "05_marl_analysis.ipynb": build_05,
    "06_final_results.ipynb": build_06,
}


def main():
    for fname, builder in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = builder()
        path = NB_DIR / fname
        nbf.write(nb, path)
        print(f"built {path}")

        client = NotebookClient(nb, timeout=120, kernel_name="python3",
                                 resources={"metadata": {"path": str(NB_DIR)}})
        client.execute()
        nbf.write(nb, path)  # write back with outputs
        print(f"  executed OK, outputs saved -> {path}")


if __name__ == "__main__":
    main()
