"""Phase (handoff) Section J: generate clean CSV + Markdown tables from
already-verified result artifacts. No recomputation of models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = REPO_ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = REPO_ROOT / "results"


def save(df: pd.DataFrame, name: str):
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    (TABLES_DIR / f"{name}.md").write_text(df.to_markdown(index=False))
    print(f"wrote {name}.csv / {name}.md ({len(df)} rows)")


def table1_dataset_summary():
    summary = json.loads((REPO_ROOT / "docs" / "dataset_summary.json").read_text())
    rows = [
        {"item": "file", "value": summary["file"]["path"]},
        {"item": "size_bytes", "value": summary["file"]["size_bytes"]},
        {"item": "physical_line_count", "value": summary["file"]["physical_line_count_including_header"]},
        {"item": "logical_record_count", "value": summary["file"]["logical_csv_record_count"]},
        {"item": "n_columns", "value": summary["shape"]["n_cols"]},
        {"item": "n_exact_duplicate_rows", "value": summary["duplicates"]["n_exact_duplicate_rows_excluding_index_col"]},
        {"item": "n_distinct_clusters", "value": len(summary["categorical_summary"]["cluster"])},
        {"item": "n_distinct_machines", "value": summary["entity_cardinality"]["machine_id"]},
        {"item": "n_distinct_collections", "value": summary["entity_cardinality"]["collection_id"]},
    ]
    save(pd.DataFrame(rows), "01_dataset_summary")


def table2_forecast_metrics():
    m = json.loads((RESULTS_DIR / "forecasting" / "metrics.json").read_text())
    rows = []
    for model, d in m["overall"].items():
        rows.append({"model": model, "n": d["n"], "mae": d["mae"], "rmse": d["rmse"], "r2": d["r2"]})
    save(pd.DataFrame(rows), "02_forecast_metrics")


def table3_thermal_validation():
    text = (RESULTS_DIR / "thermal_validation" / "validation_report.md").read_text()
    rows = []
    for line in text.splitlines():
        if line.startswith("- **"):
            status = "PASS" if line.startswith("- **PASS**") else "FAIL"
            name = line.split("—", 1)[-1].strip()
            rows.append({"test": name, "status": status})
    save(pd.DataFrame(rows), "03_thermal_validation")


def table4_controller_comparison():
    comp = json.loads((RESULTS_DIR / "final_evaluation" / "controller_comparison.json").read_text())
    main = ["fixed", "threshold", "pid", "mpc", "ppo_seed0", "ppo_seed1", "ppo_seed2",
            "mappo_seed0", "mappo_seed1", "mappo_seed2"]
    rows = []
    for c in main:
        d = comp[c]
        rows.append({
            "controller": c,
            "energy_kwh_mean": d["energy.total_kwh"]["mean"],
            "energy_kwh_std": d["energy.total_kwh"]["std"],
            "violation_pct": d["thermal.violation_pct"]["mean"],
            "max_temp_c": d["thermal.max_temp_c"]["mean"],
            "mean_action_change": d["control.mean_action_change"]["mean"],
            "shield_interventions": d["safety.n_interventions"]["mean"],
            "n_episodes": d["energy.total_kwh"]["n_episodes"],
        })
    save(pd.DataFrame(rows), "04_controller_comparison")


def table5_robustness():
    rob = json.loads((RESULTS_DIR / "robustness" / "robustness_results.json").read_text())
    rows = []
    for scale, res in rob["experiment_B_workload_spike"].items():
        for ctrl, m in res.items():
            rows.append({"experiment": "B_workload_spike", "condition": scale, "controller": ctrl, **m})
    for noise, res in rob["experiment_C_forecast_error"].items():
        for ctrl, m in res.items():
            rows.append({"experiment": "C_forecast_error", "condition": noise, "controller": ctrl, **m})
    save(pd.DataFrame(rows), "05_robustness_comparison")


def table6_seed_level_rl():
    comp = json.loads((RESULTS_DIR / "final_evaluation" / "controller_comparison.json").read_text())
    rows = []
    for algo in ["ppo", "mappo"]:
        for seed in [0, 1, 2]:
            d = comp[f"{algo}_seed{seed}"]
            rows.append({
                "algorithm": algo, "seed": seed,
                "energy_kwh_mean": d["energy.total_kwh"]["mean"],
                "violation_pct": d["thermal.violation_pct"]["mean"],
                "max_temp_c": d["thermal.max_temp_c"]["mean"],
            })
    save(pd.DataFrame(rows), "06_seed_level_rl_results")


def table7_ablations():
    comp = json.loads((RESULTS_DIR / "final_evaluation" / "controller_comparison.json").read_text())
    rows = [
        {"ablation": "forecast: with", "energy_kwh": comp["mappo_seed0_forecast_ablation_with"]["energy.total_kwh"]["mean"]},
        {"ablation": "forecast: without", "energy_kwh": comp["mappo_seed0_forecast_ablation_without"]["energy.total_kwh"]["mean"]},
        {"ablation": "safety shield: on", "energy_kwh": comp["mappo_seed0_shield_on"]["energy.total_kwh"]["mean"],
         "violation_pct": comp["mappo_seed0_shield_on"]["thermal.violation_pct"]["mean"]},
        {"ablation": "safety shield: off", "energy_kwh": comp["mappo_seed0_shield_off"]["energy.total_kwh"]["mean"],
         "violation_pct": comp["mappo_seed0_shield_off"]["thermal.violation_pct"]["mean"]},
    ]
    save(pd.DataFrame(rows), "07_ablation_results")


def table8_experiment_config():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())
    rows = [
        {"parameter": "bin_seconds", "value": cfg["data"]["bin_seconds"]},
        {"parameter": "n_zones", "value": cfg["data"]["n_zones"]},
        {"parameter": "cluster_for_zones", "value": cfg["data"]["cluster_for_zones"]},
        {"parameter": "train/val/test split", "value": f"{cfg['split']['train_frac']}/{cfg['split']['val_frac']}/{cfg['split']['test_frac']}"},
        {"parameter": "forecast horizon_steps", "value": cfg["forecasting"]["horizon_steps"]},
        {"parameter": "forecast lookback_steps", "value": cfg["forecasting"]["lookback_steps"]},
        {"parameter": "thermal_mass_kwh_per_c", "value": cfg["thermal"]["thermal_mass_kwh_per_c"]},
        {"parameter": "cooling_max_kw", "value": cfg["thermal"]["cooling_max_kw"]},
        {"parameter": "safety_limit_c", "value": cfg["thermal"]["safety_limit_c"]},
        {"parameter": "reward: energy_weight", "value": cfg["reward"]["energy_weight"]},
        {"parameter": "reward: safety_penalty_weight", "value": cfg["reward"]["safety_penalty_weight"]},
        {"parameter": "reward: proximity_weight", "value": cfg["reward"]["proximity_weight"]},
        {"parameter": "reward: shield_intervention_penalty_weight", "value": cfg["reward"]["shield_intervention_penalty_weight"]},
        {"parameter": "PPO total_timesteps", "value": cfg["rl"]["ppo"]["total_timesteps"]},
        {"parameter": "MAPPO total_updates", "value": cfg["rl"]["mappo"]["total_updates"]},
        {"parameter": "RL seeds", "value": str(cfg["seeds"])},
    ]
    save(pd.DataFrame(rows), "08_experiment_configuration")


def main():
    table1_dataset_summary()
    table2_forecast_metrics()
    table3_thermal_validation()
    table4_controller_comparison()
    table5_robustness()
    table6_seed_level_rl()
    table7_ablations()
    table8_experiment_config()


if __name__ == "__main__":
    main()
