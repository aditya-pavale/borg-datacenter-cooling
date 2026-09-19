"""Readiness checks for switching from pilot (cells a-d) to final
(cells a-h): the single-source-of-truth config wiring, path isolation,
and the absence of hard-coded pilot assumptions. See
docs/final_pipeline_readiness_audit.md for the full audit this codifies.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from environment.v2_cooling_core import load_v2_config, load_data_config  # noqa: E402
from evaluation.v2_common import results_dir, models_dir, disclaimer  # noqa: E402


def test_active_cell_set_has_a_single_source_of_truth():
    # v2_config.yaml must NOT redeclare active_cell_set/n_zones -- that
    # duplication is exactly the bug this test guards against (see
    # docs/version2_research_design.md).
    import yaml
    raw_v2 = yaml.safe_load((REPO_ROOT / "configs" / "v2_config.yaml").read_text())
    assert "active_cell_set" not in raw_v2["data"], (
        "configs/v2_config.yaml re-declares data.active_cell_set -- this must "
        "only live in configs/data_config.yaml (single source of truth)"
    )
    assert "n_zones" not in raw_v2["data"], (
        "configs/v2_config.yaml re-declares data.n_zones -- this must be "
        "DERIVED from the loaded data, never manually configured"
    )


def test_load_v2_config_merges_active_cell_set_from_data_config():
    data_cfg = load_data_config()
    cfg = load_v2_config()
    assert cfg["data"]["active_cell_set"] == data_cfg["active_cell_set"]
    assert cfg["data"]["active_cells"] == data_cfg["cells"][data_cfg["active_cell_set"]]


def test_pilot_and_final_cell_lists_are_disjoint_supersets_correctly():
    data_cfg = load_data_config()
    pilot = set(data_cfg["cells"]["pilot"])
    final = set(data_cfg["cells"]["final"])
    assert pilot.issubset(final), "pilot cells must all be present in the final cell list"
    assert final - pilot == {"e", "f", "g", "h"}


def test_results_and_models_dirs_are_isolated_by_cell_set():
    cfg = load_v2_config()
    cfg_final = {**cfg, "data": {**cfg["data"], "active_cell_set": "final"}}
    assert results_dir(cfg, "x") != results_dir(cfg_final, "x")
    assert models_dir(cfg, "x") != models_dir(cfg_final, "x")
    assert "pilot" in str(results_dir(cfg, "x"))
    assert "final" in str(results_dir(cfg_final, "x"))


def test_disclaimer_text_differs_between_pilot_and_final():
    cfg = load_v2_config()
    cfg_final = {**cfg, "data": {**cfg["data"], "active_cell_set": "final",
                                   "active_cells": load_data_config()["cells"]["final"]}}
    pilot_text = disclaimer(cfg)
    final_text = disclaimer(cfg_final)
    assert "PILOT" in pilot_text and "NOT a final" in pilot_text
    assert "FINAL" in final_text
    assert pilot_text != final_text


def test_extraction_script_is_parameterized_not_hardcoded_to_pilot_cells():
    script = (REPO_ROOT / "scripts" / "bigquery" / "run_clusterdata_extraction.sh").read_text()
    assert "for cell in a b c d e f g h" in script, (
        "extraction runner must loop over all 8 cells, not just a-d"
    )
    assert "a b c d\n" not in script or "for cell in a b c d\n" not in script


def test_no_v2_script_hardcodes_a_pilot_only_results_path():
    """Grep-level guard: no script should write to a literal
    'results/pilot/...' or 'models/v2/ppo' etc. string -- everything
    must go through v2_common.results_dir/models_dir so a switch to
    active_cell_set=final is a one-line config change, not a code
    change."""
    v2_scripts = sorted((REPO_ROOT / "scripts").glob("v2_*.py"))
    offenders = []
    for path in v2_scripts:
        if path.name == "v2_final_evaluation_pilot.py":
            continue  # explicitly pilot-only by design, and guarded by its own assert
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if '"results" / "pilot"' in line or "'results' / 'pilot'" in line:
                offenders.append((path.name, line.strip()))
            if '"models" / "v2" / "ppo"' in line or '"models" / "v2" / "mappo"' in line:
                offenders.append((path.name, line.strip()))
            if '"models" / "v2" / "forecasting"' in line or '"models" / "v2" / "power_model"' in line:
                offenders.append((path.name, line.strip()))
    assert not offenders, f"hard-coded pilot-only paths found: {offenders}"
