"""Shared helpers for V2 scripts so that "which cell set produced this"
is computed once, consistently, rather than hand-typed (and easy to
forget to update) in every script -- see
docs/version2_research_design.md and docs/final_pipeline_readiness_audit.md.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def results_dir(cfg: dict, *subdirs: str) -> Path:
    """results/<active_cell_set>/<subdirs...> -- NEVER a hard-coded
    'results/pilot/...'. Pilot and final outputs land in physically
    separate directories automatically, keyed off the single source of
    truth (configs/data_config.yaml's active_cell_set)."""
    cell_set = cfg["data"]["active_cell_set"]
    return REPO_ROOT / "results" / cell_set / Path(*subdirs)


def models_dir(cfg: dict, *subdirs: str) -> Path:
    cell_set = cfg["data"]["active_cell_set"]
    return REPO_ROOT / "models" / "v2" / cell_set / Path(*subdirs)


def disclaimer(cfg: dict) -> str:
    cell_set = cfg["data"]["active_cell_set"]
    cells = cfg["data"].get("active_cells", [])
    if cell_set == "pilot":
        return (
            f"PILOT/DEVELOPMENT data -- cells {','.join(cells)} only. "
            f"See results/pilot/README.md. NOT a final V2 result."
        )
    return (
        f"FINAL data -- all {len(cells)} cells ({','.join(cells)}). "
        f"Produced by the frozen final pipeline (docs/final_pipeline_readiness_audit.md)."
    )
