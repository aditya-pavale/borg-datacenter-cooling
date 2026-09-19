"""Common metrics computed identically for every controller (Phase 29)."""

from __future__ import annotations

import numpy as np


def compute_episode_metrics(trajectory: list[dict], safety_limit_c: float, dt_hours: float = 0.25) -> dict:
    """trajectory: list of per-step info dicts from CoolingCore.step(),
    each containing temps, energy_kw, violation, action, smoothness,
    shield_intervened, shield_n_interventions."""
    temps = np.array([s["temps"] for s in trajectory])         # (T, n_zones)
    energy = np.array([s["energy_kw"] for s in trajectory])    # (T,)
    violation = np.array([s["violation"] for s in trajectory]) # (T, n_zones)
    actions = np.array([s["action"] for s in trajectory])      # (T, n_zones)
    smoothness = np.array([s["smoothness"] for s in trajectory])  # (T, n_zones)
    shield_interventions = np.array([s["shield_n_interventions"] for s in trajectory])

    violation_steps = (violation > 0).any(axis=1)
    action_changes = np.abs(np.diff(actions, axis=0))

    return {
        "energy": {
            "total_kwh": float(energy.sum() * dt_hours),
            "mean_kw": float(energy.mean()),
        },
        "thermal": {
            "mean_temp_c": float(temps.mean()),
            "max_temp_c": float(temps.max()),
            "min_temp_c": float(temps.min()),
            "temp_variance": float(temps.var()),
            "violation_steps": int(violation_steps.sum()),
            "violation_pct": float(100 * violation_steps.mean()),
            "cumulative_violation_severity": float(violation.sum()),
            "max_overshoot_c": float(violation.max()),
        },
        "control": {
            "mean_action_change": float(action_changes.mean()) if len(action_changes) else 0.0,
            "max_action_change": float(action_changes.max()) if len(action_changes) else 0.0,
            "mean_action": float(actions.mean()),
        },
        "safety": {
            "n_interventions": int(shield_interventions.sum()),
            "intervention_pct_steps": float(100 * (shield_interventions > 0).mean()),
        },
        "n_steps": len(trajectory),
    }


def aggregate_across_episodes(episode_metrics: list[dict]) -> dict:
    """mean/std across episodes for every scalar leaf metric."""
    import collections

    flat_keys = set()
    def flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                yield from flatten(v, key + ".")
            else:
                yield key, v

    values = collections.defaultdict(list)
    for m in episode_metrics:
        for k, v in flatten(m):
            values[k].append(v)

    agg = {}
    for k, vs in values.items():
        agg[k] = {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "n_episodes": len(vs)}
    return agg
