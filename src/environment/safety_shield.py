"""Phase 12: safety shield -- a layer AFTER the learned/classical policy,
independent of the reward function, that corrects an action if the
thermal model predicts it would leave a zone above `safety_limit_c -
margin_c` after one step.

Because the one-step Euler thermal update only involves EACH zone's OWN
cooling action (the inter-zone coupling term uses only the current,
already-known temperatures, not other zones' next actions), the minimum
sufficient action for zone i can be solved in closed form, independent
of the other zones' actions:

    T_i[t+1] = T_i[t] + (dt/C_i) * (Q_i - a_i*cooling_max*eff + coupling_i + ambient_i)

Solving T_i[t+1] = threshold_i for a_i gives the exact minimum action
that keeps zone i at or under the threshold, given the other terms.
This is NOT a general nonlinear MPC solver; it is an exact solution to
this specific linear one-step model, valid only because of that
linearity -- documented explicitly, no claim of a formal safety
guarantee beyond the accuracy of the thermal model itself and the
one-step horizon.
"""

from __future__ import annotations

import numpy as np

from thermal.thermal_model import ThermalParams


def predict_next_temps(params: ThermalParams, adj: np.ndarray, temps: np.ndarray,
                        heat_kw: np.ndarray, action: np.ndarray) -> np.ndarray:
    cooling_kw = np.clip(action, 0.0, 1.0) * params.cooling_max_kw * params.cooling_efficiency
    coupling_term = params.coupling_kw_per_c * (adj @ temps - adj.sum(axis=1) * temps)
    ambient_term = params.ambient_coupling_kw_per_c * (params.ambient_temp_c - temps)
    dT = (params.dt_hours / params.thermal_mass_kwh_per_c) * (
        heat_kw - cooling_kw + coupling_term + ambient_term
    )
    return temps + dT


def shield_action(
    params: ThermalParams,
    adj: np.ndarray,
    temps: np.ndarray,
    heat_kw: np.ndarray,
    proposed_action: np.ndarray,
    margin_c: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (safe_action, predicted_temps_with_safe_action, info)."""
    proposed_action = np.clip(proposed_action, 0.0, 1.0)
    predicted = predict_next_temps(params, adj, temps, heat_kw, proposed_action)
    threshold = params.safety_limit_c - margin_c

    coupling_term = params.coupling_kw_per_c * (adj @ temps - adj.sum(axis=1) * temps)
    ambient_term = params.ambient_coupling_kw_per_c * (params.ambient_temp_c - temps)
    denom = params.cooling_max_kw * params.cooling_efficiency

    # a_i_needed solves: threshold = T_i + dt/C*(Q_i - a_i*denom + coupling_i + ambient_i)
    a_needed = (
        heat_kw + coupling_term + ambient_term
        - (threshold - temps) * params.thermal_mass_kwh_per_c / params.dt_hours
    ) / np.where(denom > 0, denom, 1.0)
    a_needed = np.clip(a_needed, 0.0, 1.0)

    unsafe_mask = predicted > threshold
    safe_action = np.where(unsafe_mask, np.maximum(proposed_action, a_needed), proposed_action)
    predicted_safe = predict_next_temps(params, adj, temps, heat_kw, safe_action)

    info = {
        "intervened": bool(unsafe_mask.any()),
        "n_interventions": int(unsafe_mask.sum()),
        "unsafe_mask": unsafe_mask.copy(),
        "predicted_temps_unshielded": predicted.copy(),
        "predicted_temps_shielded": predicted_safe.copy(),
        "action_delta": (safe_action - proposed_action).copy(),
    }
    return safe_action, predicted_safe, info
