# Safety Shield — Phase 12/26

## Architecture
`proposed_action -> thermal one-step-ahead prediction -> safety check -> corrected action -> environment`,
implemented in `src/environment/safety_shield.py`, called from
`CoolingCore.step()` (`src/environment/cooling_core.py`) **after** any
policy (classical or learned) produces an action and **before** it is
applied to the thermal model. It is architecturally independent of the
reward function — it never sees or affects the reward, only the action.

## Method
Because the one-step Euler thermal update depends on each zone's own
cooling action only (inter-zone coupling uses the *current*, already-known
temperatures, not other zones' next actions), the minimum action needed
to keep a zone at or under `safety_limit_c - margin_c` after one step
has an exact closed-form solution per zone (derived in the module
docstring). The shield computes this, and only overrides the proposed
action where the closed-form solution exceeds it.

## Important, empirically-discovered limitation
This is a **one-step-ahead, actuator-bounded** shield — it does **not**
guarantee compliance in general. Testing (`tests/test_environment.py::
test_shield_intervenes_when_unsafe`) surfaced a real case: with the
project's thermal parameters (originally tested against the
pre-recalibration `thermal_mass_kwh_per_c=2.0`; the recalibrated
default is 0.5, see `docs/experiment_log.md` — the qualitative
limitation holds either way, just with a different numeric bound):
with `cooling_max_kw=0.5`, `dt_seconds=900`, the maximum possible
temperature change achievable in a single 15-minute step, even at full
cooling, is bounded by `(dt_hours/C) * cooling_max_kw`, which is
`0.25°C` at the current, recalibrated thermal mass. If a
zone is already close to the limit with high heat load, the shield
correctly saturates the action at 1.0 and strictly improves the outcome,
but **cannot instantaneously enforce the margin** if the required
correction exceeds what the actuator can deliver in one step. This is
disclosed here rather than papered over; **no formal safety guarantee
is claimed**, only "best achievable one-step correction given actuator
and thermal-inertia limits." A practical consequence is that any
controller (classical or learned) must act proactively, before
temperatures approach the limit, rather than relying on the shield to
rescue a late reaction.

## Measured, not claimed
`n_interventions`, `intervened` (bool), and the resulting temperature
before/after shielding are recorded every step (`info` dict from
`CoolingCore.step`) and aggregated per-episode in the evaluation
pipeline (`src/evaluation/run_controller.py`), reported for every
controller and for the shield-on/shield-off ablation (Experiment F).
