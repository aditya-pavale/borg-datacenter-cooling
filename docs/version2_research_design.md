# Version 2 Research Design Decisions

This document records the design decisions made while adapting V1's
architecture to official Google 2019 data, and why each one was made.
It is referenced from code comments throughout `src/` and `configs/`
rather than duplicated there.

## 1. Zones are Borg cells, not a synthetic hash partition

V1 had no real partition of machines to build "zones" from (the Kaggle
export carried no cell/rack/room identifier), so it hashed `machine_id
% 3` into 3 synthetic zones — an explicitly-labeled simulation
abstraction (`docs/thermal_model.md`).

V2 has a real, Google-defined partition: the Borg cell, established in
`docs/official_data_alignment_audit.md` as the finest scientifically
defensible join key between ClusterData2019 and PowerData2019. Re-mixing
cells into an arbitrary fixed zone count (e.g. hashing 4 or 8 cells back
down to 3 "zones") would reintroduce exactly the kind of unjustified
aggregation that audit exists to prevent, while also discarding the
real per-cell power measurement's meaning. **V2 therefore sets
`n_zones == len(active cells)`**: 4 for the pilot (`a,b,c,d`), 8 once
`e-h` are extracted. `src/environment/multi_agent_env.py`'s
`AGENT_NAMES` was extended from 3 to 8 entries (backward compatible —
V1 still gets `zone_0..zone_2` when it slices the first 3) to
accommodate this.

## 2. The power model output drives heat directly

V1 had no power measurements and used an ASSUMED linear
`P = P_idle + (P_max - P_idle) * cpu_util` formula
(`src/thermal/power_model.py`), explicitly labeled as unvalidated.

V2 has real PowerData2019 measurements. `scripts/v2_fit_power_model.py`
fits candidate regressors (constant, linear, linear+cell-fixed-effect,
random forest) against them and selects the best on **validation
RMSE only**: random forest, R²=0.780 (pilot, cells a-d — see
`results/pilot/power_model/metrics.json`). Its predictions
(`predicted_power_util`, a fraction of PDU rated capacity, precomputed
per split by `scripts/v2_precompute_power_predictions.py`) drive the
thermal model's heat input directly in
`src/environment/v2_cooling_core.py`, replacing V1's assumed formula
with a fitted one.

**heat_scale_kw remains an ASSUMED simulation constant** — no PDU
capacity in kW exists anywhere in the public PowerData2019 schema
(`docs/official_data_alignment_audit.md` §3), so `predicted_power_util`
(a 0-1-ish fraction) cannot be converted to absolute watts, only scaled
by an assumed reference. This plays the same role, and carries the same
epistemic status, as V1's `p_max_kw`.

## 3. heat_scale_kw recalibration (empirical, documented)

The default `heat_scale_kw = 0.5` (chosen to roughly match V1's power
magnitude) made the safety constraint nearly vacuous: with cooling off,
simulated temperature plateaus at ~26.5°C and never crosses the 27°C
safety limit across 26 sampled full-day pilot episodes — an
uninteresting control problem where every controller trivially "passes"
safety, the same failure mode V1 hit and fixed with its
`thermal_mass_kwh_per_c` parameter (`docs/experiment_log.md`).

A sweep over `{0.5, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 1.0, 1.3, 1.6,
2.0}` (cooling-off vs. full-cooling max temperature, cells a-d, 14
sampled full-day episodes per value) found:

| heat_scale_kw | cooling-off max temp (mean) | full-cooling max temp (mean) |
|---|---|---|
| 0.5 | 26.50°C | 24.04°C |
| 0.8 | 26.62°C | 25.30°C |
| **0.85** | **27.33°C** | **26.41°C** |
| 0.9 | 28.35°C | 27.53°C |
| 1.0 | 30.49°C | 29.77°C (full cooling ALSO fails) |
| 2.0 | 52.49°C | 52.18°C |

At 1.0 and above, `cooling_max_kw` (0.5, unchanged from V1) is
undersized for the heat load — even full cooling cannot hold the
safety limit, an unsolvable rather than non-trivial problem. **0.85**
is the smallest tested value at which cooling-off already exceeds the
limit while full cooling stays comfortably under it — set in
`configs/v2_config.yaml`. This is a pilot-data (cells a-d) calibration;
it should be re-checked, not assumed to transfer unchanged, once cells
e-h are available (per the master plan's "final model selection must be
performed again on the complete dataset" rule).

**A real methodology bug was found and fixed while investigating this**
(not a bug in the environment physics itself, but in how this
calibration was first measured): the original 14-point sweep
constructed `V2CoolingCore` with its default `use_safety_shield=None`,
which reads `configs/v2_config.yaml`'s `safety_shield.enabled: true`.
So the "cooling-off" and "full-cooling" runs were never actually
open-loop — the safety shield was silently overriding the commanded
`action=0` whenever it predicted a threshold breach, and the amount of
override scales with `cooling_max_kw`. This was caught by a diagnostic
showing the "cooling-off" trajectory's max temperature changing when
`cooling_max_kw` was varied, which is physically impossible for a truly
open-loop, action-independent cooling-off run
(`cooling_kw = action * cooling_max_kw`, and `action=0` should zero
that term regardless of `cooling_max_kw`). **Fix**: recalibrate with
`use_safety_shield=False` explicitly, isolating the raw thermal/heat
dynamics from the shield.

**Corrected, dense (251-window, every-20th-bin) sweep, shield
disabled**:

| heat_scale_kw | cooling-off max temp (mean/max) | full-cooling max temp (mean/max) |
|---|---|---|
| 0.5 | 33.99 / 34.15 (100% violate) | 24.04 / 24.04 (0% violate) |
| 0.75 | 39.60 / 39.84 | 24.29 / 24.46 |
| **0.85** | **41.84 / 42.12 (100% violate)** | **26.44 / 26.71 (0% violate)** |
| 0.9 | 42.97 / 43.26 | 27.56 / 27.85 (98% violate) |
| 1.0 | 45.21 / 45.53 | 29.81 / 30.13 (100% violate) |

**0.85 is re-confirmed as correctly calibrated**, now on the full dense
sweep rather than the original confounded 14-point one: cooling-off
violates on every one of 251 tested windows, full cooling stays safe on
every one, with real margin before 0.9 breaks it. This supersedes (and
corrects) the earlier table in this section, which was measured under
the same shield confound.

**Separately — a genuine, disclosed control-difficulty finding, not a
bug**: given this corrected calibration, PID *with the shield enabled*
(the intended, real evaluation configuration) still incurs violations
on 33.9% of test-split steps
(`results/pilot/controllers/classical_controller_comparison.json`).
This is real and was investigated, not swept away: PID only commands
strong cooling once temperature error is already large (a reactive, not
proactive, controller), the actuator is rate-limited
(`action.max_rate_per_step=0.3`), and the safety shield's correction is
a one-step-ahead, myopic patch, not a guarantee against multi-step
accumulated overshoot — exactly the documented limitation in
`docs/safety_design.md` ("cannot always fully restore the margin in one
step under strong thermal inertia"), now empirically observed on real
official-Google-derived data rather than only asserted from V1. **This
was not fixed by loosening the calibration** — the pilot control
problem is genuinely harder than a cleanly bimodal always-safe/
always-unsafe split, which is arguably more informative for comparing
controllers than a trivial problem would be, and is reported as such.

## 4. Native 5-minute resolution, not V1's assumed 15-minute

The BigQuery extraction (`scripts/bigquery/`) already produces 5-minute
buckets, matching PowerData2019's native cadence. V2's pilot pipeline
runs the full stack at 5 minutes rather than assuming a coarser bin is
needed, since the sparsity problem that forced V1 to 15-minute bins
(94% zero-valued 5-minute cells for a single Kaggle cluster) does not
apply here — cell-level aggregation across thousands of machines is
never near-zero (`results/pilot/forecasting/metrics.json` confirms the
5-minute series is highly learnable, GRU R²=0.824 vs. persistence
R²=0.772). 10- and 15-minute resolutions remain documented candidates
(`configs/data_config.yaml`) for a resolution-selection experiment
before any final freeze, per the master plan's Phase 11 requirement,
but were not blocking for pilot development.
