# Experiment / Correction Log

Chronological log of corrections, bugs found, and design pivots during
autonomous development. Per the master plan, nothing here is hidden or
retroactively erased.

---

## [2026-09-19] Phase 1 audit corrections (from independent review)

An independent adversarial review of the initial `docs/dataset_audit.md`
re-derived every material statistic via different code paths and found
two corrections required (both applied directly to `docs/dataset_audit.md`,
logged in its own §20):

1. **Null-pattern scope** — original draft checked only 8 columns and
   reported "99.81% fully populated." Full 34-column check found 6
   patterns, 69.2% fully non-null. Conclusion (evidence for a joined
   table) unchanged; framing corrected.
2. **Task-instance grain** — original draft used
   `(collection_id, instance_index)` as the task-instance key. Corrected
   to `(collection_id, instance_index, machine_id)` after discovering one
   `(collection_id, instance_index)` pair spanned 6,640 distinct
   machines. A corroborating check (0 duplicates at the corrected grain)
   was added, positively supporting the sum/mean-based aggregation plan.

Severity: MEDIUM (documentation precision), no impact on downstream
pipeline correctness since the final aggregation recommendation already
used the correct grain.

## [2026-09-19] Phase 2 aggregation design pivot — workload sparsity

**What happened:** the originally planned aggregation (single real Borg
`cluster`, 5-minute bins) produced a workload time series with 94.36%
of (zone, bin) cells at exactly zero. Root cause (evidence-based, see
`docs/data_preprocessing.md` §3.1): usage-report rows in this Kaggle
export are tied to lifecycle events, not sampled on a fixed per-instance
cadence, and the total row count (405,894 for an 8-cluster, 31-day,
96,174-machine trace) is far too small for a full periodic-telemetry
table — this file is inferred to be a subsample/event-filtered extract
of a larger original trace.

**Fix:** pooled all 8 Borg clusters as the workload source and widened
the bin from 5 to 15 minutes (`configs/config.yaml`), reducing zero-cells
to 30.3%. The 30-minute forecast objective was preserved by setting
`horizon_steps=2` (2 × 15 min). This is a stronger simulation assumption
than the original single-cluster design (weakens the "one physical
facility" narrative) and is disclosed as such in
`docs/data_preprocessing.md` rather than hidden.

Severity: HIGH (would have produced a degenerate, near-trivial workload
signal for forecasting/RL if not caught). Caught before any downstream
model was trained, by inspecting actual aggregation output rather than
trusting the pipeline to "just work" because it ran without error.

## [2026-09-19] Phase 6 thermal validation test miscalibration

**What happened:** Test 2 ("constant workload, no cooling should
plateau") failed on first run: 300 steps was far shorter than the
model's ~100-hour ambient-only time constant, so temperature was still
rising (correctly, just slowly) rather than having plateaued.

**Fix:** derived the test's run length from the model's own analytical
time constant (`tau = C/K_amb`) and checked convergence toward the
closed-form equilibrium (`T_amb + Q/K_amb`) instead of an arbitrary
short window. The thermal model's dynamics were correct throughout;
only the test was miscalibrated. Documented in `docs/thermal_model.md`.

Severity: LOW (test bug, not a model bug) — logged per the master
plan's instruction not to hide any correction, however small.

## [2026-09-19] Thermal-mass recalibration — safety constraint was vacuous

**What happened:** before running any controller, a stress test (fully
disabled cooling, for a full 24h episode, over every real test-split
workload window) was run against the default thermal parameters
(`thermal_mass_kwh_per_c=2.0`). The maximum temperature ever reached,
anywhere in the entire test split, was 25.19°C — never within reach of
the 27°C safety limit. This meant every controller (including doing
nothing) would trivially "pass" the safety objective, making the
safety-shield ablation (Experiment F) and safety-related metrics
vacuous by construction, not a genuine test of anything.

**Fix:** re-ran the same cooling-off stress test at
`thermal_mass_kwh_per_c ∈ {2.0, 0.5, 0.3, 0.15}` (all other parameters
held fixed) and picked 0.5 kWh/°C — the largest (i.e. least aggressive)
value at which cooling-off already risks exceeding the limit purely
from idle-level heat, while full cooling can still comfortably keep
zones safe. Verified afterward: (a) all 9 thermal validation
experiments (`scripts/validate_thermal_model.py`) still pass with the
new value; (b) with the safety shield disabled, a genuinely bad
controller (no cooling) now violates the limit 20.5% of the time
(cumulative severity 13.8), while reasonable controllers (threshold,
PID, moderate fixed action) remain safe — confirming the environment
now poses a real, solvable, non-vacuous safety-constrained control
problem; (c) with the shield enabled, the same bad controller is
rescued (111 mean interventions/episode, 0 violations), which is the
intended behavior for the shield ablation.

Severity: HIGH (would have silently invalidated the safety-related
experiments and the safety-shield ablation) — caught before any RL
training was run, by stress-testing the environment's boundary
behavior rather than assuming the default config was adequate.

## [2026-09-19] RL policies exploiting the safety shield, then a real PPO bug

**What happened (round 1 — apparent shield exploitation):** the first
full controller comparison (`scripts/final_evaluation.py`) showed both
MAPPO and single-agent PPO achieving suspiciously low energy (~1.7 kWh
vs. PID's 6.4 kWh) while reporting 0% violations. The shield-off
ablation immediately exposed why: the SAME trained policy caused
violations 20.05% of the time with the shield removed. The policy had
learned to cool almost nothing and let the safety shield silently fix
the consequences for free, since the reward only penalized violations
*after* shielding.

**Fix attempt 1 (reward redesign):** added a direct
`shield_intervention_penalty_weight` term and increased/broadened the
`proximity_weight`/`proximity_start_below_limit_c` penalty
(`configs/config.yaml`). This did NOT change the trained policy's
behavior at all across several retrains — a strong signal that the
problem was not (only) reward shaping.

**Root cause (round 2 — a real PPO implementation bug):** direct
inspection of the trained actor showed its output was a **constant**
(~-3 to -8 pre-clamp) regardless of the input observation, and manually
sweeping fixed actions against the actual environment reward (with the
shield both on and off) showed the reward landscape clearly and
strongly favored HIGH cooling action (best around 0.7-1.0; action=0
was the worst possible choice by a wide margin, -9363 vs -246 per
episode) — i.e. the reward was correctly designed, but PPO was
converging to the WORST region of it. Tracing the custom MAPPO
implementation (`src/marl/mappo.py`, `src/marl/networks.py`) found the
actual bug: `GaussianActor.act()` sampled a raw (unbounded) action for
the environment-facing clamped action, but the PPO importance-sampling
ratio was computed by evaluating `log_prob()` at the **clamped**
action, not the original raw sample, both when it was first collected
and again during the update. Since the Gaussian's density at a clamped
boundary value can be arbitrarily different from its density at the
actual sampled point, this silently corrupted the PPO ratio
`exp(new_logp - old_logp)` even when the policy had not changed at
all, producing essentially arbitrary, unstable gradients. A secondary
issue (an earlier version of `GaussianActor` also squashed its mean
through `sigmoid()`, which independently created a vanishing-gradient
trap once the pre-sigmoid value drifted very negative) was found and
removed in the same pass, reverting to the standard "unbounded mean,
clip only at the environment boundary" convention already documented
(but not correctly implemented) in `docs/marl_design.md`.

**Fix:** `GaussianActor.act()` now returns and the training buffer now
stores the **raw, pre-clamp** sampled action; `evaluate()` computes
`log_prob` on that raw action, matching what was actually sampled.
After this fix, MAPPO training immediately became stable and
non-degenerate across all 3 seeds: mean training reward improved
monotonically by roughly 28x in scaled terms, the shield-off ablation
for the retrained policy now matches the shield-on result exactly
(25.300 kWh both ways, 0% violations either way) — i.e. the retrained
policy is genuinely safe on its own, not shield-dependent — and the
converged energy usage (~25-35 kWh/episode) sits in the same
high-cooling region the manual reward sweep identified as
near-reward-optimal, rather than the previous worst-possible region.

**Result, reported honestly, not adjusted further:** with this fixed,
correct implementation and the project's CPU-only, reduced training
budget (300 on-policy updates / ~29k environment steps per MAPPO run,
100k timesteps for PPO), both MAPPO and single-agent PPO converge to a
*safe but energy-inefficient* policy (roughly "cool close to maximum
most of the time"), clearly **worse** on energy than every classical
baseline, especially PID (6.4 kWh vs. 25-35 kWh). This is reported as
the genuine result of this experimental setup, not corrected further
by additional hyperparameter search aimed at making MAPPO look
better — per the master plan's explicit instruction that if a classical
controller outperforms MAPPO, that must be reported, not hidden.

Severity: CRITICAL (round 2, the PPO log-prob bug) — this was an
actual correctness bug in the learning algorithm itself, not a
modeling or reward-design choice, and would have invalidated any
claim about MAPPO's or PPO's learning ability had it gone unnoticed.
Caught by refusing to accept "it converges to a low-energy number" as
sufficient evidence of good behavior, and instead checking the
shield-off ablation and the actor's raw output on real inputs.

## [2026-09-19] Leakage audit false positive

`scripts/leakage_audit.py` initially reported the safety-shield check
as FAILED because a naive text search for the word "future" in
`safety_shield.py` matched Python's own
`from __future__ import annotations` statement, unrelated to data
leakage. Fixed by excluding that specific line from the scan; all 8/8
leakage checks pass. Severity: LOW (test-harness bug, not a leakage
issue) — logged for completeness.

## [2026-09-19] Final handoff audit — leakage-check scope extended and one check corrected

During the final submission/handoff audit, two more leakage checks
were added to `scripts/leakage_audit.py` (no training script touches
the test split for fitting; classical/thermal parameters are static
config values, not fit from data). The first version of the new
"no training script references the test split" check was too blunt —
it flagged `train_forecaster.py` for loading `test.parquet`, which is
legitimate (the test split is loaded once, upfront, but never used
until after the best-validation checkpoint is restored, purely to
report final held-out metrics — that is what a test set is for, not
leakage). Corrected to check specifically whether `X_test`/`y_test` is
referenced *inside* the training/early-stopping loop (it is not).
Final leakage audit: **11/11 checks PASS** (`docs/leakage_audit.md`).
Severity: LOW (audit-script false positive, caught and fixed before
being reported as a real finding).

## [2026-09-19] Final handoff audit — cleanup and .gitignore fix

Removed `src/rl/`, an empty directory left over from initial project
scaffolding (single-agent PPO uses Stable-Baselines3 directly, MAPPO
lives in `src/marl/`; nothing was ever placed in `src/rl/`, and no
file anywhere imports from it — verified before deletion). Added
`.pytest_cache/` and `*.egg-info/` to `.gitignore`, which were
previously uncovered. Both changes logged in
`docs/repository_inventory.md`. Severity: LOW (housekeeping, zero
functional/reproducibility impact — full test suite re-verified
passing after the removal).
