# Three-Zone Thermal Model — Phase 6

## Equations

Continuous-time, per zone `i`:

```
C_i * dT_i/dt = Q_i - Q_cooling,i + sum_j K_ij (T_j - T_i) + K_amb (T_amb - T_i)
```

Discretized with explicit Euler at `dt = thermal.dt_seconds` (900s,
matching the empirically-chosen 15-minute control interval — see
`docs/data_preprocessing.md`):

```
T_i[t+1] = T_i[t] + (dt_hours / C_i) * (Q_i[t] - Q_cooling_i[t] + coupling_i[t] + ambient_i[t])
```

Topology: line graph `zone0 -- zone1 -- zone2` (zone 1 couples to both
neighbors; zones 0 and 2 do not couple directly), matching the master
plan's stated coupling structure.

`Q_cooling_i = action_i * cooling_max_kw * cooling_efficiency`,
`action_i ∈ [0,1]`.

## Parameter provenance — ALL simulation assumptions (category D)

No thermal, power, or cooling measurement exists anywhere in the Borg
dataset (`docs/dataset_audit.md` §14). Every parameter below is an
assumed, illustrative value, not calibrated:

| Parameter | Value | Status |
|---|---|---|
| `thermal_mass_kwh_per_c` (C) | 0.5 kWh/°C | ASSUMED, empirically recalibrated (see `docs/experiment_log.md`) so the safety limit is reachable within one episode |
| `coupling_kw_per_c` (K_ij) | 0.05 kW/°C | ASSUMED |
| `ambient_coupling_kw_per_c` (K_amb) | 0.02 kW/°C | ASSUMED |
| `ambient_temp_c` | 22.0 °C | ASSUMED |
| `cooling_max_kw` | 0.5 kW | ASSUMED |
| `cooling_efficiency` | 1.0 (kW removed per kW action) | ASSUMED simplification |
| `safety_limit_c` | 27.0 °C | project-specified target, used consistently everywhere (env, reward, baselines, plots) |

Forward-Euler stability bound: `dt < 2*tau` where
`tau = C / (K_amb + n_neighbors*K_ij)`. Checked automatically at model
construction (`ThreeZoneThermalModel.__init__` asserts this). For the
current (recalibrated, see below) config, `tau ≈ 4.2h` for a middle
zone (2 neighbors) and `dt = 0.25h` — comfortably stable.

## Validation (Phase 19) — required before any RL training

Nine experiments in `scripts/validate_thermal_model.py`, results in
`results/thermal_validation/validation_report.md` and per-test plots
in the same directory. **All 9 currently PASS.**

1. Zero heat + zero cooling → relax toward ambient.
2. Constant workload, no cooling → rises monotonically toward the
   analytical equilibrium `T_amb + Q/K_amb` (no blow-up).
3. Step workload increase → temperature increases.
4. Step workload decrease → temperature decreases.
5. Cooling step increase → temperature decreases.
6. Cooling step decrease → temperature increases.
7. Zone coupling: heating only zone 0 warms zones 1 and 2, with
   attenuation by graph distance (z0 rise > z1 rise > z2 rise).
8. Ambient disturbance: a hotter ambient temperature produces a
   higher steady-state zone temperature.
9. Numerical stability: no non-finite values, no single-step jump
   larger than 2°C, across a 500-step constant-high-heat run.

**One bug was found and fixed during validation** (logged in
`docs/experiment_log.md`): the original Test 2 used a fixed 300-step
window and a tight plateau tolerance, and failed — not because the
model was wrong, but because with zero active cooling the only heat
sink is the weak ambient-coupling term (`K_amb=0.02`), giving a ~100
hour time constant; 300 steps (75 hours) is well short of the ~5-6
time constants needed to reach steady state. The fix was to derive the
test's run length from the model's own time constant and check
convergence toward the analytically-derived equilibrium temperature
(`T_amb + Q/K_amb = 37.0°C`) rather than an arbitrary short window —
the model's dynamics were correct throughout; only the test's horizon
was miscalibrated.

## Physical validation status

**None.** This is a numerically-stable, behaviorally-sane simulation
model, not a physically validated one — there is no real thermal data
in this project to validate against. This limitation is repeated in
`docs/assumptions_and_limitations.md` and must not be elided in the
final report.

## Reproduce
```bash
.venv/bin/python scripts/validate_thermal_model.py
```
