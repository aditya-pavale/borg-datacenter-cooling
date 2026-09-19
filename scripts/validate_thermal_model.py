"""Phase 6/19: mandatory thermal-model validation experiments.

Must pass before any RL training proceeds (master plan S19: "If the
model behaves incorrectly: diagnose, fix, rerun. Do not continue with
an invalid thermal model.").
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thermal.thermal_model import ThermalParams, ThreeZoneThermalModel  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "thermal_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_params() -> ThermalParams:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())["thermal"]
    return ThermalParams(
        dt_seconds=cfg["dt_seconds"],
        thermal_mass_kwh_per_c=cfg["thermal_mass_kwh_per_c"],
        coupling_kw_per_c=cfg["coupling_kw_per_c"],
        ambient_coupling_kw_per_c=cfg["ambient_coupling_kw_per_c"],
        ambient_temp_c=cfg["ambient_temp_c"],
        cooling_max_kw=cfg["cooling_max_kw"],
        cooling_efficiency=cfg["cooling_efficiency"],
        initial_temp_c=cfg["initial_temp_c"],
        safety_limit_c=cfg["safety_limit_c"],
    )


def run_episode(model, heat_fn, cooling_fn, n_steps=200):
    temps = [model.temps.copy()]
    for t in range(n_steps):
        h = heat_fn(t)
        c = cooling_fn(t)
        model.step(h, c)
        temps.append(model.temps.copy())
    return np.array(temps)


def plot(temps, title, fname):
    plt.figure(figsize=(7, 4))
    for z in range(temps.shape[1]):
        plt.plot(temps[:, z], label=f"zone {z}")
    plt.xlabel("step")
    plt.ylabel("temperature (C)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname)
    plt.close()


def main():
    p = load_params()
    report = []

    # --- Test 1: zero cooling, zero workload -> temps relax toward ambient ---
    m = ThreeZoneThermalModel(p)
    m.reset()
    temps = run_episode(m, lambda t: np.zeros(3), lambda t: np.zeros(3), n_steps=100)
    plot(temps, "Test 1: zero heat, zero cooling", "test1_zero_zero.png")
    passed = np.all(np.abs(temps[-1] - p.ambient_temp_c) < np.abs(temps[0] - p.ambient_temp_c))
    report.append(("1. Zero cooling + zero workload -> relax toward ambient", passed,
                    f"final temps {temps[-1]}, ambient {p.ambient_temp_c}"))

    # --- Test 2: constant workload, no cooling -> temps rise and approach the
    # analytical equilibrium (no blow-up). With uniform heat across all zones
    # the inter-zone coupling term is identically zero (all zones equal), so
    # the only heat sink is ambient coupling: at equilibrium
    # 0 = Q - K_amb*(T_eq - T_amb)  =>  T_eq = T_amb + Q / K_amb.
    # Time constant tau = C / K_amb, which for the default config is
    # 2.0/0.02 = 100 hours -- so this test must run long enough (>= ~5 tau)
    # to actually reach steady state; a short run that is merely "still
    # rising but decelerating" is NOT a model failure, it is too short a
    # test window, which is why n_steps is derived from tau below rather
    # than a fixed small constant.
    m.reset()
    q = 0.3
    t_eq_analytical = p.ambient_temp_c + q / p.ambient_coupling_kw_per_c
    tau_hours = p.thermal_mass_kwh_per_c / p.ambient_coupling_kw_per_c
    n_steps_needed = int(6 * tau_hours * 3600 / p.dt_seconds)  # ~6 tau to settle
    temps = run_episode(m, lambda t: np.full(3, q), lambda t: np.zeros(3), n_steps=n_steps_needed)
    plot(temps, "Test 2: constant heat, no cooling (long horizon)", "test2_constant_heat.png")
    finite = np.all(np.isfinite(temps))
    monotonic_rise = np.all(np.diff(temps[:, 0]) >= -1e-9)  # never overshoots/oscillates
    close_to_equilibrium = np.abs(temps[-1, 0] - t_eq_analytical) < 0.5
    passed = finite and monotonic_rise and close_to_equilibrium and (temps[-1] > temps[0]).all()
    report.append(("2. Constant workload, no cooling -> rises monotonically toward the "
                    "analytical equilibrium T_amb + Q/K_amb (no blow-up)",
                    passed,
                    f"final temp={temps[-1,0]:.3f}C, analytical equilibrium={t_eq_analytical:.3f}C, "
                    f"tau={tau_hours:.1f}h, n_steps={n_steps_needed}, monotonic={monotonic_rise}, finite={finite}"))

    # --- Test 3: step workload increase mid-episode -> temp increases after the step ---
    m.reset()
    def heat_step_up(t):
        return np.full(3, 0.1) if t < 50 else np.full(3, 0.4)
    temps = run_episode(m, heat_step_up, lambda t: np.zeros(3), n_steps=150)
    plot(temps, "Test 3: step workload increase", "test3_step_up.png")
    before = temps[49]
    after = temps[100]
    passed = (after > before).all()
    report.append(("3. Step workload increase -> temperature increases", passed,
                    f"before={before}, after={after}"))

    # --- Test 4: step workload decrease mid-episode -> temp decreases after the step ---
    m.reset()
    def heat_step_down(t):
        return np.full(3, 0.4) if t < 50 else np.full(3, 0.05)
    temps = run_episode(m, heat_step_down, lambda t: np.zeros(3), n_steps=150)
    plot(temps, "Test 4: step workload decrease", "test4_step_down.png")
    before = temps[49]
    after = temps[130]
    passed = (after < before).all()
    report.append(("4. Step workload decrease -> temperature decreases", passed,
                    f"before={before}, after={after}"))

    # --- Test 5: cooling step increase (constant heat) -> temp drops after cooling turns on ---
    m.reset()
    def cooling_step_up(t):
        return np.zeros(3) if t < 50 else np.ones(3)
    temps = run_episode(m, lambda t: np.full(3, 0.3), cooling_step_up, n_steps=200)
    plot(temps, "Test 5: cooling step increase", "test5_cooling_up.png")
    before = temps[49]
    after = temps[150]
    passed = (after < before).all()
    report.append(("5. Cooling step increase -> temperature decreases", passed,
                    f"before={before}, after={after}"))

    # --- Test 6: cooling step decrease -> temp rises after cooling turns off ---
    m.reset()
    def cooling_step_down(t):
        return np.ones(3) if t < 50 else np.zeros(3)
    temps = run_episode(m, lambda t: np.full(3, 0.3), cooling_step_down, n_steps=200)
    plot(temps, "Test 6: cooling step decrease", "test6_cooling_down.png")
    before = temps[49]
    after = temps[150]
    passed = (after > before).all()
    report.append(("6. Cooling step decrease -> temperature increases", passed,
                    f"before={before}, after={after}"))

    # --- Test 7: zone coupling -- heat only zone 0, check zones 1,2 also warm (with lag/attenuation) ---
    m.reset()
    def heat_only_zone0(t):
        h = np.zeros(3)
        h[0] = 0.4
        return h
    temps = run_episode(m, heat_only_zone0, lambda t: np.zeros(3), n_steps=300)
    plot(temps, "Test 7: heat only zone 0 (coupling test)", "test7_coupling.png")
    z0_rise = temps[-1, 0] - temps[0, 0]
    z1_rise = temps[-1, 1] - temps[0, 1]
    z2_rise = temps[-1, 2] - temps[0, 2]
    # Expect z0 > z1 > z2 (attenuation with distance), all > 0, z2 rises least (not directly coupled)
    passed = (z0_rise > z1_rise > z2_rise > 0.001)
    report.append(("7. Zone coupling: heating zone 0 warms 1 and 2 with attenuation by distance",
                    passed, f"rises: z0={z0_rise:.4f} z1={z1_rise:.4f} z2={z2_rise:.4f}"))

    # --- Test 8: ambient temperature disturbance ---
    m.reset()
    def ambient_disturbance_step(t):
        return np.zeros(3)
    m2 = ThreeZoneThermalModel(ThermalParams(**{**p.__dict__, "ambient_temp_c": 35.0}))
    m2.reset()
    temps_hot_ambient = run_episode(m2, ambient_disturbance_step, lambda t: np.zeros(3), n_steps=200)
    temps_normal_ambient = run_episode(ThreeZoneThermalModel(p), ambient_disturbance_step, lambda t: np.zeros(3), n_steps=200)
    plot(temps_hot_ambient, "Test 8: hot ambient (35C) disturbance", "test8_hot_ambient.png")
    passed = (temps_hot_ambient[-1] > temps_normal_ambient[-1]).all()
    report.append(("8. Ambient disturbance: hotter ambient -> higher steady-state zone temps",
                    passed, f"hot_ambient_final={temps_hot_ambient[-1]}, normal_ambient_final={temps_normal_ambient[-1]}"))

    # --- Stability / no absurd jumps check across all tests ---
    all_finite = True
    max_single_step_jump = 0.0
    m.reset()
    temps = run_episode(m, lambda t: np.full(3, 0.4), lambda t: np.zeros(3), n_steps=500)
    all_finite = np.all(np.isfinite(temps))
    max_single_step_jump = np.abs(np.diff(temps, axis=0)).max()
    passed = all_finite and max_single_step_jump < 2.0  # no >2C jump in one 15-min step
    report.append(("9. Numerical stability: no divergence, no absurd single-step jumps",
                    passed, f"all_finite={all_finite}, max_step_jump={max_single_step_jump:.4f}C"))

    # ---- write report ----
    lines = ["# Thermal Model Validation Results\n"]
    all_passed = True
    for name, passed, evidence in report:
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        lines.append(f"- **{status}** — {name}\n  - Evidence: {evidence}\n")
    lines.append(f"\n## Overall: {'ALL TESTS PASSED' if all_passed else 'FAILURES DETECTED'}\n")
    (RESULTS_DIR / "validation_report.md").write_text("\n".join(lines))
    print("\n".join(lines))

    if not all_passed:
        raise SystemExit("Thermal model validation FAILED -- see results/thermal_validation/validation_report.md")

    return all_passed


if __name__ == "__main__":
    main()
