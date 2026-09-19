# Pre-E-H Checkpoint

Status: clean checkpoint, created after the V2 pilot pipeline, tests,
readiness audit, and progress presentation were complete — taken while
cells E-H remain blocked, before any final 8-cell work begins.

## Git state

- Current commit: `eb0a544`
- Checkpoint tag: `v2-pilot-ready-e-h` → commit `eb0a544` (annotated,
  "V2 pilot pipeline complete and validated; ready for final E-H
  extraction"), pushed to `origin`.
- Branch: `v2-official-google-data`
- GitHub synchronization: local `HEAD` and `origin/v2-official-google-data`
  identical (`eb0a544`); confirmed via `git fetch origin` immediately
  before tagging.
- V1 preservation: `main` unchanged at `6c7f01f`; tag `v1-kaggle-baseline`
  → `6c7f01f`, pushed. No V2 commit has touched `main` or that tag.

## Notebook metadata decision

`notebooks/01_dataset_analysis.ipynb` had an uncommitted, metadata-only
diff (a `kernelspec` block: `display_name`/`language`/`name`), almost
certainly added by an editor opening the notebook rather than by any
research work. It touched no cells, code, or outputs, and was not
required for the notebook's existing (V1) execution record or for
reproducibility. **Reverted** via `git restore`, not committed — the
working tree was already clean afterward, so no cleanup commit was
needed.

## Test status

`pytest tests/ -q` → **62 passed** (46 inherited V1 tests + 9
`test_v2_pipeline.py` + 7 `test_v2_config_consistency.py`). 0 failed,
0 skipped.

## Official datasets

- **ClusterData2019** (workload): cells a, b, c, d extracted and
  validated (pilot). Cells e, f, g, h blocked on BigQuery Sandbox's
  free-tier monthly scan quota — a deliberate decision to stay off
  billing, not a data-access or data-quality problem
  (`docs/gcp_billing_blocker.md`).
- **PowerData2019** (power): all 8 cells, all 50 available PDU tables,
  fully extracted.
- **machine_events**: all 8 cells, fully extracted.

## A–D pilot status: what's done

- Workload↔power alignment resolved at Borg-cell granularity, verified
  against Google's own documentation and schema
  (`docs/official_data_alignment_audit.md`).
- GRU forecaster: test R² ≈ 0.824 vs. persistence 0.772 / moving-average
  0.787 (`results/pilot/forecasting/metrics.json`).
- Empirical power model: Random Forest validation R² ≈ 0.780 against
  real PowerData2019 measurements — V1 never had real power data at all
  (`results/pilot/power_model/metrics.json`).
- Thermal model calibrated (`heat_scale_kw = 0.85`) on a dense
  251-window sweep, with the safety shield explicitly disabled for the
  open-loop diagnostic (see bugs below).
- Classical controllers (Fixed, Threshold, PID), MPC, PPO, and MAPPO all
  run end-to-end on the real pilot data; PPO/MAPPO at smoke-test budgets
  only.
- Safety-shield ON/OFF ablation completed, with a reliance analysis.
- Robustness infrastructure (workload spike, forecast noise) exercised.
- 62/62 engineering tests passing, including a config-consistency suite
  written specifically to catch pilot/final drift before E-H arrives.

## E-H status: what's blocked

Workload extraction for cells e, f, g, h only. The extraction query is
written, cost-estimated (~627 GiB combined for all four, verified via
dry-run), and was live-tested during the readiness audit (correctly
attempted cell e, failed only on the still-unreset quota). No code
change is needed once the quota resets.

## Disk space

79 GB total root partition, 16 GB free (80% full) at last check —
stable since the pilot extraction; no large new files added.

## Major pilot findings

1. Cell-level aggregation produces a genuinely learnable workload signal
   (GRU R² 0.824), unlike V1's near-zero R² on the sparse Kaggle export.
2. A real, measured-power-anchored empirical power model is now
   possible (R² 0.780) — a capability V1 never had.
3. With the safety shield enabled, PID still violates the thermal
   safety limit on 33.9% of test steps — a genuine, disclosed control
   finding (reactive lag + rate-limited actuator + a myopic one-step
   shield), not artificially tuned away.
4. The safety-shield ablation shows PID's shield-on/off results are
   identical, but for a specific, non-obvious reason: its cooling is
   already saturated in the failure regime, not because it is
   proactively safe — the two are not the same claim.

## Major bugs found and fixed

1. **Thermal calibration shield confound**: the original calibration
   sweep measured "cooling off" with the safety shield still active by
   default, so the shield was silently overriding the commanded
   zero action. Caught via an impossible symptom (the "off" trajectory
   changing when a cooling-capacity parameter was varied). Fixed by
   explicitly disabling the shield for the diagnostic, re-run on a
   dense 251-window sweep, and locked in with a regression test.
2. **Pilot/final config and path drift**: `configs/v2_config.yaml`
   independently duplicated `configs/data_config.yaml`'s
   `active_cell_set`, and several scripts wrote to hard-coded
   `results/pilot/...` / `models/v2/{forecasting,ppo,mappo,power_model}/...`
   paths regardless of which cell set was active — a final run could
   have silently desynced configs or overwritten pilot artifacts at an
   identical path. Fixed: single source of truth for the active cell
   set, `n_zones` derived from loaded data instead of manually
   configured, and every script routed through
   `results/<cell_set>/...` / `models/v2/<cell_set>/...` helpers. Locked
   in by 7 new tests, including a grep-level guard against any future
   hard-coded pilot path.

## Exact remaining work

1. Wait for the BigQuery Sandbox free-tier monthly quota to reset (a
   time-based external dependency, not an action either party can
   force).
2. Extract cells e-h with the existing, unmodified extraction script.
3. Validate the 8-cell dataset with the same checks used for the pilot.
4. Re-run the full pipeline (preprocessing → forecasting → power model
   → thermal calibration → controllers → PPO/MAPPO) on the complete
   dataset, with model selection touching only train/validation.
5. Freeze the final configuration and evaluate the untouched final test
   split exactly once.
6. Run full multi-seed (≥5 where feasible) PPO/MAPPO, ablations
   (forecast on/off, shield on/off, reward/thermal sensitivity), and
   robustness experiments on the final data.
7. Independent final audit, then the final DOCX report and final PPTX
   (distinct from the pilot progress deck already published).

## Explicit final-claim statement

**A-D results are development/pilot results only. Final Version 2
scientific claims require the complete A-H dataset and a fresh final
evaluation.** No result in this repository, at this commit, has been
presented as a final Version 2 finding.
