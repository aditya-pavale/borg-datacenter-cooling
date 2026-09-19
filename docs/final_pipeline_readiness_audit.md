# Final Pipeline Readiness Audit (pre-E-H)

Status: this is a static/structural readiness pass, run while cells e-h
remain blocked on the BigQuery Sandbox free-tier monthly quota
(`docs/gcp_billing_blocker.md`). **No PPO/MAPPO comparison was run
against cells a-d as part of this audit** — per instruction, that would
be exactly the kind of "final-looking" result on pilot data this audit
exists to prevent. Everything below is either (a) a live test that
passes today, or (b) a code-level guarantee verified by reading the
actual code, not by running the final pipeline against substitute data.
**No synthetic or substitute e-h data was created at any point** —
nothing here weakens or bypasses the requirement that a real final
result needs the real, complete 8-cell dataset.

## 1. Extraction script retrieves e-h without code changes

**Verified live, today**: `bash scripts/bigquery/run_clusterdata_extraction.sh`
was re-run during this session. It correctly skipped cells a-d
(existing output), dry-ran cell e (estimate: 155,765,859,864 bytes,
matching the pre-computed estimate in
`scripts/bigquery/extract_clusterdata_workload_lite.sql`), and failed
only on the live BigQuery quota check — the exact, expected failure
mode, not a code error. `tests/test_v2_config_consistency.py::test_extraction_script_is_parameterized_not_hardcoded_to_pilot_cells`
locks in that the script's cell loop covers `a b c d e f g h`, not a
hard-coded subset. **No code change is needed** — only the quota
resetting.

## 2. The A-H merge is deterministic

`src/data/v2_build_timeseries.py` builds the cell list from
`configs/data_config.yaml`'s `cells.<active_cell_set>` (alphabetically
sorted for the `zone` code assignment) and joins workload to power on
`(cell, bucket_5min)` with an inner join plus an explicit >=95% coverage
assertion per cell (raises `AssertionError` rather than silently
proceeding on a bad join). There is no random sampling, shuffling, or
per-run nondeterminism anywhere in this path — re-running it on
identical inputs produces byte-identical output. This was true before
this audit and is unchanged; the audit's contribution is confirming no
step depends on `len(cells)` being 4.

## 3. Final train/val/test split is automatically generated

`src/data/v2_split.py` reads `cfg['active_cell_set']` and computes one
shared chronological bucket cutoff (train_frac/val_frac/test_frac from
`configs/data_config.yaml`) applied identically across however many
cells are present, writing `{split}_{cell_set}.parquet`. Switching
`active_cell_set: final` in `configs/data_config.yaml` and re-running
`v2_build_timeseries.py` then `v2_split.py` produces
`train_final.parquet` / `val_final.parquet` / `test_final.parquet`
automatically, alongside (not overwriting) the existing `*_pilot.parquet`
files.

## 4. All model pipelines can be reset/retrained from scratch

Every V2 training/fitting script (`v2_train_forecaster.py`,
`v2_fit_power_model.py`, `v2_precompute_power_predictions.py`,
`v2_precompute_forecasts.py`, `v2_train_ppo.py`, `v2_train_mappo.py`)
takes no positional dependency on a previous pilot run except reading
its own declared inputs (the `{split}_{cell_set}.parquet` files and,
where relevant, a model file it itself will (re)produce). None of them
skip work if an output already exists — every run retrains/refits from
scratch and overwrites its own output path. Verified by re-running the
entire pilot chain end-to-end during this audit (`v2_build_timeseries`
-> `v2_split` -> `v2_fit_power_model` -> `v2_precompute_power_predictions`
-> `v2_train_forecaster` -> `v2_precompute_forecasts` -> classical
controllers -> MPC -> safety ablation -> robustness -> PPO -> MAPPO ->
combined evaluation) and confirming every numeric result matched the
prior run exactly (deterministic, reproducible).

## 5. Pilot artifacts remain isolated from final artifacts

**A real gap was found and fixed here.** Before this audit,
`configs/v2_config.yaml` independently duplicated
`data.active_cell_set` (separately from `configs/data_config.yaml`'s
copy), several scripts wrote to hard-coded `results/pilot/...` and
`models/v2/{forecasting,power_model,ppo,mappo}/...` paths regardless of
which cell set was active, and `n_zones` was a manually-set config
integer rather than derived from the data. Together these meant: (a)
flipping `active_cell_set` to `"final"` in one config file but not the
other would silently desync, and (b) a final run would **overwrite**
pilot model/result files at the same path rather than producing
separate ones.

**Fixed**: `load_v2_config()` (`src/environment/v2_cooling_core.py`) now
reads `active_cell_set` from `configs/data_config.yaml` only (removed
from `v2_config.yaml` entirely) and derives `n_zones` from the actual
loaded parquet's distinct zone count, asserting it against
`data_config.yaml`'s declared cell list rather than trusting either
blindly. Every script now writes through
`src/evaluation/v2_common.py`'s `results_dir(cfg, ...)` /
`models_dir(cfg, ...)`, which resolve to `results/<cell_set>/...` and
`models/v2/<cell_set>/...` — pilot and final can never collide.
`tests/test_v2_config_consistency.py` (7 tests) locks this in,
including a grep-level guard
(`test_no_v2_script_hardcodes_a_pilot_only_results_path`) that fails the
test suite if any future script reintroduces a hard-coded pilot path.

## 6. Final experiment configurations are reproducible

`configs/data_config.yaml` and `configs/v2_config.yaml` are both
version-controlled; every script reads all parameters from them (no
inline magic numbers for anything that varies between pilot and final).
Seeds are fixed (`cfg["seed"]`, `cfg["seeds"]`). The one config value
that is NOT yet re-validated for 8 cells is `heat_scale_kw=0.85`
(calibrated on the pilot's 4-cell heat distribution,
`docs/version2_research_design.md` S3) — this is flagged explicitly
below (S8) as something that must be re-checked, not assumed, once real
e-h data exists.

## 7. Final DOCX/PPT generation will consume only final result files

Not yet built (per instruction: DOCX/PPTX generation is deliberately
deferred). The readiness property established now: every result file
produced by a V2 script already lands under `results/<cell_set>/...`
with an explicit `disclaimer` field distinguishing pilot from final
(`src/evaluation/v2_common.py::disclaimer`). When the report/deck
generation scripts are written, they should read exclusively from
`results/final/**` and can assert on that `disclaimer` field's presence
to fail loudly if a pilot file is accidentally included — the
directory-and-field-level separation this audit put in place is what
makes that assertion possible.

## 8. No hard-coded A-D assumptions remain anywhere in the final pipeline

Verified by:
- `tests/test_v2_config_consistency.py::test_no_v2_script_hardcodes_a_pilot_only_results_path`
  (grep-level, part of the standard test suite, 62/62 passing including
  this one)
- Manual grep sweep for `cell_a`/`cell_b`/`cell_c`/`cell_d`,
  `"cells a-d"` (docstring-only, non-functional), and any `n_zones = 4`
  literal outside of documentation/comments — none found in executable
  code paths
- `src/environment/multi_agent_env.py`'s `AGENT_NAMES` already generates
  8 entries (`[f"zone_{i}" for i in range(8)]`), sliced to `n_zones` at
  runtime — not a fixed-length-4 (or 3) list
- `n_zones` is derived from `workload_pivot.shape[1]` in
  `V2CoolingCore.__init__`, not read from a config integer anywhere in
  the execution path

**One honest, disclosed limitation this audit does NOT resolve**: the
`heat_scale_kw=0.85` thermal calibration and the observed 34% PID
violation rate were characterized only on the pilot's 4-cell heat
distribution. The code that computes them is fully generic (re-running
`docs/version2_research_design.md`'s calibration sweep against 8-cell
data requires no code change), but the *numeric result* of that sweep
on 8 cells is genuinely unknown until e-h exist — this audit verifies
the pipeline is ready to re-run it, not that the existing pilot number
will hold.

## Test suite status

`pytest tests/ -q` → **62 passed** (46 inherited V1 tests + 9
`test_v2_pipeline.py` + 7 `test_v2_config_consistency.py`). No skips, no
xfails.

## What happens next, exactly

Nothing, until the BigQuery Sandbox quota resets. At that point:
`scripts/bigquery/run_clusterdata_extraction.sh` (no changes) → flip
`active_cell_set: final` in `configs/data_config.yaml` (the single
line) → re-run the full chain (S4 above) → re-run the S3 thermal
calibration sweep on 8-cell data → freeze → evaluate the untouched
final test split once. This document is the checkpoint that pipeline is
mechanically ready for that sequence; it does not run any part of it
against cells a-d as a substitute.
