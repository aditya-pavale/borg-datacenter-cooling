# Version Control State

Snapshot taken 2026-09-19. First written during Phase 0/5 of the V2
master plan; updated after the pilot pipeline, readiness audit, and
progress presentation were completed (same day, later commits).

## Branches
- `main` — Version 1 (Kaggle baseline), 1 commit (`6c7f01f`), pushed to
  `origin/main`. Unchanged since V2 work began.
- `v2-official-google-data` — active V2 branch, pushed to `origin`.
  Current head: `ecd9ba8` ("docs: add V2 pilot progress presentation
  for faculty review"). Commit history on top of `main`:
  1. `070bad3` — official data access audit + alignment resolution + pilot cell extraction
  2. `bec2a45` — V2 pilot pipeline end-to-end (forecasting, power model, thermal, classical controllers, PPO/MAPPO smoke tests)
  3. `f493b6c` — V2 unit tests, MPC controller, safety ablation, thermal calibration bug found+fixed
  4. `b83ce2a` — final-pipeline readiness audit; fixed a real pilot/final config+path isolation bug
  5. `ecd9ba8` — V2 pilot progress presentation (39-slide PPTX, generated from real result files)

## Tags
- `v1-kaggle-baseline` → commit `6c7f01f`, **already pushed to
  `origin`** (`git ls-remote --tags origin` confirms
  `refs/tags/v1-kaggle-baseline`). Version 1 is preserved and
  recoverable independent of any V2 work.

## Remote
`origin` → `https://github.com/aditya-pavale/borg-datacenter-cooling.git`
(fetch + push).

## Preservation status
Version 1 is fully protected: it is tagged, the tag is pushed, and no
V2 commit rewrites or force-pushes any V1 history. All V2 work happens
as new commits on `v2-official-google-data`, which is never merged back
into `main` in a way that would need `main` to change.

## What changed getting here
No destructive git operations were performed at any point. Every commit
on `v2-official-google-data` is additive on top of `main`'s single
commit; `main` and the `v1-kaggle-baseline` tag have never been
touched. Cells E–H remain unextracted (workload only — power and
machine data for all 8 cells were extracted in commit `070bad3`); no
commit fabricates or substitutes data for them.
