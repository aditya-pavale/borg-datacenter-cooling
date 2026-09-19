# Version Control State

Snapshot taken 2026-09-19, during Phase 0/5 of the V2 master plan.

## Branches
- `main` — Version 1 (Kaggle baseline), 1 commit (`6c7f01f`), pushed to
  `origin/main`.
- `v2-official-google-data` — active branch for Version 2, branched from
  `main` at the same commit. Not yet pushed to `origin` at the time this
  document was first written (pushed later in this same session — see
  git log for the actual push).

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
No destructive git operations were performed. `v2-official-google-data`
existed (per the initial `git status` at the start of this session) with
zero commits ahead of `main`; this session's Phase 0/5 work is the first
content added to it.
