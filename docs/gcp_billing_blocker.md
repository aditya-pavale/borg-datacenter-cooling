# BigQuery Free-Tier Status (cells e-h)

## Current state (2026-09-19)
- **Project**: `borg-datacenter-cooling-v2`
- **Authentication**: working (`gcloud auth list`, ADC credentials present)
- **Billing**: `gcloud beta billing projects describe borg-datacenter-cooling-v2`
  → `billingEnabled: false` — **intentional, by explicit user decision**.
  This project stays on BigQuery Sandbox (free tier) rather than linking
  a payment method.
- **Cells extracted**: a, b, c, d (workload), all 8 cells (power,
  machine_events).
- **Cells blocked**: e, f, g, h (workload only) — blocked purely by the
  Sandbox free monthly bytes-scanned allowance, not by any technical or
  scientific issue.

## Approximate cost / quota accounting
- Sandbox allowance consumed so far: ~724 GB (cells a-d, heavier query
  that included `COUNT(DISTINCT machine_id)`).
- Cell e's attempt at the same heavier query (another ~181 GB) failed
  with `Quota exceeded: ... free query bytes scanned` — so the cap sits
  somewhere between ~724 GB and ~905 GB, empirically.
- The query was optimized afterward (`extract_clusterdata_workload_lite.sql`,
  drops the `COUNT(DISTINCT machine_id)` column): cells e+f+g+h now
  cost an estimated **673,348,771,176 bytes (~627 GiB) combined**
  (dry-run verified), which should fit one full reset cycle's allowance.
- No real-money cost has been, or will be, incurred — this project has
  no billing account attached, so there is no possibility of an
  unexpected charge.

## Exact remaining requirement
Nothing from the user is required except time: BigQuery Sandbox's free
monthly quota resets automatically. Once it does,
`scripts/bigquery/run_clusterdata_extraction.sh` extracts e-h with no
code changes (it already skips cells with existing output). No GCP
console action, payment method, or account change is needed under the
"stay free" decision.

## If this decision is ever revisited
Should the free allowance turn out to reset on a longer cycle than
expected, or a smaller amount than estimated, the only way to get e-h
sooner would be linking a billing account at
console.cloud.google.com/billing/linkedaccount?project=borg-datacenter-cooling-v2
— a human-only, account-level action. This is not being pursued per the
user's explicit instruction to stay completely free.
