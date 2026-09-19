# Multi-Agent RL Design — Phase 10/11/23

## Algorithm: MAPPO (implemented from scratch)

Stable-Baselines3 does not support multi-agent training, and no
verified MARL framework was pre-installed. Per the master plan's
explicit prohibition on "pseudo-MAPPO" (three PPO heads relabeled as
agents), this project implements the core MAPPO components directly
in `src/marl/mappo.py`:

- **Centralized training / decentralized execution (CTDE).** At
  execution time, each agent's actor sees ONLY its own zone's local
  observation (`MultiAgentCoolingEnv._local_obs`, dimension 8: own
  temp, own margin, own workload, own forecast (2 steps), own previous
  action, ambient temp, time-of-day sin/cos). During training only, a
  centralized critic sees the concatenation of all 3 zones' local
  observations (`get_global_state()`, dimension 24). No global state,
  and no other agent's observation, is ever passed to an actor at
  execution time — verified in `tests/test_marl.py`.
- **Parameter sharing across agents.** A single actor network (and a
  single centralized critic) is used for all 3 zones, rather than 3
  separate networks. Justification: the 3 zones are structurally
  near-symmetric (a line-graph coupling where zone 1 has 2 neighbors
  and zones 0/2 have 1), the per-zone observation is already
  self-contained (zone identity is implicit in which zone's data is
  fed in), and with only ~1,786 training bins the data budget favors
  fewer parameters. This is validated empirically, not simply assumed
  — see the ablation note in `docs/experiment_plan.md`.
- **PPO update.** Standard clipped-surrogate PPO
  (`clip_ratio=0.2`), GAE(λ) advantage estimation from the centralized
  critic's value estimates, entropy bonus, on-policy rollouts of
  `rollout_length=96` steps (one episode) per update.
- **Action distribution.** A diagonal Gaussian in unbounded space,
  with the environment clipping to `[0,1]` at execution (the same
  convention SB3's default continuous-action `MlpPolicy` uses) — the
  boundary distortion this introduces to the log-probability is a
  standard, documented PPO simplification, not unique to this project.

## Credit assignment: shared/global reward

Every agent receives the **identical, global** reward at every step
(`MultiAgentCoolingEnv.step`: `rewards = {a: global_reward for a in
agents}`). Alternatives considered:

- **Local/per-zone reward** (each agent penalized only for its own
  zone's energy/violation): rejected as the primary signal because the
  thermal model is coupled — zone 1's temperature depends partly on
  zones 0 and 2's heat and cooling choices (`docs/thermal_model.md`).
  A local reward would systematically misattribute credit/blame for
  coupling-driven temperature changes to the wrong agent (e.g. zone 1
  could be penalized for a violation actually caused by zone 0
  underscooling).
- **Global reward** (chosen): correctly reflects that the true
  objective (`docs/problem_statement.md`) is facility-wide energy and
  safety, and sidesteps the coupling misattribution problem, at the
  cost of a noisier per-agent learning signal (each agent's action
  affects only 1/3 of the global reward's information content).
  This is the standard, simplest defensible choice for a small
  (3-agent), tightly-coupled system, and is what is implemented and
  evaluated. No counterfactual/difference-reward mechanism was added,
  per the master plan's instruction not to add complexity "merely
  because it sounds advanced" without validation showing it is needed.

## No future-data leakage in the multi-agent state

Verified in `tests/test_marl.py` and `tests/test_environment.py`:
neither the local observation nor the global critic state ever
contains a value from `workload[t+1:]` — the `forecast` field is a
pre-computed, causal GRU output (`scripts/precompute_forecasts.py`),
and `workload` is only ever indexed at the current step.

## Single-agent PPO baseline (Phase 9/24)

Uses Stable-Baselines3's `PPO` with the default continuous
`MlpPolicy`, on `SingleAgentCoolingEnv` — built on the identical
`CoolingCore` as the multi-agent env (`docs/data_preprocessing.md`,
`tests/test_environment.py::test_single_and_multi_agent_envs_agree_on_physics`),
so the physics, workload windows, reward, and safety threshold are
byte-identical between the PPO and MAPPO comparisons (master plan
Phase 28 fairness requirement).
