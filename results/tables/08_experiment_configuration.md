| parameter                                  | value       |
|:-------------------------------------------|:------------|
| bin_seconds                                | 900         |
| n_zones                                    | 3           |
| cluster_for_zones                          | all         |
| train/val/test split                       | 0.6/0.2/0.2 |
| forecast horizon_steps                     | 2           |
| forecast lookback_steps                    | 16          |
| thermal_mass_kwh_per_c                     | 0.5         |
| cooling_max_kw                             | 0.5         |
| safety_limit_c                             | 27.0        |
| reward: energy_weight                      | 1.0         |
| reward: safety_penalty_weight              | 50.0        |
| reward: proximity_weight                   | 10.0        |
| reward: shield_intervention_penalty_weight | 5.0         |
| PPO total_timesteps                        | 100000      |
| MAPPO total_updates                        | 300         |
| RL seeds                                   | [0, 1, 2]   |