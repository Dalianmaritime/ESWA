# ESWA-Oriented Pilot Review

## Framing
- Recommended manuscript framing: maritime non-traditional security, offshore critical infrastructure protection, and service continuity management.
- Avoid military, combat, or warfighting wording in the manuscript narrative.
- This result set should be described as a pilot study with two random seeds and no statistical significance claims.

## Pilot Gate
- Overall pilot status: REQUIRES DEBUG
- Cheat-FlipIt: PASS
- FlipIt: FAIL
  - DRL attacker success rate is higher than baseline

## Algorithm Review
### Cheat-FlipIt
- DRL avg_defender_return: 93.550
- Baseline avg_defender_return: -12.895
- DRL avg_defender_control_per_cost: 0.613
- Baseline avg_defender_control_per_cost: 0.389
- DRL attacker_success_rate: 15.50%
- Baseline attacker_success_rate: 27.50%
- DRL false/missed: 0.026 / 0.016
- Baseline false/missed: 0.196 / 0.127
- DRL collapse rates (A/D): 8.50% / 0.00%
- Baseline collapse rates (A/D): 19.00% / 1.50%

### FlipIt
- DRL avg_defender_return: 77.160
- Baseline avg_defender_return: 39.295
- DRL avg_defender_control_per_cost: 0.613
- Baseline avg_defender_control_per_cost: 0.586
- DRL attacker_success_rate: 19.50%
- Baseline attacker_success_rate: 17.50%
- DRL false/missed: 0.042 / 0.047
- Baseline false/missed: 0.158 / 0.107
- DRL collapse rates (A/D): 0.00% / 0.00%
- Baseline collapse rates (A/D): 0.00% / 2.00%

## Debug Recommendation
- Decision case: C
- Action: extend_training_then_retune_shaping
- Summary: Constrained checkpoint selection is active, but operational superiority is still not consistent enough across scenarios.
- First change: increase DRL training to 1200 episodes and evaluation_frequency=100.
- Only if instability remains should reward shaping be tuned.

## DRL Run Diagnostics
- paper_main_cheat_drl_seed42: strategy=constrained_operational, best_checkpoint_episode=300, final_validation_score=111.136, feasible_under_baseline=True, last_eval_rising=False
- paper_main_cheat_drl_seed123: strategy=constrained_operational, best_checkpoint_episode=800, final_validation_score=73.327, feasible_under_baseline=True, last_eval_rising=False
- paper_main_flipit_drl_seed42: strategy=constrained_operational, best_checkpoint_episode=150, final_validation_score=49.437, feasible_under_baseline=False, last_eval_rising=False
- paper_main_flipit_drl_seed123: strategy=constrained_operational, best_checkpoint_episode=100, final_validation_score=98.475, feasible_under_baseline=True, last_eval_rising=False