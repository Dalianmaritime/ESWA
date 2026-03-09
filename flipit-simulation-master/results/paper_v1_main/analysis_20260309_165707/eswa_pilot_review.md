# ESWA-Oriented Pilot Review

## Framing
- Recommended manuscript framing: maritime non-traditional security, offshore critical infrastructure protection, and service continuity management.
- Avoid military, combat, or warfighting wording in the manuscript narrative.
- This result set should be described as a pilot study with two random seeds and no statistical significance claims.

## Pilot Gate
- Overall pilot status: PASS
- Cheat-FlipIt: PASS
- FlipIt: PASS

## Algorithm Review
### Cheat-FlipIt
- DRL avg_defender_return: 37.945
- Baseline avg_defender_return: -103.395
- DRL attacker_success_rate: 53.00%
- Baseline attacker_success_rate: 86.00%
- DRL false/missed: 0.024 / 0.048
- Baseline false/missed: 0.205 / 0.218

### FlipIt
- DRL avg_defender_return: 11.630
- Baseline avg_defender_return: -153.030
- DRL attacker_success_rate: 62.50%
- Baseline attacker_success_rate: 75.00%
- DRL false/missed: 0.038 / 0.084
- Baseline false/missed: 0.335 / 0.230

## Debug Recommendation
- Decision case: A
- Action: keep_current_algorithm
- Summary: Current DRL results already satisfy the pilot-level ESWA screening criteria.
- No immediate algorithm change is required before adding more random seeds.

## DRL Run Diagnostics
- paper_main_cheat_drl_seed42: best_checkpoint_episode=500, final_validation_score=46.123, last_eval_rising=True
- paper_main_cheat_drl_seed123: best_checkpoint_episode=799, final_validation_score=23.973, last_eval_rising=False
- paper_main_flipit_drl_seed42: best_checkpoint_episode=799, final_validation_score=10.057, last_eval_rising=False
- paper_main_flipit_drl_seed123: best_checkpoint_episode=650, final_validation_score=3.252, last_eval_rising=False