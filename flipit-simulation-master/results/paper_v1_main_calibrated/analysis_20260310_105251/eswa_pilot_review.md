# ESWA-Oriented Pilot Review

## Framing
- Recommended manuscript framing: maritime non-traditional security, offshore critical infrastructure protection, and service continuity management.
- Avoid military, combat, or warfighting wording in the manuscript narrative.
- This result set should be described as a pilot study with two random seeds and no statistical significance claims.

## Pilot Gate
- Overall pilot status: REQUIRES DEBUG
- Cheat-FlipIt: FAIL
  - DRL attacker success rate is higher than baseline
- FlipIt: FAIL
  - DRL attacker success rate is higher than baseline

## Algorithm Review
### Cheat-FlipIt
- DRL avg_defender_return: 59.085
- Baseline avg_defender_return: -12.895
- DRL attacker_success_rate: 30.50%
- Baseline attacker_success_rate: 27.50%
- DRL false/missed: 0.056 / 0.023
- Baseline false/missed: 0.196 / 0.127
- DRL collapse rates (A/D): 14.50% / 0.00%
- Baseline collapse rates (A/D): 19.00% / 1.50%

### FlipIt
- DRL avg_defender_return: 48.315
- Baseline avg_defender_return: 39.295
- DRL attacker_success_rate: 37.00%
- Baseline attacker_success_rate: 17.50%
- DRL false/missed: 0.047 / 0.046
- Baseline false/missed: 0.158 / 0.107
- DRL collapse rates (A/D): 0.00% / 0.00%
- Baseline collapse rates (A/D): 0.00% / 2.00%

## Debug Recommendation
- Decision case: B
- Action: switch_checkpoint_selection_metric
- Summary: Raw-return advantage exists, but operational stability is not consistent enough across scenarios.
- Validation selection metric to add: `avg_defender_return - 50 * avg_missed_response_rate - 20 * avg_false_response_rate`
- Keep training reward unchanged; only adjust checkpoint selection and rerun DRL.

## DRL Run Diagnostics
- paper_main_cheat_drl_seed42: best_checkpoint_episode=100, final_validation_score=58.526, last_eval_rising=False
- paper_main_cheat_drl_seed123: best_checkpoint_episode=50, final_validation_score=55.102, last_eval_rising=False
- paper_main_flipit_drl_seed42: best_checkpoint_episode=200, final_validation_score=45.048, last_eval_rising=True
- paper_main_flipit_drl_seed123: best_checkpoint_episode=50, final_validation_score=45.130, last_eval_rising=False