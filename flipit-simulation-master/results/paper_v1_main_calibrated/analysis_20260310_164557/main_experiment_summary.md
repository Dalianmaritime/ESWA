# Paper V1 Main Experiment Summary

- manifest: E:\1031\flipit-simulation-master\results\paper_v1_main_calibrated\paper_main_manifest_20260310_164556.json
- Note: this is a pilot experiment with the currently selected seeds; it is suitable for algorithm screening, not final statistical claims.

## Cheat-FlipIt
### DRL
- seeds: [42, 123]
- defender_control_rate: 0.749 +- 0.079
- attacker_success_rate: 15.50% +- 0.092
- avg_defender_return: 93.550 +- 26.842
- avg_defender_training_return: 55.393 +- 24.231
- avg_defender_spent_budget: 74.980 +- 6.774
- avg_defender_control_per_cost: 0.613 +- 0.112
- avg_false_response_rate: 0.026 +- 0.010
- avg_missed_response_rate: 0.016 +- 0.006
- avg_inspection_precision: 0.176 +- 0.060
- attacker_resource_collapse_rate: 8.50% +- 0.021
- defender_resource_collapse_rate: 0.00% +- 0.000
- avg_final_attacker_budget: -0.443 +- 0.180
- avg_final_defender_budget: 129.440 +- 19.288
- avg_attacker_below_guarantee_steps: 3.335 +- 0.403
- avg_defender_below_guarantee_steps: 0.010 +- 0.014

### Threshold baseline
- seeds: [42, 123]
- defender_control_rate: 0.571 +- 0.016
- attacker_success_rate: 27.50% +- 0.007
- avg_defender_return: -12.895 +- 10.628
- avg_defender_training_return: -35.801 +- 7.300
- avg_defender_spent_budget: 91.275 +- 2.906
- avg_defender_control_per_cost: 0.389 +- 0.019
- avg_false_response_rate: 0.196 +- 0.009
- avg_missed_response_rate: 0.127 +- 0.003
- avg_inspection_precision: 0.274 +- 0.010
- attacker_resource_collapse_rate: 19.00% +- 0.028
- defender_resource_collapse_rate: 1.50% +- 0.007
- avg_final_attacker_budget: -0.247 +- 0.047
- avg_final_defender_budget: 83.626 +- 6.435
- avg_attacker_below_guarantee_steps: 1.850 +- 0.099
- avg_defender_below_guarantee_steps: 0.030 +- 0.014

## FlipIt
### DRL
- seeds: [42, 123]
- defender_control_rate: 0.791 +- 0.004
- attacker_success_rate: 4.50% +- 0.021
- avg_defender_return: 143.435 +- 3.543
- avg_defender_training_return: 79.305 +- 0.591
- avg_defender_spent_budget: 34.675 +- 3.981
- avg_defender_control_per_cost: 1.606 +- 0.149
- avg_false_response_rate: 0.018 +- 0.008
- avg_missed_response_rate: 0.040 +- 0.002
- avg_inspection_precision: 0.198 +- 0.010
- attacker_resource_collapse_rate: 0.00% +- 0.000
- defender_resource_collapse_rate: 0.00% +- 0.000
- avg_final_attacker_budget: -0.374 +- 0.147
- avg_final_defender_budget: 180.506 +- 3.428
- avg_attacker_below_guarantee_steps: 4.115 +- 0.247
- avg_defender_below_guarantee_steps: 0.000 +- 0.000

### Threshold baseline
- seeds: [42, 123]
- defender_control_rate: 0.677 +- 0.002
- attacker_success_rate: 17.50% +- 0.021
- avg_defender_return: 39.295 +- 3.316
- avg_defender_training_return: -3.011 +- 1.567
- avg_defender_spent_budget: 79.345 +- 2.270
- avg_defender_control_per_cost: 0.586 +- 0.002
- avg_false_response_rate: 0.158 +- 0.003
- avg_missed_response_rate: 0.107 +- 0.000
- avg_inspection_precision: 0.204 +- 0.005
- attacker_resource_collapse_rate: 0.00% +- 0.000
- defender_resource_collapse_rate: 2.00% +- 0.000
- avg_final_attacker_budget: -0.369 +- 0.112
- avg_final_defender_budget: 119.048 +- 2.486
- avg_attacker_below_guarantee_steps: 3.215 +- 0.163
- avg_defender_below_guarantee_steps: 0.040 +- 0.000

## ESWA Review
- This study is framed as a pilot study with two random seeds for maritime non-traditional security and critical maritime infrastructure protection.
- The current results should be interpreted as initial consistency trends rather than statistical significance claims.
- Pilot screening result: PASS
- Recommended debug path: Case A (keep_current_algorithm)
- Review summary: Current DRL results already satisfy the pilot-level ESWA screening criteria.
