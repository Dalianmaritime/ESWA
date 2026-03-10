# Paper V1 Main Experiment Summary

- manifest: E:\1031\flipit-simulation-master\results\paper_v1_main_calibrated\paper_main_manifest_20260310_124739.json
- Note: this is a pilot experiment with the currently selected seeds; it is suitable for algorithm screening, not final statistical claims.

## Cheat-FlipIt
### DRL
- seeds: [42, 123]
- defender_control_rate: 0.507 +- 0.041
- attacker_success_rate: 48.00% +- 0.085
- avg_defender_return: 53.565 +- 10.585
- avg_defender_training_return: -10.690 +- 10.958
- avg_false_response_rate: 0.016 +- 0.000
- avg_missed_response_rate: 0.034 +- 0.004
- avg_inspection_precision: 0.221 +- 0.008
- attacker_resource_collapse_rate: 5.50% +- 0.035
- defender_resource_collapse_rate: 0.00% +- 0.000
- avg_final_attacker_budget: 0.269 +- 0.819
- avg_final_defender_budget: 117.396 +- 4.779
- avg_attacker_below_guarantee_steps: 1.985 +- 0.021
- avg_defender_below_guarantee_steps: 0.000 +- 0.000

### Threshold baseline
- seeds: [42, 123]
- defender_control_rate: 0.571 +- 0.016
- attacker_success_rate: 27.50% +- 0.007
- avg_defender_return: -12.895 +- 10.628
- avg_defender_training_return: -35.801 +- 7.300
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
- defender_control_rate: 0.546 +- 0.102
- attacker_success_rate: 43.00% +- 0.184
- avg_defender_return: 48.185 +- 17.161
- avg_defender_training_return: -1.299 +- 24.729
- avg_false_response_rate: 0.031 +- 0.008
- avg_missed_response_rate: 0.047 +- 0.014
- avg_inspection_precision: 0.288 +- 0.016
- attacker_resource_collapse_rate: 0.00% +- 0.000
- defender_resource_collapse_rate: 0.00% +- 0.000
- avg_final_attacker_budget: -1.182 +- 0.229
- avg_final_defender_budget: 113.516 +- 4.392
- avg_attacker_below_guarantee_steps: 3.195 +- 0.940
- avg_defender_below_guarantee_steps: 0.000 +- 0.000

### Threshold baseline
- seeds: [42, 123]
- defender_control_rate: 0.677 +- 0.002
- attacker_success_rate: 17.50% +- 0.021
- avg_defender_return: 39.295 +- 3.316
- avg_defender_training_return: -3.011 +- 1.567
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
- Pilot screening result: REQUIRES DEBUG
- Recommended debug path: Case B (switch_checkpoint_selection_metric)
- Review summary: Raw-return advantage exists, but operational stability is not consistent enough across scenarios.
