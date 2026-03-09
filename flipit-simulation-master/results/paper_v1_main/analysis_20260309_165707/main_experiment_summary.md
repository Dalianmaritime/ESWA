# Paper V1 Main Experiment Summary

- manifest: E:\1031\flipit-simulation-master\results\paper_v1_main\paper_main_manifest_latest.json
- Note: this is a pilot experiment with the currently selected seeds; it is suitable for algorithm screening, not final statistical claims.

## Cheat-FlipIt
### DRL
- seeds: [42, 123]
- defender_control_rate: 0.476 +- 0.045
- attacker_success_rate: 53.00% +- 0.071
- avg_defender_return: 37.945 +- 14.743
- avg_defender_training_return: -19.791 +- 12.412
- avg_false_response_rate: 0.024 +- 0.001
- avg_missed_response_rate: 0.048 +- 0.019
- avg_inspection_precision: 0.238 +- 0.028

### Threshold baseline
- seeds: [42, 123]
- defender_control_rate: 0.361 +- 0.023
- attacker_success_rate: 86.00% +- 0.057
- avg_defender_return: -103.395 +- 12.212
- avg_defender_training_return: -104.283 +- 8.938
- avg_false_response_rate: 0.205 +- 0.011
- avg_missed_response_rate: 0.218 +- 0.008
- avg_inspection_precision: 0.302 +- 0.002

## FlipIt
### DRL
- seeds: [42, 123]
- defender_control_rate: 0.450 +- 0.020
- attacker_success_rate: 62.50% +- 0.007
- avg_defender_return: 11.630 +- 4.441
- avg_defender_training_return: -33.530 +- 4.776
- avg_false_response_rate: 0.038 +- 0.002
- avg_missed_response_rate: 0.084 +- 0.008
- avg_inspection_precision: 0.260 +- 0.033

### Threshold baseline
- seeds: [42, 123]
- defender_control_rate: 0.393 +- 0.010
- attacker_success_rate: 75.00% +- 0.028
- avg_defender_return: -153.030 +- 2.051
- avg_defender_training_return: -126.708 +- 2.577
- avg_false_response_rate: 0.335 +- 0.002
- avg_missed_response_rate: 0.230 +- 0.003
- avg_inspection_precision: 0.253 +- 0.002

## ESWA Review
- This study is framed as a pilot study with two random seeds for maritime non-traditional security and critical maritime infrastructure protection.
- The current results should be interpreted as initial consistency trends rather than statistical significance claims.
- Pilot screening result: PASS
- Recommended debug path: Case A (keep_current_algorithm)
- Review summary: Current DRL results already satisfy the pilot-level ESWA screening criteria.
