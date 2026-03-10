# Maritime Cheat-FlipIt V2 Experiment Summary

## Experiment
- ID: smoke_flipit_drl_seed42
- Type: drl
- Mode: flipit
- Timestamp: 2026-03-10T15:18:50.643756
- Sample size: 5

## Metric Conventions
- Training objective: defender_training_return (shaped reward used for optimization)
- Reported utility: avg_defender_return (raw utility used for reporting)
- Checkpoint selection metric: constrained_operational
- Best checkpoint episode: 49
- Early stopping enabled: True
- Early stopping triggered: False
- Completed training episodes: 50

## Final Performance
- Attacker success rate: 60.00%
- Defender control rate: 0.383
- Attacker control rate: 0.617
- Avg defender return: -60.400
- Avg attacker return: -26.900
- Avg false response rate: 0.247
- Avg missed response rate: 0.087
- Avg inspection precision: 0.243
- Avg episode length: 60.00
- Attacker resource collapse rate: 0.00%
- Defender resource collapse rate: 0.00%
- Avg final attacker budget: -1.500
- Avg final defender budget: 56.900
- Avg attacker below-guarantee steps: 2.800
- Avg defender below-guarantee steps: 0.000
- Avg defender spent budget: 102.000
- Avg defender control per cost: 0.209

## Training Diagnostics
- Avg defender training return: -106.610
