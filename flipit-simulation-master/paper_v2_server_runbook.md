# Paper V2 Server Runbook

## Recommended Results Roots

- `results/paper_v1_main_5seed`
- `results/paper_v1_ablation_cheat_3seed`
- `results/paper_v1_robustness_cheat_3seed`

Each runner will also create:

- `*_manifest_latest.json`
- `*_progress_latest.json`
- `analysis_YYYYMMDD_HHMMSS/`
- `policy_analysis_YYYYMMDD_HHMMSS/` when `analysis/analyze_policy_patterns_v2.py` is run

## Recommended Execution Order

1. Run the formal 5-seed main package.
2. Run policy-pattern analysis on the 5-seed main package.
3. Run the cheat-only 3-seed ablation package.
4. Optionally run policy-pattern analysis on the ablation package.
5. Run the cheat-only 3-seed robustness package.
6. Run policy-pattern analysis on the robustness package.

## Commands

### 1. Formal 5-seed main

```bash
python run_paper_main_v2.py \
  --seeds 42 123 2027 3407 8848 \
  --results-subdir paper_v1_main_5seed \
  --resume-missing
```

### 2. Main managerial interpretation

```bash
python analysis/analyze_policy_patterns_v2.py \
  --results-root results/paper_v1_main_5seed \
  --paper-group main
```

### 3. Ablation package

Use the new 5-seed main results as the baseline-reference source for constrained checkpoint selection.

```bash
python run_paper_ablation_v2.py \
  --seeds 42 123 2027 \
  --results-subdir paper_v1_ablation_cheat_3seed \
  --baseline-reference-root results/paper_v1_main_5seed \
  --resume-missing
```

### 4. Optional ablation managerial interpretation

```bash
python analysis/analyze_policy_patterns_v2.py \
  --results-root results/paper_v1_ablation_cheat_3seed \
  --paper-group ablation
```

### 5. Robustness package

```bash
python run_paper_robustness_v2.py \
  --seeds 42 123 2027 \
  --results-subdir paper_v1_robustness_cheat_3seed \
  --resume-missing
```

### 6. Robustness managerial interpretation

```bash
python analysis/analyze_policy_patterns_v2.py \
  --results-root results/paper_v1_robustness_cheat_3seed \
  --paper-group robustness
```

## Manual Re-analysis Commands

The runners already auto-run their paired analysis scripts. Use these only when you need to regenerate summaries or figures without rerunning experiments.

```bash
python analysis/analyze_paper_main_v2.py --results-root results/paper_v1_main_5seed
python analysis/analyze_paper_ablation_v2.py --results-root results/paper_v1_ablation_cheat_3seed
python analysis/analyze_paper_robustness_v2.py --results-root results/paper_v1_robustness_cheat_3seed
```

## Notes

- Do not use any `legacy/` runner, config, markdown, or result root.
- `run_paper_ablation_v2.py` is cheat-only and DRL-only in the current V2 package.
- `run_paper_robustness_v2.py` is cheat-only and runs both baseline and DRL for each family/level pair.
- `run_paper_robustness_v2.py` must keep baseline runs in the same results root because DRL checkpoint selection builds same-condition baseline references from those baseline outputs.
