# GPT-5.4 Experiment Task Brief for Current V2 Codebase

## Purpose
This brief is the authoritative execution note for continuing the paper experiments on the current V2 codebase.

Use this brief instead of older notes, `legacy/` scripts, or outdated README examples.

The goal is to extend the current paper-ready V2 experiment chain from:
- a completed `main` pilot study

to:
- a publishable `main + ablation + robustness + managerial analysis` package

for a paper on:
- maritime non-traditional security
- offshore critical infrastructure protection
- resource-constrained adversarial defense
- signal deception and takeover games
- DQN-based adaptive defender policies


## Critical Rule
Only use the **current V2 chain**.

Do **not** reuse:
- `legacy/`
- old TRC scripts in `legacy/run_*`
- old conclusions or markdown claims in `legacy/section_*.md`
- old result directories under `results/paper_v1_main/`

The current valid chain is the calibrated V2 branch centered on:
- `run_trc_full_training_v2.py`
- `run_traditional_experiment_v2.py`
- `run_paper_main_v2.py`
- `signal_v2_utils.py`
- `analysis/analyze_paper_main_v2.py`
- `configs/paper_v1/main/*.yml`


## Current Repository Truth
As of the current repo state, the following are already implemented and valid:

### Main experiment code
- `e:/1031/flipit-simulation-master/run_paper_main_v2.py`
- `e:/1031/flipit-simulation-master/run_trc_full_training_v2.py`
- `e:/1031/flipit-simulation-master/run_traditional_experiment_v2.py`
- `e:/1031/flipit-simulation-master/analysis/analyze_paper_main_v2.py`

### Current valid main configs
- `e:/1031/flipit-simulation-master/configs/paper_v1/main/main_cheat_drl.yml`
- `e:/1031/flipit-simulation-master/configs/paper_v1/main/main_flipit_drl.yml`
- `e:/1031/flipit-simulation-master/configs/paper_v1/main/main_cheat_baseline.yml`
- `e:/1031/flipit-simulation-master/configs/paper_v1/main/main_flipit_baseline.yml`

### Current validated result root
- `e:/1031/flipit-simulation-master/results/paper_v1_main_calibrated`

### Current latest main artifacts
- manifest:
  - `e:/1031/flipit-simulation-master/results/paper_v1_main_calibrated/paper_main_manifest_latest.json`
- summary:
  - `e:/1031/flipit-simulation-master/results/paper_v1_main_calibrated/analysis_20260310_164557/main_experiment_summary.md`
- JSON summary:
  - `e:/1031/flipit-simulation-master/results/paper_v1_main_calibrated/analysis_20260310_164557/main_experiment_summary.json`
- ESWA review:
  - `e:/1031/flipit-simulation-master/results/paper_v1_main_calibrated/analysis_20260310_164557/eswa_pilot_review.md`

### Current valid experiment status
- The calibrated 2-seed main pilot now passes the ESWA-style pilot gate.
- Both `cheat` and `flipit` currently pass.
- The codebase now includes:
  - constrained checkpoint selection
  - resource sustainability mechanism
  - efficiency metrics such as `avg_defender_control_per_cost`
  - early stopping for `flipit`


## Important Missing Pieces
The following V2 paper components do **not** yet exist as real current-code entry points and should be created fresh:

### Missing experiment runners
- `run_paper_ablation_v2.py`
- `run_paper_robustness_v2.py`

### Missing analysis scripts
- `analysis/analyze_paper_ablation_v2.py`
- `analysis/analyze_paper_robustness_v2.py`

### Missing config directories
- `configs/paper_v1/ablation/`
- `configs/paper_v1/robustness/`

Do not search `legacy/` for replacements and treat them as reusable. They belong to the old chain and are not consistent with the current V2 environment, rewards, resource mechanics, or checkpoint-selection logic.


## Scientific Positioning
The paper should be framed as:
- maritime non-traditional security
- offshore critical infrastructure protection
- service continuity under deceptive and budget-constrained threats

Avoid:
- military
- warfighting
- combat-system framing

Use terms such as:
- deceptive signal
- false alarm induction
- takeover attempt
- inspection-response scheduling
- budget sustainability
- defender control continuity


## Current V2 Mechanics That Must Stay Fixed
Do not redesign these unless explicitly asked:

1. Resource sustainability mechanism
- base income + control bonus
- mild overdraft allowed
- guarantee-line failure rule

2. Training/report separation
- DRL optimizes shaped `training reward`
- paper reports raw `avg_defender_return` and operational metrics

3. Checkpoint selection
- must stay `constrained_operational`
- must stay baseline-aware

4. Main calibrated configs
- use the current `paper_v1/main` configs as the baseline reference implementation


## Immediate Objective
Build the next-stage paper experiment package on top of the current calibrated main branch.

The next-stage objective is:
1. keep the current main pipeline reproducible
2. extend the main study from pilot to formal seeds
3. add ablation
4. add robustness
5. add managerial interpretation outputs


## Phase 1: Main Experiment Extension
### Goal
Extend the current `main` experiment from `2 seeds` to `5 seeds`.

### Keep unchanged
- current V2 environment
- current constrained checkpoint selection
- current baseline logic
- current calibrated `cheat` and `flipit` configs

### Required seeds
Use:
- `42`
- `123`
- `2027`
- `3407`
- `8848`

### Required command path
Continue using:
- `e:/1031/flipit-simulation-master/run_paper_main_v2.py`

### Required results root
Use a new subdir, not the current pilot directory:
- `paper_v1_main_5seed`

### Expected deliverables
- 20 total runs:
  - `2 scenarios x 2 methods x 5 seeds`
- updated manifest
- updated main summary
- updated ESWA-style review
- same figure set as current main analysis

### Acceptance criteria
- all runs complete
- manifest contains all 20 entries
- analysis outputs regenerate without manual patching
- metrics include:
  - `attacker_success_rate`
  - `defender_control_rate`
  - `avg_defender_return`
  - `avg_false_response_rate`
  - `avg_missed_response_rate`
  - `avg_defender_spent_budget`
  - `avg_defender_control_per_cost`


## Phase 2: Ablation Study
### Goal
Quantify where the current DRL performance comes from.

### Scope
Only use V2.

### Recommended ablation scenarios
Start with `cheat` only.
If time permits, later mirror on `flipit`.

### Required ablation variants
Create configs and runner support for:

1. `full`
- current calibrated V2 DRL

2. `no_constrained_selection`
- keep training identical
- checkpoint selection falls back to raw validation score or old validation scalar

3. `no_resource_sustainability`
- use the same environment code path but disable the new sustainability mechanism via config switch if feasible
- if the environment is too entangled, implement this carefully as a config-driven compatibility mode
- if not feasible without environment instability, document the blocker clearly before modifying

4. `no_reward_shaping`
- DRL uses raw defender reward as training reward

5. `no_action_mask`
- disable budget/floor-aware action masking

6. `no_signal_features`
- remove signal/belief-sensitive inputs from the defender observation before policy inference

### Deliverables
Create:
- `configs/paper_v1/ablation/`
- `run_paper_ablation_v2.py`
- `analysis/analyze_paper_ablation_v2.py`

### Seeds
Start with:
- `42`
- `123`
- `2027`

### Output metrics
Must match main metrics.

### Acceptance criteria
- ablation scripts run end-to-end
- each ablation produces a valid result directory
- analysis compares all ablation variants in one summary


## Phase 3: Robustness Study
### Goal
Test whether the current DRL remains effective under scenario shifts.

### Priority order
Implement in this order:

1. budget stress
2. deception intensity
3. attack strength

### Required robustness families
Create configs for:

#### A. Budget stress
- low defender budget
- nominal budget
- high defender budget

#### B. Deception intensity
- low `cheat_emit_prob`
- nominal `cheat_emit_prob`
- high `cheat_emit_prob`

#### C. Attack strength
- lower takeover success
- nominal takeover success
- higher takeover success

### Deliverables
Create:
- `configs/paper_v1/robustness/`
- `run_paper_robustness_v2.py`
- `analysis/analyze_paper_robustness_v2.py`

### Seeds
Use:
- `42`
- `123`
- `2027`

### Scope
Start with:
- `cheat`
- DRL and baseline only

Later extension:
- add `flipit`


## Phase 4: Managerial Interpretation Layer
### Goal
Make the paper readable as a maritime decision-support study, not only an RL comparison.

### Required outputs
Add analysis for:
- action distribution by zone
- hold / inspect / respond ratios
- average resource trajectory
- collapse-event frequency and timing
- response timing after suspicious signal
- inspection-response behavior under `cheat` vs `flipit`

### Suggested files
- `analysis/analyze_policy_patterns_v2.py`
- or extend the existing analysis scripts if cleaner

### Required figure ideas
- zone-level action distribution
- budget trajectory plot
- collapse timing histogram
- response timing plot after suspicious signal


## Code Rules
When editing:
- prefer current V2 files only
- do not reintroduce legacy imports
- use `apply_patch` for file edits
- keep configs under `configs/paper_v1/...`
- keep result roots under `results/...`
- preserve current metric names

When implementing new runners:
- follow the style of `run_paper_main_v2.py`
- support:
  - `--seeds`
  - `--resume-missing`
  - `--results-subdir`
- write manifest files
- write progress files
- auto-run analysis at the end unless disabled

When implementing new analyses:
- follow the style of `analysis/analyze_paper_main_v2.py`
- always produce:
  - `.md` summary
  - `.json` summary
  - figures


## Testing Requirements
All new work must include tests or extend current tests.

At minimum:
- extend `tests/test_signal_v2_utils.py` for new utility logic
- add tests for any new config-driven logic if utilities are added

Before handing back results, run:
- `pytest e:/1031/flipit-simulation-master/tests -q`


## Current Main Result Reference
The new experiment package should treat the following as the current trusted baseline reference:

- `Cheat-FlipIt`
  - DRL `attacker_success_rate = 15.5%`
  - baseline `27.5%`
  - DRL `avg_defender_return = 93.55`
  - baseline `-12.895`
  - DRL `avg_defender_control_per_cost = 0.613`
  - baseline `0.389`

- `FlipIt`
  - DRL `attacker_success_rate = 4.5%`
  - baseline `17.5%`
  - DRL `avg_defender_return = 143.435`
  - baseline `39.295`
  - DRL `avg_defender_control_per_cost = 1.606`
  - baseline `0.586`

These numbers come from:
- `e:/1031/flipit-simulation-master/results/paper_v1_main_calibrated/analysis_20260310_164557/main_experiment_summary.md`


## Explicit Non-Goals
Do not spend time on:
- rewriting old README claims
- fixing non-V2 old scripts
- improving old `legacy` markdown
- changing the scientific framing to military language
- redesigning the environment before ablation and robustness are finished


## Final Expected Deliverable Set
After completion, the repo should contain:

1. a formal 5-seed main package
2. a working V2 ablation package
3. a working V2 robustness package
4. managerial interpretation figures
5. updated summaries suitable for drafting the experimental section of a journal paper


## Short Execution Order
Follow this order exactly:

1. preserve the current main calibrated branch
2. extend `main` to 5 seeds
3. implement `ablation`
4. implement `robustness`
5. add managerial interpretation analysis
6. regenerate summaries and figures

