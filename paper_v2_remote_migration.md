# Paper V2 Remote Migration Guide

## Short Answer

Yes. You can move the project to the school remote machine, but you should move the workspace as a whole, not only `flipit-simulation-master`.

The current V2 entrypoints expect this sibling layout:

```text
<workspace-root>/
  flipit-simulation-master/
  gym-flipit-master/
```

This matters because the V2 runners insert `../gym-flipit-master` into `sys.path` at runtime.

## What To Copy

Required:

- `flipit-simulation-master/`
- `gym-flipit-master/`

Optional:

- `.git/` if you want history on the remote machine
- `flipit-simulation-master/results/` if you want to resume previous runs
- `flipit-simulation-master/results/paper_v1_main_5seed/` if you want to run ablation against an existing main baseline reference without rerunning main first

Safe to skip if you only want a clean rerun:

- old result folders
- `.pytest_cache/`
- temporary folders

## Python Version

Minimum:

- Python `3.10+`

Recommended:

- Python `3.10` or `3.11`

Reason:

- the codebase uses `int | None` style type syntax, which requires Python 3.10 or newer
- PyTorch wheels are usually easiest to install on Windows for Python 3.10 or 3.11

## Required Dependencies For The Current V2 Paper Chain

Core runtime:

- `torch`
- `numpy`
- `matplotlib`
- `PyYAML`
- `gymnasium`
- `gym`

Validation:

- `pytest`

Not required for the current V2 main/ablation/robustness chain:

- `torchvision`
- `torchaudio`
- `pandas`
- `seaborn`
- `scipy`
- `pettingzoo`
- `supersuit`
- `wandb`
- `plotly`
- `scikit-learn`
- `optuna`

Those extra packages appear in older or optional workflows, but they are not needed for the V2 paper pipeline you described.

## Automated Setup Script

From the copied workspace root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\paper_v2_remote_setup.ps1 -RunSmokeTests
```

Default behavior:

- creates `.venv-paper-v2`
- installs CPU PyTorch by default
- installs the minimal V2 dependency set
- installs local `gym_flipit` with `pip install -e`
- runs the three V2 smoke test files

Useful variants:

```powershell
powershell -ExecutionPolicy Bypass -File .\paper_v2_remote_setup.ps1 -CheckOnly
powershell -ExecutionPolicy Bypass -File .\paper_v2_remote_setup.ps1 -TorchChannel cu121
powershell -ExecutionPolicy Bypass -File .\paper_v2_remote_setup.ps1 -TorchChannel cu124
```

Use a CUDA channel only if the remote machine actually has a compatible NVIDIA driver and CUDA runtime.

## Recommended Run Flow On The Remote Machine

1. Copy the workspace with the sibling layout preserved.
2. Run `paper_v2_remote_setup.ps1`.
3. Change directory into `flipit-simulation-master/`.
4. Run the formal `main` package first.
5. Run policy-pattern analysis for `main`.
6. Run `ablation`.
7. Run `robustness`.
8. Run policy-pattern analysis for each package as needed.

The exact experiment commands are already documented here:

- `flipit-simulation-master/paper_v2_server_runbook.md`

## Important Resume And Reference Rules

If you want `--resume-missing` to continue an existing remote run:

- copy the corresponding result root to the remote machine first

If you want ablation to use an already existing `main` baseline reference:

- copy `flipit-simulation-master/results/paper_v1_main_5seed/`
- or rerun `main` on the remote machine before ablation

For robustness:

- baseline and DRL must stay in the same results root
- do not split the baseline runs into another folder if you want same-condition baseline-aware checkpoint selection to work

## Recommended Commands After Setup

```powershell
Set-Location .\flipit-simulation-master
$py = (Resolve-Path ..\.venv-paper-v2\Scripts\python.exe).Path
& $py run_paper_main_v2.py --seeds 42 123 2027 3407 8848 --results-subdir paper_v1_main_5seed --resume-missing
& $py analysis\analyze_policy_patterns_v2.py --results-root results/paper_v1_main_5seed --paper-group main
```

## Common Failure Modes

`ModuleNotFoundError: gym_flipit`

- `gym-flipit-master` was not copied as a sibling directory
- or `pip install -e gym-flipit-master` was not run

`Python 3.10+ is required`

- the remote machine is using an older interpreter

`baseline-aware` or reference-target related failures during ablation or robustness

- required baseline outputs are missing from the expected results root

PyTorch install problems on a machine without GPU support

- rerun the setup script with the default CPU channel

## Minimal Migration Checklist

- copy both repositories with sibling layout preserved
- use Python 3.10 or 3.11 if possible
- run `paper_v2_remote_setup.ps1`
- run the three smoke tests
- run `main` before `ablation`
- keep robustness baseline and DRL in the same results root
