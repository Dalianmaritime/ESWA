#!/usr/bin/env python3
"""Run the full V2 experiment bundle and trigger V2 analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(command):
    print(f"[RUN] {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run all V2 TRC experiments.")
    parser.add_argument("--smoke", action="store_true", help="Run reduced-episode smoke tests.")
    args = parser.parse_args()

    python = sys.executable

    drl_flags = []
    baseline_flags = []
    if args.smoke:
        print("[INFO] Smoke mode validates the V2 pipeline only; use full runs for policy-quality comparisons.", flush=True)
        drl_flags = [
            "--training-episodes",
            "12",
            "--evaluation-episodes",
            "4",
            "--final-evaluation-episodes",
            "6",
            "--evaluation-frequency",
            "3",
        ]
        baseline_flags = ["--episodes", "6"]

    runs = [
        [python, "run_trc_full_training_v2.py", "configs/trc_signal_cheat_v2.yml", *drl_flags],
        [python, "run_trc_full_training_v2.py", "configs/trc_signal_flipit_v2.yml", *drl_flags],
        [python, "run_traditional_experiment_v2.py", "configs/trc_signal_baseline_cheat_v2.yml", *baseline_flags],
        [python, "run_traditional_experiment_v2.py", "configs/trc_signal_baseline_flipit_v2.yml", *baseline_flags],
    ]
    for command in runs:
        run_command(command)

    analysis_command = [python, "analysis/trc_signal_analysis_v2.py"]
    if args.smoke:
        analysis_command.append("--latest-only")
    run_command(analysis_command)


if __name__ == "__main__":
    main()
