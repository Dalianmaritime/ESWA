#!/usr/bin/env python3
"""Run the paper-v1 V2 ablation matrix with explicit per-seed entries."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from paper_v2_batch_utils import (
    build_results_root,
    collect_completed_runs,
    format_duration,
    render_progress_bar,
    run_analysis,
    write_manifest,
    write_progress,
)
from run_trc_full_training_v2 import SignalDRLTrainingExperimentV2
from signal_v2_utils import write_baseline_reference_targets_from_source

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_SUBDIR = "paper_v1_ablation"
DEFAULT_BASELINE_REFERENCE_ROOT = SCRIPT_DIR / "results" / "paper_v1_main_calibrated"
MANIFEST_STEM = "paper_ablation_manifest"
PROGRESS_FILENAME = "paper_ablation_progress_latest.json"
VARIANT_ORDER = [
    "full",
    "no_constrained_selection",
    "no_resource_sustainability",
    "no_reward_shaping",
    "no_action_mask",
    "no_signal_features",
]

EXPERIMENT_MATRIX = [
    {
        "scenario_id": "cheat",
        "method_id": "drl",
        "variant_id": "full",
        "variant_label": "Full calibrated V2",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "ablation" / "ablation_cheat_full.yml",
    },
    {
        "scenario_id": "cheat",
        "method_id": "drl",
        "variant_id": "no_constrained_selection",
        "variant_label": "No constrained selection",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "ablation" / "ablation_cheat_no_constrained_selection.yml",
    },
    {
        "scenario_id": "cheat",
        "method_id": "drl",
        "variant_id": "no_resource_sustainability",
        "variant_label": "No resource sustainability",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "ablation" / "ablation_cheat_no_resource_sustainability.yml",
    },
    {
        "scenario_id": "cheat",
        "method_id": "drl",
        "variant_id": "no_reward_shaping",
        "variant_label": "No reward shaping",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "ablation" / "ablation_cheat_no_reward_shaping.yml",
    },
    {
        "scenario_id": "cheat",
        "method_id": "drl",
        "variant_id": "no_action_mask",
        "variant_label": "No action mask",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "ablation" / "ablation_cheat_no_action_mask.yml",
    },
    {
        "scenario_id": "cheat",
        "method_id": "drl",
        "variant_id": "no_signal_features",
        "variant_label": "No signal features",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "ablation" / "ablation_cheat_no_signal_features.yml",
    },
]


def build_experiment_id(scenario_id: str, variant_id: str, seed: int) -> str:
    return f"paper_ablation_{scenario_id}_{variant_id}_seed{seed}"


def build_run_plan(selected_entries: List[Dict[str, Any]], seeds: List[int]) -> List[Tuple[Dict[str, Any], int]]:
    scenario_rank = {"cheat": 0, "flipit": 1}
    variant_rank = {variant_id: index for index, variant_id in enumerate(VARIANT_ORDER)}
    planned_runs = [(entry, int(seed)) for entry in selected_entries for seed in seeds]
    planned_runs.sort(
        key=lambda item: (
            scenario_rank.get(item[0]["scenario_id"], 99),
            variant_rank.get(item[0]["variant_id"], 99),
            int(item[1]),
        )
    )
    return planned_runs


def ensure_baseline_reference(results_root: Path, baseline_reference_root: Path, scenarios: List[str]) -> Path:
    return write_baseline_reference_targets_from_source(
        source_results_root=baseline_reference_root,
        destination_results_root=results_root,
        scenarios=scenarios,
    )


def run_single_experiment(entry: Dict[str, Any], args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    experiment_id = build_experiment_id(entry["scenario_id"], entry["variant_id"], seed)
    print(
        f"[RUN] scenario={entry['scenario_id']} variant={entry['variant_id']} seed={seed} config={entry['config_path'].name}",
        flush=True,
    )
    experiment = SignalDRLTrainingExperimentV2(
        config_path=str(entry["config_path"]),
        training_episodes=args.training_episodes,
        evaluation_episodes=args.evaluation_episodes,
        final_evaluation_episodes=args.final_evaluation_episodes,
        evaluation_frequency=args.evaluation_frequency,
        routine_eval_episodes=args.routine_eval_episodes,
        confirmation_eval_episodes=args.confirmation_eval_episodes,
        random_seed=seed,
        experiment_id=experiment_id,
        results_subdir=str(args.results_subdir),
    )
    results = experiment.run()
    performance = results["final_performance"]
    print(
        f"[DONE] {experiment_id} control={performance['defender_control_rate']:.3f} raw={performance['avg_defender_return']:.3f}",
        flush=True,
    )
    return {
        "experiment_id": experiment_id,
        "paper_group": "ablation",
        "scenario_id": entry["scenario_id"],
        "method_id": entry["method_id"],
        "variant_id": entry["variant_id"],
        "variant_label": entry["variant_label"],
        "seed": int(seed),
        "config_path": str(Path(entry["config_path"]).resolve()),
        "result_dir": str(experiment.results_dir.resolve()),
        "final_performance": performance,
    }


def collect_run(
    entry: Dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    completed_runs: Dict[Tuple[Any, ...], Dict[str, Any]],
) -> Dict[str, Any]:
    run_key = ("ablation", entry["scenario_id"], entry["method_id"], entry["variant_id"], None, None, int(seed))
    if args.resume_missing and run_key in completed_runs:
        existing = completed_runs[run_key]
        print(
            f"[SKIP] scenario={entry['scenario_id']} variant={entry['variant_id']} seed={seed} reuse={Path(existing['result_dir']).name}",
            flush=True,
        )
        return existing
    return run_single_experiment(entry, args, int(seed))


def main():
    parser = argparse.ArgumentParser(description="Run paper-v1 ablation experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2027])
    parser.add_argument("--scenarios", nargs="+", choices=["cheat"], default=["cheat"])
    parser.add_argument("--variants", nargs="+", choices=VARIANT_ORDER, default=list(VARIANT_ORDER))
    parser.add_argument("--training-episodes", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--final-evaluation-episodes", type=int, default=None)
    parser.add_argument("--evaluation-frequency", type=int, default=None)
    parser.add_argument("--routine-eval-episodes", type=int, default=None)
    parser.add_argument("--confirmation-eval-episodes", type=int, default=None)
    parser.add_argument("--baseline-reference-root", default=str(DEFAULT_BASELINE_REFERENCE_ROOT))
    parser.add_argument("--resume-missing", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    args = parser.parse_args()

    selected_entries = [
        entry
        for entry in EXPERIMENT_MATRIX
        if entry["scenario_id"] in args.scenarios and entry["variant_id"] in args.variants
    ]
    if not selected_entries:
        raise SystemExit("No ablation experiments selected. Check --scenarios and --variants.")

    results_root = build_results_root(SCRIPT_DIR, args.results_subdir)
    completed_runs = collect_completed_runs(results_root) if args.resume_missing else {}
    planned_runs = build_run_plan(selected_entries, [int(seed) for seed in args.seeds])
    total_runs = len(planned_runs)
    batch_started_at = time.time()
    manifest = {
        "paper_group": "ablation",
        "timestamp": datetime.now().isoformat(),
        "scenarios": sorted({entry["scenario_id"] for entry in selected_entries}),
        "variants": [variant_id for variant_id in VARIANT_ORDER if variant_id in args.variants],
        "seeds": [int(seed) for seed in args.seeds],
        "resume_missing": bool(args.resume_missing),
        "results_subdir": str(args.results_subdir),
        "baseline_reference_root": str(Path(args.baseline_reference_root).resolve()),
        "runs": [],
    }
    completed_count = 0
    run_durations: List[float] = []
    selected_scenarios = sorted({entry["scenario_id"] for entry in selected_entries})
    baseline_reference_path = ensure_baseline_reference(
        results_root=results_root,
        baseline_reference_root=Path(args.baseline_reference_root).resolve(),
        scenarios=selected_scenarios,
    )
    write_progress(
        results_root,
        PROGRESS_FILENAME,
        {
            "status": "running",
            "results_subdir": str(args.results_subdir),
            "baseline_reference_path": str(baseline_reference_path),
            "completed_runs": 0,
            "total_runs": total_runs,
            "current_run": None,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "updated_at": datetime.now().isoformat(),
        },
    )

    for entry, seed in planned_runs:
        print(
            f"[PROGRESS] {render_progress_bar(completed_count, total_runs)} next={entry['scenario_id']}/{entry['variant_id']}/seed{seed}",
            flush=True,
        )
        current_experiment_id = build_experiment_id(entry["scenario_id"], entry["variant_id"], int(seed))
        average_duration = sum(run_durations) / len(run_durations) if run_durations else 0.0
        remaining_runs = total_runs - completed_count
        eta_seconds = average_duration * remaining_runs if run_durations else None
        write_progress(
            results_root,
            PROGRESS_FILENAME,
            {
                "status": "running",
                "results_subdir": str(args.results_subdir),
                "baseline_reference_path": str(baseline_reference_path),
                "completed_runs": completed_count,
                "total_runs": total_runs,
                "current_run": {
                    "experiment_id": current_experiment_id,
                    "scenario_id": entry["scenario_id"],
                    "method_id": entry["method_id"],
                    "variant_id": entry["variant_id"],
                    "seed": int(seed),
                    "result_dir_glob": str(results_root / f"{current_experiment_id}_*"),
                },
                "elapsed_seconds": float(time.time() - batch_started_at),
                "eta_seconds": None if eta_seconds is None else float(max(0.0, eta_seconds)),
                "updated_at": datetime.now().isoformat(),
            },
        )
        start_time = time.time()
        run_key = ("ablation", entry["scenario_id"], entry["method_id"], entry["variant_id"], None, None, int(seed))
        reused = bool(args.resume_missing and run_key in completed_runs)
        run_payload = collect_run(entry, args, int(seed), completed_runs)
        manifest["runs"].append(run_payload)
        elapsed = time.time() - start_time
        if not reused:
            run_durations.append(elapsed)
        completed_count += 1
        average_duration = sum(run_durations) / len(run_durations) if run_durations else 0.0
        remaining_runs = total_runs - completed_count
        eta_seconds = average_duration * remaining_runs
        write_progress(
            results_root,
            PROGRESS_FILENAME,
            {
                "status": "running" if completed_count < total_runs else "completed",
                "results_subdir": str(args.results_subdir),
                "baseline_reference_path": str(baseline_reference_path),
                "completed_runs": completed_count,
                "total_runs": total_runs,
                "current_run": None,
                "elapsed_seconds": float(time.time() - batch_started_at),
                "eta_seconds": float(max(0.0, eta_seconds)),
                "updated_at": datetime.now().isoformat(),
            },
        )
        print(
            f"[PROGRESS] {render_progress_bar(completed_count, total_runs)} elapsed={format_duration(elapsed)} eta={format_duration(eta_seconds)}",
            flush=True,
        )

    manifest_path = write_manifest(results_root, MANIFEST_STEM, manifest)
    print(f"[INFO] Manifest written to {manifest_path}", flush=True)
    if not args.skip_analysis:
        run_analysis(
            script_dir=SCRIPT_DIR,
            analysis_script=SCRIPT_DIR / "analysis" / "analyze_paper_ablation_v2.py",
            manifest_path=manifest_path,
            results_root=results_root,
        )


if __name__ == "__main__":
    main()
