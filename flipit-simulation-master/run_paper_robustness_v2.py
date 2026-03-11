#!/usr/bin/env python3
"""Run the paper-v1 V2 robustness matrix with explicit per-seed entries."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from paper_v2_batch_utils import (
    build_results_root,
    collect_completed_runs,
    format_duration,
    render_progress_bar,
    run_analysis,
    write_manifest,
    write_progress,
)
from run_traditional_experiment_v2 import TraditionalExperimentV2
from run_trc_full_training_v2 import SignalDRLTrainingExperimentV2
from signal_v2_utils import save_json

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_SUBDIR = "paper_v1_robustness"
MANIFEST_STEM = "paper_robustness_manifest"
PROGRESS_FILENAME = "paper_robustness_progress_latest.json"
FAMILY_ORDER = ["budget_stress", "deception_intensity", "attack_strength"]
LEVEL_ORDER = ["low", "nominal", "high"]
METHOD_ORDER = ["baseline", "drl"]


def _build_matrix_entry(family: str, level: str, method_id: str) -> Dict[str, Any]:
    return {
        "paper_group": "robustness",
        "scenario_id": "cheat",
        "method_id": method_id,
        "variant_id": f"{family}_{level}",
        "robustness_family": family,
        "robustness_level": level,
        "config_path": (
            SCRIPT_DIR
            / "configs"
            / "paper_v1"
            / "robustness"
            / f"robustness_cheat_{family}_{level}_{method_id}.yml"
        ),
    }


EXPERIMENT_MATRIX = [
    _build_matrix_entry(family, level, method_id)
    for family in FAMILY_ORDER
    for level in LEVEL_ORDER
    for method_id in METHOD_ORDER
]


def build_experiment_id(scenario_id: str, family: str, level: str, method_id: str, seed: int) -> str:
    return f"paper_robustness_{scenario_id}_{family}_{level}_{method_id}_seed{seed}"


def build_run_plan(selected_entries: List[Dict[str, Any]], seeds: List[int]) -> List[Tuple[Dict[str, Any], int]]:
    method_rank = {method_id: index for index, method_id in enumerate(METHOD_ORDER)}
    family_rank = {family_id: index for index, family_id in enumerate(FAMILY_ORDER)}
    level_rank = {level_id: index for index, level_id in enumerate(LEVEL_ORDER)}
    planned_runs = [(entry, int(seed)) for entry in selected_entries for seed in seeds]
    planned_runs.sort(
        key=lambda item: (
            method_rank.get(item[0]["method_id"], 99),
            family_rank.get(item[0]["robustness_family"], 99),
            level_rank.get(item[0]["robustness_level"], 99),
            int(item[1]),
        )
    )
    return planned_runs


def write_local_baseline_reference(
    results_root: Path,
    completed_runs: Dict[Tuple[Any, ...], Dict[str, Any]],
    scenario_id: str,
    robustness_family: str,
    robustness_level: str,
) -> Path:
    matching_runs = [
        payload
        for payload in completed_runs.values()
        if payload.get("paper_group") == "robustness"
        and payload.get("scenario_id") == scenario_id
        and payload.get("method_id") == "baseline"
        and payload.get("robustness_family") == robustness_family
        and payload.get("robustness_level") == robustness_level
    ]
    if not matching_runs:
        raise FileNotFoundError(
            "Missing baseline references for "
            f"scenario={scenario_id} family={robustness_family} level={robustness_level} under {results_root}"
        )

    performances = [run["final_performance"] for run in matching_runs]
    payload = {
        "generated_at": datetime.now().isoformat(),
        "results_root": str(results_root.resolve()),
        "reference_scope": {
            "scenario_id": scenario_id,
            "robustness_family": robustness_family,
            "robustness_level": robustness_level,
        },
        "scenarios": {
            scenario_id: {
                "num_runs": len(matching_runs),
                "seeds": sorted(int(run["seed"]) for run in matching_runs),
                "source_result_dirs": [
                    str(run["result_dir"])
                    for run in sorted(matching_runs, key=lambda item: int(item["seed"]))
                ],
                "attacker_success_rate": float(np.mean([item["attacker_success_rate"] for item in performances])),
                "defender_control_rate": float(np.mean([item["defender_control_rate"] for item in performances])),
                "avg_defender_return": float(np.mean([item["avg_defender_return"] for item in performances])),
            }
        },
    }
    output_path = results_root / "baseline_reference_targets.json"
    save_json(output_path, payload)
    return output_path


def run_single_experiment(entry: Dict[str, Any], args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    experiment_id = build_experiment_id(
        entry["scenario_id"],
        entry["robustness_family"],
        entry["robustness_level"],
        entry["method_id"],
        seed,
    )
    print(
        f"[RUN] family={entry['robustness_family']} level={entry['robustness_level']} method={entry['method_id']} seed={seed} config={entry['config_path'].name}",
        flush=True,
    )

    if entry["method_id"] == "drl":
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
    else:
        experiment = TraditionalExperimentV2(
            config_path=str(entry["config_path"]),
            episodes=args.final_evaluation_episodes,
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
        "paper_group": "robustness",
        "scenario_id": entry["scenario_id"],
        "method_id": entry["method_id"],
        "variant_id": entry["variant_id"],
        "robustness_family": entry["robustness_family"],
        "robustness_level": entry["robustness_level"],
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
    run_key = (
        "robustness",
        entry["scenario_id"],
        entry["method_id"],
        entry["variant_id"],
        entry["robustness_family"],
        entry["robustness_level"],
        int(seed),
    )
    if args.resume_missing and run_key in completed_runs:
        existing = completed_runs[run_key]
        print(
            f"[SKIP] family={entry['robustness_family']} level={entry['robustness_level']} method={entry['method_id']} seed={seed} reuse={Path(existing['result_dir']).name}",
            flush=True,
        )
        return existing
    return run_single_experiment(entry, args, int(seed))


def main():
    parser = argparse.ArgumentParser(description="Run paper-v1 robustness experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2027])
    parser.add_argument("--scenarios", nargs="+", choices=["cheat"], default=["cheat"])
    parser.add_argument("--families", nargs="+", choices=FAMILY_ORDER, default=list(FAMILY_ORDER))
    parser.add_argument("--levels", nargs="+", choices=LEVEL_ORDER, default=list(LEVEL_ORDER))
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--training-episodes", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--final-evaluation-episodes", type=int, default=None)
    parser.add_argument("--evaluation-frequency", type=int, default=None)
    parser.add_argument("--routine-eval-episodes", type=int, default=None)
    parser.add_argument("--confirmation-eval-episodes", type=int, default=None)
    parser.add_argument("--resume-missing", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    args = parser.parse_args()

    selected_entries = [
        entry
        for entry in EXPERIMENT_MATRIX
        if entry["scenario_id"] in args.scenarios
        and entry["robustness_family"] in args.families
        and entry["robustness_level"] in args.levels
        and entry["method_id"] in args.methods
    ]
    if not selected_entries:
        raise SystemExit("No robustness experiments selected. Check --families, --levels, and --methods.")

    results_root = build_results_root(SCRIPT_DIR, args.results_subdir)
    completed_runs = collect_completed_runs(results_root) if args.resume_missing else {}
    planned_runs = build_run_plan(selected_entries, [int(seed) for seed in args.seeds])
    total_runs = len(planned_runs)
    batch_started_at = time.time()
    manifest = {
        "paper_group": "robustness",
        "timestamp": datetime.now().isoformat(),
        "scenarios": sorted({entry["scenario_id"] for entry in selected_entries}),
        "families": [family for family in FAMILY_ORDER if family in args.families],
        "levels": [level for level in LEVEL_ORDER if level in args.levels],
        "methods": [method for method in METHOD_ORDER if method in args.methods],
        "seeds": [int(seed) for seed in args.seeds],
        "resume_missing": bool(args.resume_missing),
        "results_subdir": str(args.results_subdir),
        "runs": [],
    }
    completed_count = 0
    run_durations: List[float] = []
    write_progress(
        results_root,
        PROGRESS_FILENAME,
        {
            "status": "running",
            "results_subdir": str(args.results_subdir),
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
            f"[PROGRESS] {render_progress_bar(completed_count, total_runs)} next={entry['robustness_family']}/{entry['robustness_level']}/{entry['method_id']}/seed{seed}",
            flush=True,
        )
        baseline_reference_path = None
        if entry["method_id"] == "drl":
            baseline_reference_path = write_local_baseline_reference(
                results_root=results_root,
                completed_runs=completed_runs,
                scenario_id=entry["scenario_id"],
                robustness_family=entry["robustness_family"],
                robustness_level=entry["robustness_level"],
            )
            print(f"[INFO] Baseline reference targets: {baseline_reference_path}", flush=True)

        current_experiment_id = build_experiment_id(
            entry["scenario_id"],
            entry["robustness_family"],
            entry["robustness_level"],
            entry["method_id"],
            int(seed),
        )
        average_duration = sum(run_durations) / len(run_durations) if run_durations else 0.0
        remaining_runs = total_runs - completed_count
        eta_seconds = average_duration * remaining_runs if run_durations else None
        write_progress(
            results_root,
            PROGRESS_FILENAME,
            {
                "status": "running",
                "results_subdir": str(args.results_subdir),
                "completed_runs": completed_count,
                "total_runs": total_runs,
                "current_run": {
                    "experiment_id": current_experiment_id,
                    "scenario_id": entry["scenario_id"],
                    "method_id": entry["method_id"],
                    "variant_id": entry["variant_id"],
                    "robustness_family": entry["robustness_family"],
                    "robustness_level": entry["robustness_level"],
                    "seed": int(seed),
                    "result_dir_glob": str(results_root / f"{current_experiment_id}_*"),
                    "baseline_reference_path": None if baseline_reference_path is None else str(baseline_reference_path),
                },
                "elapsed_seconds": float(time.time() - batch_started_at),
                "eta_seconds": None if eta_seconds is None else float(max(0.0, eta_seconds)),
                "updated_at": datetime.now().isoformat(),
            },
        )
        start_time = time.time()
        run_key = (
            "robustness",
            entry["scenario_id"],
            entry["method_id"],
            entry["variant_id"],
            entry["robustness_family"],
            entry["robustness_level"],
            int(seed),
        )
        reused = bool(args.resume_missing and run_key in completed_runs)
        run_payload = collect_run(entry, args, int(seed), completed_runs)
        completed_runs[run_key] = run_payload
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
            analysis_script=SCRIPT_DIR / "analysis" / "analyze_paper_robustness_v2.py",
            manifest_path=manifest_path,
            results_root=results_root,
        )


if __name__ == "__main__":
    main()
