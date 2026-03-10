#!/usr/bin/env python3
"""Run the paper-v1 main experiment matrix with explicit per-seed entries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from run_traditional_experiment_v2 import TraditionalExperimentV2
from run_trc_full_training_v2 import SignalDRLTrainingExperimentV2
from signal_v2_utils import write_baseline_reference_targets

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_SUBDIR = "paper_v1_main"

EXPERIMENT_MATRIX = [
    {
        "scenario_id": "cheat",
        "method_id": "drl",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "main" / "main_cheat_drl.yml",
        "runner": "drl",
    },
    {
        "scenario_id": "flipit",
        "method_id": "drl",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "main" / "main_flipit_drl.yml",
        "runner": "drl",
    },
    {
        "scenario_id": "cheat",
        "method_id": "baseline",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "main" / "main_cheat_baseline.yml",
        "runner": "baseline",
    },
    {
        "scenario_id": "flipit",
        "method_id": "baseline",
        "config_path": SCRIPT_DIR / "configs" / "paper_v1" / "main" / "main_flipit_baseline.yml",
        "runner": "baseline",
    },
]


def build_experiment_id(scenario_id: str, method_id: str, seed: int) -> str:
    return f"paper_main_{scenario_id}_{method_id}_seed{seed}"


def build_results_root(results_subdir: str) -> Path:
    return SCRIPT_DIR / "results" / str(results_subdir)


def render_progress_bar(completed: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    filled = int(width * completed / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {completed}/{total}"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def build_run_plan(selected_entries: List[Dict[str, Any]], seeds: List[int]) -> List[Tuple[Dict[str, Any], int]]:
    method_rank = {"baseline": 0, "drl": 1}
    scenario_rank = {scenario_id: index for index, scenario_id in enumerate(["cheat", "flipit"])}
    planned_runs = [(entry, int(seed)) for entry in selected_entries for seed in seeds]
    planned_runs.sort(
        key=lambda item: (
            method_rank.get(item[0]["method_id"], 99),
            scenario_rank.get(item[0]["scenario_id"], 99),
            int(item[1]),
        )
    )
    return planned_runs


def ensure_baseline_reference(results_root: Path, scenarios: List[str]) -> Path:
    return write_baseline_reference_targets(results_root, scenarios=scenarios)


def write_manifest(manifest: Dict[str, Any], results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = results_root / f"paper_main_manifest_{timestamp}.json"
    latest_path = results_root / "paper_main_manifest_latest.json"
    payload = json.dumps(manifest, indent=2, ensure_ascii=False)
    manifest_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return manifest_path


def write_batch_progress(results_root: Path, payload: Dict[str, Any]):
    results_root.mkdir(parents=True, exist_ok=True)
    progress_path = results_root / "paper_main_progress_latest.json"
    progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_complete_result(result_dir: Path) -> Dict[str, Any]:
    with open(result_dir / "complete_training_results.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_completed_runs(results_root: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    completed_runs: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    if not results_root.exists():
        return completed_runs

    for result_dir in results_root.iterdir():
        if not result_dir.is_dir():
            continue
        result_path = result_dir / "complete_training_results.json"
        if not result_path.exists():
            continue

        try:
            complete_results = load_complete_result(result_dir)
        except (json.JSONDecodeError, OSError, KeyError):
            continue

        experiment_info = complete_results.get("experiment_info", {})
        scenario_id = experiment_info.get("scenario_id")
        method_id = experiment_info.get("method_id")
        random_seed = experiment_info.get("random_seed")
        experiment_id = experiment_info.get("experiment_id")
        final_performance = complete_results.get("final_performance")
        if scenario_id is None or method_id is None or random_seed is None or experiment_id is None or final_performance is None:
            continue

        run_key = (str(scenario_id), str(method_id), int(random_seed))
        candidate = {
            "experiment_id": str(experiment_id),
            "scenario_id": str(scenario_id),
            "method_id": str(method_id),
            "seed": int(random_seed),
            "config_path": str(experiment_info.get("config_path", "")),
            "result_dir": str(result_dir.resolve()),
            "final_performance": final_performance,
            "_sort_key": result_dir.stat().st_mtime,
        }
        existing = completed_runs.get(run_key)
        if existing is None or candidate["_sort_key"] > existing["_sort_key"]:
            completed_runs[run_key] = candidate

    for payload in completed_runs.values():
        payload.pop("_sort_key", None)
    return completed_runs


def run_single_experiment(entry: Dict[str, Any], args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    experiment_id = build_experiment_id(entry["scenario_id"], entry["method_id"], seed)
    print(
        f"[RUN] scenario={entry['scenario_id']} method={entry['method_id']} seed={seed} config={entry['config_path'].name}",
        flush=True,
    )

    if entry["runner"] == "drl":
        experiment = SignalDRLTrainingExperimentV2(
            config_path=str(entry["config_path"]),
            training_episodes=args.training_episodes,
            evaluation_episodes=args.evaluation_episodes,
            final_evaluation_episodes=args.final_evaluation_episodes,
            evaluation_frequency=args.evaluation_frequency,
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
        "scenario_id": entry["scenario_id"],
        "method_id": entry["method_id"],
        "seed": seed,
        "config_path": str(Path(entry["config_path"]).resolve()),
        "result_dir": str(experiment.results_dir.resolve()),
        "final_performance": performance,
    }


def maybe_run_analysis(manifest_path: Path, results_root: Path):
    command = [
        sys.executable,
        str(SCRIPT_DIR / "analysis" / "analyze_paper_main_v2.py"),
        "--manifest",
        str(manifest_path),
        "--results-root",
        str(results_root),
    ]
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def collect_run(entry: Dict[str, Any], args: argparse.Namespace, seed: int, completed_runs: Dict[Tuple[str, str, int], Dict[str, Any]]) -> Dict[str, Any]:
    run_key = (entry["scenario_id"], entry["method_id"], int(seed))
    if args.resume_missing and run_key in completed_runs:
        existing = completed_runs[run_key]
        print(
            f"[SKIP] scenario={entry['scenario_id']} method={entry['method_id']} seed={seed} reuse={Path(existing['result_dir']).name}",
            flush=True,
        )
        return existing
    return run_single_experiment(entry, args, seed)


def main():
    parser = argparse.ArgumentParser(description="Run paper-v1 main experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--scenarios", nargs="+", choices=["cheat", "flipit"], default=["cheat", "flipit"])
    parser.add_argument("--methods", nargs="+", choices=["drl", "baseline"], default=["drl", "baseline"])
    parser.add_argument("--training-episodes", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--final-evaluation-episodes", type=int, default=None)
    parser.add_argument("--evaluation-frequency", type=int, default=None)
    parser.add_argument("--resume-missing", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    args = parser.parse_args()

    selected_entries = [
        entry
        for entry in EXPERIMENT_MATRIX
        if entry["scenario_id"] in args.scenarios and entry["method_id"] in args.methods
    ]
    if not selected_entries:
        raise SystemExit("No experiments selected. Check --scenarios and --methods.")

    results_root = build_results_root(args.results_subdir)
    completed_runs = find_completed_runs(results_root) if args.resume_missing else {}
    planned_runs = build_run_plan(selected_entries, [int(seed) for seed in args.seeds])
    total_runs = len(planned_runs)
    batch_started_at = time.time()
    manifest = {
        "paper_group": "main",
        "timestamp": datetime.now().isoformat(),
        "seeds": [int(seed) for seed in args.seeds],
        "resume_missing": bool(args.resume_missing),
        "results_subdir": str(args.results_subdir),
        "runs": [],
    }
    completed_count = 0
    run_durations: List[float] = []
    write_batch_progress(
        results_root,
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
    selected_scenarios = sorted({entry["scenario_id"] for entry in selected_entries})
    baseline_reference_path: Path | None = None
    for entry, seed in planned_runs:
        print(
            f"[PROGRESS] {render_progress_bar(completed_count, total_runs)} next={entry['scenario_id']}/{entry['method_id']}/seed{seed}",
            flush=True,
        )
        if entry["method_id"] == "drl":
            baseline_reference_path = ensure_baseline_reference(results_root, scenarios=selected_scenarios)
            print(f"[INFO] Baseline reference targets: {baseline_reference_path}", flush=True)
        current_experiment_id = build_experiment_id(entry["scenario_id"], entry["method_id"], int(seed))
        average_duration = sum(run_durations) / len(run_durations) if run_durations else 0.0
        remaining_runs = total_runs - completed_count
        eta_seconds = average_duration * remaining_runs if run_durations else None
        write_batch_progress(
            results_root,
            {
                "status": "running",
                "results_subdir": str(args.results_subdir),
                "completed_runs": completed_count,
                "total_runs": total_runs,
                "current_run": {
                    "experiment_id": current_experiment_id,
                    "scenario_id": entry["scenario_id"],
                    "method_id": entry["method_id"],
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
        run_key = (entry["scenario_id"], entry["method_id"], int(seed))
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
        write_batch_progress(
            results_root,
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

    manifest_path = write_manifest(manifest, results_root)
    print(f"[INFO] Manifest written to {manifest_path}", flush=True)
    if not args.skip_analysis:
        maybe_run_analysis(manifest_path, results_root)


if __name__ == "__main__":
    main()
