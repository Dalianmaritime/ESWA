#!/usr/bin/env python3
"""Run the paper-v1 main experiment matrix with explicit per-seed entries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from run_traditional_experiment_v2 import TraditionalExperimentV2
from run_trc_full_training_v2 import SignalDRLTrainingExperimentV2

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results" / "paper_v1_main"

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


def write_manifest(manifest: Dict[str, Any]) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = RESULTS_ROOT / f"paper_main_manifest_{timestamp}.json"
    latest_path = RESULTS_ROOT / "paper_main_manifest_latest.json"
    payload = json.dumps(manifest, indent=2, ensure_ascii=False)
    manifest_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return manifest_path


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
            results_subdir="paper_v1_main",
        )
    else:
        experiment = TraditionalExperimentV2(
            config_path=str(entry["config_path"]),
            episodes=args.final_evaluation_episodes,
            random_seed=seed,
            experiment_id=experiment_id,
            results_subdir="paper_v1_main",
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


def maybe_run_analysis(manifest_path: Path):
    command = [
        sys.executable,
        str(SCRIPT_DIR / "analysis" / "analyze_paper_main_v2.py"),
        "--manifest",
        str(manifest_path),
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
    args = parser.parse_args()

    selected_entries = [
        entry
        for entry in EXPERIMENT_MATRIX
        if entry["scenario_id"] in args.scenarios and entry["method_id"] in args.methods
    ]
    if not selected_entries:
        raise SystemExit("No experiments selected. Check --scenarios and --methods.")

    completed_runs = find_completed_runs(RESULTS_ROOT) if args.resume_missing else {}
    manifest = {
        "paper_group": "main",
        "timestamp": datetime.now().isoformat(),
        "seeds": [int(seed) for seed in args.seeds],
        "resume_missing": bool(args.resume_missing),
        "runs": [],
    }
    for seed in args.seeds:
        for entry in selected_entries:
            manifest["runs"].append(collect_run(entry, args, int(seed), completed_runs))

    manifest_path = write_manifest(manifest)
    print(f"[INFO] Manifest written to {manifest_path}", flush=True)
    if not args.skip_analysis:
        maybe_run_analysis(manifest_path)


if __name__ == "__main__":
    main()
