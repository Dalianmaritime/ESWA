"""Shared helpers for V2 paper batch runners."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def build_results_root(script_dir: Path, results_subdir: str) -> Path:
    return script_dir / "results" / str(results_subdir)


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


def write_manifest(results_root: Path, manifest_stem: str, manifest: Dict[str, Any]) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = results_root / f"{manifest_stem}_{timestamp}.json"
    latest_path = results_root / f"{manifest_stem}_latest.json"
    payload = json.dumps(manifest, indent=2, ensure_ascii=False)
    manifest_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return manifest_path


def write_progress(results_root: Path, filename: str, payload: Dict[str, Any]):
    results_root.mkdir(parents=True, exist_ok=True)
    progress_path = results_root / filename
    progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_complete_result(result_dir: Path) -> Dict[str, Any]:
    with open(result_dir / "complete_training_results.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def _pick_experiment_value(complete_results: Dict[str, Any], key: str) -> Any:
    experiment_info = dict(complete_results.get("experiment_info", {}))
    if experiment_info.get(key) is not None:
        return experiment_info.get(key)
    config_snapshot = dict(complete_results.get("config_snapshot", {}))
    experiment_config = dict(config_snapshot.get("experiment", {}))
    return experiment_config.get(key)


def extract_run_identity(result_dir: Path, complete_results: Dict[str, Any]) -> Dict[str, Any] | None:
    scenario_id = _pick_experiment_value(complete_results, "scenario_id")
    method_id = _pick_experiment_value(complete_results, "method_id")
    random_seed = _pick_experiment_value(complete_results, "random_seed")
    experiment_id = _pick_experiment_value(complete_results, "experiment_id")
    final_performance = complete_results.get("final_performance")
    if scenario_id is None or method_id is None or random_seed is None or experiment_id is None or final_performance is None:
        return None

    return {
        "experiment_id": str(experiment_id),
        "paper_group": _pick_experiment_value(complete_results, "paper_group"),
        "scenario_id": str(scenario_id),
        "method_id": str(method_id),
        "seed": int(random_seed),
        "variant_id": _pick_experiment_value(complete_results, "variant_id"),
        "variant_label": _pick_experiment_value(complete_results, "variant_label"),
        "robustness_family": _pick_experiment_value(complete_results, "robustness_family"),
        "robustness_level": _pick_experiment_value(complete_results, "robustness_level"),
        "config_path": str(_pick_experiment_value(complete_results, "config_path") or ""),
        "result_dir": str(result_dir.resolve()),
        "final_performance": final_performance,
        "variant_controls": _pick_experiment_value(complete_results, "variant_controls"),
    }


def collect_completed_runs(results_root: Path) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    completed_runs: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
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
        except (OSError, json.JSONDecodeError, KeyError):
            continue

        identity = extract_run_identity(result_dir, complete_results)
        if identity is None:
            continue

        run_key = (
            identity["paper_group"],
            identity["scenario_id"],
            identity["method_id"],
            identity["variant_id"],
            identity["robustness_family"],
            identity["robustness_level"],
            identity["seed"],
        )
        candidate = {
            **identity,
            "_sort_key": result_dir.stat().st_mtime,
        }
        existing = completed_runs.get(run_key)
        if existing is None or candidate["_sort_key"] > existing["_sort_key"]:
            completed_runs[run_key] = candidate

    for payload in completed_runs.values():
        payload.pop("_sort_key", None)
    return completed_runs


def run_analysis(script_dir: Path, analysis_script: Path, manifest_path: Path, results_root: Path):
    command = [
        sys.executable,
        str(analysis_script),
        "--manifest",
        str(manifest_path),
        "--results-root",
        str(results_root),
    ]
    subprocess.run(command, cwd=script_dir, check=True)
