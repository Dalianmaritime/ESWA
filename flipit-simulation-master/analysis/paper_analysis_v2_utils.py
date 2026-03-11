"""Shared helpers for V2 paper analysis scripts."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


KEY_METRICS = [
    "defender_control_rate",
    "attacker_success_rate",
    "avg_defender_return",
    "avg_defender_training_return",
    "avg_defender_spent_budget",
    "avg_defender_control_per_cost",
    "avg_false_response_rate",
    "avg_missed_response_rate",
    "avg_inspection_precision",
    "attacker_resource_collapse_rate",
    "defender_resource_collapse_rate",
    "avg_final_attacker_budget",
    "avg_final_defender_budget",
    "avg_attacker_below_guarantee_steps",
    "avg_defender_below_guarantee_steps",
]


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_manifest(results_root: Path, manifest_stem: str) -> Path:
    latest_path = results_root / f"{manifest_stem}_latest.json"
    if latest_path.exists():
        return latest_path
    matches = sorted(results_root.glob(f"{manifest_stem}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No manifest found under {results_root} for stem '{manifest_stem}'")
    return matches[-1]


def load_complete_result(result_dir: Path) -> Dict[str, Any]:
    with open(result_dir / "complete_training_results.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runs_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    loaded_runs: List[Dict[str, Any]] = []
    for run_entry in manifest["runs"]:
        result_dir = Path(run_entry["result_dir"])
        complete_results = load_complete_result(result_dir)
        loaded_runs.append(
            {
                **run_entry,
                "complete_results": complete_results,
                "final_performance": complete_results["final_performance"],
            }
        )
    return loaded_runs


def aggregate_runs(run_entries: Iterable[Dict[str, Any]], key_fields: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for run in run_entries:
        group_key = tuple(run.get(field) for field in key_fields)
        grouped[group_key].append(run)
    return grouped


def summarize_group(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_seeds": len(runs),
        "seeds": sorted(int(run["seed"]) for run in runs),
    }
    for metric in KEY_METRICS:
        values = np.asarray([run["final_performance"].get(metric, 0.0) for run in runs], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def ensure_output_dir(results_root: Path, prefix: str) -> Path:
    output_dir = results_root / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(output_dir: Path, filename: str):
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
