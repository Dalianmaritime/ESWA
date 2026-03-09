"""Shared utilities for V2 signal-region experiments."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_results_directory(base_dir: Path, config_path: str, config: Dict[str, Any]) -> Path:
    experiment_id = config["experiment"]["experiment_id"]
    results_subdir = config["experiment"].get("results_subdir")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = base_dir / "results"
    if results_subdir:
        results_root = results_root / str(results_subdir)
    results_dir = results_root / f"{experiment_id}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.yml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    with open(results_dir / "config_path.txt", "w", encoding="utf-8") as handle:
        handle.write(str(config_path))
    return results_dir


def make_step_record(step_index: int, info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step": int(step_index),
        "mode": info["mode"],
        "true_controller": info["true_controller"],
        "true_zone": info["true_zone"],
        "current_signal": info["current_signal"],
        "signal_source_action": info["signal_source_action"],
        "mu_breach": float(info["mu_breach"]),
        "zone_beliefs": dict(info["zone_beliefs"]),
        "attacker_action_label": info["attacker_action_label"],
        "defender_action_label": info["defender_action_label"],
        "inspection_result": info["inspection_result"],
        "false_response": bool(info["false_response"]),
        "missed_response": bool(info["missed_response"]),
        "takeover_attempted": bool(info["takeover_attempted"]),
        "takeover_success": bool(info["takeover_success"]),
        "response_success": bool(info["response_success"]),
        "attacker_budget_remaining": float(info["attacker_budget_remaining"]),
        "defender_budget_remaining": float(info["defender_budget_remaining"]),
        "attacker_reward": float(info.get("attacker_reward", 0.0)),
        "defender_reward": float(info.get("defender_reward", 0.0)),
        "training_reward": float(info.get("training_reward", 0.0)),
    }


def summarize_episode(
    episode_index: int,
    step_records: List[Dict[str, Any]],
    last_info: Dict[str, Any],
    attacker_return: float,
    defender_return: float,
    defender_training_return: float | None = None,
) -> Dict[str, Any]:
    metrics = dict(last_info.get("episode_metrics_snapshot", {}))
    steps = max(len(step_records), int(metrics.get("tick", 0)))
    attacker_control_steps = int(metrics.get("attacker_control_steps", 0))
    defender_control_steps = int(metrics.get("defender_control_steps", 0))
    inspect_actions = int(metrics.get("inspect_actions", 0))
    positive_inspections = int(metrics.get("positive_inspections", 0))
    false_responses = int(metrics.get("false_responses", 0))
    missed_responses = int(metrics.get("missed_responses", 0))

    if steps == 0:
        attacker_control_rate = 0.0
        defender_control_rate = 0.0
        false_response_rate = 0.0
        missed_response_rate = 0.0
    else:
        attacker_control_rate = attacker_control_steps / steps
        defender_control_rate = defender_control_steps / steps
        false_response_rate = false_responses / steps
        missed_response_rate = missed_responses / steps

    inspection_precision = positive_inspections / inspect_actions if inspect_actions else 0.0
    winner = last_info.get("winner") or (
        "attacker"
        if attacker_control_steps > defender_control_steps
        else "defender"
        if defender_control_steps > attacker_control_steps
        else "draw"
    )
    return {
        "episode_index": int(episode_index),
        "winner": winner,
        "attacker_success": winner == "attacker",
        "attacker_return": float(attacker_return),
        "defender_return": float(defender_return),
        "defender_training_return": float(
            defender_return if defender_training_return is None else defender_training_return
        ),
        "attacker_control_rate": attacker_control_rate,
        "defender_control_rate": defender_control_rate,
        "false_response_rate": false_response_rate,
        "missed_response_rate": missed_response_rate,
        "inspection_precision": inspection_precision,
        "episode_length": int(steps),
        "episode_metrics_snapshot": metrics,
        "step_records": step_records,
    }


def compute_final_performance(episodes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    episode_list = list(episodes)
    sample_size = len(episode_list)
    if sample_size == 0:
        return {
            "attacker_success_rate": 0.0,
            "defender_control_rate": 0.0,
            "attacker_control_rate": 0.0,
            "avg_defender_return": 0.0,
            "avg_defender_training_return": 0.0,
            "avg_attacker_return": 0.0,
            "avg_false_response_rate": 0.0,
            "avg_missed_response_rate": 0.0,
            "avg_inspection_precision": 0.0,
            "avg_episode_length": 0.0,
            "sample_size": 0,
        }

    return {
        "attacker_success_rate": float(np.mean([episode["attacker_success"] for episode in episode_list])),
        "defender_control_rate": float(np.mean([episode["defender_control_rate"] for episode in episode_list])),
        "attacker_control_rate": float(np.mean([episode["attacker_control_rate"] for episode in episode_list])),
        "avg_defender_return": float(np.mean([episode["defender_return"] for episode in episode_list])),
        "avg_defender_training_return": float(
            np.mean([episode.get("defender_training_return", episode["defender_return"]) for episode in episode_list])
        ),
        "avg_attacker_return": float(np.mean([episode["attacker_return"] for episode in episode_list])),
        "avg_false_response_rate": float(np.mean([episode["false_response_rate"] for episode in episode_list])),
        "avg_missed_response_rate": float(np.mean([episode["missed_response_rate"] for episode in episode_list])),
        "avg_inspection_precision": float(np.mean([episode["inspection_precision"] for episode in episode_list])),
        "avg_episode_length": float(np.mean([episode["episode_length"] for episode in episode_list])),
        "sample_size": int(sample_size),
    }


def get_metric_value(metrics: Dict[str, Any], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Metric '{metric_name}' is not available in metrics: {sorted(metrics.keys())}")
    return float(metrics[metric_name])


def save_json(path: Path, payload: Any):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def generate_summary_markdown(results: Dict[str, Any]) -> str:
    info = results["experiment_info"]
    perf = results["final_performance"]
    metric_conventions = results.get("metric_conventions", {})
    checkpoint_selection = results.get("checkpoint_selection", {})

    lines = [
        "# Maritime Cheat-FlipIt V2 Experiment Summary",
        "",
        "## Experiment",
        f"- ID: {info['experiment_id']}",
        f"- Type: {info['experiment_type']}",
        f"- Mode: {info['mode']}",
        f"- Timestamp: {info['timestamp']}",
        f"- Sample size: {perf['sample_size']}",
        "",
        "## Metric Conventions",
        f"- Training objective: {metric_conventions.get('training_metric', 'defender_training_return (shaped reward)')}",
        f"- Reported utility: {metric_conventions.get('report_metric', 'avg_defender_return (raw utility)')}",
    ]

    if checkpoint_selection:
        lines.extend(
            [
                f"- Checkpoint selection metric: {checkpoint_selection.get('metric', 'avg_defender_training_return')}",
                f"- Best checkpoint episode: {checkpoint_selection.get('best_episode', 'n/a')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Final Performance",
            f"- Attacker success rate: {perf['attacker_success_rate']:.2%}",
            f"- Defender control rate: {perf['defender_control_rate']:.3f}",
            f"- Attacker control rate: {perf['attacker_control_rate']:.3f}",
            f"- Avg defender return: {perf['avg_defender_return']:.3f}",
            f"- Avg attacker return: {perf['avg_attacker_return']:.3f}",
            f"- Avg false response rate: {perf['avg_false_response_rate']:.3f}",
            f"- Avg missed response rate: {perf['avg_missed_response_rate']:.3f}",
            f"- Avg inspection precision: {perf['avg_inspection_precision']:.3f}",
            f"- Avg episode length: {perf['avg_episode_length']:.2f}",
            "",
            "## Training Diagnostics",
            f"- Avg defender training return: {perf['avg_defender_training_return']:.3f}",
        ]
    )
    return "\n".join(lines) + "\n"
