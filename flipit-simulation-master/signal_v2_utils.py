"""Shared utilities for V2 signal-region experiments."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml


def compute_validation_selection_score(
    avg_defender_return: float,
    avg_missed_response_rate: float,
    avg_false_response_rate: float,
) -> float:
    return float(avg_defender_return - 50.0 * avg_missed_response_rate - 20.0 * avg_false_response_rate)


def compute_economic_score(
    performance: Dict[str, Any],
    false_response_penalty: float = 20.0,
    missed_response_penalty: float = 35.0,
) -> float:
    return float(
        performance["avg_defender_return"]
        - false_response_penalty * performance["avg_false_response_rate"]
        - missed_response_penalty * performance["avg_missed_response_rate"]
    )


def compute_security_deficit(performance: Dict[str, Any], reference_targets: Dict[str, float]) -> float:
    attacker_gap = max(0.0, float(performance["attacker_success_rate"]) - float(reference_targets["attacker_success_rate"]))
    defender_gap = max(0.0, float(reference_targets["defender_control_rate"]) - float(performance["defender_control_rate"]))
    return float(attacker_gap + defender_gap)


def compute_checkpoint_selection_metrics(
    performance: Dict[str, Any],
    reference_targets: Dict[str, float],
    attacker_success_tolerance: float = 0.03,
    defender_control_tolerance: float = 0.03,
    false_response_penalty: float = 20.0,
    missed_response_penalty: float = 35.0,
    strict_require_raw_return_advantage: bool = True,
) -> Dict[str, Any]:
    performance_attacker_success = float(performance["attacker_success_rate"])
    performance_defender_control = float(performance["defender_control_rate"])
    performance_raw_return = float(performance["avg_defender_return"])
    reference_attacker_success = float(reference_targets["attacker_success_rate"])
    reference_defender_control = float(reference_targets["defender_control_rate"])
    reference_raw_return = float(reference_targets["avg_defender_return"])

    attacker_success_with_tolerance = performance_attacker_success <= reference_attacker_success + attacker_success_tolerance
    defender_control_with_tolerance = performance_defender_control >= reference_defender_control - defender_control_tolerance
    attacker_success_under_baseline = performance_attacker_success <= reference_attacker_success
    defender_control_over_baseline = performance_defender_control >= reference_defender_control
    raw_return_over_baseline = performance_raw_return > reference_raw_return if strict_require_raw_return_advantage else True

    return {
        "security_pass_count": int(attacker_success_with_tolerance) + int(defender_control_with_tolerance),
        "security_deficit": compute_security_deficit(performance, reference_targets),
        "economic_score": compute_economic_score(
            performance,
            false_response_penalty=false_response_penalty,
            missed_response_penalty=missed_response_penalty,
        ),
        "selection_feasible_under_tolerance": bool(attacker_success_with_tolerance and defender_control_with_tolerance),
        "selection_feasible_under_baseline": bool(
            attacker_success_under_baseline and defender_control_over_baseline and raw_return_over_baseline
        ),
        "security_margin_vs_baseline": {
            "attacker_success_rate": float(reference_attacker_success - performance_attacker_success),
            "defender_control_rate": float(performance_defender_control - reference_defender_control),
        },
        "economic_margin_vs_baseline": float(performance_raw_return - reference_raw_return),
        "raw_return_over_baseline": bool(raw_return_over_baseline),
    }


def routine_candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, ...]:
    selection_metrics = candidate["selection_metrics"]
    performance = candidate["performance"]
    return (
        -float(selection_metrics["security_pass_count"]),
        float(selection_metrics["security_deficit"]),
        -float(performance["avg_defender_return"]),
        -float(performance.get("avg_defender_control_per_cost", 0.0)),
        float(performance["avg_missed_response_rate"]),
        float(performance["avg_false_response_rate"]),
        float(candidate.get("episode", 0)),
    )


def confirmation_candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, ...]:
    selection_metrics = candidate["selection_metrics"]
    performance = candidate["performance"]
    return (
        0.0 if selection_metrics["selection_feasible_under_baseline"] else 1.0,
        float(selection_metrics["security_deficit"]),
        -float(performance["avg_defender_return"]),
        -float(performance.get("avg_defender_control_per_cost", 0.0)),
        float(performance["avg_missed_response_rate"]),
        float(performance["avg_false_response_rate"]),
        float(candidate.get("episode", 0)),
    )


def constrained_candidate_improves(
    candidate: Dict[str, Any],
    current_best_candidate: Dict[str, Any] | None,
) -> bool:
    if current_best_candidate is None:
        return True
    return routine_candidate_sort_key(candidate) < routine_candidate_sort_key(current_best_candidate)


def should_trigger_early_stopping(
    enabled: bool,
    current_episode: int,
    min_training_episodes: int,
    evaluations_since_improvement: int,
    patience_evaluations: int,
) -> bool:
    if not enabled or patience_evaluations <= 0:
        return False
    completed_episodes = int(current_episode) + 1
    if completed_episodes < int(min_training_episodes):
        return False
    return int(evaluations_since_improvement) >= int(patience_evaluations)


def _load_complete_result_from_dir(result_dir: Path) -> Dict[str, Any]:
    with open(result_dir / "complete_training_results.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_baseline_reference_targets(results_root: Path, scenarios: Iterable[str] | None = None) -> Dict[str, Any]:
    scenario_filter = {str(item) for item in scenarios} if scenarios is not None else None
    latest_runs: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    for result_dir in results_root.iterdir():
        if not result_dir.is_dir():
            continue
        result_path = result_dir / "complete_training_results.json"
        if not result_path.exists():
            continue
        try:
            complete_results = _load_complete_result_from_dir(result_dir)
        except (OSError, json.JSONDecodeError, KeyError):
            continue

        experiment_info = complete_results.get("experiment_info", {})
        if str(experiment_info.get("method_id")) != "baseline":
            continue

        scenario_id = str(experiment_info.get("scenario_id", ""))
        random_seed = experiment_info.get("random_seed")
        final_performance = complete_results.get("final_performance")
        if not scenario_id or random_seed is None or final_performance is None:
            continue
        if scenario_filter is not None and scenario_id not in scenario_filter:
            continue

        key = (scenario_id, int(random_seed))
        candidate = {
            "scenario_id": scenario_id,
            "seed": int(random_seed),
            "result_dir": str(result_dir.resolve()),
            "final_performance": final_performance,
            "_sort_key": result_dir.stat().st_mtime,
        }
        existing = latest_runs.get(key)
        if existing is None or candidate["_sort_key"] > existing["_sort_key"]:
            latest_runs[key] = candidate

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in latest_runs.values():
        grouped.setdefault(candidate["scenario_id"], []).append(candidate)

    if scenario_filter is not None:
        missing = sorted(scenario_filter.difference(grouped.keys()))
        if missing:
            raise FileNotFoundError(
                f"Missing baseline references for scenarios {missing} under {results_root}"
            )

    scenarios_payload = {}
    for scenario_id, run_entries in grouped.items():
        performances = [entry["final_performance"] for entry in run_entries]
        scenarios_payload[scenario_id] = {
            "num_runs": len(run_entries),
            "seeds": sorted(entry["seed"] for entry in run_entries),
            "source_result_dirs": [entry["result_dir"] for entry in sorted(run_entries, key=lambda item: item["seed"])],
            "attacker_success_rate": float(np.mean([item["attacker_success_rate"] for item in performances])),
            "defender_control_rate": float(np.mean([item["defender_control_rate"] for item in performances])),
            "avg_defender_return": float(np.mean([item["avg_defender_return"] for item in performances])),
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "results_root": str(results_root.resolve()),
        "scenarios": scenarios_payload,
    }


def write_baseline_reference_targets(results_root: Path, scenarios: Iterable[str] | None = None) -> Path:
    payload = build_baseline_reference_targets(results_root, scenarios=scenarios)
    results_root.mkdir(parents=True, exist_ok=True)
    output_path = results_root / "baseline_reference_targets.json"
    save_json(output_path, payload)
    return output_path


def load_baseline_reference_targets(results_root: Path, scenario_id: str) -> Dict[str, Any]:
    reference_path = results_root / "baseline_reference_targets.json"
    if not reference_path.exists():
        write_baseline_reference_targets(results_root, scenarios=[scenario_id])
    with open(reference_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenario_payload = payload.get("scenarios", {}).get(str(scenario_id))
    if scenario_payload is None:
        raise FileNotFoundError(f"No baseline reference targets found for scenario '{scenario_id}' in {reference_path}")
    return scenario_payload


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
        "attacker_action_cost": float(info.get("attacker_action_cost", 0.0)),
        "defender_action_cost": float(info.get("defender_action_cost", 0.0)),
        "attacker_base_income_applied": float(info.get("attacker_base_income_applied", 0.0)),
        "defender_base_income_applied": float(info.get("defender_base_income_applied", 0.0)),
        "attacker_control_bonus_applied": float(info.get("attacker_control_bonus_applied", 0.0)),
        "defender_control_bonus_applied": float(info.get("defender_control_bonus_applied", 0.0)),
        "attacker_below_guarantee_streak": int(info.get("attacker_below_guarantee_streak", 0)),
        "defender_below_guarantee_streak": int(info.get("defender_below_guarantee_streak", 0)),
        "attacker_budget_collapse": bool(info.get("attacker_budget_collapse", False)),
        "defender_budget_collapse": bool(info.get("defender_budget_collapse", False)),
        "termination_reason": info.get("termination_reason"),
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
    attacker_below_guarantee_steps = int(metrics.get("attacker_below_guarantee_steps", 0))
    defender_below_guarantee_steps = int(metrics.get("defender_below_guarantee_steps", 0))
    total_attacker_action_cost = float(metrics.get("total_attacker_action_cost", 0.0))
    total_defender_action_cost = float(metrics.get("total_defender_action_cost", 0.0))

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
    defender_control_per_cost = defender_control_steps / max(total_defender_action_cost, 1.0)
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
        "total_attacker_action_cost": total_attacker_action_cost,
        "total_defender_action_cost": total_defender_action_cost,
        "defender_control_per_cost": float(defender_control_per_cost),
        "attacker_control_rate": attacker_control_rate,
        "defender_control_rate": defender_control_rate,
        "false_response_rate": false_response_rate,
        "missed_response_rate": missed_response_rate,
        "inspection_precision": inspection_precision,
        "episode_length": int(steps),
        "termination_reason": last_info.get("termination_reason"),
        "attacker_resource_collapse": bool(last_info.get("attacker_budget_collapse", False)),
        "defender_resource_collapse": bool(last_info.get("defender_budget_collapse", False)),
        "final_attacker_budget": float(last_info.get("attacker_budget_remaining", 0.0)),
        "final_defender_budget": float(last_info.get("defender_budget_remaining", 0.0)),
        "attacker_below_guarantee_steps": attacker_below_guarantee_steps,
        "defender_below_guarantee_steps": defender_below_guarantee_steps,
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
            "attacker_resource_collapse_rate": 0.0,
            "defender_resource_collapse_rate": 0.0,
            "avg_final_attacker_budget": 0.0,
            "avg_final_defender_budget": 0.0,
            "avg_attacker_below_guarantee_steps": 0.0,
            "avg_defender_below_guarantee_steps": 0.0,
            "avg_defender_spent_budget": 0.0,
            "avg_defender_control_per_cost": 0.0,
            "validation_selection_score": 0.0,
            "sample_size": 0,
        }

    performance = {
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
        "attacker_resource_collapse_rate": float(
            np.mean([episode.get("attacker_resource_collapse", False) for episode in episode_list])
        ),
        "defender_resource_collapse_rate": float(
            np.mean([episode.get("defender_resource_collapse", False) for episode in episode_list])
        ),
        "avg_final_attacker_budget": float(np.mean([episode.get("final_attacker_budget", 0.0) for episode in episode_list])),
        "avg_final_defender_budget": float(np.mean([episode.get("final_defender_budget", 0.0) for episode in episode_list])),
        "avg_attacker_below_guarantee_steps": float(
            np.mean([episode.get("attacker_below_guarantee_steps", 0.0) for episode in episode_list])
        ),
        "avg_defender_below_guarantee_steps": float(
            np.mean([episode.get("defender_below_guarantee_steps", 0.0) for episode in episode_list])
        ),
        "avg_defender_spent_budget": float(np.mean([episode.get("total_defender_action_cost", 0.0) for episode in episode_list])),
        "avg_defender_control_per_cost": float(
            np.mean([episode.get("defender_control_per_cost", 0.0) for episode in episode_list])
        ),
        "sample_size": int(sample_size),
    }
    performance["validation_selection_score"] = compute_validation_selection_score(
        performance["avg_defender_return"],
        performance["avg_missed_response_rate"],
        performance["avg_false_response_rate"],
    )
    return performance


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
    early_stopping = results.get("early_stopping", {})
    attacker_resource_collapse_rate = float(perf.get("attacker_resource_collapse_rate", 0.0))
    defender_resource_collapse_rate = float(perf.get("defender_resource_collapse_rate", 0.0))
    avg_final_attacker_budget = float(perf.get("avg_final_attacker_budget", 0.0))
    avg_final_defender_budget = float(perf.get("avg_final_defender_budget", 0.0))
    avg_attacker_below_guarantee_steps = float(perf.get("avg_attacker_below_guarantee_steps", 0.0))
    avg_defender_below_guarantee_steps = float(perf.get("avg_defender_below_guarantee_steps", 0.0))
    avg_defender_spent_budget = float(perf.get("avg_defender_spent_budget", 0.0))
    avg_defender_control_per_cost = float(perf.get("avg_defender_control_per_cost", 0.0))

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
    if early_stopping:
        lines.extend(
            [
                f"- Early stopping enabled: {bool(early_stopping.get('enabled', False))}",
                f"- Early stopping triggered: {bool(early_stopping.get('triggered', False))}",
                f"- Completed training episodes: {int(early_stopping.get('completed_training_episodes', 0))}",
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
            f"- Attacker resource collapse rate: {attacker_resource_collapse_rate:.2%}",
            f"- Defender resource collapse rate: {defender_resource_collapse_rate:.2%}",
            f"- Avg final attacker budget: {avg_final_attacker_budget:.3f}",
            f"- Avg final defender budget: {avg_final_defender_budget:.3f}",
            f"- Avg attacker below-guarantee steps: {avg_attacker_below_guarantee_steps:.3f}",
            f"- Avg defender below-guarantee steps: {avg_defender_below_guarantee_steps:.3f}",
            f"- Avg defender spent budget: {avg_defender_spent_budget:.3f}",
            f"- Avg defender control per cost: {avg_defender_control_per_cost:.3f}",
            "",
            "## Training Diagnostics",
            f"- Avg defender training return: {perf['avg_defender_training_return']:.3f}",
        ]
    )
    return "\n".join(lines) + "\n"
