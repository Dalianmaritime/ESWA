#!/usr/bin/env python3
"""Aggregate paper-v1 main experiment results across seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / "results" / "paper_v1_main"
KEY_METRICS = [
    "defender_control_rate",
    "attacker_success_rate",
    "avg_defender_return",
    "avg_defender_training_return",
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
SCENARIO_ORDER = ["cheat", "flipit"]
METHOD_ORDER = ["drl", "baseline"]
METHOD_LABELS = {"drl": "DRL", "baseline": "Threshold baseline"}
SCENARIO_LABELS = {"cheat": "Cheat-FlipIt", "flipit": "FlipIt"}
LAST_EVAL_WINDOW = 3


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_manifest(results_root: Path) -> Path:
    latest_path = results_root / "paper_main_manifest_latest.json"
    if latest_path.exists():
        return latest_path
    matches = sorted(results_root.glob("paper_main_manifest_*.json"))
    if not matches:
        raise FileNotFoundError(f"No manifest found under {results_root}")
    return matches[-1]


def load_complete_result(result_dir: Path) -> Dict[str, Any]:
    with open(result_dir / "complete_training_results.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_runs(run_entries: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for run in run_entries:
        grouped[(run["scenario_id"], run["method_id"])].append(run)
    return grouped


def summarize_group(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_seeds": len(runs), "seeds": sorted(run["seed"] for run in runs)}
    for metric in KEY_METRICS:
        values = np.asarray([run["final_performance"].get(metric, 0.0) for run in runs], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def compute_validation_score(performance: Dict[str, Any]) -> float:
    return float(
        performance["avg_defender_return"]
        - 50.0 * performance["avg_missed_response_rate"]
        - 20.0 * performance["avg_false_response_rate"]
    )


def analyze_drl_run(run: Dict[str, Any]) -> Dict[str, Any]:
    complete_results = run["complete_results"]
    checkpoint_selection = complete_results.get("checkpoint_selection", {})
    evaluation_history = complete_results.get("evaluation_history", [])
    training_config = complete_results.get("config_snapshot", {}).get("drl", {})
    training_episodes = int(training_config.get("training_episodes", 0))
    best_episode = checkpoint_selection.get("best_episode")

    last_window = evaluation_history[-LAST_EVAL_WINDOW:] if evaluation_history else []
    last_raw_returns = [item["performance"]["avg_defender_return"] for item in last_window]
    last_eval_rising = (
        len(last_raw_returns) == LAST_EVAL_WINDOW
        and all(last_raw_returns[index] < last_raw_returns[index + 1] for index in range(len(last_raw_returns) - 1))
    )

    return {
        "experiment_id": run["experiment_id"],
        "scenario_id": run["scenario_id"],
        "seed": run["seed"],
        "best_checkpoint_episode": best_episode,
        "training_episodes": training_episodes,
        "checkpoint_selected_last_episode": best_episode == training_episodes - 1 if best_episode is not None else False,
        "last_three_raw_returns": last_raw_returns,
        "last_eval_rising": bool(last_eval_rising),
        "final_validation_score": compute_validation_score(run["final_performance"]),
    }


def assess_pilot_readiness(grouped_summary: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    scenario_checks = {}
    overall_pass = True
    for scenario_id in SCENARIO_ORDER:
        drl = grouped_summary.get((scenario_id, "drl"))
        baseline = grouped_summary.get((scenario_id, "baseline"))
        if drl is None or baseline is None:
            scenario_checks[scenario_id] = {
                "available": False,
                "passes": False,
                "reasons": ["missing comparison pair"],
            }
            overall_pass = False
            continue

        reasons = []
        passes = True

        if drl["avg_defender_return"]["mean"] <= baseline["avg_defender_return"]["mean"]:
            passes = False
            reasons.append("DRL raw return does not exceed baseline")
        if drl["attacker_success_rate"]["mean"] > baseline["attacker_success_rate"]["mean"]:
            passes = False
            reasons.append("DRL attacker success rate is higher than baseline")

        false_improved = drl["avg_false_response_rate"]["mean"] < baseline["avg_false_response_rate"]["mean"]
        missed_improved = drl["avg_missed_response_rate"]["mean"] < baseline["avg_missed_response_rate"]["mean"]
        false_not_worse = drl["avg_false_response_rate"]["mean"] <= baseline["avg_false_response_rate"]["mean"] + 0.02
        missed_not_worse = drl["avg_missed_response_rate"]["mean"] <= baseline["avg_missed_response_rate"]["mean"] + 0.02
        if not ((false_improved and missed_not_worse) or (missed_improved and false_not_worse)):
            passes = False
            reasons.append("DRL does not achieve the required false/missed-response tradeoff")

        scenario_checks[scenario_id] = {
            "available": True,
            "passes": passes,
            "reasons": reasons,
        }
        overall_pass &= passes

    return {
        "overall_pass": bool(overall_pass),
        "scenario_checks": scenario_checks,
    }


def determine_debug_recommendation(
    pilot_assessment: Dict[str, Any],
    drl_run_diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if pilot_assessment["overall_pass"]:
        scenarios_to_extend = sorted(
            {
                item["scenario_id"]
                for item in drl_run_diagnostics
                if item["checkpoint_selected_last_episode"] and item["last_eval_rising"]
            }
        )
        return {
            "case": "A",
            "action": "keep_current_algorithm",
            "summary": "Current DRL results already satisfy the pilot-level ESWA screening criteria.",
            "scenarios_to_extend_training": scenarios_to_extend,
            "recommended_training_episodes": 1200 if scenarios_to_extend else None,
            "recommended_evaluation_frequency": 100 if scenarios_to_extend else None,
        }

    raw_advantage_all = all(
        scenario_data.get("available") and "DRL raw return does not exceed baseline" not in scenario_data["reasons"]
        for scenario_data in pilot_assessment["scenario_checks"].values()
    )
    if raw_advantage_all:
        return {
            "case": "B",
            "action": "switch_checkpoint_selection_metric",
            "summary": "Raw-return advantage exists, but operational stability is not consistent enough across scenarios.",
            "validation_selection_metric": "avg_defender_return - 50 * avg_missed_response_rate - 20 * avg_false_response_rate",
        }

    return {
        "case": "C",
        "action": "extend_training_then_retune_shaping",
        "summary": "Both raw return and operational metrics remain unstable, so training horizon should be extended before changing reward shaping.",
        "recommended_training_episodes": 1200,
        "recommended_evaluation_frequency": 100,
        "follow_up_reward_tuning": {
            "training_missed_threat_penalty": "increase slightly",
            "training_false_response_penalty": "increase slightly",
            "training_positive_inspection_bonus": "decrease slightly",
        },
    }


def ensure_output_dir(results_root: Path) -> Path:
    output_dir = results_root / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(output_dir: Path, filename: str):
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_grouped_metric(
    grouped_summary: Dict[Tuple[str, str], Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
):
    x_axis = np.arange(len(SCENARIO_ORDER))
    width = 0.35
    plt.figure(figsize=(10, 6))
    for method_index, method_id in enumerate(METHOD_ORDER):
        means = []
        stds = []
        for scenario_id in SCENARIO_ORDER:
            group = grouped_summary.get((scenario_id, method_id))
            if group is None:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(group[metric]["mean"])
                stds.append(group[metric]["std"])
        offsets = x_axis + (method_index - 0.5) * width
        plt.bar(offsets, means, width=width, yerr=stds, capsize=4, label=METHOD_LABELS[method_id])
    plt.xticks(x_axis, [SCENARIO_LABELS[item] for item in SCENARIO_ORDER])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    save_figure(output_dir, filename)


def plot_false_vs_missed(grouped_summary: Dict[Tuple[str, str], Dict[str, Any]], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x_axis = np.arange(len(SCENARIO_ORDER))
    width = 0.35
    metric_specs = [
        ("avg_false_response_rate", "False-response rate"),
        ("avg_missed_response_rate", "Missed-response rate"),
    ]
    for axis, (metric, title) in zip(axes, metric_specs):
        for method_index, method_id in enumerate(METHOD_ORDER):
            means = []
            stds = []
            for scenario_id in SCENARIO_ORDER:
                group = grouped_summary.get((scenario_id, method_id))
                if group is None:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    means.append(group[metric]["mean"])
                    stds.append(group[metric]["std"])
            offsets = x_axis + (method_index - 0.5) * width
            axis.bar(offsets, means, width=width, yerr=stds, capsize=4, label=METHOD_LABELS[method_id])
        axis.set_xticks(x_axis)
        axis.set_xticklabels([SCENARIO_LABELS[item] for item in SCENARIO_ORDER])
        axis.set_title(title)
        axis.set_ylabel("Rate")
    axes[1].legend()
    save_figure(output_dir, "main_false_missed_by_scenario.png")


def plot_drl_learning_curves(runs: List[Dict[str, Any]], output_dir: Path):
    grouped_curves: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for run in runs:
        if run["method_id"] != "drl":
            continue
        evaluation_history = run["complete_results"].get("evaluation_history", [])
        for item in evaluation_history:
            grouped_curves[run["scenario_id"]].append(
                {
                    "episode": item["episode"],
                    "avg_defender_return": item["performance"]["avg_defender_return"],
                    "avg_defender_training_return": item["performance"]["avg_defender_training_return"],
                }
            )

    if not grouped_curves:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    metric_specs = [
        ("avg_defender_return", "Avg defender raw return", "main_drl_raw_return_curve.png"),
        ("avg_defender_training_return", "Avg defender training return", "main_drl_training_return_curve.png"),
    ]
    for axis, (metric, ylabel, _) in zip(axes, metric_specs):
        for scenario_id in SCENARIO_ORDER:
            scenario_runs = [run for run in runs if run["scenario_id"] == scenario_id and run["method_id"] == "drl"]
            if not scenario_runs:
                continue
            episodes = sorted(
                {
                    item["episode"]
                    for run in scenario_runs
                    for item in run["complete_results"].get("evaluation_history", [])
                }
            )
            means = []
            for episode in episodes:
                values = [
                    item["performance"][metric]
                    for run in scenario_runs
                    for item in run["complete_results"].get("evaluation_history", [])
                    if item["episode"] == episode
                ]
                means.append(float(np.mean(values)))
            axis.plot(episodes, means, marker="o", label=SCENARIO_LABELS[scenario_id])
        axis.set_xlabel("Training episode")
        axis.set_ylabel(ylabel)
    axes[0].set_title("DRL raw-return evaluation curve")
    axes[1].set_title("DRL training-return curve")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "main_drl_learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_summary_markdown(
    grouped_summary: Dict[Tuple[str, str], Dict[str, Any]],
    output_dir: Path,
    manifest_path: Path,
    pilot_assessment: Dict[str, Any],
    debug_recommendation: Dict[str, Any],
):
    lines = [
        "# Paper V1 Main Experiment Summary",
        "",
        f"- manifest: {manifest_path}",
        "- Note: this is a pilot experiment with the currently selected seeds; it is suitable for algorithm screening, not final statistical claims.",
        "",
    ]
    for scenario_id in SCENARIO_ORDER:
        lines.append(f"## {SCENARIO_LABELS[scenario_id]}")
        for method_id in METHOD_ORDER:
            group = grouped_summary.get((scenario_id, method_id))
            if group is None:
                continue
            lines.extend(
                [
                    f"### {METHOD_LABELS[method_id]}",
                    f"- seeds: {group['seeds']}",
                    f"- defender_control_rate: {group['defender_control_rate']['mean']:.3f} +- {group['defender_control_rate']['std']:.3f}",
                    f"- attacker_success_rate: {group['attacker_success_rate']['mean']:.2%} +- {group['attacker_success_rate']['std']:.3f}",
                    f"- avg_defender_return: {group['avg_defender_return']['mean']:.3f} +- {group['avg_defender_return']['std']:.3f}",
                    f"- avg_defender_training_return: {group['avg_defender_training_return']['mean']:.3f} +- {group['avg_defender_training_return']['std']:.3f}",
                    f"- avg_false_response_rate: {group['avg_false_response_rate']['mean']:.3f} +- {group['avg_false_response_rate']['std']:.3f}",
                    f"- avg_missed_response_rate: {group['avg_missed_response_rate']['mean']:.3f} +- {group['avg_missed_response_rate']['std']:.3f}",
                    f"- avg_inspection_precision: {group['avg_inspection_precision']['mean']:.3f} +- {group['avg_inspection_precision']['std']:.3f}",
                    f"- attacker_resource_collapse_rate: {group['attacker_resource_collapse_rate']['mean']:.2%} +- {group['attacker_resource_collapse_rate']['std']:.3f}",
                    f"- defender_resource_collapse_rate: {group['defender_resource_collapse_rate']['mean']:.2%} +- {group['defender_resource_collapse_rate']['std']:.3f}",
                    f"- avg_final_attacker_budget: {group['avg_final_attacker_budget']['mean']:.3f} +- {group['avg_final_attacker_budget']['std']:.3f}",
                    f"- avg_final_defender_budget: {group['avg_final_defender_budget']['mean']:.3f} +- {group['avg_final_defender_budget']['std']:.3f}",
                    f"- avg_attacker_below_guarantee_steps: {group['avg_attacker_below_guarantee_steps']['mean']:.3f} +- {group['avg_attacker_below_guarantee_steps']['std']:.3f}",
                    f"- avg_defender_below_guarantee_steps: {group['avg_defender_below_guarantee_steps']['mean']:.3f} +- {group['avg_defender_below_guarantee_steps']['std']:.3f}",
                    "",
                ]
            )
    lines.extend(
        [
            "## ESWA Review",
            "- This study is framed as a pilot study with two random seeds for maritime non-traditional security and critical maritime infrastructure protection.",
            "- The current results should be interpreted as initial consistency trends rather than statistical significance claims.",
            f"- Pilot screening result: {'PASS' if pilot_assessment['overall_pass'] else 'REQUIRES DEBUG'}",
            f"- Recommended debug path: Case {debug_recommendation['case']} ({debug_recommendation['action']})",
            f"- Review summary: {debug_recommendation['summary']}",
            "",
        ]
    )
    (output_dir / "main_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_eswa_review_report(
    output_dir: Path,
    grouped_summary: Dict[Tuple[str, str], Dict[str, Any]],
    pilot_assessment: Dict[str, Any],
    debug_recommendation: Dict[str, Any],
    drl_run_diagnostics: List[Dict[str, Any]],
):
    lines = [
        "# ESWA-Oriented Pilot Review",
        "",
        "## Framing",
        "- Recommended manuscript framing: maritime non-traditional security, offshore critical infrastructure protection, and service continuity management.",
        "- Avoid military, combat, or warfighting wording in the manuscript narrative.",
        "- This result set should be described as a pilot study with two random seeds and no statistical significance claims.",
        "",
        "## Pilot Gate",
        f"- Overall pilot status: {'PASS' if pilot_assessment['overall_pass'] else 'REQUIRES DEBUG'}",
    ]

    for scenario_id in SCENARIO_ORDER:
        scenario_data = pilot_assessment["scenario_checks"].get(scenario_id, {})
        status = "PASS" if scenario_data.get("passes") else "FAIL"
        lines.append(f"- {SCENARIO_LABELS[scenario_id]}: {status}")
        for reason in scenario_data.get("reasons", []):
            lines.append(f"  - {reason}")

    lines.extend(
        [
            "",
            "## Algorithm Review",
        ]
    )
    for scenario_id in SCENARIO_ORDER:
        drl = grouped_summary.get((scenario_id, "drl"))
        baseline = grouped_summary.get((scenario_id, "baseline"))
        if drl is None or baseline is None:
            continue
        lines.extend(
            [
                f"### {SCENARIO_LABELS[scenario_id]}",
                f"- DRL avg_defender_return: {drl['avg_defender_return']['mean']:.3f}",
                f"- Baseline avg_defender_return: {baseline['avg_defender_return']['mean']:.3f}",
                f"- DRL attacker_success_rate: {drl['attacker_success_rate']['mean']:.2%}",
                f"- Baseline attacker_success_rate: {baseline['attacker_success_rate']['mean']:.2%}",
                f"- DRL false/missed: {drl['avg_false_response_rate']['mean']:.3f} / {drl['avg_missed_response_rate']['mean']:.3f}",
                f"- Baseline false/missed: {baseline['avg_false_response_rate']['mean']:.3f} / {baseline['avg_missed_response_rate']['mean']:.3f}",
                f"- DRL collapse rates (A/D): {drl['attacker_resource_collapse_rate']['mean']:.2%} / {drl['defender_resource_collapse_rate']['mean']:.2%}",
                f"- Baseline collapse rates (A/D): {baseline['attacker_resource_collapse_rate']['mean']:.2%} / {baseline['defender_resource_collapse_rate']['mean']:.2%}",
                "",
            ]
        )

    lines.extend(
        [
            "## Debug Recommendation",
            f"- Decision case: {debug_recommendation['case']}",
            f"- Action: {debug_recommendation['action']}",
            f"- Summary: {debug_recommendation['summary']}",
        ]
    )
    if debug_recommendation["case"] == "A":
        scenarios_to_extend = debug_recommendation.get("scenarios_to_extend_training", [])
        if scenarios_to_extend:
            lines.append(
                f"- Training-horizon follow-up: extend {', '.join(scenarios_to_extend)} DRL training to {debug_recommendation['recommended_training_episodes']} episodes with evaluation_frequency={debug_recommendation['recommended_evaluation_frequency']}."
            )
        else:
            lines.append("- No immediate algorithm change is required before adding more random seeds.")
    elif debug_recommendation["case"] == "B":
        lines.append(f"- Validation selection metric to add: `{debug_recommendation['validation_selection_metric']}`")
        lines.append("- Keep training reward unchanged; only adjust checkpoint selection and rerun DRL.")
    else:
        lines.append(
            f"- First change: increase DRL training to {debug_recommendation['recommended_training_episodes']} episodes and evaluation_frequency={debug_recommendation['recommended_evaluation_frequency']}."
        )
        lines.append("- Only if instability remains should reward shaping be tuned.")

    lines.extend(
        [
            "",
            "## DRL Run Diagnostics",
        ]
    )
    for diagnostic in sorted(drl_run_diagnostics, key=lambda item: (item["scenario_id"], item["seed"])):
        lines.extend(
            [
                f"- {diagnostic['experiment_id']}: best_checkpoint_episode={diagnostic['best_checkpoint_episode']}, final_validation_score={diagnostic['final_validation_score']:.3f}, last_eval_rising={diagnostic['last_eval_rising']}",
            ]
        )

    (output_dir / "eswa_pilot_review.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze paper-v1 main experiment results.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else find_latest_manifest(results_root)
    manifest = load_manifest(manifest_path)

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

    grouped_runs = aggregate_runs(loaded_runs)
    grouped_summary = {group_key: summarize_group(group_runs) for group_key, group_runs in grouped_runs.items()}
    drl_run_diagnostics = [analyze_drl_run(run) for run in loaded_runs if run["method_id"] == "drl"]
    pilot_assessment = assess_pilot_readiness(grouped_summary)
    debug_recommendation = determine_debug_recommendation(pilot_assessment, drl_run_diagnostics)

    output_dir = ensure_output_dir(results_root)
    with open(output_dir / "main_experiment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "manifest_path": str(manifest_path),
                "pilot_assessment": pilot_assessment,
                "debug_recommendation": debug_recommendation,
                "drl_run_diagnostics": drl_run_diagnostics,
                "grouped_summary": {
                    f"{scenario_id}_{method_id}": summary
                    for (scenario_id, method_id), summary in grouped_summary.items()
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    plot_grouped_metric(
        grouped_summary,
        metric="defender_control_rate",
        ylabel="Defender control rate",
        title="Paper V1 main experiment: control rate",
        filename="main_control_rate_by_scenario.png",
        output_dir=output_dir,
    )
    plot_grouped_metric(
        grouped_summary,
        metric="avg_defender_return",
        ylabel="Avg defender raw return",
        title="Paper V1 main experiment: raw return",
        filename="main_raw_return_by_scenario.png",
        output_dir=output_dir,
    )
    plot_grouped_metric(
        grouped_summary,
        metric="attacker_success_rate",
        ylabel="Attacker success rate",
        title="Paper V1 main experiment: attacker success",
        filename="main_attacker_success_by_scenario.png",
        output_dir=output_dir,
    )
    plot_false_vs_missed(grouped_summary, output_dir)
    plot_drl_learning_curves(loaded_runs, output_dir)
    write_summary_markdown(grouped_summary, output_dir, manifest_path, pilot_assessment, debug_recommendation)
    write_eswa_review_report(output_dir, grouped_summary, pilot_assessment, debug_recommendation, drl_run_diagnostics)
    print(f"Paper V1 main analysis written to {output_dir}")


if __name__ == "__main__":
    main()
