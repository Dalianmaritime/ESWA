#!/usr/bin/env python3
"""Analyze V2 signal-region experiment outputs without synthetic fallbacks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

RESULT_PREFIXES = [
    "trc_signal_cheat_drl_v2_",
    "trc_signal_flipit_drl_v2_",
    "trc_signal_cheat_baseline_v2_",
    "trc_signal_flipit_baseline_v2_",
]
ZONES = ("outer", "lane", "core")


def find_latest_result_dirs(results_root: Path) -> Dict[str, Path]:
    latest = {}
    for prefix in RESULT_PREFIXES:
        matches = sorted(
            [path for path in results_root.glob(f"{prefix}*") if path.is_dir()],
            key=lambda item: item.stat().st_mtime,
        )
        if matches:
            latest[prefix] = matches[-1]
    return latest


def load_result(result_dir: Path) -> Dict:
    with open(result_dir / "complete_training_results.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    if not values:
        return np.array([])
    window = min(window, len(values))
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")


def aggregate_zone_counts(details: List[Dict]) -> Dict[str, Dict[str, int]]:
    signal_counts = defaultdict(int)
    response_counts = defaultdict(int)
    for episode in details:
        for record in episode.get("step_records", []):
            signal = record.get("current_signal")
            if signal in ZONES:
                signal_counts[signal] += 1
            defender_action_label = record.get("defender_action_label", "")
            for zone in ZONES:
                if defender_action_label == f"respond_{zone}":
                    response_counts[zone] += 1
    return {
        "signals": {zone: int(signal_counts[zone]) for zone in ZONES},
        "responses": {zone: int(response_counts[zone]) for zone in ZONES},
    }


def make_output_dir(script_dir: Path) -> Path:
    output_dir = script_dir.parent / "results" / f"v2_signal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(output_dir: Path, name: str):
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_control_rates(labels: List[str], results: Dict[str, Dict], output_dir: Path):
    x_axis = np.arange(len(labels))
    width = 0.35
    defender_rates = [results[label]["final_performance"]["defender_control_rate"] for label in labels]
    attacker_rates = [results[label]["final_performance"]["attacker_control_rate"] for label in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(x_axis - width / 2, defender_rates, width=width, label="Defender control rate")
    plt.bar(x_axis + width / 2, attacker_rates, width=width, label="Attacker control rate")
    plt.xticks(x_axis, labels, rotation=15)
    plt.ylabel("Control rate")
    plt.title("V2 control-rate comparison")
    plt.legend()
    save_figure(output_dir, "control_rate_comparison")


def plot_attacker_success(labels: List[str], results: Dict[str, Dict], output_dir: Path):
    success_rates = [results[label]["final_performance"]["attacker_success_rate"] for label in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, success_rates, color="#cc5f5f")
    plt.ylabel("Attacker success rate")
    plt.title("V2 attacker success-rate comparison")
    plt.xticks(rotation=15)
    save_figure(output_dir, "attacker_success_rate_comparison")


def plot_false_vs_missed(labels: List[str], results: Dict[str, Dict], output_dir: Path):
    x_axis = np.arange(len(labels))
    width = 0.35
    false_rates = [results[label]["final_performance"]["avg_false_response_rate"] for label in labels]
    missed_rates = [results[label]["final_performance"]["avg_missed_response_rate"] for label in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(x_axis - width / 2, false_rates, width=width, label="False response rate")
    plt.bar(x_axis + width / 2, missed_rates, width=width, label="Missed response rate")
    plt.xticks(x_axis, labels, rotation=15)
    plt.ylabel("Rate")
    plt.title("V2 false-response vs missed-response")
    plt.legend()
    save_figure(output_dir, "false_vs_missed_response")


def plot_raw_defender_returns(labels: List[str], results: Dict[str, Dict], output_dir: Path):
    raw_returns = [results[label]["final_performance"]["avg_defender_return"] for label in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, raw_returns, color="#4d79a7")
    plt.ylabel("Avg defender raw return")
    plt.title("V2 raw-return comparison")
    plt.xticks(rotation=15)
    save_figure(output_dir, "raw_defender_return_comparison")


def plot_inspection_precision(labels: List[str], results: Dict[str, Dict], output_dir: Path):
    plt.figure(figsize=(10, 6))
    for label in labels:
        evaluation_history = results[label]["evaluation_history"]
        if not evaluation_history:
            continue
        x_axis = [item["episode"] for item in evaluation_history]
        y_axis = [item["performance"]["avg_inspection_precision"] for item in evaluation_history]
        plt.plot(x_axis, y_axis, marker="o", label=label)
    plt.xlabel("Training episode")
    plt.ylabel("Inspection precision")
    plt.title("V2 inspection precision curve")
    plt.legend()
    save_figure(output_dir, "inspection_precision_curve")


def plot_defender_raw_return_curve(labels: List[str], results: Dict[str, Dict], output_dir: Path):
    plt.figure(figsize=(10, 6))
    for label in labels:
        evaluation_history = results[label]["evaluation_history"]
        if not evaluation_history:
            continue
        x_axis = [item["episode"] for item in evaluation_history]
        y_axis = [item["performance"]["avg_defender_return"] for item in evaluation_history]
        plt.plot(x_axis, y_axis, marker="o", label=label)
    plt.xlabel("Training episode")
    plt.ylabel("Avg defender raw return")
    plt.title("V2 defender raw-return evaluation curve")
    plt.legend()
    save_figure(output_dir, "defender_return_learning_curve")


def plot_defender_training_reward_curve(labels: List[str], results: Dict[str, Dict], output_dir: Path):
    plt.figure(figsize=(10, 6))
    for label in labels:
        training_history = results[label]["training_history"]
        if not training_history:
            continue
        training_rewards = [item.get("defender_training_return", item["defender_return"]) for item in training_history]
        smoothed = moving_average(training_rewards, window=10)
        if smoothed.size == 0:
            continue
        x_axis = list(range(len(smoothed)))
        plt.plot(x_axis, smoothed, label=label)
    plt.xlabel("Training episode (smoothed)")
    plt.ylabel("Defender training reward")
    plt.title("V2 defender training-reward curve")
    plt.legend()
    save_figure(output_dir, "defender_training_reward_curve")


def plot_zone_distribution(labels: List[str], results: Dict[str, Dict], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x_axis = np.arange(len(labels))
    width = 0.22
    signal_data = {label: aggregate_zone_counts(results[label]["final_evaluation_details"])["signals"] for label in labels}
    response_data = {label: aggregate_zone_counts(results[label]["final_evaluation_details"])["responses"] for label in labels}

    for index, zone in enumerate(ZONES):
        axes[0].bar(
            x_axis + (index - 1) * width,
            [signal_data[label][zone] for label in labels],
            width=width,
            label=zone,
        )
        axes[1].bar(
            x_axis + (index - 1) * width,
            [response_data[label][zone] for label in labels],
            width=width,
            label=zone,
        )

    axes[0].set_xticks(x_axis)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_title("Signal distribution by zone")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].set_xticks(x_axis)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_title("Respond distribution by zone")
    axes[1].legend()

    save_figure(output_dir, "zone_level_signal_response_distribution")


def write_summary(output_dir: Path, selected_dirs: Dict[str, Path], results: Dict[str, Dict]):
    lines = [
        "# V2 Signal Analysis",
        "",
        "- Report metrics below use raw utility and behavioral outcomes.",
        "- `avg_defender_training_return` is included only as a training diagnostic for DRL/baseline comparison.",
        "",
    ]
    for label, result_dir in selected_dirs.items():
        result = results[label]
        perf = result["final_performance"]
        checkpoint_selection = result.get("checkpoint_selection", {})
        lines.extend(
            [
                f"## {label}",
                f"- result_dir: {result_dir}",
                f"- attacker_success_rate: {perf['attacker_success_rate']:.2%}",
                f"- defender_control_rate: {perf['defender_control_rate']:.3f}",
                f"- avg_defender_return: {perf['avg_defender_return']:.3f}",
                f"- avg_defender_training_return: {perf['avg_defender_training_return']:.3f}",
                f"- avg_false_response_rate: {perf['avg_false_response_rate']:.3f}",
                f"- avg_missed_response_rate: {perf['avg_missed_response_rate']:.3f}",
                f"- attacker_resource_collapse_rate: {perf.get('attacker_resource_collapse_rate', 0.0):.2%}",
                f"- defender_resource_collapse_rate: {perf.get('defender_resource_collapse_rate', 0.0):.2%}",
                f"- avg_final_attacker_budget: {perf.get('avg_final_attacker_budget', 0.0):.3f}",
                f"- avg_final_defender_budget: {perf.get('avg_final_defender_budget', 0.0):.3f}",
                f"- avg_attacker_below_guarantee_steps: {perf.get('avg_attacker_below_guarantee_steps', 0.0):.3f}",
                f"- avg_defender_below_guarantee_steps: {perf.get('avg_defender_below_guarantee_steps', 0.0):.3f}",
            ]
        )
        if checkpoint_selection:
            lines.extend(
                [
                    f"- checkpoint_selection_metric: {checkpoint_selection.get('metric')}",
                    f"- best_checkpoint_episode: {checkpoint_selection.get('best_episode')}",
                    f"- best_checkpoint_score: {checkpoint_selection.get('best_score')}",
                ]
            )
        lines.append("")
    (output_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze V2 signal-region experiment results.")
    parser.add_argument("--results-root", default=str(Path(__file__).resolve().parent.parent / "results"))
    parser.add_argument("--latest-only", action="store_true", help="Accepted for compatibility; analysis already uses latest runs.")
    args = parser.parse_args()

    del args.latest_only
    results_root = Path(args.results_root).resolve()
    selected_dirs = find_latest_result_dirs(results_root)
    if not selected_dirs:
        raise SystemExit(f"No V2 result directories found under {results_root}")

    result_payloads = {}
    label_mapping = {}
    for prefix, result_dir in selected_dirs.items():
        payload = load_result(result_dir)
        label = prefix.removesuffix("_").replace("trc_signal_", "").replace("_v2", "").replace("_", " ")
        result_payloads[label] = payload
        label_mapping[label] = result_dir

    output_dir = make_output_dir(Path(__file__).resolve().parent)
    labels = list(result_payloads.keys())
    plot_control_rates(labels, result_payloads, output_dir)
    plot_attacker_success(labels, result_payloads, output_dir)
    plot_false_vs_missed(labels, result_payloads, output_dir)
    plot_raw_defender_returns(labels, result_payloads, output_dir)
    plot_inspection_precision(labels, result_payloads, output_dir)
    plot_defender_raw_return_curve(labels, result_payloads, output_dir)
    plot_defender_training_reward_curve(labels, result_payloads, output_dir)
    plot_zone_distribution(labels, result_payloads, output_dir)
    write_summary(output_dir, label_mapping, result_payloads)
    print(f"V2 analysis written to {output_dir}")


if __name__ == "__main__":
    main()
