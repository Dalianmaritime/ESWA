#!/usr/bin/env python3
"""Aggregate paper-v1 ablation results across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from paper_analysis_v2_utils import (
    aggregate_runs,
    ensure_output_dir,
    find_latest_manifest,
    load_manifest,
    load_runs_from_manifest,
    save_figure,
    summarize_group,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / "results" / "paper_v1_ablation"
MANIFEST_STEM = "paper_ablation_manifest"
SCENARIO_ORDER = ["cheat"]
SCENARIO_LABELS = {"cheat": "Cheat-FlipIt"}
VARIANT_ORDER = [
    "full",
    "no_constrained_selection",
    "no_resource_sustainability",
    "no_reward_shaping",
    "no_action_mask",
    "no_signal_features",
]
VARIANT_LABELS = {
    "full": "Full",
    "no_constrained_selection": "No constrained\nselection",
    "no_resource_sustainability": "No resource\nsustainability",
    "no_reward_shaping": "No reward\nshaping",
    "no_action_mask": "No action\nmask",
    "no_signal_features": "No signal\nfeatures",
}
COMPARISON_METRICS = [
    ("avg_defender_return", "Avg defender raw return"),
    ("attacker_success_rate", "Attacker success rate"),
    ("defender_control_rate", "Defender control rate"),
    ("avg_defender_control_per_cost", "Defender control per cost"),
]


def build_delta_vs_full(grouped_summary: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    full_summary = grouped_summary.get(("cheat", "full"))
    if full_summary is None:
        return {}

    deltas: Dict[str, Dict[str, float]] = {}
    for variant_id in VARIANT_ORDER:
        variant_summary = grouped_summary.get(("cheat", variant_id))
        if variant_summary is None:
            continue
        deltas[variant_id] = {}
        for metric, _ in COMPARISON_METRICS:
            deltas[variant_id][metric] = float(
                variant_summary[metric]["mean"] - full_summary[metric]["mean"]
            )
    return deltas


def plot_metric_by_variant(
    grouped_summary: Dict[Tuple[str, str], Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
):
    means = []
    stds = []
    labels = []
    for variant_id in VARIANT_ORDER:
        summary = grouped_summary.get(("cheat", variant_id))
        if summary is None:
            continue
        labels.append(VARIANT_LABELS[variant_id])
        means.append(summary[metric]["mean"])
        stds.append(summary[metric]["std"])

    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(labels))
    plt.bar(x_axis, means, yerr=stds, capsize=4, color="#3057d5")
    plt.xticks(x_axis, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    save_figure(output_dir, filename)


def plot_false_vs_missed(grouped_summary: Dict[Tuple[str, str], Dict[str, Any]], output_dir: Path):
    labels = []
    false_means = []
    missed_means = []
    for variant_id in VARIANT_ORDER:
        summary = grouped_summary.get(("cheat", variant_id))
        if summary is None:
            continue
        labels.append(VARIANT_LABELS[variant_id])
        false_means.append(summary["avg_false_response_rate"]["mean"])
        missed_means.append(summary["avg_missed_response_rate"]["mean"])

    x_axis = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x_axis - width / 2, false_means, width=width, label="False response rate")
    plt.bar(x_axis + width / 2, missed_means, width=width, label="Missed response rate")
    plt.xticks(x_axis, labels)
    plt.ylabel("Rate")
    plt.title("Paper V1 ablation: false vs missed response rates")
    plt.legend()
    save_figure(output_dir, "ablation_false_missed_by_variant.png")


def write_summary_markdown(
    grouped_summary: Dict[Tuple[str, str], Dict[str, Any]],
    delta_vs_full: Dict[str, Dict[str, float]],
    output_dir: Path,
    manifest_path: Path,
):
    lines = [
        "# Paper V1 Ablation Summary",
        "",
        f"- manifest: {manifest_path}",
        "- Scope: cheat-only V2 ablation with DRL variants and seed aggregation.",
        "",
        f"## {SCENARIO_LABELS['cheat']}",
        "",
    ]
    for variant_id in VARIANT_ORDER:
        summary = grouped_summary.get(("cheat", variant_id))
        if summary is None:
            continue
        delta_payload = delta_vs_full.get(variant_id, {})
        lines.extend(
            [
                f"### {VARIANT_LABELS[variant_id].replace(chr(10), ' ')}",
                f"- seeds: {summary['seeds']}",
                f"- avg_defender_return: {summary['avg_defender_return']['mean']:.3f} +- {summary['avg_defender_return']['std']:.3f}",
                f"- attacker_success_rate: {summary['attacker_success_rate']['mean']:.2%} +- {summary['attacker_success_rate']['std']:.3f}",
                f"- defender_control_rate: {summary['defender_control_rate']['mean']:.3f} +- {summary['defender_control_rate']['std']:.3f}",
                f"- avg_defender_control_per_cost: {summary['avg_defender_control_per_cost']['mean']:.3f} +- {summary['avg_defender_control_per_cost']['std']:.3f}",
                f"- avg_false_response_rate: {summary['avg_false_response_rate']['mean']:.3f} +- {summary['avg_false_response_rate']['std']:.3f}",
                f"- avg_missed_response_rate: {summary['avg_missed_response_rate']['mean']:.3f} +- {summary['avg_missed_response_rate']['std']:.3f}",
                f"- delta_vs_full.avg_defender_return: {delta_payload.get('avg_defender_return', 0.0):+.3f}",
                f"- delta_vs_full.attacker_success_rate: {delta_payload.get('attacker_success_rate', 0.0):+.3f}",
                f"- delta_vs_full.defender_control_rate: {delta_payload.get('defender_control_rate', 0.0):+.3f}",
                f"- delta_vs_full.avg_defender_control_per_cost: {delta_payload.get('avg_defender_control_per_cost', 0.0):+.3f}",
                "",
            ]
        )
    (output_dir / "ablation_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze paper-v1 ablation results.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else find_latest_manifest(results_root, MANIFEST_STEM)
    manifest = load_manifest(manifest_path)
    loaded_runs = load_runs_from_manifest(manifest)
    grouped_runs = aggregate_runs(loaded_runs, ["scenario_id", "variant_id"])
    grouped_summary = {group_key: summarize_group(group_runs) for group_key, group_runs in grouped_runs.items()}
    delta_vs_full = build_delta_vs_full(grouped_summary)
    output_dir = ensure_output_dir(results_root, "analysis")

    with open(output_dir / "ablation_experiment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "manifest_path": str(manifest_path),
                "delta_vs_full": delta_vs_full,
                "grouped_summary": {
                    f"{scenario_id}_{variant_id}": summary
                    for (scenario_id, variant_id), summary in grouped_summary.items()
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    plot_metric_by_variant(
        grouped_summary,
        metric="avg_defender_return",
        ylabel="Avg defender raw return",
        title="Paper V1 ablation: raw return by variant",
        filename="ablation_raw_return_by_variant.png",
        output_dir=output_dir,
    )
    plot_metric_by_variant(
        grouped_summary,
        metric="attacker_success_rate",
        ylabel="Attacker success rate",
        title="Paper V1 ablation: attacker success by variant",
        filename="ablation_attacker_success_by_variant.png",
        output_dir=output_dir,
    )
    plot_metric_by_variant(
        grouped_summary,
        metric="avg_defender_control_per_cost",
        ylabel="Defender control per cost",
        title="Paper V1 ablation: control efficiency by variant",
        filename="ablation_control_per_cost_by_variant.png",
        output_dir=output_dir,
    )
    plot_false_vs_missed(grouped_summary, output_dir)
    write_summary_markdown(grouped_summary, delta_vs_full, output_dir, manifest_path)
    print(f"Paper V1 ablation analysis written to {output_dir}")


if __name__ == "__main__":
    main()
