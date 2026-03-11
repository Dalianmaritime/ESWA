#!/usr/bin/env python3
"""Aggregate paper-v1 robustness results across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

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
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / "results" / "paper_v1_robustness"
MANIFEST_STEM = "paper_robustness_manifest"
FAMILY_ORDER = ["budget_stress", "deception_intensity", "attack_strength"]
LEVEL_ORDER = ["low", "nominal", "high"]
METHOD_ORDER = ["baseline", "drl"]
METHOD_LABELS = {"baseline": "Threshold baseline", "drl": "DRL"}
FAMILY_LABELS = {
    "budget_stress": "Budget stress",
    "deception_intensity": "Deception intensity",
    "attack_strength": "Attack strength",
}
LEVEL_LABELS = {"low": "Low", "nominal": "Nominal", "high": "High"}
PANEL_METRICS = [
    ("avg_defender_return", "Avg defender raw return"),
    ("attacker_success_rate", "Attacker success rate"),
    ("avg_defender_control_per_cost", "Defender control per cost"),
]


def build_method_deltas(grouped_summary: Dict[Tuple[str, str, str, str], Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    for family in FAMILY_ORDER:
        deltas[family] = {}
        for level in LEVEL_ORDER:
            baseline = grouped_summary.get((family, level, "baseline", "cheat"))
            drl = grouped_summary.get((family, level, "drl", "cheat"))
            if baseline is None or drl is None:
                continue
            deltas[family][level] = {}
            for metric, _ in PANEL_METRICS:
                deltas[family][level][metric] = float(drl[metric]["mean"] - baseline[metric]["mean"])
    return deltas


def plot_family_comparison(
    grouped_summary: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    family: str,
    output_dir: Path,
):
    fig, axes = plt.subplots(1, len(PANEL_METRICS), figsize=(16, 5))
    x_axis = np.arange(len(LEVEL_ORDER))
    width = 0.35
    for axis, (metric, ylabel) in zip(axes, PANEL_METRICS):
        for method_index, method_id in enumerate(METHOD_ORDER):
            means = []
            stds = []
            for level in LEVEL_ORDER:
                summary = grouped_summary.get((family, level, method_id, "cheat"))
                if summary is None:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    means.append(summary[metric]["mean"])
                    stds.append(summary[metric]["std"])
            offsets = x_axis + (method_index - 0.5) * width
            axis.bar(offsets, means, width=width, yerr=stds, capsize=4, label=METHOD_LABELS[method_id])
            axis.set_xticks(x_axis)
            axis.set_xticklabels([LEVEL_LABELS[level] for level in LEVEL_ORDER])
            axis.set_title(ylabel)
            axis.set_ylabel(ylabel)
    axes[-1].legend()
    plt.suptitle(f"Paper V1 robustness: {FAMILY_LABELS[family]}")
    save_figure(output_dir, f"robustness_{family}_comparison.png")


def write_summary_markdown(
    grouped_summary: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    method_deltas: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
    manifest_path: Path,
):
    lines = [
        "# Paper V1 Robustness Summary",
        "",
        f"- manifest: {manifest_path}",
        "- Scope: cheat-only robustness families with DRL and baseline comparisons across seeds.",
        "",
    ]
    for family in FAMILY_ORDER:
        lines.extend([f"## {FAMILY_LABELS[family]}", ""])
        for level in LEVEL_ORDER:
            lines.append(f"### {LEVEL_LABELS[level]}")
            for method_id in METHOD_ORDER:
                summary = grouped_summary.get((family, level, method_id, "cheat"))
                if summary is None:
                    continue
                lines.extend(
                    [
                        f"- {METHOD_LABELS[method_id]} seeds: {summary['seeds']}",
                        f"- {METHOD_LABELS[method_id]} avg_defender_return: {summary['avg_defender_return']['mean']:.3f} +- {summary['avg_defender_return']['std']:.3f}",
                        f"- {METHOD_LABELS[method_id]} attacker_success_rate: {summary['attacker_success_rate']['mean']:.2%} +- {summary['attacker_success_rate']['std']:.3f}",
                        f"- {METHOD_LABELS[method_id]} defender_control_rate: {summary['defender_control_rate']['mean']:.3f} +- {summary['defender_control_rate']['std']:.3f}",
                        f"- {METHOD_LABELS[method_id]} avg_defender_control_per_cost: {summary['avg_defender_control_per_cost']['mean']:.3f} +- {summary['avg_defender_control_per_cost']['std']:.3f}",
                    ]
                )
            delta_payload = method_deltas.get(family, {}).get(level)
            if delta_payload:
                lines.extend(
                    [
                        f"- DRL minus baseline avg_defender_return: {delta_payload['avg_defender_return']:+.3f}",
                        f"- DRL minus baseline attacker_success_rate: {delta_payload['attacker_success_rate']:+.3f}",
                        f"- DRL minus baseline avg_defender_control_per_cost: {delta_payload['avg_defender_control_per_cost']:+.3f}",
                    ]
                )
            lines.append("")
    (output_dir / "robustness_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze paper-v1 robustness results.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else find_latest_manifest(results_root, MANIFEST_STEM)
    manifest = load_manifest(manifest_path)
    loaded_runs = load_runs_from_manifest(manifest)
    grouped_runs = aggregate_runs(loaded_runs, ["robustness_family", "robustness_level", "method_id", "scenario_id"])
    grouped_summary = {group_key: summarize_group(group_runs) for group_key, group_runs in grouped_runs.items()}
    method_deltas = build_method_deltas(grouped_summary)
    output_dir = ensure_output_dir(results_root, "analysis")

    with open(output_dir / "robustness_experiment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "manifest_path": str(manifest_path),
                "method_deltas": method_deltas,
                "grouped_summary": {
                    f"{family}_{level}_{method_id}_{scenario_id}": summary
                    for (family, level, method_id, scenario_id), summary in grouped_summary.items()
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    for family in FAMILY_ORDER:
        plot_family_comparison(grouped_summary, family, output_dir)
    write_summary_markdown(grouped_summary, method_deltas, output_dir, manifest_path)
    print(f"Paper V1 robustness analysis written to {output_dir}")


if __name__ == "__main__":
    main()
