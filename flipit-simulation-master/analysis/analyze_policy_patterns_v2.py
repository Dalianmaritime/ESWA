#!/usr/bin/env python3
"""Generate managerial policy-pattern analysis from V2 result manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from paper_analysis_v2_utils import ensure_output_dir, find_latest_manifest, load_manifest, load_runs_from_manifest, save_figure

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / "results" / "paper_v1_main"
MANIFEST_STEMS = {
    "main": "paper_main_manifest",
    "ablation": "paper_ablation_manifest",
    "robustness": "paper_robustness_manifest",
}
ZONE_ORDER = ["outer", "lane", "core"]


def infer_group_fields(manifest: Dict[str, Any]) -> List[str]:
    paper_group = str(manifest.get("paper_group", "main"))
    if paper_group == "ablation":
        return ["scenario_id", "variant_id"]
    if paper_group == "robustness":
        return ["robustness_family", "robustness_level", "method_id"]
    return ["scenario_id", "method_id"]


def build_group_label(run: Dict[str, Any], group_fields: Iterable[str]) -> str:
    parts = [str(run.get(field, "unknown")) for field in group_fields]
    return " / ".join(parts)


def iter_episode_entries(runs: List[Dict[str, Any]], group_fields: Iterable[str]):
    for run in runs:
        group_label = build_group_label(run, group_fields)
        for episode in run["complete_results"].get("final_evaluation_details", []):
            yield group_label, episode


def compute_action_summary(runs: List[Dict[str, Any]], group_fields: Iterable[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for group_label, episode in iter_episode_entries(runs, group_fields):
        group_summary = summary.setdefault(
            group_label,
            {
                "episodes": 0,
                "action_category_counts": {"hold": 0, "inspect": 0, "respond": 0},
                "zone_action_counts": {f"inspect_{zone}": 0 for zone in ZONE_ORDER} | {f"respond_{zone}": 0 for zone in ZONE_ORDER},
            },
        )
        group_summary["episodes"] += 1
        for step in episode.get("step_records", []):
            defender_action_label = str(step.get("defender_action_label", "hold"))
            if defender_action_label == "hold":
                group_summary["action_category_counts"]["hold"] += 1
            elif defender_action_label.startswith("inspect_"):
                group_summary["action_category_counts"]["inspect"] += 1
                group_summary["zone_action_counts"][defender_action_label] += 1
            elif defender_action_label.startswith("respond_"):
                group_summary["action_category_counts"]["respond"] += 1
                group_summary["zone_action_counts"][defender_action_label] += 1

    for group_summary in summary.values():
        total_steps = max(1, sum(group_summary["action_category_counts"].values()))
        group_summary["action_category_ratios"] = {
            key: float(value / total_steps) for key, value in group_summary["action_category_counts"].items()
        }
        zone_total = max(1, sum(group_summary["zone_action_counts"].values()))
        group_summary["zone_action_ratios"] = {
            key: float(value / zone_total) for key, value in group_summary["zone_action_counts"].items()
        }
    return summary


def compute_budget_trajectories(runs: List[Dict[str, Any]], group_fields: Iterable[str]) -> Dict[str, Any]:
    grouped_steps: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: {"defender": [], "attacker": []}))
    for group_label, episode in iter_episode_entries(runs, group_fields):
        for step in episode.get("step_records", []):
            step_index = int(step["step"])
            grouped_steps[group_label][step_index]["defender"].append(float(step.get("defender_budget_remaining", 0.0)))
            grouped_steps[group_label][step_index]["attacker"].append(float(step.get("attacker_budget_remaining", 0.0)))

    trajectories: Dict[str, Any] = {}
    for group_label, per_step in grouped_steps.items():
        trajectories[group_label] = {
            "steps": sorted(per_step.keys()),
            "mean_defender_budget": [float(np.mean(per_step[index]["defender"])) for index in sorted(per_step.keys())],
            "mean_attacker_budget": [float(np.mean(per_step[index]["attacker"])) for index in sorted(per_step.keys())],
        }
    return trajectories


def compute_collapse_summary(runs: List[Dict[str, Any]], group_fields: Iterable[str]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"episodes": 0, "collapse_events": 0, "collapse_steps": []})
    for group_label, episode in iter_episode_entries(runs, group_fields):
        grouped[group_label]["episodes"] += 1
        if str(episode.get("termination_reason", "")).endswith("resource_collapse"):
            grouped[group_label]["collapse_events"] += 1
            grouped[group_label]["collapse_steps"].append(int(episode.get("episode_length", 0)))

    summary: Dict[str, Any] = {}
    for group_label, payload in grouped.items():
        collapse_steps = payload["collapse_steps"]
        summary[group_label] = {
            "episodes": payload["episodes"],
            "collapse_frequency": float(payload["collapse_events"] / max(1, payload["episodes"])),
            "mean_collapse_step": float(np.mean(collapse_steps)) if collapse_steps else None,
        }
    return summary


def _signal_followup(step_records: List[Dict[str, Any]], start_index: int, zone: str) -> Tuple[int | None, str]:
    inspect_lag = None
    respond_lag = None
    for current_index in range(start_index, len(step_records)):
        label = str(step_records[current_index].get("defender_action_label", "hold"))
        if label == f"inspect_{zone}" and inspect_lag is None:
            inspect_lag = current_index - start_index
        if label == f"respond_{zone}" and respond_lag is None:
            respond_lag = current_index - start_index
        if inspect_lag is not None and respond_lag is not None:
            break
    if respond_lag is None and inspect_lag is None:
        return None, "no_followup"
    if respond_lag is not None and (inspect_lag is None or respond_lag < inspect_lag):
        return respond_lag, "respond_first"
    return respond_lag, "inspect_first"


def compute_signal_response_summary(runs: List[Dict[str, Any]], group_fields: Iterable[str]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "suspicious_signal_count": 0,
            "response_lags": [],
            "followup_counts": {"inspect_first": 0, "respond_first": 0, "no_followup": 0},
        }
    )
    for group_label, episode in iter_episode_entries(runs, group_fields):
        step_records = episode.get("step_records", [])
        for index, step in enumerate(step_records):
            signal_zone = str(step.get("current_signal", "null"))
            if signal_zone not in ZONE_ORDER:
                continue
            lag, followup = _signal_followup(step_records, index, signal_zone)
            grouped[group_label]["suspicious_signal_count"] += 1
            grouped[group_label]["followup_counts"][followup] += 1
            if lag is not None:
                grouped[group_label]["response_lags"].append(int(lag))

    summary: Dict[str, Any] = {}
    for group_label, payload in grouped.items():
        count = max(1, payload["suspicious_signal_count"])
        response_lags = payload["response_lags"]
        summary[group_label] = {
            "suspicious_signal_count": payload["suspicious_signal_count"],
            "mean_response_lag": float(np.mean(response_lags)) if response_lags else None,
            "median_response_lag": float(np.median(response_lags)) if response_lags else None,
            "responded_signal_count": len(response_lags),
            "inspect_first_rate": float(payload["followup_counts"]["inspect_first"] / count),
            "respond_first_rate": float(payload["followup_counts"]["respond_first"] / count),
            "no_followup_rate": float(payload["followup_counts"]["no_followup"] / count),
        }
    return summary


def plot_action_category_ratios(action_summary: Dict[str, Any], output_dir: Path):
    group_labels = list(action_summary.keys())
    categories = ["hold", "inspect", "respond"]
    x_axis = np.arange(len(group_labels))
    bottom = np.zeros(len(group_labels))
    plt.figure(figsize=(12, 6))
    for category in categories:
        values = [action_summary[group]["action_category_ratios"][category] for group in group_labels]
        plt.bar(x_axis, values, bottom=bottom, label=category)
        bottom += np.asarray(values)
    plt.xticks(x_axis, group_labels, rotation=20, ha="right")
    plt.ylabel("Ratio")
    plt.title("Defender action category ratios")
    plt.legend()
    save_figure(output_dir, "policy_action_category_ratios.png")


def plot_zone_action_distribution(action_summary: Dict[str, Any], output_dir: Path):
    group_labels = list(action_summary.keys())
    zone_labels = [f"inspect_{zone}" for zone in ZONE_ORDER] + [f"respond_{zone}" for zone in ZONE_ORDER]
    x_axis = np.arange(len(group_labels))
    bottom = np.zeros(len(group_labels))
    plt.figure(figsize=(12, 6))
    for zone_label in zone_labels:
        values = [action_summary[group]["zone_action_ratios"][zone_label] for group in group_labels]
        plt.bar(x_axis, values, bottom=bottom, label=zone_label)
        bottom += np.asarray(values)
    plt.xticks(x_axis, group_labels, rotation=20, ha="right")
    plt.ylabel("Ratio within inspect/respond actions")
    plt.title("Zone-level action distribution")
    plt.legend(ncol=2)
    save_figure(output_dir, "policy_zone_action_distribution.png")


def plot_budget_trajectories(trajectories: Dict[str, Any], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for group_label, payload in trajectories.items():
        axes[0].plot(payload["steps"], payload["mean_defender_budget"], label=group_label)
        axes[1].plot(payload["steps"], payload["mean_attacker_budget"], label=group_label)
    axes[0].set_title("Average defender budget trajectory")
    axes[0].set_ylabel("Budget")
    axes[1].set_title("Average attacker budget trajectory")
    for axis in axes:
        axis.set_xlabel("Step")
    axes[1].legend()
    save_figure(output_dir, "policy_budget_trajectories.png")


def plot_collapse_summary(collapse_summary: Dict[str, Any], output_dir: Path):
    group_labels = list(collapse_summary.keys())
    x_axis = np.arange(len(group_labels))
    frequencies = [collapse_summary[group]["collapse_frequency"] for group in group_labels]
    timings = [collapse_summary[group]["mean_collapse_step"] or 0.0 for group in group_labels]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x_axis, frequencies, color="#bf4e30")
    ax1.set_ylabel("Collapse frequency")
    ax1.set_xticks(x_axis)
    ax1.set_xticklabels(group_labels, rotation=20, ha="right")
    ax1.set_title("Collapse-event frequency and timing")
    ax2 = ax1.twinx()
    ax2.plot(x_axis, timings, color="#1c7d54", marker="o")
    ax2.set_ylabel("Mean collapse step")
    save_figure(output_dir, "policy_collapse_summary.png")


def plot_response_timing(signal_response_summary: Dict[str, Any], output_dir: Path):
    group_labels = list(signal_response_summary.keys())
    x_axis = np.arange(len(group_labels))
    mean_lags = [signal_response_summary[group]["mean_response_lag"] or 0.0 for group in group_labels]
    plt.figure(figsize=(12, 6))
    plt.bar(x_axis, mean_lags, color="#3057d5")
    plt.xticks(x_axis, group_labels, rotation=20, ha="right")
    plt.ylabel("Mean lag (steps)")
    plt.title("Response timing after suspicious signal")
    save_figure(output_dir, "policy_response_timing.png")


def plot_signal_followup(signal_response_summary: Dict[str, Any], output_dir: Path):
    group_labels = list(signal_response_summary.keys())
    x_axis = np.arange(len(group_labels))
    categories = ["inspect_first_rate", "respond_first_rate", "no_followup_rate"]
    labels = {
        "inspect_first_rate": "Inspect first",
        "respond_first_rate": "Respond first",
        "no_followup_rate": "No follow-up",
    }
    bottom = np.zeros(len(group_labels))
    plt.figure(figsize=(12, 6))
    for category in categories:
        values = [signal_response_summary[group][category] for group in group_labels]
        plt.bar(x_axis, values, bottom=bottom, label=labels[category])
        bottom += np.asarray(values)
    plt.xticks(x_axis, group_labels, rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.title("Inspection-response behavior after suspicious signal")
    plt.legend()
    save_figure(output_dir, "policy_signal_followup_behavior.png")


def write_summary_markdown(
    output_dir: Path,
    manifest_path: Path,
    action_summary: Dict[str, Any],
    collapse_summary: Dict[str, Any],
    signal_response_summary: Dict[str, Any],
):
    lines = [
        "# Policy Pattern Summary",
        "",
        f"- manifest: {manifest_path}",
        "- Scope: managerial interpretation of defender behavior from final evaluation traces.",
        "",
    ]
    for group_label in action_summary.keys():
        collapse_payload = collapse_summary.get(group_label, {})
        signal_payload = signal_response_summary.get(group_label, {})
        lines.extend(
            [
                f"## {group_label}",
                f"- hold / inspect / respond: {action_summary[group_label]['action_category_ratios']['hold']:.3f} / {action_summary[group_label]['action_category_ratios']['inspect']:.3f} / {action_summary[group_label]['action_category_ratios']['respond']:.3f}",
                f"- collapse frequency: {collapse_payload.get('collapse_frequency', 0.0):.3f}",
                f"- mean collapse step: {collapse_payload.get('mean_collapse_step')}",
                f"- suspicious signal count: {signal_payload.get('suspicious_signal_count', 0)}",
                f"- mean response lag: {signal_payload.get('mean_response_lag')}",
                f"- inspect-first / respond-first / no-followup: {signal_payload.get('inspect_first_rate', 0.0):.3f} / {signal_payload.get('respond_first_rate', 0.0):.3f} / {signal_payload.get('no_followup_rate', 0.0):.3f}",
                "",
            ]
        )
    (output_dir / "policy_pattern_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze V2 policy patterns from result manifests.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--paper-group", choices=["main", "ablation", "robustness"], default="main")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else find_latest_manifest(results_root, MANIFEST_STEMS[args.paper_group])
    )
    manifest = load_manifest(manifest_path)
    group_fields = infer_group_fields(manifest)
    loaded_runs = load_runs_from_manifest(manifest)

    action_summary = compute_action_summary(loaded_runs, group_fields)
    trajectories = compute_budget_trajectories(loaded_runs, group_fields)
    collapse_summary = compute_collapse_summary(loaded_runs, group_fields)
    signal_response_summary = compute_signal_response_summary(loaded_runs, group_fields)

    output_dir = ensure_output_dir(results_root, "policy_analysis")
    with open(output_dir / "policy_pattern_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "manifest_path": str(manifest_path),
                "group_fields": group_fields,
                "action_summary": action_summary,
                "budget_trajectories": trajectories,
                "collapse_summary": collapse_summary,
                "signal_response_summary": signal_response_summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    if action_summary:
        plot_action_category_ratios(action_summary, output_dir)
        plot_zone_action_distribution(action_summary, output_dir)
    if trajectories:
        plot_budget_trajectories(trajectories, output_dir)
    if collapse_summary:
        plot_collapse_summary(collapse_summary, output_dir)
    if signal_response_summary:
        plot_response_timing(signal_response_summary, output_dir)
        plot_signal_followup(signal_response_summary, output_dir)
    write_summary_markdown(output_dir, manifest_path, action_summary, collapse_summary, signal_response_summary)
    print(f"V2 policy pattern analysis written to {output_dir}")


if __name__ == "__main__":
    main()
