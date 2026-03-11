from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flipit-simulation-master"))

from paper_v2_batch_utils import collect_completed_runs


def _write_complete_results(
    run_dir: Path,
    *,
    experiment_info: dict,
    final_performance: dict,
    final_evaluation_details: list | None = None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_info": experiment_info,
        "config_snapshot": {
            "experiment": {
                key: experiment_info.get(key)
                for key in [
                    "experiment_id",
                    "paper_group",
                    "scenario_id",
                    "method_id",
                    "random_seed",
                    "variant_id",
                    "robustness_family",
                    "robustness_level",
                    "config_path",
                ]
            }
        },
        "final_performance": final_performance,
        "training_history": [],
        "evaluation_history": [],
        "final_evaluation_details": [] if final_evaluation_details is None else final_evaluation_details,
    }
    (run_dir / "complete_training_results.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _performance(avg_defender_return: float, attacker_success_rate: float, defender_control_rate: float) -> dict:
    return {
        "avg_defender_return": avg_defender_return,
        "attacker_success_rate": attacker_success_rate,
        "defender_control_rate": defender_control_rate,
        "avg_defender_training_return": avg_defender_return + 1.0,
        "avg_defender_spent_budget": 10.0,
        "avg_defender_control_per_cost": defender_control_rate / 2.0,
        "avg_false_response_rate": 0.1,
        "avg_missed_response_rate": 0.05,
    }


def test_collect_completed_runs_prefers_latest_run_with_same_identity(tmp_path: Path):
    results_root = tmp_path / "results"
    older_dir = results_root / "run_old"
    newer_dir = results_root / "run_new"
    common_info = {
        "paper_group": "robustness",
        "scenario_id": "cheat",
        "method_id": "baseline",
        "random_seed": 42,
        "variant_id": "budget_stress_nominal",
        "robustness_family": "budget_stress",
        "robustness_level": "nominal",
        "config_path": "demo.yml",
    }
    _write_complete_results(
        older_dir,
        experiment_info={**common_info, "experiment_id": "older"},
        final_performance=_performance(10.0, 0.3, 0.5),
    )
    _write_complete_results(
        newer_dir,
        experiment_info={**common_info, "experiment_id": "newer"},
        final_performance=_performance(12.0, 0.2, 0.6),
    )
    os.utime(older_dir, (100, 100))
    os.utime(newer_dir, (200, 200))
    completed_runs = collect_completed_runs(results_root)
    run_key = ("robustness", "cheat", "baseline", "budget_stress_nominal", "budget_stress", "nominal", 42)
    assert completed_runs[run_key]["experiment_id"] == "newer"
    assert completed_runs[run_key]["final_performance"]["avg_defender_return"] == 12.0


def test_ablation_analysis_script_reads_manifest_and_writes_outputs(tmp_path: Path):
    results_root = tmp_path / "ablation_results"
    run_full = results_root / "paper_ablation_cheat_full_seed42_20260311"
    run_mask = results_root / "paper_ablation_cheat_no_action_mask_seed42_20260311"
    _write_complete_results(
        run_full,
        experiment_info={
            "experiment_id": "paper_ablation_cheat_full_seed42",
            "paper_group": "ablation",
            "scenario_id": "cheat",
            "method_id": "drl",
            "random_seed": 42,
            "variant_id": "full",
            "config_path": "full.yml",
        },
        final_performance=_performance(80.0, 0.15, 0.7),
    )
    _write_complete_results(
        run_mask,
        experiment_info={
            "experiment_id": "paper_ablation_cheat_no_action_mask_seed42",
            "paper_group": "ablation",
            "scenario_id": "cheat",
            "method_id": "drl",
            "random_seed": 42,
            "variant_id": "no_action_mask",
            "config_path": "no_action_mask.yml",
        },
        final_performance=_performance(55.0, 0.25, 0.6),
    )
    manifest_path = results_root / "paper_ablation_manifest_latest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "paper_group": "ablation",
                "runs": [
                    {
                        "experiment_id": "paper_ablation_cheat_full_seed42",
                        "scenario_id": "cheat",
                        "method_id": "drl",
                        "variant_id": "full",
                        "seed": 42,
                        "result_dir": str(run_full),
                    },
                    {
                        "experiment_id": "paper_ablation_cheat_no_action_mask_seed42",
                        "scenario_id": "cheat",
                        "method_id": "drl",
                        "variant_id": "no_action_mask",
                        "seed": 42,
                        "result_dir": str(run_mask),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "flipit-simulation-master" / "analysis" / "analyze_paper_ablation_v2.py"),
            "--manifest",
            str(manifest_path),
            "--results-root",
            str(results_root),
        ],
        cwd=ROOT / "flipit-simulation-master",
        check=True,
    )
    output_dirs = sorted(results_root.glob("analysis_*"))
    assert output_dirs
    assert (output_dirs[-1] / "ablation_experiment_summary.md").exists()
    assert (output_dirs[-1] / "ablation_experiment_summary.json").exists()
    assert (output_dirs[-1] / "ablation_raw_return_by_variant.png").exists()


def test_robustness_analysis_script_reads_manifest_and_writes_outputs(tmp_path: Path):
    results_root = tmp_path / "robustness_results"
    run_baseline = results_root / "paper_robustness_cheat_budget_stress_nominal_baseline_seed42_20260311"
    run_drl = results_root / "paper_robustness_cheat_budget_stress_nominal_drl_seed42_20260311"
    _write_complete_results(
        run_baseline,
        experiment_info={
            "experiment_id": "paper_robustness_cheat_budget_stress_nominal_baseline_seed42",
            "paper_group": "robustness",
            "scenario_id": "cheat",
            "method_id": "baseline",
            "random_seed": 42,
            "variant_id": "budget_stress_nominal",
            "robustness_family": "budget_stress",
            "robustness_level": "nominal",
            "config_path": "baseline.yml",
        },
        final_performance=_performance(35.0, 0.28, 0.58),
    )
    _write_complete_results(
        run_drl,
        experiment_info={
            "experiment_id": "paper_robustness_cheat_budget_stress_nominal_drl_seed42",
            "paper_group": "robustness",
            "scenario_id": "cheat",
            "method_id": "drl",
            "random_seed": 42,
            "variant_id": "budget_stress_nominal",
            "robustness_family": "budget_stress",
            "robustness_level": "nominal",
            "config_path": "drl.yml",
        },
        final_performance=_performance(70.0, 0.12, 0.74),
    )
    manifest_path = results_root / "paper_robustness_manifest_latest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "paper_group": "robustness",
                "runs": [
                    {
                        "experiment_id": "paper_robustness_cheat_budget_stress_nominal_baseline_seed42",
                        "scenario_id": "cheat",
                        "method_id": "baseline",
                        "variant_id": "budget_stress_nominal",
                        "robustness_family": "budget_stress",
                        "robustness_level": "nominal",
                        "seed": 42,
                        "result_dir": str(run_baseline),
                    },
                    {
                        "experiment_id": "paper_robustness_cheat_budget_stress_nominal_drl_seed42",
                        "scenario_id": "cheat",
                        "method_id": "drl",
                        "variant_id": "budget_stress_nominal",
                        "robustness_family": "budget_stress",
                        "robustness_level": "nominal",
                        "seed": 42,
                        "result_dir": str(run_drl),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "flipit-simulation-master" / "analysis" / "analyze_paper_robustness_v2.py"),
            "--manifest",
            str(manifest_path),
            "--results-root",
            str(results_root),
        ],
        cwd=ROOT / "flipit-simulation-master",
        check=True,
    )
    output_dirs = sorted(results_root.glob("analysis_*"))
    assert output_dirs
    assert (output_dirs[-1] / "robustness_experiment_summary.md").exists()
    assert (output_dirs[-1] / "robustness_experiment_summary.json").exists()
    assert (output_dirs[-1] / "robustness_budget_stress_comparison.png").exists()


def test_policy_pattern_analysis_script_reads_step_records_and_writes_outputs(tmp_path: Path):
    results_root = tmp_path / "policy_results"
    run_dir = results_root / "paper_main_cheat_drl_seed42_20260311"
    _write_complete_results(
        run_dir,
        experiment_info={
            "experiment_id": "paper_main_cheat_drl_seed42",
            "paper_group": "main",
            "scenario_id": "cheat",
            "method_id": "drl",
            "random_seed": 42,
            "config_path": "main.yml",
        },
        final_performance=_performance(90.0, 0.1, 0.8),
        final_evaluation_details=[
            {
                "episode_length": 3,
                "termination_reason": "max_steps",
                "step_records": [
                    {
                        "step": 1,
                        "current_signal": "outer",
                        "defender_action_label": "inspect_outer",
                        "defender_budget_remaining": 30.0,
                        "attacker_budget_remaining": 28.0,
                    },
                    {
                        "step": 2,
                        "current_signal": "null",
                        "defender_action_label": "respond_outer",
                        "defender_budget_remaining": 25.0,
                        "attacker_budget_remaining": 24.0,
                    },
                    {
                        "step": 3,
                        "current_signal": "null",
                        "defender_action_label": "hold",
                        "defender_budget_remaining": 26.0,
                        "attacker_budget_remaining": 23.0,
                    },
                ],
            }
        ],
    )
    manifest_path = results_root / "paper_main_manifest_latest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "paper_group": "main",
                "runs": [
                    {
                        "experiment_id": "paper_main_cheat_drl_seed42",
                        "scenario_id": "cheat",
                        "method_id": "drl",
                        "seed": 42,
                        "result_dir": str(run_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "flipit-simulation-master" / "analysis" / "analyze_policy_patterns_v2.py"),
            "--manifest",
            str(manifest_path),
            "--results-root",
            str(results_root),
            "--paper-group",
            "main",
        ],
        cwd=ROOT / "flipit-simulation-master",
        check=True,
    )
    output_dirs = sorted(results_root.glob("policy_analysis_*"))
    assert output_dirs
    assert (output_dirs[-1] / "policy_pattern_summary.md").exists()
    assert (output_dirs[-1] / "policy_pattern_summary.json").exists()
    assert (output_dirs[-1] / "policy_action_category_ratios.png").exists()
