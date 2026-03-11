from __future__ import annotations

import sys
import json
import os
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gym-flipit-master"))
sys.path.insert(0, str(ROOT / "flipit-simulation-master"))

from signal_v2_utils import (
    ablate_signal_features,
    build_experiment_tags,
    build_baseline_reference_targets,
    compute_checkpoint_selection_metrics,
    compute_final_performance,
    confirmation_candidate_sort_key,
    constrained_candidate_improves,
    generate_summary_markdown,
    get_metric_value,
    get_variant_controls,
    routine_candidate_sort_key,
    should_trigger_early_stopping,
    summarize_episode,
    write_baseline_reference_targets_from_source,
)
from strategies.signal_rainbow_dqn_v2 import SignalRainbowDQNAgentV2, project_distribution


def test_summarize_episode_uses_metrics_tick_when_trace_is_disabled():
    last_info = {
        "winner": "defender",
        "episode_metrics_snapshot": {
            "tick": 12,
            "attacker_control_steps": 3,
            "defender_control_steps": 9,
            "inspect_actions": 4,
            "positive_inspections": 3,
            "false_responses": 1,
            "missed_responses": 2,
        },
    }
    summary = summarize_episode(
        episode_index=0,
        step_records=[],
        last_info=last_info,
        attacker_return=-5.0,
        defender_return=7.0,
    )
    assert summary["episode_length"] == 12
    assert summary["attacker_control_rate"] == 3 / 12
    assert summary["defender_control_rate"] == 9 / 12
    assert summary["false_response_rate"] == 1 / 12
    assert summary["missed_response_rate"] == 2 / 12


def test_get_variant_controls_defaults_to_current_v2_behavior():
    controls = get_variant_controls({"experiment": {"experiment_id": "demo"}})
    assert controls == {
        "disable_resource_sustainability": False,
        "disable_reward_shaping": False,
        "disable_action_mask": False,
        "disable_signal_features": False,
    }


def test_ablate_signal_features_replaces_signal_and_belief_inputs_with_neutral_values():
    observation = [0.1] * 17
    sanitized = ablate_signal_features(observation)
    assert sanitized[2] == 0.0
    assert sanitized[3:6].tolist() == pytest.approx([1.0 / 3.0] * 3)
    assert sanitized[6:10].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert sanitized[10:14].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert sanitized[14] == 0.0
    assert sanitized[15] == pytest.approx(0.1)
    assert sanitized[16] == pytest.approx(0.1)


def test_build_experiment_tags_preserves_variant_and_robustness_metadata():
    config = {
        "experiment": {
            "paper_group": "robustness",
            "variant_id": "budget_low",
            "variant_label": "Low defender budget",
            "robustness_family": "budget_stress",
            "robustness_level": "low",
        },
        "variant_controls": {
            "disable_action_mask": True,
        },
    }
    tags = build_experiment_tags(config)
    assert tags["paper_group"] == "robustness"
    assert tags["variant_id"] == "budget_low"
    assert tags["variant_label"] == "Low defender budget"
    assert tags["robustness_family"] == "budget_stress"
    assert tags["robustness_level"] == "low"
    assert tags["variant_controls"]["disable_action_mask"] is True


def test_distribution_projection_preserves_probability_mass_for_same_bin_case():
    support = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    next_target_dist = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float32)
    rewards = torch.tensor([0.0], dtype=torch.float32)
    dones = torch.tensor([0.0], dtype=torch.float32)
    projected = project_distribution(
        support=support,
        next_target_dist=next_target_dist,
        rewards=rewards,
        dones=dones,
        gamma=1.0,
        v_min=-1.0,
        v_max=1.0,
    )
    assert torch.allclose(projected.sum(dim=1), torch.tensor([1.0]), atol=1e-6)


def test_summarize_episode_keeps_training_return_separate_from_raw_return():
    last_info = {
        "winner": "defender",
        "episode_metrics_snapshot": {
            "tick": 4,
            "attacker_control_steps": 1,
            "defender_control_steps": 3,
            "inspect_actions": 1,
            "positive_inspections": 1,
            "false_responses": 0,
            "missed_responses": 0,
        },
    }
    summary = summarize_episode(
        episode_index=1,
        step_records=[],
        last_info=last_info,
        attacker_return=-2.0,
        defender_return=1.0,
        defender_training_return=5.5,
    )
    assert summary["defender_return"] == 1.0
    assert summary["defender_training_return"] == 5.5


def test_compute_final_performance_keeps_training_reward_separate_from_raw_return():
    performance = compute_final_performance(
        [
            {
                "attacker_success": False,
                "defender_control_rate": 0.8,
                "attacker_control_rate": 0.2,
                "defender_return": 1.0,
                "defender_training_return": 5.0,
                "attacker_return": -1.0,
                "false_response_rate": 0.1,
                "missed_response_rate": 0.05,
                "inspection_precision": 0.6,
                "episode_length": 10,
                "total_defender_action_cost": 3.0,
                "defender_control_per_cost": 0.8,
            },
            {
                "attacker_success": True,
                "defender_control_rate": 0.4,
                "attacker_control_rate": 0.6,
                "defender_return": -3.0,
                "defender_training_return": 1.0,
                "attacker_return": 3.0,
                "false_response_rate": 0.2,
                "missed_response_rate": 0.15,
                "inspection_precision": 0.4,
                "episode_length": 12,
                "total_defender_action_cost": 5.0,
                "defender_control_per_cost": 0.4,
            },
        ]
    )
    assert performance["avg_defender_return"] == -1.0
    assert performance["avg_defender_training_return"] == 3.0
    assert performance["avg_defender_spent_budget"] == 4.0
    assert performance["avg_defender_control_per_cost"] == pytest.approx(0.6)
    assert performance["validation_selection_score"] == pytest.approx(-1.0 - 50.0 * 0.1 - 20.0 * 0.15)


def test_get_metric_value_returns_requested_metric_and_rejects_missing_metric():
    metrics = {"avg_defender_return": -2.0, "avg_defender_training_return": 4.5}
    assert get_metric_value(metrics, "avg_defender_training_return") == 4.5
    with pytest.raises(KeyError):
        get_metric_value(metrics, "missing_metric")


def test_generate_summary_markdown_documents_training_and_report_metric_split():
    markdown = generate_summary_markdown(
        {
            "experiment_info": {
                "experiment_id": "demo",
                "experiment_type": "drl",
                "mode": "cheat",
                "timestamp": "2026-03-09T00:00:00",
            },
            "metric_conventions": {
                "training_metric": "defender_training_return (shaped reward used for optimization)",
                "report_metric": "avg_defender_return (raw utility used for reporting)",
            },
            "checkpoint_selection": {
                "metric": "avg_defender_training_return",
                "best_episode": 12,
            },
            "final_performance": {
                "attacker_success_rate": 0.25,
                "defender_control_rate": 0.75,
                "attacker_control_rate": 0.25,
                "avg_defender_return": -6.0,
                "avg_defender_training_return": 8.0,
                "avg_attacker_return": -2.0,
                "avg_false_response_rate": 0.1,
                "avg_missed_response_rate": 0.05,
                "avg_inspection_precision": 0.7,
                "avg_episode_length": 60.0,
                "sample_size": 8,
            },
        }
    )
    assert "Training objective" in markdown
    assert "Reported utility" in markdown
    assert "Avg defender return: -6.000" in markdown
    assert "Avg defender training return: 8.000" in markdown


def test_valid_action_mask_blocks_unaffordable_inspect_and_respond_actions():
    agent = SignalRainbowDQNAgentV2(
        obs_dim=17,
        action_dim=7,
        defender_initial_budget=120.0,
        defender_inspect_cost=1.0,
        defender_respond_cost_by_zone={"outer": 4.0, "lane": 5.0, "core": 6.0},
        defender_action_floor=-6.0,
    )
    observation = torch.zeros((1, 17), dtype=torch.float32)
    observation[0, 0] = -2.5 / 120.0
    mask = agent._valid_action_mask(observation)
    assert mask.shape == (1, 7)
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is True
    assert mask[0, 4].item() is False


def test_constrained_selection_prefers_security_feasible_candidate():
    reference_targets = {
        "attacker_success_rate": 0.18,
        "defender_control_rate": 0.67,
        "avg_defender_return": 40.0,
    }
    higher_raw_but_less_secure = {
        "episode": 600,
        "performance": {
            "attacker_success_rate": 0.27,
            "defender_control_rate": 0.61,
            "avg_defender_return": 70.0,
            "avg_false_response_rate": 0.02,
            "avg_missed_response_rate": 0.04,
            "avg_defender_control_per_cost": 0.50,
        },
    }
    safer_candidate = {
        "episode": 300,
        "performance": {
            "attacker_success_rate": 0.20,
            "defender_control_rate": 0.65,
            "avg_defender_return": 54.0,
            "avg_false_response_rate": 0.04,
            "avg_missed_response_rate": 0.05,
            "avg_defender_control_per_cost": 0.62,
        },
    }
    for candidate in (higher_raw_but_less_secure, safer_candidate):
        candidate["selection_metrics"] = compute_checkpoint_selection_metrics(
            candidate["performance"],
            reference_targets,
            attacker_success_tolerance=0.03,
            defender_control_tolerance=0.03,
        )

    ranked = sorted([higher_raw_but_less_secure, safer_candidate], key=routine_candidate_sort_key)
    assert ranked[0]["episode"] == 300
    assert ranked[0]["selection_metrics"]["selection_feasible_under_tolerance"] is True


def test_confirmation_selection_falls_back_to_minimum_security_deficit():
    reference_targets = {
        "attacker_success_rate": 0.18,
        "defender_control_rate": 0.67,
        "avg_defender_return": 40.0,
    }
    candidate_a = {
        "episode": 200,
        "performance": {
            "attacker_success_rate": 0.22,
            "defender_control_rate": 0.61,
            "avg_defender_return": 60.0,
            "avg_false_response_rate": 0.02,
            "avg_missed_response_rate": 0.03,
            "avg_defender_control_per_cost": 0.55,
        },
    }
    candidate_b = {
        "episode": 350,
        "performance": {
            "attacker_success_rate": 0.20,
            "defender_control_rate": 0.63,
            "avg_defender_return": 55.0,
            "avg_false_response_rate": 0.03,
            "avg_missed_response_rate": 0.04,
            "avg_defender_control_per_cost": 0.58,
        },
    }
    for candidate in (candidate_a, candidate_b):
        candidate["selection_metrics"] = compute_checkpoint_selection_metrics(candidate["performance"], reference_targets)
        assert candidate["selection_metrics"]["selection_feasible_under_baseline"] is False

    ranked = sorted([candidate_a, candidate_b], key=confirmation_candidate_sort_key)
    assert ranked[0]["episode"] == 350
    assert ranked[0]["selection_metrics"]["security_deficit"] < ranked[1]["selection_metrics"]["security_deficit"]


def test_constrained_candidate_improves_tracks_security_first_improvement():
    reference_targets = {
        "attacker_success_rate": 0.18,
        "defender_control_rate": 0.67,
        "avg_defender_return": 40.0,
    }
    current_best = {
        "episode": 600,
        "performance": {
            "attacker_success_rate": 0.27,
            "defender_control_rate": 0.61,
            "avg_defender_return": 70.0,
            "avg_false_response_rate": 0.02,
            "avg_missed_response_rate": 0.04,
            "avg_defender_control_per_cost": 0.50,
        },
    }
    safer_candidate = {
        "episode": 300,
        "performance": {
            "attacker_success_rate": 0.20,
            "defender_control_rate": 0.65,
            "avg_defender_return": 54.0,
            "avg_false_response_rate": 0.04,
            "avg_missed_response_rate": 0.05,
            "avg_defender_control_per_cost": 0.62,
        },
    }
    for candidate in (current_best, safer_candidate):
        candidate["selection_metrics"] = compute_checkpoint_selection_metrics(
            candidate["performance"],
            reference_targets,
            attacker_success_tolerance=0.03,
            defender_control_tolerance=0.03,
        )

    assert constrained_candidate_improves(safer_candidate, current_best) is True
    assert constrained_candidate_improves(current_best, safer_candidate) is False


def test_should_trigger_early_stopping_requires_minimum_training_and_patience():
    assert should_trigger_early_stopping(
        enabled=False,
        current_episode=499,
        min_training_episodes=200,
        evaluations_since_improvement=8,
        patience_evaluations=6,
    ) is False
    assert should_trigger_early_stopping(
        enabled=True,
        current_episode=149,
        min_training_episodes=200,
        evaluations_since_improvement=6,
        patience_evaluations=6,
    ) is False
    assert should_trigger_early_stopping(
        enabled=True,
        current_episode=249,
        min_training_episodes=200,
        evaluations_since_improvement=5,
        patience_evaluations=6,
    ) is False
    assert should_trigger_early_stopping(
        enabled=True,
        current_episode=249,
        min_training_episodes=200,
        evaluations_since_improvement=6,
        patience_evaluations=6,
    ) is True


def test_build_baseline_reference_targets_aggregates_latest_baseline_runs(tmp_path: Path):
    results_root = tmp_path / "results"
    results_root.mkdir()
    payloads = [
        ("paper_main_flipit_baseline_seed42_a", "flipit", 42, 0.16, 0.68, 41.6),
        ("paper_main_flipit_baseline_seed42_b", "flipit", 42, 0.18, 0.60, 10.0),
        ("paper_main_flipit_baseline_seed123", "flipit", 123, 0.19, 0.67, 36.9),
        ("paper_main_cheat_baseline_seed42", "cheat", 42, 0.27, 0.58, -5.0),
    ]
    for name, scenario_id, seed, attacker_success, defender_control, raw_return in payloads:
        run_dir = results_root / name
        run_dir.mkdir()
        (run_dir / "complete_training_results.json").write_text(
            json.dumps(
                {
                    "experiment_info": {
                        "scenario_id": scenario_id,
                        "method_id": "baseline",
                        "random_seed": seed,
                    },
                    "final_performance": {
                        "attacker_success_rate": attacker_success,
                        "defender_control_rate": defender_control,
                        "avg_defender_return": raw_return,
                    },
                }
            ),
            encoding="utf-8",
        )
        timestamp = 100 if name.endswith("_a") else 200 if name.endswith("_b") else 300
        os.utime(run_dir, (timestamp, timestamp))
        os.utime(run_dir / "complete_training_results.json", (timestamp, timestamp))

    reference_payload = build_baseline_reference_targets(results_root, scenarios=["flipit"])
    flipit = reference_payload["scenarios"]["flipit"]
    assert flipit["num_runs"] == 2
    assert flipit["seeds"] == [42, 123]
    assert flipit["attacker_success_rate"] == pytest.approx((0.18 + 0.19) / 2.0)
    assert flipit["defender_control_rate"] == pytest.approx((0.60 + 0.67) / 2.0)
    assert flipit["avg_defender_return"] == pytest.approx((10.0 + 36.9) / 2.0)


def test_write_baseline_reference_targets_from_source_copies_payload_to_new_results_root(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    run_dir = source_root / "paper_main_cheat_baseline_seed42"
    run_dir.mkdir()
    (run_dir / "complete_training_results.json").write_text(
        json.dumps(
            {
                "experiment_info": {
                    "scenario_id": "cheat",
                    "method_id": "baseline",
                    "random_seed": 42,
                },
                "final_performance": {
                    "attacker_success_rate": 0.25,
                    "defender_control_rate": 0.61,
                    "avg_defender_return": -8.0,
                },
            }
        ),
        encoding="utf-8",
    )
    destination_root = tmp_path / "destination"
    output_path = write_baseline_reference_targets_from_source(
        source_results_root=source_root,
        destination_results_root=destination_root,
        scenarios=["cheat"],
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["results_root"] == str(source_root.resolve())
    assert payload["copied_to_results_root"] == str(destination_root.resolve())
    assert payload["scenarios"]["cheat"]["seeds"] == [42]
