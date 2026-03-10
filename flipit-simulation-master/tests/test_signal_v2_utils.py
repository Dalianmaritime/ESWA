from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gym-flipit-master"))
sys.path.insert(0, str(ROOT / "flipit-simulation-master"))

from signal_v2_utils import compute_final_performance, generate_summary_markdown, get_metric_value, summarize_episode
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
            },
        ]
    )
    assert performance["avg_defender_return"] == -1.0
    assert performance["avg_defender_training_return"] == 3.0


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
