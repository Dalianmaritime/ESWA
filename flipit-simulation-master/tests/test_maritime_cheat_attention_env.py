from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gym-flipit-master"))
sys.path.insert(0, str(ROOT / "flipit-simulation-master"))

from gym_flipit.envs.maritime_cheat_attention_env import MaritimeCheatAttentionEnv
from strategies.belief_threshold_defender_v2 import BeliefThresholdDefenderV2
from strategies.signal_cheat_greedy_attacker_v2 import SignalCheatGreedyAttackerV2
from strategies.signal_rainbow_dqn_v2 import SignalRainbowDQNAgentV2


def load_base_config():
    config_path = ROOT / "flipit-simulation-master" / "configs" / "trc_signal_cheat_v2.yml"
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def deterministic_config():
    config = copy.deepcopy(load_base_config())
    config["environment"]["max_steps"] = 5
    config["signal_model"]["background_noise_by_zone"] = {"outer": 0.0, "lane": 0.0, "core": 0.0}
    config["signal_model"]["inspection_accuracy_by_zone"] = {"outer": 1.0, "lane": 1.0, "core": 1.0}
    config["transition_model"]["base_takeover_success_by_zone"] = {"outer": 1.0, "lane": 1.0, "core": 1.0}
    config["transition_model"]["recapture_success_by_zone"] = {"outer": 1.0, "lane": 1.0, "core": 1.0}
    return config


def test_cheat_does_not_change_controller(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    _, _, _, _, info = env.step((1, 0))
    assert info["true_controller"] == "defender"
    assert info["takeover_attempted"] is False
    assert info["current_signal"] in {"outer", "null"}


def test_takeover_changes_controller(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    _, _, _, _, info = env.step((4, 0))
    assert info["takeover_attempted"] is True
    assert info["takeover_success"] is True
    assert info["true_controller"] == "attacker"


def test_inspect_updates_belief_without_control_change(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    env.true_controller = "attacker"
    env.true_zone = "outer"
    prior_mu = env.mu_breach
    _, _, _, _, info = env.step((0, 1))
    assert info["true_controller"] == "attacker"
    assert info["inspection_result"] == "positive"
    assert env.mu_breach != prior_mu
    assert env.last_inspect_result == 1


def test_respond_same_zone_recaptures(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    env.true_controller = "attacker"
    env.true_zone = "core"
    _, _, _, _, info = env.step((0, 6))
    assert info["response_success"] is True
    assert info["true_controller"] == "defender"


def test_false_response_flag(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    _, _, _, _, info = env.step((1, 4))
    assert info["false_response"] is True
    assert env.prev_false_response_flag == 1


def test_missed_response_flag(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    _, _, _, _, info = env.step((6, 0))
    assert info["missed_response"] is True
    assert env.prev_missed_response_flag == 1


def test_failed_takeover_without_coverage_is_not_counted_as_missed_response(deterministic_config):
    config = copy.deepcopy(deterministic_config)
    config["transition_model"]["base_takeover_success_by_zone"] = {"outer": 0.0, "lane": 0.0, "core": 0.0}
    env = MaritimeCheatAttentionEnv(config)
    _, _ = env.reset(seed=7)
    _, _, _, _, info = env.step((6, 0))
    assert info["takeover_attempted"] is True
    assert info["takeover_success"] is False
    assert info["missed_response"] is False
    assert env.prev_missed_response_flag == 0


def test_environment_returns_training_reward_while_preserving_raw_defender_reward(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    _, reward, _, _, info = env.step((1, 4))
    assert reward == pytest.approx(info["training_reward"])
    assert info["defender_reward"] < reward
    assert info["false_response"] is True


def test_disable_reward_shaping_uses_raw_defender_reward_for_training(deterministic_config):
    config = copy.deepcopy(deterministic_config)
    config["variant_controls"] = {"disable_reward_shaping": True}
    env = MaritimeCheatAttentionEnv(config)
    _, _ = env.reset(seed=7)
    _, reward, _, _, info = env.step((1, 4))
    assert reward == pytest.approx(info["defender_reward"])
    assert info["training_reward"] == pytest.approx(info["defender_reward"])


def test_flipit_mode_disables_cheat():
    config = copy.deepcopy(load_base_config())
    config["signal_model"]["mode"] = "flipit"
    config["signal_model"]["cheat_emit_prob"] = 0.0
    config["policies"]["attacker"]["allow_cheat"] = False
    config["signal_model"]["background_noise_by_zone"] = {"outer": 0.0, "lane": 0.0, "core": 0.0}

    env = MaritimeCheatAttentionEnv(config)
    strategy = SignalCheatGreedyAttackerV2(
        allow_cheat=True,
        cheat_cost=config["costs_and_rewards"]["attacker_cheat_cost"],
        takeover_cost_by_zone=config["costs_and_rewards"]["attacker_takeover_cost_by_zone"],
        takeover_trigger_belief=config["policies"]["attacker"]["takeover_trigger_belief"],
        exploit_false_response=config["policies"]["attacker"]["exploit_false_response"],
    )
    _, _ = env.reset(seed=7)
    action = strategy.select_action(env.get_public_state())
    assert action not in {1, 2, 3}

    _, _, _, _, info = env.step((1, 0))
    assert info["attacker_action_label"] == "wait"
    assert info["signal_source_action"] == "background_noise"


def test_observation_dim_and_action_space_from_config():
    env = MaritimeCheatAttentionEnv(load_base_config())
    observation, info = env.reset(seed=7)
    assert env.action_space.n == 7
    assert observation.shape == (17,)
    assert info["mode"] == "cheat"


def test_defender_can_overdraft_if_action_stays_above_floor(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    env.defender_budget = 0.5
    _, _, terminated, truncated, info = env.step((0, 4))
    assert terminated is False
    assert truncated is False
    assert info["defender_action_label"] == "respond_outer"
    assert info["defender_budget_remaining"] == pytest.approx(0.0)


def test_actions_below_floor_are_downgraded_to_wait_or_hold(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    env.attacker_budget = -5.5
    env.defender_budget = -3.0
    _, _, _, _, info = env.step((1, 4))
    assert info["attacker_action_label"] == "wait"
    assert info["defender_action_label"] == "hold"


def test_control_bonus_applies_only_to_current_controller(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    _, _, _, _, info = env.step((0, 0))
    assert info["attacker_base_income_applied"] == pytest.approx(deterministic_config["resources"]["attacker_base_income_per_step"])
    assert info["attacker_control_bonus_applied"] == pytest.approx(0.0)
    assert info["defender_base_income_applied"] == pytest.approx(deterministic_config["resources"]["defender_base_income_per_step"])
    assert info["defender_control_bonus_applied"] == pytest.approx(
        deterministic_config["resources"]["defender_control_bonus_per_step"]
    )

    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    env.true_controller = "attacker"
    env.true_zone = "lane"
    _, _, _, _, info = env.step((0, 0))
    assert info["attacker_control_bonus_applied"] == pytest.approx(
        deterministic_config["resources"]["attacker_control_bonus_per_step"]
    )
    assert info["defender_control_bonus_applied"] == pytest.approx(0.0)


def test_consecutive_below_guarantee_steps_trigger_resource_collapse(deterministic_config):
    config = copy.deepcopy(deterministic_config)
    config["resources"]["defender_base_income_per_step"] = 0.0
    config["resources"]["defender_control_bonus_per_step"] = 0.0
    config["resources"]["defender_guarantee_line"] = -2.0
    config["resources"]["guarantee_breach_patience"] = 2
    env = MaritimeCheatAttentionEnv(config)
    _, _ = env.reset(seed=7)
    env.defender_budget = -2.5
    _, _, terminated, truncated, info = env.step((0, 0))
    assert terminated is False
    assert truncated is False
    assert info["defender_below_guarantee_streak"] == 1

    _, _, terminated, truncated, info = env.step((0, 0))
    assert terminated is True
    assert truncated is False
    assert info["winner"] == "attacker"
    assert info["defender_budget_collapse"] is True
    assert info["termination_reason"] == "defender_resource_collapse"


def test_disable_resource_sustainability_removes_income_and_overdraft_path(deterministic_config):
    config = copy.deepcopy(deterministic_config)
    config["variant_controls"] = {"disable_resource_sustainability": True}
    env = MaritimeCheatAttentionEnv(config)
    _, _ = env.reset(seed=7)
    env.defender_budget = 0.5
    _, _, terminated, truncated, info = env.step((0, 4))
    assert terminated is False
    assert truncated is False
    assert info["defender_action_label"] == "hold"
    assert info["defender_base_income_applied"] == pytest.approx(0.0)
    assert info["defender_control_bonus_applied"] == pytest.approx(0.0)
    assert info["defender_budget_remaining"] == pytest.approx(0.5)
    assert info["defender_below_guarantee_streak"] == 0


def test_guarantee_streak_resets_after_budget_recovery(deterministic_config):
    config = copy.deepcopy(deterministic_config)
    config["resources"]["defender_base_income_per_step"] = 0.0
    config["resources"]["defender_control_bonus_per_step"] = 0.0
    config["resources"]["defender_guarantee_line"] = -2.0
    env = MaritimeCheatAttentionEnv(config)
    _, _ = env.reset(seed=7)
    env.defender_budget = -2.5
    _, _, _, _, info = env.step((0, 0))
    assert info["defender_below_guarantee_streak"] == 1

    env.defender_budget = -1.0
    _, _, _, _, info = env.step((0, 0))
    assert info["defender_below_guarantee_streak"] == 0


def test_dqn_mask_uses_action_floor_instead_of_cash_coverage():
    agent = SignalRainbowDQNAgentV2(
        obs_dim=17,
        action_dim=7,
        defender_initial_budget=28.0,
        defender_inspect_cost=1.0,
        defender_respond_cost_by_zone={"outer": 4.0, "lane": 5.0, "core": 6.0},
        defender_action_floor=-6.0,
    )
    observation = torch.zeros((1, 17), dtype=torch.float32)
    observation[0, 0] = -2.5 / 28.0
    mask = agent._valid_action_mask(observation)[0].tolist()
    assert mask[0] is True
    assert mask[1] is True
    assert mask[4] is False
    assert mask[5] is False
    assert mask[6] is False


def test_dqn_can_disable_action_mask_for_ablation():
    agent = SignalRainbowDQNAgentV2(
        obs_dim=17,
        action_dim=7,
        defender_initial_budget=28.0,
        defender_inspect_cost=1.0,
        defender_respond_cost_by_zone={"outer": 4.0, "lane": 5.0, "core": 6.0},
        defender_action_floor=-6.0,
        use_action_mask=False,
    )
    observation = torch.zeros((1, 17), dtype=torch.float32)
    observation[0, 0] = -2.5 / 28.0
    mask = agent._valid_action_mask(observation)[0].tolist()
    assert mask == [True] * 7


def test_attacker_strategy_allows_feasible_overdraft_actions():
    strategy = SignalCheatGreedyAttackerV2(
        allow_cheat=True,
        cheat_cost=1.5,
        takeover_cost_by_zone={"outer": 6.0, "lane": 7.5, "core": 9.0},
        action_floor=-6.0,
        takeover_trigger_belief=0.42,
        exploit_false_response=True,
    )
    action = strategy.select_action(
        {
            "mode": "cheat",
            "attacker_budget_remaining": 0.0,
            "mu_breach": 0.0,
            "current_signal": "null",
            "focus_zone": "none",
            "zone_beliefs": {"outer": 0.1, "lane": 0.6, "core": 0.3},
            "prev_false_response_flag": 0,
            "last_deception_zone": None,
        }
    )
    assert action != 0


def test_baseline_falls_back_from_unaffordable_respond_to_inspect():
    defender = BeliefThresholdDefenderV2(inspect_threshold=0.2, respond_threshold=0.55)
    defender.configure_budget_constraints(
        defender_initial_budget=28.0,
        defender_inspect_cost=1.0,
        defender_respond_cost_by_zone={"outer": 4.0, "lane": 5.0, "core": 6.0},
        defender_action_floor=-6.0,
    )
    observation = [0.0] * 17
    observation[0] = -2.5 / 28.0
    observation[2] = 0.8
    observation[7] = 1.0
    action = defender.select_action(observation, training=False)
    assert action == 1


def test_termination_info_reports_budget_collapse_metadata(deterministic_config):
    config = copy.deepcopy(deterministic_config)
    config["resources"]["defender_base_income_per_step"] = 0.0
    config["resources"]["defender_control_bonus_per_step"] = 0.0
    config["resources"]["defender_guarantee_line"] = -2.0
    config["resources"]["guarantee_breach_patience"] = 1
    env = MaritimeCheatAttentionEnv(config)
    _, _ = env.reset(seed=7)
    env.defender_budget = -3.0
    _, _, terminated, truncated, info = env.step((0, 0))
    assert terminated is True
    assert truncated is False
    assert info["defender_budget_collapse"] is True
    assert info["termination_reason"] == "defender_resource_collapse"
    assert "attacker_below_guarantee_streak" in info
    assert "defender_below_guarantee_streak" in info


def test_environment_reports_action_costs_and_cumulative_defender_spend(deterministic_config):
    env = MaritimeCheatAttentionEnv(deterministic_config)
    _, _ = env.reset(seed=7)
    _, _, _, _, info = env.step((0, 4))
    assert info["attacker_action_cost"] == pytest.approx(0.0)
    assert info["defender_action_cost"] == pytest.approx(
        deterministic_config["costs_and_rewards"]["defender_respond_cost_by_zone"]["outer"]
    )
    assert info["episode_metrics_snapshot"]["total_defender_action_cost"] == pytest.approx(
        deterministic_config["costs_and_rewards"]["defender_respond_cost_by_zone"]["outer"]
    )
