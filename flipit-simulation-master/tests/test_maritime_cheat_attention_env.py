from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gym-flipit-master"))
sys.path.insert(0, str(ROOT / "flipit-simulation-master"))

from gym_flipit.envs.maritime_cheat_attention_env import MaritimeCheatAttentionEnv
from strategies.signal_cheat_greedy_attacker_v2 import SignalCheatGreedyAttackerV2


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
