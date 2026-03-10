"""Signal-region Maritime Cheat-FlipIt environment."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

ZONES: Tuple[str, str, str] = ("outer", "lane", "core")
NULL_SIGNAL = "null"
NO_INSPECT = "none"

ATTACKER_ACTION_LABELS: Dict[int, str] = {
    0: "wait",
    1: "cheat_outer",
    2: "cheat_lane",
    3: "cheat_core",
    4: "takeover_outer",
    5: "takeover_lane",
    6: "takeover_core",
}

DEFENDER_ACTION_LABELS: Dict[int, str] = {
    0: "hold",
    1: "inspect_outer",
    2: "inspect_lane",
    3: "inspect_core",
    4: "respond_outer",
    5: "respond_lane",
    6: "respond_core",
}

ZONE_TO_ATTACKER_CHEAT_ACTION = {zone: index + 1 for index, zone in enumerate(ZONES)}
ZONE_TO_ATTACKER_TAKEOVER_ACTION = {zone: index + 4 for index, zone in enumerate(ZONES)}
ZONE_TO_DEFENDER_INSPECT_ACTION = {zone: index + 1 for index, zone in enumerate(ZONES)}
ZONE_TO_DEFENDER_RESPOND_ACTION = {zone: index + 4 for index, zone in enumerate(ZONES)}

OBS_INDEX: Dict[str, int] = {
    "defender_budget_ratio": 0,
    "attacker_budget_ratio": 1,
    "mu_breach": 2,
    "nu_outer": 3,
    "nu_lane": 4,
    "nu_core": 5,
    "signal_is_null": 6,
    "signal_is_outer": 7,
    "signal_is_lane": 8,
    "signal_is_core": 9,
    "last_inspect_none": 10,
    "last_inspect_outer": 11,
    "last_inspect_lane": 12,
    "last_inspect_core": 13,
    "last_inspect_result": 14,
    "prev_false_response_flag": 15,
    "prev_missed_response_flag": 16,
}


def _clamp_probability(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def attacker_action_label(action_id: int) -> str:
    return ATTACKER_ACTION_LABELS.get(int(action_id), "wait")


def defender_action_label(action_id: int) -> str:
    return DEFENDER_ACTION_LABELS.get(int(action_id), "hold")


def attacker_action_zone(action_id: int) -> Optional[str]:
    action_id = int(action_id)
    if action_id in (1, 2, 3):
        return ZONES[action_id - 1]
    if action_id in (4, 5, 6):
        return ZONES[action_id - 4]
    return None


def defender_action_zone(action_id: int) -> Optional[str]:
    action_id = int(action_id)
    if action_id in (1, 2, 3):
        return ZONES[action_id - 1]
    if action_id in (4, 5, 6):
        return ZONES[action_id - 4]
    return None


def is_cheat_action(action_id: int) -> bool:
    return int(action_id) in (1, 2, 3)


def is_takeover_action(action_id: int) -> bool:
    return int(action_id) in (4, 5, 6)


def is_inspect_action(action_id: int) -> bool:
    return int(action_id) in (1, 2, 3)


def is_respond_action(action_id: int) -> bool:
    return int(action_id) in (4, 5, 6)


class MaritimeCheatAttentionEnv(gym.Env):
    """Parallel V2 environment with signal manipulation and limited attention."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_source: Any):
        super().__init__()
        self.config_source = config_source
        self.config = self._load_config(config_source)

        self.mode = self.config["signal_model"]["mode"]
        self.zone_names = tuple(self.config.get("zones", ZONES))
        if tuple(self.zone_names) != ZONES:
            raise ValueError("V2 environment requires zones ordered as outer/lane/core")

        self.max_steps = int(self.config["environment"]["max_steps"])
        self.default_seed = int(self.config["experiment"].get("random_seed", 42))
        self.rng = np.random.RandomState(self.default_seed)

        self.attacker_initial_budget = float(self.config["resources"]["attacker_initial_budget"])
        self.defender_initial_budget = float(self.config["resources"]["defender_initial_budget"])
        self.attacker_base_income_per_step = float(self.config["resources"]["attacker_base_income_per_step"])
        self.defender_base_income_per_step = float(self.config["resources"]["defender_base_income_per_step"])
        self.attacker_control_bonus_per_step = float(self.config["resources"]["attacker_control_bonus_per_step"])
        self.defender_control_bonus_per_step = float(self.config["resources"]["defender_control_bonus_per_step"])
        self.attacker_action_floor = float(self.config["resources"]["attacker_action_floor"])
        self.defender_action_floor = float(self.config["resources"]["defender_action_floor"])
        self.attacker_guarantee_line = float(self.config["resources"]["attacker_guarantee_line"])
        self.defender_guarantee_line = float(self.config["resources"]["defender_guarantee_line"])
        self.guarantee_breach_patience = int(self.config["resources"]["guarantee_breach_patience"])

        self.attacker_cheat_cost = float(self.config["costs_and_rewards"]["attacker_cheat_cost"])
        self.attacker_takeover_cost_by_zone = {
            zone: float(value)
            for zone, value in self.config["costs_and_rewards"]["attacker_takeover_cost_by_zone"].items()
        }
        self.defender_inspect_cost = float(self.config["costs_and_rewards"]["defender_inspect_cost"])
        self.defender_respond_cost_by_zone = {
            zone: float(value)
            for zone, value in self.config["costs_and_rewards"]["defender_respond_cost_by_zone"].items()
        }
        self.defender_control_reward = float(self.config["costs_and_rewards"]["defender_control_reward"])
        self.attacker_control_reward = float(self.config["costs_and_rewards"]["attacker_control_reward"])
        self.false_response_penalty = float(self.config["costs_and_rewards"]["false_response_penalty"])
        self.missed_threat_penalty = float(self.config["costs_and_rewards"]["missed_threat_penalty"])
        self.attacker_resource_collapse_penalty = float(
            self.config["costs_and_rewards"]["attacker_resource_collapse_penalty"]
        )
        self.defender_resource_collapse_penalty = float(
            self.config["costs_and_rewards"]["defender_resource_collapse_penalty"]
        )
        self.attacker_resource_collapse_bonus = float(
            self.config["costs_and_rewards"]["attacker_resource_collapse_bonus"]
        )
        self.defender_resource_collapse_bonus = float(
            self.config["costs_and_rewards"]["defender_resource_collapse_bonus"]
        )

        self.cheat_emit_prob = _clamp_probability(self.config["signal_model"]["cheat_emit_prob"])
        self.takeover_detect_prob_by_zone = {
            zone: _clamp_probability(value)
            for zone, value in self.config["signal_model"]["takeover_detect_prob_by_zone"].items()
        }
        self.background_noise_by_zone = {
            zone: _clamp_probability(value)
            for zone, value in self.config["signal_model"]["background_noise_by_zone"].items()
        }
        self.inspection_accuracy_by_zone = {
            zone: _clamp_probability(value)
            for zone, value in self.config["signal_model"]["inspection_accuracy_by_zone"].items()
        }

        self.base_takeover_success_by_zone = {
            zone: _clamp_probability(value)
            for zone, value in self.config["transition_model"]["base_takeover_success_by_zone"].items()
        }
        self.respond_block_multiplier_same_zone = {
            zone: float(value)
            for zone, value in self.config["transition_model"]["respond_block_multiplier_same_zone"].items()
        }
        self.inspect_block_multiplier_same_zone = {
            zone: float(value)
            for zone, value in self.config["transition_model"]["inspect_block_multiplier_same_zone"].items()
        }
        self.off_zone_vulnerability_multiplier = float(
            self.config["transition_model"]["off_zone_vulnerability_multiplier"]
        )
        self.recapture_success_by_zone = {
            zone: _clamp_probability(value)
            for zone, value in self.config["transition_model"]["recapture_success_by_zone"].items()
        }

        self.belief_signal_sensitivity = float(self.config["policies"].get("belief_signal_sensitivity", 0.65))
        self.belief_decay = float(self.config["policies"].get("belief_decay", 0.92))
        self.breach_prior = float(self.config["environment"].get("breach_prior", 0.08))
        self.uniform_zone_prior = np.full(len(ZONES), 1.0 / len(ZONES), dtype=np.float32)
        drl_config = self.config.get("drl", {})
        self.training_control_bonus = float(drl_config.get("training_control_bonus", 2.0))
        self.training_attacker_control_penalty = float(drl_config.get("training_attacker_control_penalty", 2.0))
        self.training_positive_inspection_bonus = float(drl_config.get("training_positive_inspection_bonus", 0.8))
        self.training_successful_response_bonus = float(drl_config.get("training_successful_response_bonus", 2.5))
        self.training_false_response_penalty = float(drl_config.get("training_false_response_penalty", 1.5))
        self.training_missed_threat_penalty = float(drl_config.get("training_missed_threat_penalty", 2.5))
        self.training_inspect_cost_weight = float(drl_config.get("training_inspect_cost_weight", 0.25))
        self.training_respond_cost_weight = float(drl_config.get("training_respond_cost_weight", 0.35))
        self.training_resource_collapse_penalty = float(drl_config.get("training_resource_collapse_penalty", 8.0))
        self.training_opponent_resource_collapse_bonus = float(
            drl_config.get("training_opponent_resource_collapse_bonus", 4.0)
        )

        self.action_space = spaces.Discrete(7)
        self.attacker_action_space = spaces.Discrete(7)
        observation_low = np.array(
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            dtype=np.float32,
        )
        observation_high = np.array(
            [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            shape=(17,),
            dtype=np.float32,
        )

        self.tick = 0
        self.reset()

    @staticmethod
    def _load_config(config_source: Any) -> Dict[str, Any]:
        if isinstance(config_source, dict):
            return copy.deepcopy(config_source)
        with open(config_source, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = self.default_seed
        self.default_seed = int(seed)
        self.rng = np.random.RandomState(self.default_seed)
        return [self.default_seed]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.tick = 0
        self.true_controller = "defender"
        self.true_zone = "outer"
        self.current_signal = NULL_SIGNAL
        self.signal_source_action = "none"
        self.focus_zone = NO_INSPECT
        self.active_threat_zone = None
        self.last_deception_zone = None
        self.attacker_budget = self.attacker_initial_budget
        self.defender_budget = self.defender_initial_budget
        self.attacker_below_guarantee_streak = 0
        self.defender_below_guarantee_streak = 0

        self.mu_breach = self.breach_prior
        self.zone_beliefs = self.uniform_zone_prior.copy()
        self.last_inspect_zone = NO_INSPECT
        self.last_inspect_result = 0
        self.prev_false_response_flag = 0
        self.prev_missed_response_flag = 0

        self.metrics = {
            "attacker_control_steps": 0,
            "defender_control_steps": 0,
            "cheat_actions": 0,
            "takeover_attempts": 0,
            "takeover_successes": 0,
            "response_successes": 0,
            "inspect_actions": 0,
            "positive_inspections": 0,
            "false_responses": 0,
            "missed_responses": 0,
            "attacker_below_guarantee_steps": 0,
            "defender_below_guarantee_steps": 0,
            "attacker_budget_collapse": False,
            "defender_budget_collapse": False,
            "signal_counts": {zone: 0 for zone in ZONES},
            "respond_counts": {zone: 0 for zone in ZONES},
            "inspect_counts": {zone: 0 for zone in ZONES},
        }
        return self._get_observation(), self._reset_info()

    def step(self, action: Tuple[int, int]):
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("V2 environment expects action=(attacker_action_id, defender_action_id)")

        self.tick += 1

        requested_attacker_action = int(action[0])
        requested_defender_action = int(action[1])
        attacker_action = self._normalize_attacker_action(requested_attacker_action)
        defender_action = self._normalize_defender_action(requested_defender_action)

        attacker_cost = self._attacker_action_cost(attacker_action)
        defender_cost = self._defender_action_cost(defender_action)

        if self.attacker_budget - attacker_cost < self.attacker_action_floor:
            attacker_action = 0
            attacker_cost = 0.0
        if self.defender_budget - defender_cost < self.defender_action_floor:
            defender_action = 0
            defender_cost = 0.0

        attacker_zone = attacker_action_zone(attacker_action)
        defender_zone = defender_action_zone(defender_action)
        self.focus_zone = defender_zone or NO_INSPECT
        self.last_inspect_zone = NO_INSPECT
        self.last_inspect_result = 0
        self.last_deception_zone = attacker_zone if is_cheat_action(attacker_action) else None

        current_signal, signal_source = self._generate_signal(attacker_action, attacker_zone, self.focus_zone)
        self.current_signal = current_signal
        self.signal_source_action = signal_source

        if current_signal in ZONES:
            self.metrics["signal_counts"][current_signal] += 1

        takeover_attempted = is_takeover_action(attacker_action)
        takeover_success = False
        response_success = False
        false_response = False
        missed_response = False
        inspection_result = "none"

        if is_cheat_action(attacker_action):
            self.metrics["cheat_actions"] += 1
        if takeover_attempted:
            self.metrics["takeover_attempts"] += 1
            self.active_threat_zone = attacker_zone
        elif self.true_controller == "attacker":
            self.active_threat_zone = self.true_zone
        else:
            self.active_threat_zone = None

        if takeover_attempted:
            takeover_success = self._resolve_takeover(attacker_zone, defender_action, defender_zone)
            if takeover_success:
                self.true_controller = "attacker"
                self.true_zone = attacker_zone or self.true_zone
                self.metrics["takeover_successes"] += 1
            elif self.true_controller != "attacker":
                self.active_threat_zone = None

        if is_inspect_action(defender_action):
            self.metrics["inspect_actions"] += 1
            self.metrics["inspect_counts"][defender_zone] += 1
            inspection_result = self._resolve_inspection(defender_zone)

        if is_respond_action(defender_action):
            self.metrics["respond_counts"][defender_zone] += 1
            false_response = not self._zone_has_real_threat(defender_zone, takeover_attempted, attacker_zone)
            if false_response:
                self.metrics["false_responses"] += 1
            response_success = self._resolve_response(defender_zone)
            if response_success:
                self.metrics["response_successes"] += 1
                self.active_threat_zone = None

        if takeover_attempted and takeover_success and not self._defender_covers_zone(defender_action, attacker_zone):
            missed_response = True
            self.metrics["missed_responses"] += 1

        self.prev_false_response_flag = int(false_response)
        self.prev_missed_response_flag = int(missed_response)

        self._update_beliefs_from_signal(current_signal)
        if inspection_result != "none":
            self._update_beliefs_from_inspection(defender_zone, inspection_result == "positive")

        attacker_base_income_applied = self.attacker_base_income_per_step
        defender_base_income_applied = self.defender_base_income_per_step
        attacker_control_bonus_applied = self.attacker_control_bonus_per_step if self.true_controller == "attacker" else 0.0
        defender_control_bonus_applied = self.defender_control_bonus_per_step if self.true_controller == "defender" else 0.0

        self.attacker_budget = (
            self.attacker_budget
            - attacker_cost
            + attacker_base_income_applied
            + attacker_control_bonus_applied
        )
        self.defender_budget = (
            self.defender_budget
            - defender_cost
            + defender_base_income_applied
            + defender_control_bonus_applied
        )

        if self.true_controller == "attacker":
            self.metrics["attacker_control_steps"] += 1
        else:
            self.metrics["defender_control_steps"] += 1

        attacker_reward = self.attacker_control_reward if self.true_controller == "attacker" else 0.0
        attacker_reward -= attacker_cost

        defender_reward = self.defender_control_reward if self.true_controller == "defender" else 0.0
        defender_reward -= defender_cost
        if false_response:
            defender_reward -= self.false_response_penalty
        if missed_response:
            defender_reward -= self.missed_threat_penalty

        training_reward = 0.0
        if self.true_controller == "defender":
            training_reward += self.training_control_bonus
        else:
            training_reward -= self.training_attacker_control_penalty
        if is_inspect_action(defender_action):
            training_reward -= defender_cost * self.training_inspect_cost_weight
        if is_respond_action(defender_action):
            training_reward -= defender_cost * self.training_respond_cost_weight
        if inspection_result == "positive":
            training_reward += self.training_positive_inspection_bonus
        if response_success:
            training_reward += self.training_successful_response_bonus
        if false_response:
            training_reward -= self.training_false_response_penalty
        if missed_response:
            training_reward -= self.training_missed_threat_penalty

        attacker_budget_collapse = self._update_guarantee_streak("attacker")
        defender_budget_collapse = self._update_guarantee_streak("defender")

        terminated = False
        truncated = False
        winner = None
        termination_reason = None

        if attacker_budget_collapse or defender_budget_collapse:
            terminated = True
            self.metrics["attacker_budget_collapse"] = attacker_budget_collapse
            self.metrics["defender_budget_collapse"] = defender_budget_collapse
            if attacker_budget_collapse and defender_budget_collapse:
                termination_reason = "both_resource_collapse"
                winner = self._winner_from_control_steps()
                attacker_reward -= self.attacker_resource_collapse_penalty
                defender_reward -= self.defender_resource_collapse_penalty
                if winner == "attacker":
                    attacker_reward += self.attacker_resource_collapse_bonus
                    training_reward -= self.training_resource_collapse_penalty
                elif winner == "defender":
                    defender_reward += self.defender_resource_collapse_bonus
                    training_reward += self.training_opponent_resource_collapse_bonus
            elif attacker_budget_collapse:
                termination_reason = "attacker_resource_collapse"
                winner = "defender"
                attacker_reward -= self.attacker_resource_collapse_penalty
                defender_reward += self.defender_resource_collapse_bonus
                training_reward += self.training_opponent_resource_collapse_bonus
            else:
                termination_reason = "defender_resource_collapse"
                winner = "attacker"
                defender_reward -= self.defender_resource_collapse_penalty
                attacker_reward += self.attacker_resource_collapse_bonus
                training_reward -= self.training_resource_collapse_penalty
        else:
            truncated = self.tick >= self.max_steps
            if truncated:
                termination_reason = "max_steps"
                winner = self._winner_from_control_steps()

        info = {
            "mode": self.mode,
            "true_controller": self.true_controller,
            "current_signal": self.current_signal,
            "signal_source_action": self.signal_source_action,
            "true_zone": self.true_zone,
            "mu_breach": float(self.mu_breach),
            "zone_beliefs": self._zone_belief_dict(),
            "attacker_budget_remaining": float(self.attacker_budget),
            "defender_budget_remaining": float(self.defender_budget),
            "attacker_base_income_applied": float(attacker_base_income_applied),
            "defender_base_income_applied": float(defender_base_income_applied),
            "attacker_control_bonus_applied": float(attacker_control_bonus_applied),
            "defender_control_bonus_applied": float(defender_control_bonus_applied),
            "attacker_below_guarantee_streak": int(self.attacker_below_guarantee_streak),
            "defender_below_guarantee_streak": int(self.defender_below_guarantee_streak),
            "attacker_budget_collapse": bool(attacker_budget_collapse),
            "defender_budget_collapse": bool(defender_budget_collapse),
            "focus_zone": self.focus_zone,
            "active_threat_zone": self.active_threat_zone,
            "last_deception_zone": self.last_deception_zone,
            "attacker_action_label": attacker_action_label(attacker_action),
            "defender_action_label": defender_action_label(defender_action),
            "inspection_result": inspection_result,
            "false_response": bool(false_response),
            "missed_response": bool(missed_response),
            "takeover_attempted": bool(takeover_attempted),
            "takeover_success": bool(takeover_success),
            "response_success": bool(response_success),
            "episode_metrics_snapshot": self._episode_metrics_snapshot(),
            "attacker_reward": float(attacker_reward),
            "defender_reward": float(defender_reward),
            "training_reward": float(training_reward),
            "attacker_action_id": attacker_action,
            "defender_action_id": defender_action,
            "tick": self.tick,
            "winner": winner,
            "termination_reason": termination_reason,
        }
        return self._get_observation(), float(training_reward), terminated, truncated, info

    def render(self):
        print(self.get_public_state())

    def close(self):
        return None

    def get_public_state(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "mode": self.mode,
            "current_signal": self.current_signal,
            "mu_breach": float(self.mu_breach),
            "zone_beliefs": self._zone_belief_dict(),
            "attacker_budget_remaining": float(self.attacker_budget),
            "defender_budget_remaining": float(self.defender_budget),
            "attacker_below_guarantee_streak": int(self.attacker_below_guarantee_streak),
            "defender_below_guarantee_streak": int(self.defender_below_guarantee_streak),
            "focus_zone": self.focus_zone,
            "active_threat_zone": self.active_threat_zone,
            "last_deception_zone": self.last_deception_zone,
            "last_inspect_zone": self.last_inspect_zone,
            "last_inspect_result": int(self.last_inspect_result),
            "prev_false_response_flag": int(self.prev_false_response_flag),
            "prev_missed_response_flag": int(self.prev_missed_response_flag),
        }

    def _reset_info(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "true_controller": self.true_controller,
            "current_signal": self.current_signal,
            "true_zone": self.true_zone,
            "mu_breach": float(self.mu_breach),
            "zone_beliefs": self._zone_belief_dict(),
            "attacker_budget_remaining": float(self.attacker_budget),
            "defender_budget_remaining": float(self.defender_budget),
            "focus_zone": self.focus_zone,
            "active_threat_zone": self.active_threat_zone,
            "last_deception_zone": self.last_deception_zone,
        }

    def _normalize_attacker_action(self, action_id: int) -> int:
        action_id = int(action_id)
        if action_id not in ATTACKER_ACTION_LABELS:
            return 0
        if self.mode == "flipit" and is_cheat_action(action_id):
            return 0
        return action_id

    def _normalize_defender_action(self, action_id: int) -> int:
        action_id = int(action_id)
        if action_id not in DEFENDER_ACTION_LABELS:
            return 0
        return action_id

    def _attacker_action_cost(self, action_id: int) -> float:
        if is_cheat_action(action_id):
            return self.attacker_cheat_cost
        if is_takeover_action(action_id):
            return self.attacker_takeover_cost_by_zone[attacker_action_zone(action_id)]
        return 0.0

    def _defender_action_cost(self, action_id: int) -> float:
        if is_inspect_action(action_id):
            return self.defender_inspect_cost
        if is_respond_action(action_id):
            return self.defender_respond_cost_by_zone[defender_action_zone(action_id)]
        return 0.0

    def _sample_background_noise(self) -> str:
        zone_order = list(ZONES)
        self.rng.shuffle(zone_order)
        for zone in zone_order:
            if self.rng.rand() < self.background_noise_by_zone[zone]:
                return zone
        return NULL_SIGNAL

    def _generate_signal(
        self,
        attacker_action: int,
        attacker_zone: Optional[str],
        focus_zone: str,
    ) -> Tuple[str, str]:
        if is_cheat_action(attacker_action) and attacker_zone:
            if self.rng.rand() < self.cheat_emit_prob:
                return attacker_zone, attacker_action_label(attacker_action)
            return NULL_SIGNAL, attacker_action_label(attacker_action)
        if is_takeover_action(attacker_action) and attacker_zone:
            detect_prob = self.takeover_detect_prob_by_zone[attacker_zone]
            if focus_zone == attacker_zone:
                detect_prob = _clamp_probability(detect_prob + 0.2 * (1.0 - detect_prob))
            elif focus_zone != NO_INSPECT:
                detect_prob = _clamp_probability(detect_prob * 0.85)
            if self.rng.rand() < detect_prob:
                return attacker_zone, attacker_action_label(attacker_action)
            return self._sample_background_noise(), attacker_action_label(attacker_action)
        return self._sample_background_noise(), "background_noise"

    def _resolve_takeover(self, zone: Optional[str], defender_action: int, defender_zone: Optional[str]) -> bool:
        if zone is None:
            return False
        success_prob = self.base_takeover_success_by_zone[zone]
        if is_respond_action(defender_action) and defender_zone == zone:
            success_prob *= self.respond_block_multiplier_same_zone[zone]
        elif is_inspect_action(defender_action) and defender_zone == zone:
            success_prob *= self.inspect_block_multiplier_same_zone[zone]
        elif defender_action != 0:
            success_prob *= self.off_zone_vulnerability_multiplier
        success_prob = _clamp_probability(success_prob)
        return bool(self.rng.rand() < success_prob)

    def _resolve_inspection(self, zone: Optional[str]) -> str:
        if zone is None:
            return "none"
        accuracy = self.inspection_accuracy_by_zone[zone]
        real_threat = self.true_controller == "attacker" and self.true_zone == zone
        positive = self.rng.rand() < accuracy if real_threat else self.rng.rand() >= accuracy
        self.last_inspect_zone = zone
        self.last_inspect_result = 1 if positive else -1
        if positive:
            self.metrics["positive_inspections"] += 1
        return "positive" if positive else "negative"

    def _resolve_response(self, zone: Optional[str]) -> bool:
        if zone is None:
            return False
        if self.true_controller != "attacker" or self.true_zone != zone:
            return False
        success_prob = self.recapture_success_by_zone[zone]
        success = bool(self.rng.rand() < success_prob)
        if success:
            self.true_controller = "defender"
        return success

    def _zone_has_real_threat(
        self,
        zone: Optional[str],
        takeover_attempted: bool,
        takeover_zone: Optional[str],
    ) -> bool:
        if zone is None:
            return False
        if takeover_attempted and takeover_zone == zone:
            return True
        return self.true_controller == "attacker" and self.true_zone == zone

    def _defender_covers_zone(self, defender_action: int, target_zone: Optional[str]) -> bool:
        if target_zone is None:
            return False
        if is_inspect_action(defender_action) or is_respond_action(defender_action):
            return defender_action_zone(defender_action) == target_zone
        return False

    def _update_beliefs_from_signal(self, signal: str):
        prior_zone_mass = np.array(self.zone_beliefs, dtype=np.float64) * float(self.mu_breach)
        prior_safe_mass = max(1e-6, 1.0 - float(self.mu_breach))

        posterior_zone_mass = np.zeros_like(prior_zone_mass)
        for zone_index, zone in enumerate(ZONES):
            detect_prob = self.takeover_detect_prob_by_zone[zone]
            if signal == NULL_SIGNAL:
                breach_likelihood = max(1e-3, 1.0 - detect_prob)
                safe_likelihood = max(1e-3, 1.0 - (self.cheat_emit_prob * 0.5))
            elif signal == zone:
                breach_likelihood = max(
                    1e-3,
                    detect_prob * (1.0 - 0.25 * self.background_noise_by_zone[zone]),
                )
                safe_likelihood = max(
                    1e-3,
                    (self.cheat_emit_prob * 0.5) + self.background_noise_by_zone[zone],
                )
            else:
                breach_likelihood = max(1e-3, self.background_noise_by_zone.get(signal, 0.01))
                safe_likelihood = max(
                    1e-3,
                    (self.cheat_emit_prob / len(ZONES)) + self.background_noise_by_zone.get(signal, 0.01),
                )
            posterior_zone_mass[zone_index] = prior_zone_mass[zone_index] * breach_likelihood

        if signal == NULL_SIGNAL:
            safe_likelihood = max(1e-3, 1.0 - (self.cheat_emit_prob * 0.5))
        else:
            safe_likelihood = max(
                1e-3,
                (self.cheat_emit_prob / len(ZONES)) + self.background_noise_by_zone.get(signal, 0.01),
            )
        posterior_safe_mass = prior_safe_mass * safe_likelihood

        total_mass = posterior_zone_mass.sum() + posterior_safe_mass
        self.mu_breach = float(posterior_zone_mass.sum() / total_mass)
        if posterior_zone_mass.sum() <= 1e-8:
            self.zone_beliefs = self.uniform_zone_prior.copy()
        else:
            updated = posterior_zone_mass / posterior_zone_mass.sum()
            self.zone_beliefs = (
                self.belief_signal_sensitivity * updated
                + (1.0 - self.belief_signal_sensitivity) * self.uniform_zone_prior
            ).astype(np.float32)
            self.zone_beliefs = (self.zone_beliefs / self.zone_beliefs.sum()).astype(np.float32)

        if signal == NULL_SIGNAL:
            self.mu_breach = max(self.breach_prior * 0.5, self.mu_breach * self.belief_decay)
            self.zone_beliefs = (
                self.belief_decay * self.zone_beliefs + (1.0 - self.belief_decay) * self.uniform_zone_prior
            ).astype(np.float32)
            self.zone_beliefs = (self.zone_beliefs / self.zone_beliefs.sum()).astype(np.float32)

    def _update_beliefs_from_inspection(self, zone: Optional[str], positive: bool):
        if zone is None:
            return
        zone_index = ZONES.index(zone)
        accuracy = self.inspection_accuracy_by_zone[zone]
        zone_mass = np.array(self.zone_beliefs, dtype=np.float64) * float(self.mu_breach)
        safe_mass = max(1e-6, 1.0 - float(self.mu_breach))

        if positive:
            zone_mass[zone_index] *= max(accuracy, 1e-3)
            safe_mass *= max(1.0 - accuracy, 1e-3)
            for index in range(len(zone_mass)):
                if index != zone_index:
                    zone_mass[index] *= max(1.0 - accuracy, 1e-3)
        else:
            zone_mass[zone_index] *= max(1.0 - accuracy, 1e-3)
            safe_mass *= max(accuracy, 1e-3)

        total_mass = zone_mass.sum() + safe_mass
        self.mu_breach = float(zone_mass.sum() / total_mass)
        if zone_mass.sum() <= 1e-8:
            self.zone_beliefs = self.uniform_zone_prior.copy()
        else:
            self.zone_beliefs = (zone_mass / zone_mass.sum()).astype(np.float32)

        if positive:
            self.mu_breach = max(self.mu_breach, 0.65)
        else:
            self.mu_breach = min(self.mu_breach, 0.45)

    def _zone_belief_dict(self) -> Dict[str, float]:
        return {zone: float(self.zone_beliefs[index]) for index, zone in enumerate(ZONES)}

    def _winner_from_control_steps(self) -> str:
        if self.metrics["attacker_control_steps"] > self.metrics["defender_control_steps"]:
            return "attacker"
        if self.metrics["attacker_control_steps"] < self.metrics["defender_control_steps"]:
            return "defender"
        return "draw"

    def _update_guarantee_streak(self, player: str) -> bool:
        if player == "attacker":
            below = self.attacker_budget < self.attacker_guarantee_line
            if below:
                self.attacker_below_guarantee_streak += 1
                self.metrics["attacker_below_guarantee_steps"] += 1
            else:
                self.attacker_below_guarantee_streak = 0
            return self.attacker_below_guarantee_streak >= self.guarantee_breach_patience

        below = self.defender_budget < self.defender_guarantee_line
        if below:
            self.defender_below_guarantee_streak += 1
            self.metrics["defender_below_guarantee_steps"] += 1
        else:
            self.defender_below_guarantee_streak = 0
        return self.defender_below_guarantee_streak >= self.guarantee_breach_patience

    def _episode_metrics_snapshot(self) -> Dict[str, Any]:
        inspection_precision = (
            self.metrics["positive_inspections"] / self.metrics["inspect_actions"]
            if self.metrics["inspect_actions"] > 0
            else 0.0
        )
        return {
            "tick": self.tick,
            "attacker_control_steps": self.metrics["attacker_control_steps"],
            "defender_control_steps": self.metrics["defender_control_steps"],
            "cheat_actions": self.metrics["cheat_actions"],
            "takeover_attempts": self.metrics["takeover_attempts"],
            "takeover_successes": self.metrics["takeover_successes"],
            "response_successes": self.metrics["response_successes"],
            "inspect_actions": self.metrics["inspect_actions"],
            "positive_inspections": self.metrics["positive_inspections"],
            "false_responses": self.metrics["false_responses"],
            "missed_responses": self.metrics["missed_responses"],
            "attacker_below_guarantee_steps": self.metrics["attacker_below_guarantee_steps"],
            "defender_below_guarantee_steps": self.metrics["defender_below_guarantee_steps"],
            "attacker_budget_collapse": self.metrics["attacker_budget_collapse"],
            "defender_budget_collapse": self.metrics["defender_budget_collapse"],
            "inspection_precision": inspection_precision,
            "signal_counts": copy.deepcopy(self.metrics["signal_counts"]),
            "respond_counts": copy.deepcopy(self.metrics["respond_counts"]),
            "inspect_counts": copy.deepcopy(self.metrics["inspect_counts"]),
        }

    def _get_observation(self) -> np.ndarray:
        signal_features = [1.0 if self.current_signal == NULL_SIGNAL else 0.0]
        signal_features.extend(1.0 if self.current_signal == zone else 0.0 for zone in ZONES)

        inspect_features = [1.0 if self.last_inspect_zone == NO_INSPECT else 0.0]
        inspect_features.extend(1.0 if self.last_inspect_zone == zone else 0.0 for zone in ZONES)

        obs = np.array(
            [
                float(np.clip(self.defender_budget / max(self.defender_initial_budget, 1.0), -1.0, 2.0)),
                float(np.clip(self.attacker_budget / max(self.attacker_initial_budget, 1.0), -1.0, 2.0)),
                float(self.mu_breach),
                float(self.zone_beliefs[0]),
                float(self.zone_beliefs[1]),
                float(self.zone_beliefs[2]),
                *signal_features,
                *inspect_features,
                float(self.last_inspect_result),
                float(self.prev_false_response_flag),
                float(self.prev_missed_response_flag),
            ],
            dtype=np.float32,
        )
        return obs
