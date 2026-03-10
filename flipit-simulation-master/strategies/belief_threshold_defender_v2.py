"""Belief-driven baseline defender for the V2 signal-region environment."""

from __future__ import annotations

from typing import Sequence

from gym_flipit.envs.maritime_cheat_attention_env import (
    OBS_INDEX,
    ZONES,
    ZONE_TO_DEFENDER_INSPECT_ACTION,
    ZONE_TO_DEFENDER_RESPOND_ACTION,
)


class BeliefThresholdDefenderV2:
    """Respond when breach belief is high, inspect when signals are ambiguous."""

    def __init__(self, inspect_threshold: float = 0.2, respond_threshold: float = 0.55):
        self.inspect_threshold = float(inspect_threshold)
        self.respond_threshold = float(respond_threshold)
        self.defender_initial_budget = 100.0
        self.defender_inspect_cost = 1.0
        self.defender_respond_cost_by_zone = {"outer": 4.0, "lane": 5.0, "core": 6.0}
        self.defender_action_floor = -6.0

    def configure_budget_constraints(
        self,
        defender_initial_budget: float,
        defender_inspect_cost: float,
        defender_respond_cost_by_zone,
        defender_action_floor: float,
    ):
        self.defender_initial_budget = float(defender_initial_budget)
        self.defender_inspect_cost = float(defender_inspect_cost)
        self.defender_respond_cost_by_zone = {
            zone: float(defender_respond_cost_by_zone[zone]) for zone in ZONES
        }
        self.defender_action_floor = float(defender_action_floor)

    def select_action(self, observation: Sequence[float], training: bool = False) -> int:
        del training
        mu_breach = float(observation[OBS_INDEX["mu_breach"]])
        budget = float(observation[OBS_INDEX["defender_budget_ratio"]]) * self.defender_initial_budget
        zone_beliefs = {
            "outer": float(observation[OBS_INDEX["nu_outer"]]),
            "lane": float(observation[OBS_INDEX["nu_lane"]]),
            "core": float(observation[OBS_INDEX["nu_core"]]),
        }
        signal_zone = self._decode_signal(observation)
        last_inspect_zone = self._decode_last_inspect_zone(observation)
        last_inspect_result = float(observation[OBS_INDEX["last_inspect_result"]])

        if last_inspect_zone is not None and last_inspect_result > 0:
            return self._fallback_action(
                ZONE_TO_DEFENDER_RESPOND_ACTION[last_inspect_zone],
                last_inspect_zone,
                budget,
            )

        target_zone = signal_zone or max(ZONES, key=lambda zone: zone_beliefs[zone])

        if mu_breach >= self.respond_threshold:
            return self._fallback_action(
                ZONE_TO_DEFENDER_RESPOND_ACTION[target_zone],
                target_zone,
                budget,
            )

        if signal_zone is not None or mu_breach >= self.inspect_threshold:
            return self._fallback_action(
                ZONE_TO_DEFENDER_INSPECT_ACTION[target_zone],
                target_zone,
                budget,
            )

        return 0

    def _action_is_feasible(self, budget: float, action_id: int) -> bool:
        if action_id == 0:
            return True
        if action_id in ZONE_TO_DEFENDER_INSPECT_ACTION.values():
            return budget - self.defender_inspect_cost >= self.defender_action_floor
        for zone, respond_action in ZONE_TO_DEFENDER_RESPOND_ACTION.items():
            if action_id == respond_action:
                return budget - self.defender_respond_cost_by_zone[zone] >= self.defender_action_floor
        return False

    def _fallback_action(self, preferred_action: int, zone: str, budget: float) -> int:
        if self._action_is_feasible(budget, preferred_action):
            return preferred_action
        inspect_action = ZONE_TO_DEFENDER_INSPECT_ACTION[zone]
        if preferred_action != inspect_action and self._action_is_feasible(budget, inspect_action):
            return inspect_action
        return 0

    @staticmethod
    def _decode_signal(observation: Sequence[float]):
        if observation[OBS_INDEX["signal_is_outer"]] > 0.5:
            return "outer"
        if observation[OBS_INDEX["signal_is_lane"]] > 0.5:
            return "lane"
        if observation[OBS_INDEX["signal_is_core"]] > 0.5:
            return "core"
        return None

    @staticmethod
    def _decode_last_inspect_zone(observation: Sequence[float]):
        if observation[OBS_INDEX["last_inspect_outer"]] > 0.5:
            return "outer"
        if observation[OBS_INDEX["last_inspect_lane"]] > 0.5:
            return "lane"
        if observation[OBS_INDEX["last_inspect_core"]] > 0.5:
            return "core"
        return None
