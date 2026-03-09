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

    def select_action(self, observation: Sequence[float], training: bool = False) -> int:
        del training
        mu_breach = float(observation[OBS_INDEX["mu_breach"]])
        zone_beliefs = {
            "outer": float(observation[OBS_INDEX["nu_outer"]]),
            "lane": float(observation[OBS_INDEX["nu_lane"]]),
            "core": float(observation[OBS_INDEX["nu_core"]]),
        }
        signal_zone = self._decode_signal(observation)
        last_inspect_zone = self._decode_last_inspect_zone(observation)
        last_inspect_result = float(observation[OBS_INDEX["last_inspect_result"]])

        if last_inspect_zone is not None and last_inspect_result > 0:
            return ZONE_TO_DEFENDER_RESPOND_ACTION[last_inspect_zone]

        target_zone = signal_zone or max(ZONES, key=lambda zone: zone_beliefs[zone])

        if mu_breach >= self.respond_threshold:
            return ZONE_TO_DEFENDER_RESPOND_ACTION[target_zone]

        if signal_zone is not None or mu_breach >= self.inspect_threshold:
            return ZONE_TO_DEFENDER_INSPECT_ACTION[target_zone]

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
