"""Heuristic attacker for the V2 signal-region environment."""

from __future__ import annotations

from typing import Dict

from gym_flipit.envs.maritime_cheat_attention_env import (
    NULL_SIGNAL,
    ZONES,
    ZONE_TO_ATTACKER_CHEAT_ACTION,
    ZONE_TO_ATTACKER_TAKEOVER_ACTION,
)


class SignalCheatGreedyAttackerV2:
    """Attacker that uses false signals to drag defender attention off the weakest zone."""

    def __init__(
        self,
        allow_cheat: bool = True,
        cheat_cost: float = 8.0,
        takeover_cost_by_zone: Dict[str, float] | None = None,
        takeover_trigger_belief: float = 0.42,
        exploit_false_response: bool = True,
    ):
        self.allow_cheat = bool(allow_cheat)
        self.cheat_cost = float(cheat_cost)
        self.takeover_cost_by_zone = takeover_cost_by_zone or {zone: 18.0 for zone in ZONES}
        self.takeover_trigger_belief = float(takeover_trigger_belief)
        self.exploit_false_response = bool(exploit_false_response)

    def select_action(self, public_state: Dict[str, object]) -> int:
        mode = str(public_state.get("mode", "cheat"))
        allow_cheat = self.allow_cheat and mode == "cheat"
        budget = float(public_state.get("attacker_budget_remaining", 0.0))
        mu_breach = float(public_state.get("mu_breach", 0.0))
        signal = str(public_state.get("current_signal", NULL_SIGNAL))
        focus_zone = public_state.get("focus_zone")
        zone_beliefs = public_state.get("zone_beliefs", {})
        if not zone_beliefs:
            zone_beliefs = {zone: 1.0 / len(ZONES) for zone in ZONES}

        target_zone = min(ZONES, key=lambda zone: float(zone_beliefs.get(zone, 0.0)))
        distraction_zone = max(ZONES, key=lambda zone: float(zone_beliefs.get(zone, 0.0)))
        if distraction_zone == target_zone:
            distraction_zone = "lane" if target_zone != "lane" else "outer"

        target_cost = float(self.takeover_cost_by_zone[target_zone])
        if budget < min(target_cost, self.cheat_cost if allow_cheat else target_cost):
            return 0

        prior_false_response = int(public_state.get("prev_false_response_flag", 0))
        last_deception_zone = public_state.get("last_deception_zone")
        target_exposed = signal == target_zone or focus_zone == target_zone

        if allow_cheat and budget >= self.cheat_cost:
            if not prior_false_response and not target_exposed and signal != distraction_zone:
                return ZONE_TO_ATTACKER_CHEAT_ACTION[distraction_zone]
            if last_deception_zone == distraction_zone and focus_zone == distraction_zone and budget >= target_cost:
                return ZONE_TO_ATTACKER_TAKEOVER_ACTION[target_zone]

        if budget >= target_cost:
            if signal == distraction_zone and self.exploit_false_response:
                return ZONE_TO_ATTACKER_TAKEOVER_ACTION[target_zone]
            if prior_false_response:
                return ZONE_TO_ATTACKER_TAKEOVER_ACTION[target_zone]
            if mu_breach <= self.takeover_trigger_belief and not target_exposed:
                return ZONE_TO_ATTACKER_TAKEOVER_ACTION[target_zone]

        if budget >= target_cost and signal == NULL_SIGNAL:
            return ZONE_TO_ATTACKER_TAKEOVER_ACTION[target_zone]

        return 0
