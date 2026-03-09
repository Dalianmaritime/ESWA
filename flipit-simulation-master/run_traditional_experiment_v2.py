#!/usr/bin/env python3
"""Run baseline experiments on the V2 signal-region Maritime Cheat-FlipIt environment."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "gym-flipit-master"))

from gym_flipit.envs.maritime_cheat_attention_env import MaritimeCheatAttentionEnv
from signal_v2_utils import (
    compute_final_performance,
    generate_summary_markdown,
    load_config,
    make_step_record,
    save_json,
    seed_everything,
    setup_results_directory,
    summarize_episode,
)
from strategies.belief_threshold_defender_v2 import BeliefThresholdDefenderV2
from strategies.signal_cheat_greedy_attacker_v2 import SignalCheatGreedyAttackerV2


class TraditionalExperimentV2:
    def __init__(
        self,
        config_path: str,
        episodes: int | None = None,
        random_seed: int | None = None,
        experiment_id: str | None = None,
        results_subdir: str | None = None,
    ):
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        if episodes is not None:
            self.config["drl"]["final_evaluation_episodes"] = int(episodes)
        if random_seed is not None:
            self.config["experiment"]["random_seed"] = int(random_seed)
        if experiment_id is not None:
            self.config["experiment"]["experiment_id"] = str(experiment_id)
        if results_subdir is not None:
            self.config["experiment"]["results_subdir"] = str(results_subdir)
        seed_everything(int(self.config["experiment"]["random_seed"]))
        self.results_dir = setup_results_directory(SCRIPT_DIR, self.config_path, self.config)

    def _build_environment(self) -> MaritimeCheatAttentionEnv:
        return MaritimeCheatAttentionEnv(self.config)

    def _build_attacker(self) -> SignalCheatGreedyAttackerV2:
        attacker_cfg = self.config["policies"]["attacker"]
        return SignalCheatGreedyAttackerV2(
            allow_cheat=attacker_cfg["allow_cheat"],
            cheat_cost=self.config["costs_and_rewards"]["attacker_cheat_cost"],
            takeover_cost_by_zone=self.config["costs_and_rewards"]["attacker_takeover_cost_by_zone"],
            takeover_trigger_belief=attacker_cfg["takeover_trigger_belief"],
            exploit_false_response=attacker_cfg["exploit_false_response"],
        )

    def _build_defender(self) -> BeliefThresholdDefenderV2:
        defender_cfg = self.config["policies"]["defender"]
        return BeliefThresholdDefenderV2(
            inspect_threshold=defender_cfg["inspect_threshold"],
            respond_threshold=defender_cfg["respond_threshold"],
        )

    def run(self) -> Dict[str, Any]:
        env = self._build_environment()
        attacker = self._build_attacker()
        defender = self._build_defender()
        episode_count = int(self.config["drl"]["final_evaluation_episodes"])

        details: List[Dict[str, Any]] = []
        for episode_index in range(episode_count):
            observation, _ = env.reset(seed=int(self.config["experiment"]["random_seed"]) + episode_index)
            done = False
            step_index = 0
            attacker_return = 0.0
            defender_return = 0.0
            defender_training_return = 0.0
            step_records: List[Dict[str, Any]] = []
            last_info: Dict[str, Any] = {}

            while not done:
                step_index += 1
                attacker_action = attacker.select_action(env.get_public_state())
                defender_action = defender.select_action(observation, training=False)
                observation, reward, terminated, truncated, info = env.step((attacker_action, defender_action))
                done = bool(terminated or truncated)
                attacker_return += float(info["attacker_reward"])
                defender_return += float(info["defender_reward"])
                defender_training_return += float(info.get("training_reward", reward))
                last_info = info
                step_records.append(make_step_record(step_index, info))

            details.append(
                summarize_episode(
                    episode_index,
                    step_records,
                    last_info,
                    attacker_return,
                    defender_return,
                    defender_training_return=defender_training_return,
                )
            )

        complete_results = {
            "experiment_info": {
                "experiment_id": self.config["experiment"]["experiment_id"],
                "experiment_type": self.config["experiment"]["experiment_type"],
                "mode": self.config["signal_model"]["mode"],
                "scenario_id": self.config["experiment"].get("scenario_id"),
                "method_id": self.config["experiment"].get("method_id"),
                "random_seed": int(self.config["experiment"]["random_seed"]),
                "config_path": self.config_path,
                "timestamp": datetime.now().isoformat(),
                "device": "baseline",
            },
            "metric_conventions": {
                "training_metric": "defender_training_return (shaped reward, recorded for comparison only)",
                "report_metric": f"{self.config['drl'].get('report_metric', 'avg_defender_return')} (raw utility used for reporting)",
            },
            "config_snapshot": self.config,
            "training_history": [],
            "evaluation_history": [],
            "final_performance": compute_final_performance(details),
            "final_evaluation_details": details,
        }

        save_json(self.results_dir / "complete_training_results.json", complete_results)
        save_json(self.results_dir / "episode_details.json", details)
        with open(self.results_dir / "training_summary.md", "w", encoding="utf-8") as handle:
            handle.write(generate_summary_markdown(complete_results))
        return complete_results


def main():
    parser = argparse.ArgumentParser(description="Run V2 baseline experiments.")
    parser.add_argument("config", help="Path to a V2 baseline config")
    parser.add_argument("--episodes", "-n", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--results-subdir", default=None)
    args = parser.parse_args()

    experiment = TraditionalExperimentV2(
        args.config,
        episodes=args.episodes,
        random_seed=args.random_seed,
        experiment_id=args.experiment_id,
        results_subdir=args.results_subdir,
    )
    results = experiment.run()
    performance = results["final_performance"]
    print("=" * 60)
    print("Maritime Cheat-FlipIt V2 baseline experiment completed")
    print("=" * 60)
    print(f"Results directory: {experiment.results_dir}")
    print(f"Attacker success rate: {performance['attacker_success_rate']:.2%}")
    print(f"Defender control rate: {performance['defender_control_rate']:.3f}")
    print(f"Avg defender return: {performance['avg_defender_return']:.3f}")
    print(f"Avg defender training return: {performance['avg_defender_training_return']:.3f}")


if __name__ == "__main__":
    main()
