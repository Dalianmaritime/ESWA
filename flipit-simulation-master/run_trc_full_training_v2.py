#!/usr/bin/env python3
"""Train the V2 defender on the signal-region Maritime Cheat-FlipIt environment."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "gym-flipit-master"))

from gym_flipit.envs.maritime_cheat_attention_env import MaritimeCheatAttentionEnv
from signal_v2_utils import (
    compute_final_performance,
    generate_summary_markdown,
    get_metric_value,
    load_config,
    make_step_record,
    save_json,
    seed_everything,
    setup_results_directory,
    summarize_episode,
)
from strategies.signal_cheat_greedy_attacker_v2 import SignalCheatGreedyAttackerV2
from strategies.signal_rainbow_dqn_v2 import SignalRainbowDQNAgentV2


class SignalDRLTrainingExperimentV2:
    def __init__(
        self,
        config_path: str,
        training_episodes: int | None = None,
        evaluation_episodes: int | None = None,
        final_evaluation_episodes: int | None = None,
        evaluation_frequency: int | None = None,
        random_seed: int | None = None,
        experiment_id: str | None = None,
        results_subdir: str | None = None,
    ):
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(self.config_path)
        if random_seed is not None:
            self.config["experiment"]["random_seed"] = int(random_seed)
        if experiment_id is not None:
            self.config["experiment"]["experiment_id"] = str(experiment_id)
        if results_subdir is not None:
            self.config["experiment"]["results_subdir"] = str(results_subdir)
        seed_everything(int(self.config["experiment"]["random_seed"]))

        drl = self.config["drl"]
        if training_episodes is not None:
            drl["training_episodes"] = int(training_episodes)
        if evaluation_episodes is not None:
            drl["evaluation_episodes"] = int(evaluation_episodes)
        if final_evaluation_episodes is not None:
            drl["final_evaluation_episodes"] = int(final_evaluation_episodes)
        if evaluation_frequency is not None:
            drl["evaluation_frequency"] = int(evaluation_frequency)

        self.results_dir = setup_results_directory(SCRIPT_DIR, self.config_path, self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_metric = "defender_training_return"
        self.report_metric = str(self.config["drl"].get("report_metric", "avg_defender_return"))
        self.checkpoint_metric = str(self.config["drl"].get("checkpoint_selection_metric", "avg_defender_training_return"))

        self.training_history: List[Dict[str, Any]] = []
        self.evaluation_history: List[Dict[str, Any]] = []

    def _build_environment(self) -> MaritimeCheatAttentionEnv:
        return MaritimeCheatAttentionEnv(self.config)

    def _build_attacker(self) -> SignalCheatGreedyAttackerV2:
        attacker_cfg = self.config["policies"]["attacker"]
        return SignalCheatGreedyAttackerV2(
            allow_cheat=attacker_cfg["allow_cheat"],
            cheat_cost=self.config["costs_and_rewards"]["attacker_cheat_cost"],
            takeover_cost_by_zone=self.config["costs_and_rewards"]["attacker_takeover_cost_by_zone"],
            action_floor=self.config["resources"]["attacker_action_floor"],
            takeover_trigger_belief=attacker_cfg["takeover_trigger_belief"],
            exploit_false_response=attacker_cfg["exploit_false_response"],
        )

    def _build_defender(self) -> SignalRainbowDQNAgentV2:
        drl = self.config["drl"]
        return SignalRainbowDQNAgentV2(
            obs_dim=drl["obs_dim"],
            action_dim=drl["action_dim"],
            lr=drl["lr"],
            gamma=drl["gamma"],
            v_min=drl["v_min"],
            v_max=drl["v_max"],
            atom_size=drl["atom_size"],
            memory_size=drl["memory_size"],
            batch_size=drl["batch_size"],
            beta_start=drl["beta_start"],
            beta_frames=drl["beta_frames"],
            target_update_freq=drl["target_update_freq"],
            defender_initial_budget=self.config["resources"]["defender_initial_budget"],
            defender_inspect_cost=self.config["costs_and_rewards"]["defender_inspect_cost"],
            defender_respond_cost_by_zone=self.config["costs_and_rewards"]["defender_respond_cost_by_zone"],
            defender_action_floor=self.config["resources"]["defender_action_floor"],
            device=str(self.device),
        )

    def _run_episode(
        self,
        env: MaritimeCheatAttentionEnv,
        attacker: SignalCheatGreedyAttackerV2,
        defender: SignalRainbowDQNAgentV2,
        episode_index: int,
        training: bool,
        store_trace: bool,
    ) -> Dict[str, Any]:
        observation, _ = env.reset(seed=int(self.config["experiment"]["random_seed"]) + episode_index)
        done = False
        step = 0
        attacker_return = 0.0
        defender_return = 0.0
        defender_training_return = 0.0
        step_records: List[Dict[str, Any]] = []
        step_losses: List[float] = []
        last_info: Dict[str, Any] = {}

        while not done:
            step += 1
            attacker_action = attacker.select_action(env.get_public_state())
            defender_action = defender.select_action(observation, training=training)
            next_observation, reward, terminated, truncated, info = env.step((attacker_action, defender_action))
            done = bool(terminated or truncated)
            attacker_return += float(info["attacker_reward"])
            defender_return += float(info["defender_reward"])
            defender_training_return += float(info.get("training_reward", reward))
            last_info = info

            if store_trace:
                step_records.append(make_step_record(step, info))

            if training:
                defender.store_transition(observation, defender_action, reward, next_observation, done)
                if (
                    len(defender.memory) >= self.config["drl"]["learning_starts"]
                    and step % self.config["drl"]["update_frequency"] == 0
                ):
                    update_stats = defender.update()
                    if "loss" in update_stats:
                        step_losses.append(float(update_stats["loss"]))

            observation = next_observation

        episode_summary = summarize_episode(
            episode_index,
            step_records,
            last_info,
            attacker_return,
            defender_return,
            defender_training_return=defender_training_return,
        )
        episode_summary["mean_training_loss"] = float(np.mean(step_losses)) if step_losses else 0.0
        episode_summary["training"] = bool(training)
        return episode_summary

    def _evaluate(
        self,
        env: MaritimeCheatAttentionEnv,
        attacker: SignalCheatGreedyAttackerV2,
        defender: SignalRainbowDQNAgentV2,
        num_episodes: int,
        seed_offset: int,
        store_trace: bool,
    ) -> List[Dict[str, Any]]:
        results = []
        for eval_index in range(num_episodes):
            results.append(
                self._run_episode(
                    env=env,
                    attacker=attacker,
                    defender=defender,
                    episode_index=seed_offset + eval_index,
                    training=False,
                    store_trace=store_trace,
                )
            )
        return results

    def run(self) -> Dict[str, Any]:
        env = self._build_environment()
        attacker = self._build_attacker()
        defender = self._build_defender()
        best_score = float("-inf")
        best_episode = -1
        best_performance: Dict[str, Any] | None = None
        best_model_dir = self.results_dir / "best_model"
        best_model_dir.mkdir(exist_ok=True)
        best_model_path = best_model_dir / "best_defender.pth"

        train_episodes = int(self.config["drl"]["training_episodes"])
        eval_frequency = max(1, int(self.config["drl"]["evaluation_frequency"]))
        eval_episodes = int(self.config["drl"]["evaluation_episodes"])

        for episode_index in range(train_episodes):
            training_episode = self._run_episode(
                env=env,
                attacker=attacker,
                defender=defender,
                episode_index=episode_index,
                training=True,
                store_trace=False,
            )
            self.training_history.append(training_episode)

            if episode_index % eval_frequency == 0 or episode_index == train_episodes - 1:
                evaluation_results = self._evaluate(
                    env=env,
                    attacker=attacker,
                    defender=defender,
                    num_episodes=eval_episodes,
                    seed_offset=10000 + episode_index * eval_episodes,
                    store_trace=False,
                )
                performance = compute_final_performance(evaluation_results)
                checkpoint_score = get_metric_value(performance, self.checkpoint_metric)
                self.evaluation_history.append(
                    {
                        "episode": episode_index,
                        "performance": performance,
                        "checkpoint_metric": self.checkpoint_metric,
                        "checkpoint_score": checkpoint_score,
                        "num_episodes": eval_episodes,
                    }
                )
                if checkpoint_score > best_score:
                    best_score = checkpoint_score
                    best_episode = episode_index
                    best_performance = dict(performance)
                    defender.save(str(best_model_path))

        if best_model_path.exists():
            defender.load(str(best_model_path))

        final_evaluation_details = self._evaluate(
            env=env,
            attacker=attacker,
            defender=defender,
            num_episodes=int(self.config["drl"]["final_evaluation_episodes"]),
            seed_offset=200000,
            store_trace=True,
        )
        final_performance = compute_final_performance(final_evaluation_details)

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
                "device": str(self.device),
            },
            "metric_conventions": {
                "training_metric": "defender_training_return (shaped reward used for optimization)",
                "report_metric": f"{self.report_metric} (raw utility used for reporting)",
            },
            "checkpoint_selection": {
                "metric": self.checkpoint_metric,
                "best_score": best_score if best_episode >= 0 else None,
                "best_episode": best_episode if best_episode >= 0 else None,
                "best_performance": best_performance,
                "best_model_path": str(best_model_path) if best_model_path.exists() else None,
            },
            "config_snapshot": self.config,
            "training_history": self.training_history,
            "evaluation_history": self.evaluation_history,
            "final_performance": final_performance,
            "final_evaluation_details": final_evaluation_details,
        }

        save_json(self.results_dir / "complete_training_results.json", complete_results)
        save_json(self.results_dir / "training_history.json", self.training_history)
        save_json(self.results_dir / "evaluation_history.json", self.evaluation_history)
        with open(self.results_dir / "training_summary.md", "w", encoding="utf-8") as handle:
            handle.write(generate_summary_markdown(complete_results))
        return complete_results


def main():
    parser = argparse.ArgumentParser(description="Train V2 DRL defender on the Maritime Cheat-FlipIt model.")
    parser.add_argument("config", help="Path to a V2 config file")
    parser.add_argument("--training-episodes", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--final-evaluation-episodes", type=int, default=None)
    parser.add_argument("--evaluation-frequency", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--results-subdir", default=None)
    args = parser.parse_args()

    experiment = SignalDRLTrainingExperimentV2(
        config_path=args.config,
        training_episodes=args.training_episodes,
        evaluation_episodes=args.evaluation_episodes,
        final_evaluation_episodes=args.final_evaluation_episodes,
        evaluation_frequency=args.evaluation_frequency,
        random_seed=args.random_seed,
        experiment_id=args.experiment_id,
        results_subdir=args.results_subdir,
    )
    results = experiment.run()
    performance = results["final_performance"]
    print("=" * 60)
    print("Maritime Cheat-FlipIt V2 DRL training completed")
    print("=" * 60)
    print(f"Results directory: {experiment.results_dir}")
    print(f"Attacker success rate: {performance['attacker_success_rate']:.2%}")
    print(f"Defender control rate: {performance['defender_control_rate']:.3f}")
    print(f"Attacker control rate: {performance['attacker_control_rate']:.3f}")
    print(f"Avg defender return: {performance['avg_defender_return']:.3f}")
    print(f"Avg defender training return: {performance['avg_defender_training_return']:.3f}")


if __name__ == "__main__":
    main()
