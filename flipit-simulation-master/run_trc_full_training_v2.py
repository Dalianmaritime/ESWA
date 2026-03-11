#!/usr/bin/env python3
"""Train the V2 defender on the signal-region Maritime Cheat-FlipIt environment."""

from __future__ import annotations

import argparse
import time
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
    ablate_signal_features,
    build_experiment_tags,
    compute_checkpoint_selection_metrics,
    compute_final_performance,
    confirmation_candidate_sort_key,
    constrained_candidate_improves,
    generate_summary_markdown,
    get_metric_value,
    get_variant_controls,
    load_baseline_reference_targets,
    load_config,
    make_step_record,
    routine_candidate_sort_key,
    save_json,
    seed_everything,
    setup_results_directory,
    should_trigger_early_stopping,
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
        routine_eval_episodes: int | None = None,
        confirmation_eval_episodes: int | None = None,
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
        self.variant_controls = get_variant_controls(self.config)

        drl = self.config["drl"]
        if training_episodes is not None:
            drl["training_episodes"] = int(training_episodes)
        if evaluation_episodes is not None:
            drl["evaluation_episodes"] = int(evaluation_episodes)
        if final_evaluation_episodes is not None:
            drl["final_evaluation_episodes"] = int(final_evaluation_episodes)
        if evaluation_frequency is not None:
            drl["evaluation_frequency"] = int(evaluation_frequency)
        checkpoint_selection = drl.setdefault("checkpoint_selection", {})
        if routine_eval_episodes is not None:
            checkpoint_selection["routine_eval_episodes"] = int(routine_eval_episodes)
        if confirmation_eval_episodes is not None:
            checkpoint_selection["confirmation_eval_episodes"] = int(confirmation_eval_episodes)

        self.results_dir = setup_results_directory(SCRIPT_DIR, self.config_path, self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_metric = "defender_return" if self.variant_controls["disable_reward_shaping"] else "defender_training_return"
        self.report_metric = str(self.config["drl"].get("report_metric", "avg_defender_return"))
        self.checkpoint_selection_config = dict(self.config["drl"].get("checkpoint_selection", {}))
        self.checkpoint_strategy = str(self.checkpoint_selection_config.get("strategy", "single_metric"))
        early_stopping_config = dict(self.config["drl"].get("early_stopping", {}))
        self.early_stopping_enabled = bool(early_stopping_config.get("enabled", False))
        self.early_stopping_min_training_episodes = int(early_stopping_config.get("min_training_episodes", 0))
        self.early_stopping_patience_evaluations = int(early_stopping_config.get("patience_evaluations", 0))
        if self.checkpoint_strategy == "constrained_operational":
            self.checkpoint_metric = "constrained_operational"
            self.results_root = self.results_dir.parent
            self.reference_targets = load_baseline_reference_targets(
                self.results_root,
                scenario_id=str(self.config["experiment"]["scenario_id"]),
            )
            self.routine_eval_episodes = int(
                self.checkpoint_selection_config.get("routine_eval_episodes", self.config["drl"]["evaluation_episodes"])
            )
            self.confirmation_eval_episodes = int(
                self.checkpoint_selection_config.get(
                    "confirmation_eval_episodes",
                    max(self.routine_eval_episodes, int(self.config["drl"]["final_evaluation_episodes"])),
                )
            )
            self.candidate_pool_size = int(self.checkpoint_selection_config.get("candidate_pool_size", 3))
            tolerance_config = dict(self.checkpoint_selection_config.get("tolerance", {}))
            self.attacker_success_tolerance = float(tolerance_config.get("attacker_success_rate", 0.03))
            self.defender_control_tolerance = float(tolerance_config.get("defender_control_rate", 0.03))
            penalty_config = dict(self.checkpoint_selection_config.get("economic_penalties", {}))
            self.selection_false_response_penalty = float(penalty_config.get("false_response_rate", 20.0))
            self.selection_missed_response_penalty = float(penalty_config.get("missed_response_rate", 35.0))
            self.strict_require_raw_return_advantage = bool(
                self.checkpoint_selection_config.get("strict_require_raw_return_advantage", True)
            )
            self.checkpoint_candidates_dir = self.results_dir / "checkpoint_candidates"
            self.checkpoint_candidates_dir.mkdir(exist_ok=True)
        else:
            self.checkpoint_metric = str(self.config["drl"].get("checkpoint_selection_metric", "avg_defender_training_return"))
            self.results_root = self.results_dir.parent
            self.reference_targets = None
            self.routine_eval_episodes = int(self.config["drl"]["evaluation_episodes"])
            self.confirmation_eval_episodes = int(self.config["drl"]["final_evaluation_episodes"])
            self.candidate_pool_size = 1
            self.attacker_success_tolerance = 0.0
            self.defender_control_tolerance = 0.0
            self.selection_false_response_penalty = 20.0
            self.selection_missed_response_penalty = 35.0
            self.strict_require_raw_return_advantage = True
            self.checkpoint_candidates_dir = self.results_dir / "checkpoint_candidates"
        self.progress_json_path = self.results_dir / "training_progress.json"
        self.progress_log_path = self.results_dir / "training_progress.log"

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
            use_action_mask=not self.variant_controls["disable_action_mask"],
            device=str(self.device),
        )

    def _prepare_observation(self, observation: np.ndarray) -> np.ndarray:
        if self.variant_controls["disable_signal_features"]:
            return ablate_signal_features(observation)
        return np.asarray(observation, dtype=np.float32)

    def _run_episode(
        self,
        env: MaritimeCheatAttentionEnv,
        attacker: SignalCheatGreedyAttackerV2,
        defender: SignalRainbowDQNAgentV2,
        episode_index: int,
        training: bool,
        store_trace: bool,
    ) -> Dict[str, Any]:
        raw_observation, _ = env.reset(seed=int(self.config["experiment"]["random_seed"]) + episode_index)
        observation = self._prepare_observation(raw_observation)
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
            raw_next_observation, reward, terminated, truncated, info = env.step((attacker_action, defender_action))
            next_observation = self._prepare_observation(raw_next_observation)
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

    def _build_selection_metrics(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        if self.reference_targets is None:
            return {}
        return compute_checkpoint_selection_metrics(
            performance,
            self.reference_targets,
            attacker_success_tolerance=self.attacker_success_tolerance,
            defender_control_tolerance=self.defender_control_tolerance,
            false_response_penalty=self.selection_false_response_penalty,
            missed_response_penalty=self.selection_missed_response_penalty,
            strict_require_raw_return_advantage=self.strict_require_raw_return_advantage,
        )

    def _save_candidate_checkpoint(
        self,
        defender: SignalRainbowDQNAgentV2,
        episode_index: int,
    ) -> Path:
        checkpoint_path = self.checkpoint_candidates_dir / f"episode_{episode_index:04d}.pth"
        defender.save(str(checkpoint_path))
        return checkpoint_path

    def _routine_candidate_record(
        self,
        episode_index: int,
        performance: Dict[str, Any],
        checkpoint_path: Path,
        num_episodes: int,
    ) -> Dict[str, Any]:
        return {
            "episode": int(episode_index),
            "performance": dict(performance),
            "selection_metrics": self._build_selection_metrics(performance),
            "checkpoint_path": str(checkpoint_path),
            "num_episodes": int(num_episodes),
        }

    def _confirm_candidate_checkpoints(
        self,
        env: MaritimeCheatAttentionEnv,
        attacker: SignalCheatGreedyAttackerV2,
        defender: SignalRainbowDQNAgentV2,
        routine_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        confirmation_results: List[Dict[str, Any]] = []
        for candidate in routine_candidates:
            defender.load(candidate["checkpoint_path"])
            evaluation_results = self._evaluate(
                env=env,
                attacker=attacker,
                defender=defender,
                num_episodes=self.confirmation_eval_episodes,
                seed_offset=300000 + int(candidate["episode"]) * self.confirmation_eval_episodes,
                store_trace=False,
            )
            confirmation_performance = compute_final_performance(evaluation_results)
            selection_metrics = self._build_selection_metrics(confirmation_performance)
            confirmation_results.append(
                {
                    "episode": int(candidate["episode"]),
                    "checkpoint_path": str(candidate["checkpoint_path"]),
                    "routine_performance": dict(candidate["performance"]),
                    "routine_selection_metrics": dict(candidate["selection_metrics"]),
                    "performance": confirmation_performance,
                    "selection_metrics": selection_metrics,
                    "num_episodes": int(self.confirmation_eval_episodes),
                }
            )
        confirmation_results.sort(key=confirmation_candidate_sort_key)
        return confirmation_results

    def _write_progress(
        self,
        status: str,
        current_episode: int,
        total_episodes: int,
        started_at: float,
        best_episode: int,
        best_score: float,
        last_performance: Dict[str, Any] | None = None,
    ):
        completed = max(0, min(current_episode + 1, total_episodes))
        elapsed_seconds = max(0.0, time.time() - started_at)
        progress_fraction = completed / max(total_episodes, 1)
        eta_seconds = (elapsed_seconds / progress_fraction - elapsed_seconds) if progress_fraction > 0 else None
        payload = {
            "experiment_id": self.config["experiment"]["experiment_id"],
            "status": status,
            "checkpoint_metric": self.checkpoint_metric,
            "current_episode": int(current_episode),
            "completed_episodes": int(completed),
            "total_episodes": int(total_episodes),
            "progress_fraction": float(progress_fraction),
            "elapsed_seconds": float(elapsed_seconds),
            "eta_seconds": None if eta_seconds is None else float(max(0.0, eta_seconds)),
            "best_episode": None if best_episode < 0 else int(best_episode),
            "best_score": None if best_episode < 0 else float(best_score),
            "last_performance": last_performance,
            "updated_at": datetime.now().isoformat(),
        }
        save_json(self.progress_json_path, payload)
        summary = (
            f"[{payload['updated_at']}] status={status} completed={completed}/{total_episodes} "
            f"metric={self.checkpoint_metric} best_episode={payload['best_episode']} best_score={payload['best_score']}"
        )
        with open(self.progress_log_path, "a", encoding="utf-8") as handle:
            handle.write(summary + "\n")
        print(summary, flush=True)

    def run(self) -> Dict[str, Any]:
        env = self._build_environment()
        attacker = self._build_attacker()
        defender = self._build_defender()
        best_score = float("-inf")
        best_episode = -1
        best_performance: Dict[str, Any] | None = None
        best_selection_metrics: Dict[str, Any] | None = None
        best_routine_candidate: Dict[str, Any] | None = None
        candidate_checkpoints: List[Dict[str, Any]] = []
        confirmation_results: List[Dict[str, Any]] = []
        selection_mode = "single_metric"
        selected_checkpoint_reason = "metric_maximization"
        early_stopping_triggered = False
        early_stopping_trigger_episode: int | None = None
        early_stopping_reason: str | None = None
        evaluations_since_improvement = 0
        last_improvement_episode: int | None = None
        completed_training_episodes = 0
        best_model_dir = self.results_dir / "best_model"
        best_model_dir.mkdir(exist_ok=True)
        best_model_path = best_model_dir / "best_defender.pth"

        train_episodes = int(self.config["drl"]["training_episodes"])
        eval_frequency = max(1, int(self.config["drl"]["evaluation_frequency"]))
        eval_episodes = int(self.routine_eval_episodes)
        started_at = time.time()
        self._write_progress(
            status="running",
            current_episode=-1,
            total_episodes=train_episodes,
            started_at=started_at,
            best_episode=best_episode,
            best_score=best_score,
            last_performance=None,
        )

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
            completed_training_episodes = episode_index + 1

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
                if self.checkpoint_strategy == "constrained_operational":
                    checkpoint_path = self._save_candidate_checkpoint(defender, episode_index)
                    candidate = self._routine_candidate_record(
                        episode_index=episode_index,
                        performance=performance,
                        checkpoint_path=checkpoint_path,
                        num_episodes=eval_episodes,
                    )
                    improved = constrained_candidate_improves(candidate, best_routine_candidate)
                    candidate_checkpoints.append(candidate)
                    candidate_checkpoints = sorted(candidate_checkpoints, key=routine_candidate_sort_key)[: self.candidate_pool_size]
                    best_candidate = candidate_checkpoints[0]
                    if improved:
                        best_routine_candidate = {
                            "episode": int(best_candidate["episode"]),
                            "performance": dict(best_candidate["performance"]),
                            "selection_metrics": dict(best_candidate["selection_metrics"]),
                        }
                        evaluations_since_improvement = 0
                        last_improvement_episode = int(best_candidate["episode"])
                    else:
                        evaluations_since_improvement += 1
                    best_episode = int(best_candidate["episode"])
                    best_performance = dict(best_candidate["performance"])
                    best_selection_metrics = dict(best_candidate["selection_metrics"])
                    best_score = float(best_selection_metrics["economic_score"])
                    self.evaluation_history.append(
                        {
                            "episode": episode_index,
                            "performance": performance,
                            "checkpoint_metric": self.checkpoint_metric,
                            "checkpoint_score": best_score,
                            "num_episodes": eval_episodes,
                            "selection_metrics": dict(candidate["selection_metrics"]),
                            "checkpoint_path": str(checkpoint_path),
                        }
                    )
                else:
                    checkpoint_score = get_metric_value(performance, self.checkpoint_metric)
                    improved = checkpoint_score > best_score
                    self.evaluation_history.append(
                        {
                            "episode": episode_index,
                            "performance": performance,
                            "checkpoint_metric": self.checkpoint_metric,
                            "checkpoint_score": checkpoint_score,
                            "num_episodes": eval_episodes,
                        }
                    )
                    if improved:
                        best_score = checkpoint_score
                        best_episode = episode_index
                        best_performance = dict(performance)
                        defender.save(str(best_model_path))
                        evaluations_since_improvement = 0
                        last_improvement_episode = int(episode_index)
                    else:
                        evaluations_since_improvement += 1
                self._write_progress(
                    status="running",
                    current_episode=episode_index,
                    total_episodes=train_episodes,
                    started_at=started_at,
                    best_episode=best_episode,
                    best_score=best_score,
                    last_performance=performance,
                )
                if should_trigger_early_stopping(
                    enabled=self.early_stopping_enabled,
                    current_episode=episode_index,
                    min_training_episodes=self.early_stopping_min_training_episodes,
                    evaluations_since_improvement=evaluations_since_improvement,
                    patience_evaluations=self.early_stopping_patience_evaluations,
                ):
                    early_stopping_triggered = True
                    early_stopping_trigger_episode = int(episode_index)
                    early_stopping_reason = (
                        "No better checkpoint candidate was found within the configured early-stopping patience."
                    )
                    self._write_progress(
                        status="stopped_early",
                        current_episode=episode_index,
                        total_episodes=train_episodes,
                        started_at=started_at,
                        best_episode=best_episode,
                        best_score=best_score,
                        last_performance=performance,
                    )
                    break

        if self.checkpoint_strategy == "constrained_operational":
            confirmation_results = self._confirm_candidate_checkpoints(
                env=env,
                attacker=attacker,
                defender=defender,
                routine_candidates=candidate_checkpoints,
            )
            selected_candidate = confirmation_results[0]
            best_episode = int(selected_candidate["episode"])
            best_performance = dict(selected_candidate["performance"])
            best_selection_metrics = dict(selected_candidate["selection_metrics"])
            best_score = float(best_selection_metrics["economic_score"])
            selection_mode = (
                "feasible_dominance"
                if best_selection_metrics.get("selection_feasible_under_baseline")
                else "fallback_min_security_deficit"
            )
            if selection_mode == "feasible_dominance":
                selected_checkpoint_reason = (
                    "Selected the checkpoint that satisfied baseline superiority and maximized raw return with control-per-cost tie-breaking."
                )
            else:
                selected_checkpoint_reason = (
                    "No checkpoint satisfied all baseline superiority constraints, so the checkpoint with minimum security deficit was selected."
                )
            defender.load(selected_candidate["checkpoint_path"])
            defender.save(str(best_model_path))
        elif best_model_path.exists():
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
                **build_experiment_tags(self.config),
            },
            "metric_conventions": {
                "training_metric": (
                    "defender_return (raw reward used for optimization)"
                    if self.variant_controls["disable_reward_shaping"]
                    else "defender_training_return (shaped reward used for optimization)"
                ),
                "report_metric": f"{self.report_metric} (raw utility used for reporting)",
            },
            "checkpoint_selection": {
                "metric": self.checkpoint_metric,
                "strategy": self.checkpoint_strategy,
                "reference_targets": self.reference_targets,
                "best_score": best_score if best_episode >= 0 else None,
                "best_episode": best_episode if best_episode >= 0 else None,
                "best_performance": best_performance,
                "best_selection_metrics": best_selection_metrics,
                "best_model_path": str(best_model_path) if best_model_path.exists() else None,
                "candidate_checkpoints": candidate_checkpoints,
                "confirmation_results": confirmation_results,
                "selected_checkpoint_reason": selected_checkpoint_reason,
                "selection_mode": selection_mode,
            },
            "early_stopping": {
                "enabled": self.early_stopping_enabled,
                "min_training_episodes": self.early_stopping_min_training_episodes,
                "patience_evaluations": self.early_stopping_patience_evaluations,
                "triggered": early_stopping_triggered,
                "trigger_episode": early_stopping_trigger_episode,
                "reason": early_stopping_reason,
                "completed_training_episodes": completed_training_episodes,
                "evaluations_since_improvement": evaluations_since_improvement,
                "last_improvement_episode": last_improvement_episode,
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
        self._write_progress(
            status="completed_early_stopping" if early_stopping_triggered else "completed",
            current_episode=max(0, completed_training_episodes - 1),
            total_episodes=train_episodes,
            started_at=started_at,
            best_episode=best_episode,
            best_score=best_score,
            last_performance=final_performance,
        )
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
    parser.add_argument("--routine-eval-episodes", type=int, default=None)
    parser.add_argument("--confirmation-eval-episodes", type=int, default=None)
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
        routine_eval_episodes=args.routine_eval_episodes,
        confirmation_eval_episodes=args.confirmation_eval_episodes,
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
