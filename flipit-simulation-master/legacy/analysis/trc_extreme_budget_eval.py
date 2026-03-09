"""
Extreme budget evaluation for the main DRL defender.

Two cases:
  1) Defender budget=10, Attacker budget=100
  2) Defender budget=20, Attacker budget=200

For each case:
  - Evaluate 20 games (no training) with the latest DRL model (Cheat mode).
  - Save per-episode results and per-step resource traces.
  - Plot the last game's resource trajectory.

Outputs:
  - results/part3_extreme_*.csv (episodes and traces)
  - Fig/part3/extreme_*_resource.png/pdf
"""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR.parent.parent / "gym-flipit-master"))

from strategies.rainbow_dqn_standalone import RainbowDQNAgent
from strategies.MaritimeDeceptiveGreedy import MaritimeDeceptiveGreedy
from gym_flipit.legacy.maritime_nontraditional_env import MaritimeNontraditionalEnv

RESULTS_DIR = BASE_DIR / "results"
TMP_CFG_DIR = RESULTS_DIR / "tmp_configs"
TMP_CFG_DIR.mkdir(exist_ok=True)

FIG_DIR = BASE_DIR / "Fig" / "part3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = BASE_DIR / "configs" / "trc_balanced_realistic.yml"
RUN_PREFIX = "trc_balanced_realistic_drl_defense_vs_greedy_attack"

EVAL_EPISODES = 20
MAX_STEPS = 30


def latest_run(prefix: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(f"{prefix}_*"), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No result dir for {prefix}_*")
    return matches[-1]


def load_best_model(run_dir: Path, obs_dim: int, action_dim: int, max_units: int) -> RainbowDQNAgent:
    perf_path = run_dir / "best_model" / "performance_record.json"
    with open(perf_path, "r", encoding="utf-8") as f:
        perf = json.load(f)
    model_path = perf["model_path"]
    agent = RainbowDQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_units=max_units,
        device="cpu",
    )
    agent.load(model_path)
    return agent


def build_attacker():
    return MaritimeDeceptiveGreedy(move_cost=15, cheat_cost=5, debug=False)


def write_temp_config(budget_att: float, budget_def: float, label: str) -> Path:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["constraints"]["budgeting"]["attacker_budget"] = budget_att
    cfg["constraints"]["budgeting"]["defender_budget"] = budget_def
    cfg["experiment_id"] = label
    path = TMP_CFG_DIR / f"{label}.yml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def evaluate(cfg_path: Path, defender) -> Dict[str, pd.DataFrame]:
    episode_rows: List[Dict] = []
    trace_rows: List[Dict] = []
    for ep in range(EVAL_EPISODES):
        env = MaritimeNontraditionalEnv(str(cfg_path))
        attacker = build_attacker()
        obs = env.reset()
        done = False
        step = 0
        last_info = {}
        while not done and step < MAX_STEPS:
            step += 1
            att_action = attacker.pre(env.tick, obs) if hasattr(attacker, "pre") else (0, 1)
            if isinstance(att_action, int):
                att_action = (0, 0) if att_action == 0 else (att_action, 1)
            elif not isinstance(att_action, tuple):
                att_action = (0, 1)

            def_action = defender.select_action(obs, training=False)
            if not isinstance(def_action, tuple):
                def_action = (def_action, 1)

            action = (att_action, def_action)
            next_obs, reward, done, info = env.step(action)
            last_info = info

            er = info.get("engagement_result")
            trace_rows.append(
                {
                    "episode": ep,
                    "step": step,
                    "att_available": info.get("attacker_budget_remaining", 0),
                    "def_available": info.get("defender_budget_remaining", 0),
                    "att_cost": er.att_cost if er else 0,
                    "def_cost": er.def_cost if er else 0,
                    "controller": info.get("current_controller", 0),
                }
            )
            obs = next_obs

        winner = last_info.get("winner")
        episode_rows.append(
            {
                "episode": ep,
                "winner": winner,
                "attacker_success": winner == "attacker",
                "att_available_end": last_info.get("attacker_budget_remaining", 0),
                "def_available_end": last_info.get("defender_budget_remaining", 0),
                "steps": step,
            }
        )

    return {
        "episodes": pd.DataFrame(episode_rows),
        "trace": pd.DataFrame(trace_rows),
    }


def plot_last_trace(df: pd.DataFrame, fname: str, title: str):
    last_ep = df["episode"].max()
    sub = df[df["episode"] == last_ep]
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=sub, x="step", y="att_available", label="Attacker available budget", color="#d62728")
    sns.lineplot(data=sub, x="step", y="def_available", label="Defender available budget", color="#1f77b4")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Available budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIG_DIR / f"{fname}.pdf")
    plt.close()


def run_case(b_att: float, b_def: float, label: str):
    cfg_path = write_temp_config(b_att, b_def, label)
    run_dir = latest_run(RUN_PREFIX)
    with open(run_dir / "config.yml", "r", encoding="utf-8") as f:
        cfg_run = yaml.safe_load(f)
    dqn_params = cfg_run.get("drl_config", {}).get("dqn", {})

    defender = load_best_model(run_dir, dqn_params.get("obs_dim", 13),
                               dqn_params.get("action_dim", 20),
                               dqn_params.get("max_units", 4))

    res = evaluate(cfg_path, defender)
    episodes = res["episodes"]
    trace = res["trace"]

    episodes.to_csv(RESULTS_DIR / f"part3_{label}_episodes.csv", index=False)
    trace.to_csv(RESULTS_DIR / f"part3_{label}_trace.csv", index=False)

    plot_last_trace(trace, f"{label}_last_trace", f"Resource trajectory (last game) - {label}")

    win_rate = episodes["attacker_success"].mean()
    print(f"{label}: attacker win rate over {EVAL_EPISODES} games = {win_rate:.3f}")


def main():
    cases = [
        (100, 10, "extreme_att100_def10"),
        (200, 20, "extreme_att200_def20"),
    ]
    for b_att, b_def, label in cases:
        run_case(b_att, b_def, label)
    print(f"Outputs saved to Fig/part3/ and results/part3_*")


if __name__ == "__main__":
    main()
