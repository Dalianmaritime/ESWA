"""
Resource trajectory probe for the main experiment (DRL defender, Cheat mode).

Purpose: visualize attacker/defender available budgets over time in the main
experiment configuration, using the latest trained DRL model. Useful to confirm
whether resource pressure is actually applied during play.

Outputs:
- CSV with per-step budgets across evaluation episodes.
- Line plot (mean ± std) of available budgets over steps.

Run:
    python analysis/trc_resource_probe.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
FIG_DIR = BASE_DIR / "Fig" / "part2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = BASE_DIR / "configs" / "trc_balanced_realistic.yml"
RUN_PREFIX = "trc_balanced_realistic_drl_defense_vs_greedy_attack"
EVAL_EPISODES = 1
MAX_STEPS = 30


def _latest_run(prefix: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(f"{prefix}_*"), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No result dir for {prefix}_*")
    return matches[-1]


def _load_best_model(run_dir: Path, obs_dim: int, action_dim: int, max_units: int) -> RainbowDQNAgent:
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


def _build_attacker():
    return MaritimeDeceptiveGreedy(move_cost=15, cheat_cost=5, debug=False)


def collect_episodes(env, attacker, defender) -> pd.DataFrame:
    rows: List[Dict] = []
    for ep in range(EVAL_EPISODES):
        obs = env.reset()
        done = False
        step = 0
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

            rows.append(
                {
                    "episode": ep,
                    "step": step,
                    "att_available": info.get("attacker_budget_remaining", 0),
                    "def_available": info.get("defender_budget_remaining", 0),
                    "controller": info.get("current_controller", 0),
                }
            )
            obs = next_obs
    return pd.DataFrame(rows)


def plot_budget(df: pd.DataFrame, fname: str):
    df_long = df.melt(id_vars=["episode", "step"], value_vars=["att_available", "def_available"],
                      var_name="side", value_name="available_budget")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_long, x="step", y="available_budget", hue="side", estimator="mean", ci="sd")
    plt.title("Available budget over steps (main experiment)")
    plt.xlabel("Step")
    plt.ylabel("Available budget")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIG_DIR / f"{fname}.pdf")
    plt.close()


def main():
    run_dir = _latest_run(RUN_PREFIX)
    with open(run_dir / "config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dqn_params = cfg.get("drl_config", {}).get("dqn", {})

    env = MaritimeNontraditionalEnv(str(CONFIG_PATH))
    attacker = _build_attacker()
    defender = _load_best_model(run_dir, dqn_params.get("obs_dim", 13),
                                dqn_params.get("action_dim", 20),
                                dqn_params.get("max_units", 4))

    df = collect_episodes(env, attacker, defender)
    df.to_csv(FIG_DIR / "resource_probe_main.csv", index=False)
    plot_budget(df, "resource_probe_main")
    print("Resource probe completed. Outputs saved to Fig/part2/")


if __name__ == "__main__":
    main()
