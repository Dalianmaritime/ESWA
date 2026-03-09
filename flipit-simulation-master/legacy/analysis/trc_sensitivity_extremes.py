"""
Sensitivity & extreme-case runner for the DRL main experiment (ESWA submission).

Sweeps:
- Nonlinear synergy (alpha / cap) ±20% for attacker / defender (separately).
- Exploration & training length: epsilon_end in {0.05, 0.01}, training_episodes in {200, 500, 800}.

Extreme cases (occupation_reward kept at 2.0):
- Half budget (initial budgets * 0.5, per-step income unchanged).
- One-third budget (initial budgets * ~0.333, per-step income unchanged).
- Zero income (per-step income = 0, budgets unchanged).

Runs use the full experiment settings (no “fast” mode). Results are saved under
`results/` with labeled experiment_id; summary figures go to `Fig/part2/`.

Usage:
    python analysis/trc_sensitivity_extremes.py
"""

import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR.parent.parent / "gym-flipit-master"))

RESULTS_DIR = BASE_DIR / "results"
TMP_CFG_DIR = RESULTS_DIR / "tmp_configs"
TMP_CFG_DIR.mkdir(exist_ok=True)

FIG_DIR = BASE_DIR / "Fig" / "part2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG_PATH = BASE_DIR / "configs" / "trc_balanced_realistic.yml"

ALPHA_CAP_SCALES = [0.8, 1.0, 1.2]
EPS_ENDS = [0.05, 0.01]
TRAIN_EPISODES = [200]


def _save_temp_config(cfg: Dict, label: str) -> Path:
    path = TMP_CFG_DIR / f"{label}.yml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def _latest_result_dir(prefix: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(f"{prefix}_*"), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No result directory found for {prefix}_*")
    return matches[-1]


def _run_experiment(cfg_path: Path, experiment_id: str):
    cmd = [
        sys.executable,
        str(BASE_DIR / "run_trc_full_training.py"),
        str(cfg_path),
    ]
    env = os.environ.copy()
    # ensure PYTHONPATH covers gym-flipit
    env["PYTHONPATH"] = str(BASE_DIR.parent.parent / "gym-flipit-master")
    subprocess.check_call(cmd, cwd=str(BASE_DIR), env=env)


def _load_metrics(run_dir: Path) -> Dict:
    with open(run_dir / "complete_training_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    perf = data["final_performance"]
    return {
        "attacker_success_rate": perf["attacker_success_rate"],
        "att_occ": perf["avg_attacker_occupation_reward"],
        "def_occ": perf["avg_defender_occupation_reward"],
        "game_length": perf["avg_game_length"],
        "result_dir": str(run_dir),
    }


def _scale_actions(actions: List[Dict], key: str, scale: float):
    for a in actions:
        if "nonlinear" in a and key in a["nonlinear"]:
            a["nonlinear"][key] = float(a["nonlinear"][key]) * scale
        if key == "cap" and a["nonlinear"][key] > 1.0:
            a["nonlinear"][key] = min(a["nonlinear"][key], 1.0)


def sweep_alpha_cap():
    base_cfg = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    rows = []
    # attacker sweep
    for alpha_s in ALPHA_CAP_SCALES:
        for cap_s in ALPHA_CAP_SCALES:
            cfg = deepcopy(base_cfg)
            _scale_actions(cfg["attacker_actions"], "alpha", alpha_s)
            _scale_actions(cfg["attacker_actions"], "cap", cap_s)
            cfg["experiment_id"] = f"sweep_att_alpha{alpha_s}_cap{cap_s}"
            cfg_path = _save_temp_config(cfg, cfg["experiment_id"])
            _run_experiment(cfg_path, cfg["experiment_id"])
            run_dir = _latest_result_dir(cfg["experiment_id"])
            m = _load_metrics(run_dir)
            m.update({"who": "attacker", "alpha_s": alpha_s, "cap_s": cap_s})
            rows.append(m)
    # defender sweep
    for alpha_s in ALPHA_CAP_SCALES:
        for cap_s in ALPHA_CAP_SCALES:
            cfg = deepcopy(base_cfg)
            _scale_actions(cfg["defender_actions"], "alpha", alpha_s)
            _scale_actions(cfg["defender_actions"], "cap", cap_s)
            cfg["experiment_id"] = f"sweep_def_alpha{alpha_s}_cap{cap_s}"
            cfg_path = _save_temp_config(cfg, cfg["experiment_id"])
            _run_experiment(cfg_path, cfg["experiment_id"])
            run_dir = _latest_result_dir(cfg["experiment_id"])
            m = _load_metrics(run_dir)
            m.update({"who": "defender", "alpha_s": alpha_s, "cap_s": cap_s})
            rows.append(m)
    return pd.DataFrame(rows)


def sweep_exploration():
    base_cfg = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    rows = []
    for eps_end in EPS_ENDS:
        for ep in TRAIN_EPISODES:
            cfg = deepcopy(base_cfg)
            cfg["drl_config"]["dqn"]["epsilon_end"] = eps_end
            cfg["drl_config"]["training_episodes"] = ep
            cfg["experiment_id"] = f"sweep_eps{eps_end}_ep{ep}"
            cfg_path = _save_temp_config(cfg, cfg["experiment_id"])
            _run_experiment(cfg_path, cfg["experiment_id"])
            run_dir = _latest_result_dir(cfg["experiment_id"])
            m = _load_metrics(run_dir)
            m.update({"epsilon_end": eps_end, "train_episodes": ep})
            rows.append(m)
    return pd.DataFrame(rows)


def run_extremes():
    base_cfg = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    scenarios = []

    # half budget
    cfg_half = deepcopy(base_cfg)
    cfg_half["constraints"]["budgeting"]["attacker_budget"] *= 0.5
    cfg_half["constraints"]["budgeting"]["defender_budget"] *= 0.5
    cfg_half["experiment_id"] = "extreme_budget_half"
    scenarios.append(("half_budget", cfg_half))

    # one-third budget
    cfg_third = deepcopy(base_cfg)
    cfg_third["constraints"]["budgeting"]["attacker_budget"] *= (1 / 3)
    cfg_third["constraints"]["budgeting"]["defender_budget"] *= (1 / 3)
    cfg_third["experiment_id"] = "extreme_budget_third"
    scenarios.append(("third_budget", cfg_third))

    # zero income
    cfg_zero = deepcopy(base_cfg)
    cfg_zero["rew_config"]["attacker_income_per_step"] = 0
    cfg_zero["rew_config"]["defender_income_per_step"] = 0
    cfg_zero["experiment_id"] = "extreme_zero_income"
    scenarios.append(("zero_income", cfg_zero))

    rows = []
    for label, cfg in scenarios:
        cfg_path = _save_temp_config(cfg, cfg["experiment_id"])
        _run_experiment(cfg_path, cfg["experiment_id"])
        run_dir = _latest_result_dir(cfg["experiment_id"])
        m = _load_metrics(run_dir)
        m.update({"scenario": label})
        rows.append(m)
    return pd.DataFrame(rows)


def plot_alpha_cap(df: pd.DataFrame):
    for who in ["attacker", "defender"]:
        sub = df[df["who"] == who]
        pivot = sub.pivot_table(index="alpha_s", columns="cap_s", values="attacker_success_rate")
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1)
        plt.title(f"Success rate vs alpha/cap ({who})")
        plt.xlabel("cap scale")
        plt.ylabel("alpha scale")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"sweep_{who}_alpha_cap_success.png", dpi=300)
        plt.savefig(FIG_DIR / f"sweep_{who}_alpha_cap_success.pdf")
        plt.close()


def plot_exploration(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="train_episodes", y="attacker_success_rate", hue="epsilon_end", marker="o")
    plt.ylim(0, 1)
    plt.title("Success rate vs episodes / epsilon_end")
    plt.xlabel("Training episodes")
    plt.ylabel("Attacker success rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sweep_exploration_success.png", dpi=300)
    plt.savefig(FIG_DIR / "sweep_exploration_success.pdf")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="train_episodes", y="def_occ", hue="epsilon_end", marker="o")
    plt.title("Defender occupation reward vs episodes / epsilon_end")
    plt.xlabel("Training episodes")
    plt.ylabel("Defender occupation reward")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sweep_exploration_defocc.png", dpi=300)
    plt.savefig(FIG_DIR / "sweep_exploration_defocc.pdf")
    plt.close()


def plot_extremes(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="scenario", y="attacker_success_rate", palette="Set2")
    plt.ylim(0, 1)
    plt.title("Extreme scenarios: attacker success rate")
    plt.xlabel("")
    plt.ylabel("Attacker success rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "extreme_success.png", dpi=300)
    plt.savefig(FIG_DIR / "extreme_success.pdf")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="scenario", y="def_occ", palette="Set2")
    plt.title("Extreme scenarios: defender occupation reward")
    plt.xlabel("")
    plt.ylabel("Defender occupation reward")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "extreme_defocc.png", dpi=300)
    plt.savefig(FIG_DIR / "extreme_defocc.pdf")
    plt.close()


def main():
    start = time.time()
    alpha_cap_df = sweep_alpha_cap()
    alpha_cap_df.to_csv(FIG_DIR / "sweep_alpha_cap.csv", index=False)
    plot_alpha_cap(alpha_cap_df)

    exp_df = sweep_exploration()
    exp_df.to_csv(FIG_DIR / "sweep_exploration.csv", index=False)
    plot_exploration(exp_df)

    ext_df = run_extremes()
    ext_df.to_csv(FIG_DIR / "extremes.csv", index=False)
    plot_extremes(ext_df)

    print(f"Completed all sweeps in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()


