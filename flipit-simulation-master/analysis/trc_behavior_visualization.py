"""
行为可视化脚本（ESWA 投稿补充图）

生成三类图：
1) 动作随时间的热力图（回合×动作ID/单位数），分别对 DRL 与传统实验
2) 资源可用余额轨迹并标注关键转折点（大额开支 / 透支报警）
3) 攻防动作协同矩阵（出现频率，并标注胜/负方向）

依赖：matplotlib、seaborn、pandas、numpy、torch
运行：
    python analysis/trc_behavior_visualization.py
输出：
    保存至 Fig/ 目录，文件名带时间戳。
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "gym-flipit-master"))

from strategies.rainbow_dqn_standalone import RainbowDQNAgent
from strategies.MaritimeDeceptiveGreedy import MaritimeDeceptiveGreedy
from strategies.AggressiveAttacker import AggressiveAttacker
from strategies.WeakDefensiveGreedy import WeakDefensiveGreedy
from gym_flipit.envs.maritime_nontraditional_env import MaritimeNontraditionalEnv

RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "Fig"
FIG_DIR.mkdir(exist_ok=True)

MAX_STEPS = 30
EVAL_EPISODES = 50


def _latest_run(prefix: str) -> Path:
    """获取指定前缀的最新结果目录"""
    candidates = sorted(RESULTS_DIR.glob(f"{prefix}_*"), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"未找到结果目录: {prefix}_*")
    return candidates[-1]


def _load_best_model(run_dir: Path, obs_dim: int, action_dim: int, max_units: int) -> RainbowDQNAgent:
    perf_path = run_dir / "best_model" / "performance_record.json"
    with open(perf_path, "r") as f:
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


def _build_attacker(deception_mode: str):
    if deception_mode == "flipit":
        return AggressiveAttacker(move_cost=15, cheat_cost=5, debug=False)
    return MaritimeDeceptiveGreedy(move_cost=15, cheat_cost=5, debug=False)


def _build_defender_traditional():
    return WeakDefensiveGreedy(move_cost=10, cheat_cost=5, debug=False)


def _collect_episode(env, attacker, defender, defender_is_drl: bool):
    """跑一局，采集动作、资源轨迹、协同统计"""
    obs = env.reset()
    done = False
    step = 0
    records = []
    coop_records = []
    resource_trace = []
    initial_att_budget = env.attacker_budget
    initial_def_budget = env.defender_budget
    big_spend_flag_att = False
    big_spend_flag_def = False
    overdraft_flag = False

    while not done and step < MAX_STEPS:
        step += 1
        # 攻击动作
        if hasattr(attacker, "pre"):
            att_action = attacker.pre(env.tick, obs)
        else:
            att_action = env.action_space.sample()
        if isinstance(att_action, int):
            att_action = (0, 0) if att_action == 0 else (att_action, 1)
        elif not isinstance(att_action, tuple):
            att_action = (0, 1)

        # 防守动作
        if defender_is_drl:
            def_action = defender.select_action(obs, training=False)
        else:
            if hasattr(defender, "pre"):
                def_action = defender.pre(env.tick, obs)
            else:
                def_action = env.action_space.sample()
        if not isinstance(def_action, tuple):
            def_action = (def_action, 1)

        combined_action = (att_action, def_action)
        next_obs, reward, done, info = env.step(combined_action)

        er = info.get("engagement_result")
        att_cost = er.att_cost if er else 0
        def_cost = er.def_cost if er else 0
        # 可用余额（含收入/占领奖励）
        att_available = info.get("attacker_budget_remaining", 0)
        def_available = info.get("defender_budget_remaining", 0)

        # 标注关键点
        big_spend_att = False
        big_spend_def = False
        if not big_spend_flag_att and att_cost >= 0.3 * initial_att_budget:
            big_spend_att = True
            big_spend_flag_att = True
        if not big_spend_flag_def and def_cost >= 0.3 * initial_def_budget:
            big_spend_def = True
            big_spend_flag_def = True
        if info.get("failure_reason") in ("attacker_budget_overdraft", "defender_budget_overdraft"):
            overdraft_flag = True

        records.append(
            {
                "step": step,
                "att_action": att_action[0],
                "att_units": att_action[1],
                "def_action": def_action[0],
                "def_units": def_action[1],
                "att_available": att_available,
                "def_available": def_available,
                "big_spend_att": big_spend_att,
                "big_spend_def": big_spend_def,
                "overdraft": overdraft_flag,
            }
        )
        coop_records.append(
            {
                "att_action": att_action[0],
                "def_action": def_action[0],
                "success": bool(er.success) if er else False,
            }
        )
        resource_trace.append(
            {
                "step": step,
                "att_available": att_available,
                "def_available": def_available,
                "big_spend_att": big_spend_att,
                "big_spend_def": big_spend_def,
                "overdraft": overdraft_flag,
            }
        )

        obs = next_obs

    return records, coop_records, resource_trace


def _collect_runs(config_path: str, run_prefix: str, defender_is_drl: bool):
    """基于配置与最新结果目录，收集评估数据"""
    run_dir = _latest_run(run_prefix)
    with open(BASE_DIR / config_path, "r", encoding="utf-8") as f:
        cfg = f.read()
    deception_mode = "cheat" if "deception_mode: \"cheat\"" in cfg else "flipit"

    env = MaritimeNontraditionalEnv(str(BASE_DIR / config_path))
    attacker = _build_attacker(deception_mode)

    if defender_is_drl:
        with open(run_dir / "config.yml", "r", encoding="utf-8") as f:
            dqn_cfg = yaml.safe_load(f)
        dqn_params = dqn_cfg.get("drl_config", {}).get("dqn", {})
        defender = _load_best_model(
            run_dir,
            obs_dim=dqn_params.get("obs_dim", 13),
            action_dim=dqn_params.get("action_dim", 20),
            max_units=dqn_params.get("max_units", 4),
        )
    else:
        defender = _build_defender_traditional()

    all_records = []
    all_coop = []
    all_resource = []
    for _ in range(EVAL_EPISODES):
        rec, coop, res = _collect_episode(env, attacker, defender, defender_is_drl)
        all_records.extend(rec)
        all_coop.extend(coop)
        all_resource.extend(res)
    return pd.DataFrame(all_records), pd.DataFrame(all_coop), pd.DataFrame(all_resource)


def plot_action_heatmap(df: pd.DataFrame, title: str, fname: str):
    # 组合动作键
    df["action_key"] = df["att_action"].astype(str) + "x" + df["att_units"].astype(str)
    pivot = df.pivot_table(index="action_key", columns="step", values="att_units", aggfunc="count", fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Attack action × units")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIG_DIR / f"{fname}.pdf")
    plt.close()

    # 防守动作热力图
    df["def_key"] = df["def_action"].astype(str) + "x" + df["def_units"].astype(str)
    pivot_def = df.pivot_table(index="def_key", columns="step", values="def_units", aggfunc="count", fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_def, cmap="YlOrRd")
    plt.title(title + " (Defense)")
    plt.xlabel("Step")
    plt.ylabel("Defense action × units")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{fname}_def.png", dpi=300)
    plt.savefig(FIG_DIR / f"{fname}_def.pdf")
    plt.close()


def plot_resource_trace(df: pd.DataFrame, title: str, fname: str):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="step", y="att_available", label="Attacker available budget", color="#d62728")
    sns.lineplot(data=df, x="step", y="def_available", label="Defender available budget", color="#1f77b4")

    # 标注关键转折点（首个大额开支、透支报警）
    for side, color in [("big_spend_att", "#d62728"), ("big_spend_def", "#1f77b4")]:
        pts = df[df[side]]
        if not pts.empty:
            plt.scatter(pts["step"], pts["att_available" if side == "big_spend_att" else "def_available"],
                        color=color, marker="o", s=40, zorder=5, label="Large spend (attack)" if side == "big_spend_att" else "Large spend (defense)")
    if df["overdraft"].any():
        od = df[df["overdraft"]]
        plt.scatter(od["step"], od["att_available"], color="black", marker="x", s=50, label="Overdraft alarm")

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Available budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{fname}.png", dpi=300)
    plt.savefig(FIG_DIR / f"{fname}.pdf")
    plt.close()


def plot_coop_matrix(df: pd.DataFrame, title: str, fname: str):
    freq = df.pivot_table(index="att_action", columns="def_action", values="success", aggfunc="count", fill_value=0)
    win = df.pivot_table(index="att_action", columns="def_action", values="success", aggfunc="mean", fill_value=0)

    plt.figure(figsize=(7, 5))
    sns.heatmap(freq, annot=True, fmt=".0f", cmap="Blues")
    plt.title(title + " (Frequency)")
    plt.xlabel("Defense action ID")
    plt.ylabel("Attack action ID")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{fname}_freq.png", dpi=300)
    plt.savefig(FIG_DIR / f"{fname}_freq.pdf")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(win, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="RdYlGn")
    plt.title(title + " (Attack success rate)")
    plt.xlabel("Defense action ID")
    plt.ylabel("Attack action ID")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{fname}_win.png", dpi=300)
    plt.savefig(FIG_DIR / f"{fname}_win.pdf")
    plt.close()


def main():
    # DRL (Cheat)
    drl_df, drl_coop, drl_res = _collect_runs(
        "configs/trc_balanced_realistic.yml",
        "trc_balanced_realistic_drl_defense_vs_greedy_attack",
        defender_is_drl=True,
    )
    plot_action_heatmap(drl_df, "DRL Defense (Cheat) Action Heatmap", "behavior_drl_cheat_actions")
    plot_resource_trace(drl_res, "DRL Defense (Cheat) Budget Trajectory", "behavior_drl_cheat_resources")
    plot_coop_matrix(drl_coop, "DRL Defense (Cheat) Attack-Defense Matrix", "behavior_drl_cheat_coop")

    # DRL (FlipIt)
    drl_df_f, drl_coop_f, drl_res_f = _collect_runs(
        "configs/trc_balanced_realistic_flipit.yml",
        "trc_balanced_realistic_drl_defense_vs_greedy_attack_flipit",
        defender_is_drl=True,
    )
    plot_action_heatmap(drl_df_f, "DRL Defense (FlipIt) Action Heatmap", "behavior_drl_flipit_actions")
    plot_resource_trace(drl_res_f, "DRL Defense (FlipIt) Budget Trajectory", "behavior_drl_flipit_resources")
    plot_coop_matrix(drl_coop_f, "DRL Defense (FlipIt) Attack-Defense Matrix", "behavior_drl_flipit_coop")

    # 传统 (Cheat)
    trad_df, trad_coop, trad_res = _collect_runs(
        "configs/trc_traditional_baseline.yml",
        "trc_traditional_baseline_vs_greedy_cheat",
        defender_is_drl=False,
    )
    plot_action_heatmap(trad_df, "Traditional Defense (Cheat) Action Heatmap", "behavior_trad_cheat_actions")
    plot_resource_trace(trad_res, "Traditional Defense (Cheat) Budget Trajectory", "behavior_trad_cheat_resources")
    plot_coop_matrix(trad_coop, "Traditional Defense (Cheat) Attack-Defense Matrix", "behavior_trad_cheat_coop")

    # 传统 (FlipIt)
    trad_df_f, trad_coop_f, trad_res_f = _collect_runs(
        "configs/trc_traditional_flipit_baseline.yml",
        "trc_traditional_flipit_baseline",
        defender_is_drl=False,
    )
    plot_action_heatmap(trad_df_f, "Traditional Defense (FlipIt) Action Heatmap", "behavior_trad_flipit_actions")
    plot_resource_trace(trad_res_f, "Traditional Defense (FlipIt) Budget Trajectory", "behavior_trad_flipit_resources")
    plot_coop_matrix(trad_coop_f, "Traditional Defense (FlipIt) Attack-Defense Matrix", "behavior_trad_flipit_coop")

    print("🎨 行为可视化已生成，存放于 Fig/")


if __name__ == "__main__":
    main()

