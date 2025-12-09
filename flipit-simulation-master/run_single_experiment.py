#!/usr/bin/env python3
"""
快速单实验运行脚本 - 用于参数调优
"""

import os
import sys
import json
import yaml
import time
import numpy as np
import torch
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../gym-flipit-master'))

# 导入环境和策略
import gym_flipit.envs
from gym_flipit.envs.maritime_nontraditional_env import MaritimeNontraditionalEnv

# 导入核心算法
from strategies.rainbow_dqn_standalone import RainbowDQNAgent  
from strategies.MaritimeDeceptiveGreedy import MaritimeDeceptiveGreedy

def run_quick_experiment(config_path: str) -> Dict:
    """快速运行单个实验用于参数调优"""
    
    print(f"[QUICK] 快速实验: {config_path}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    if 'random_seed' in config:
        seed = config['random_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # 创建环境
    env = MaritimeNontraditionalEnv(config_path)
    
    # 创建策略
    print("创建攻击方策略: 贪心算法")
    attacker = MaritimeDeceptiveGreedy(
        move_cost=config.get('p1_config', {}).get('move_cost', 15),
        cheat_cost=5,
        debug=False
    )
    
    defender_strategy = config.get('p0_strategy', 'maritime_dqn')
    defender_config = config.get('p0_config', {})
    
    if defender_strategy == "maritime_defensive_greedy":
        print("创建防守方策略: 传统贪心算法")
        defender = MaritimeDeceptiveGreedy(
            move_cost=defender_config.get('move_cost', 10),
            cheat_cost=5,
            debug=False,
            deception_threshold=defender_config.get('deception_threshold', 0.4),
            maritime_context=True
        )
    else:
        print("创建防守方策略: DRL")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        drl_config = config.get('drl_config', {})
        dqn_config = drl_config.get('dqn', {})
        
        defender = RainbowDQNAgent(
            obs_dim=dqn_config.get('obs_dim', 15),
            action_dim=dqn_config.get('action_dim', 6),
            max_units=dqn_config.get('max_units', 8),
            lr=dqn_config.get('lr', 0.0005),
            gamma=dqn_config.get('gamma', 0.95),
            memory_size=dqn_config.get('memory_size', 20000),
            batch_size=dqn_config.get('batch_size', 64),
            device=str(device)
        )
    
    print(f"攻击方: {config.get('p1_strategy')}")
    print(f"防守方: {config.get('p0_strategy')}")
    print(f"欺骗模式: {config.get('deception_mode')}")
    
    # 快速评估（无训练）
    print("进行快速评估...")
    num_tests = 5  # 少量测试快速查看效果
    
    results = []
    for i in range(num_tests):
        result = run_single_game(attacker, defender, env, config)
        results.append(result)
        print(f"  游戏{i+1}: 攻方胜利={result['attacker_success']}, 步数={result['total_steps']}")
    
    # 计算统计
    attack_successes = sum(1 for r in results if r['attacker_success'])
    attack_success_rate = attack_successes / len(results)
    avg_attacker_reward = np.mean([r['attacker_reward'] for r in results])
    avg_defender_reward = np.mean([r['defender_reward'] for r in results])
    avg_steps = np.mean([r['total_steps'] for r in results])
    
    summary = {
        'config_file': config_path,
        'deception_mode': config.get('deception_mode'),
        'attack_success_rate': attack_success_rate,
        'avg_attacker_reward': avg_attacker_reward,
        'avg_defender_reward': avg_defender_reward,
        'avg_game_length': avg_steps,
        'num_tests': num_tests
    }
    
    print(f"\n[RESULTS] 快速结果:")
    print(f"   攻击方胜率: {attack_success_rate:.1%}")
    print(f"   攻击方奖励: {avg_attacker_reward:.1f}")
    print(f"   防守方奖励: {avg_defender_reward:.1f}")
    print(f"   平均步数: {avg_steps:.1f}")
    
    return summary

def run_single_game(attacker, defender, env, config) -> Dict:
    """运行单局游戏 - 修复后的同步攻防交战逻辑"""
    
    obs = env.reset()
    done = False
    step = 0
    
    attacker_total_reward = 0
    defender_total_reward = 0
    
    max_steps = config.get('duration', 1000)
    last_info: Dict[str, Any] = {}
    
    while not done and step < max_steps:
        step += 1
        
        # 每步都需要攻防双方的动作（同步博弈）
        # 攻击方选择动作
        if hasattr(attacker, 'pre'):
            att_action = attacker.pre(env.tick, obs)
        elif hasattr(attacker, 'act'):
            att_action = attacker.act(obs, training=False)
        else:
            att_action = env.action_space.sample()
            
        # 防守方选择动作
        if hasattr(defender, 'pre'):
            def_action = defender.pre(env.tick, obs)  
        elif hasattr(defender, 'act'):
            def_action = defender.act(obs, training=False)
        else:
            def_action = env.action_space.sample()
        
        # 转换动作格式为 (action_id, n_units)
        if isinstance(att_action, int):
            att_action = (att_action, 1)
        elif not isinstance(att_action, tuple):
            att_action = (0, 1)
            
        if isinstance(def_action, int):
            def_action = (def_action, 1)
        elif not isinstance(def_action, tuple):
            def_action = (0, 1)
        
        # 构建双方动作格式
        combined_action = (att_action, def_action)
        
        # 执行动作
        try:
            next_obs, reward, done, info = env.step(combined_action)
            last_info = info
        except Exception as e:
            print(f"步骤执行错误: {e}, action: {combined_action}")
            break
        
        # 修复后的奖励分配逻辑
        # 环境返回的reward是从博弈角度计算的总体奖励
        # 我们需要根据交战结果正确分配给攻防双方
        if 'engagement_result' in info:
            result = info['engagement_result']
            if result.success:  # 攻击成功
                attacker_total_reward += abs(reward)  # 攻击方得到正奖励
                defender_total_reward -= abs(reward)  # 防守方得到负奖励
            else:  # 防守成功
                defender_total_reward += abs(reward)  # 防守方得到正奖励  
                attacker_total_reward -= abs(reward)  # 攻击方得到负奖励
        else:
            # 备用逻辑：基于reward的符号
            if reward > 0:
                attacker_total_reward += reward
                defender_total_reward -= reward
            else:
                defender_total_reward += abs(reward)
                attacker_total_reward -= abs(reward)
        
        obs = next_obs
    
    # 判断胜负
    winner = last_info.get('winner')
    if winner is None:
        if attacker_total_reward > defender_total_reward:
            winner = 'attacker'
        elif attacker_total_reward < defender_total_reward:
            winner = 'defender'
        else:
            winner = 'draw'
    attacker_success = winner == 'attacker'
    
    return {
        'attacker_reward': attacker_total_reward,
        'defender_reward': defender_total_reward,
        'winner': winner,
        'attacker_success': attacker_success,
        'total_steps': step
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='快速单实验运行')
    parser.add_argument('config', help='配置文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    start_time = time.time()
    results = run_quick_experiment(args.config)
    end_time = time.time()
    
    print(f"\n[TIME] 用时: {end_time - start_time:.1f}秒")

