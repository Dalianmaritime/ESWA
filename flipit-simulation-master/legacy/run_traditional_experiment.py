#!/usr/bin/env python3
"""
传统算法完整实验脚本 - 用于基线对比
运行多次实验并保存完整结果
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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../gym-flipit-master'))

# 导入环境和策略
import gym_flipit.legacy
from gym_flipit.legacy.maritime_nontraditional_env import MaritimeNontraditionalEnv

# 导入核心算法
from strategies.MaritimeDeceptiveGreedy import MaritimeDeceptiveGreedy
from strategies.WeakDefensiveGreedy import WeakDefensiveGreedy
from strategies.AggressiveAttacker import AggressiveAttacker

class TraditionalExperiment:
    """传统算法完整实验"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = self._setup_results_directory()
        
        # 实验统计
        self.experiment_results = []
        
    def _load_config(self) -> Dict:
        """加载配置并设置随机种子"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'random_seed' in config:
            seed = config['random_seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
                
        return config
        
    def _setup_results_directory(self) -> Path:
        """设置结果目录"""
        experiment_id = self.config.get('experiment_id', 'traditional_experiment')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = Path("results") / f"{experiment_id}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(results_dir / "config.yml", 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
        return results_dir
        
    def run_complete_experiment(self, num_episodes: int = 100) -> Dict:
        """运行完整传统算法实验"""
        
        print(f"[START] 传统算法实验: {self.config.get('experiment_id')}")
        print(f"[CONFIG] 实验轮数: {num_episodes}")
        print(f"[CONFIG] 欺骗模式: {self.config.get('deception_mode', 'cheat')}")
        
        # 创建环境
        env = MaritimeNontraditionalEnv(self.config_path)
        
        # 创建攻击方策略 - FlipIt模式使用激进攻击策略
        deception_mode = self.config.get('deception_mode', 'cheat')
        if deception_mode == 'flipit':
            print(f"[STRATEGY] 攻击方使用激进策略（FlipIt模式）")
            attacker = AggressiveAttacker(
                move_cost=self.config.get('p1_config', {}).get('move_cost', 15),
                cheat_cost=5,
                debug=False
            )
        else:
            attacker = MaritimeDeceptiveGreedy(
                move_cost=self.config.get('p1_config', {}).get('move_cost', 15),
                cheat_cost=5,
                debug=False
            )
        
        # 创建防守方策略 - 使用弱化防守策略
        defender_config = self.config.get('p0_config', {})
        
        # 使用弱化的防守策略（让攻击方更容易获胜）
        print(f"[STRATEGY] 防守方使用弱化策略，让攻击方有更高胜率")
        defender = WeakDefensiveGreedy(
            move_cost=defender_config.get('move_cost', 10),
            cheat_cost=5,
            debug=False
        )
        
        print(f"[AGENT] 攻击方: {self.config.get('p1_strategy')}")
        print(f"[AGENT] 防守方: {self.config.get('p0_strategy')}")
        
        # 运行多次实验
        print(f"[RUNNING] 开始运行 {num_episodes} 次实验...")
        
        for episode in range(num_episodes):
            if (episode + 1) % 20 == 0:
                print(f"  [PROGRESS] 已完成 {episode + 1}/{num_episodes} 次实验")
            
            result = self._run_single_game(attacker, defender, env)
            result['episode'] = episode
            self.experiment_results.append(result)
        
        # 生成完整结果
        complete_results = self._generate_complete_results()
        
        # 保存结果
        self._save_all_results(complete_results)
        
        return complete_results
        
    def _run_single_game(self, attacker, defender, env) -> Dict:
        """运行单局游戏 - 修正版：正确追踪占领奖励"""
        
        obs = env.reset()
        done = False
        step = 0
        
        # 追踪占领奖励（与DRL实验一致）
        defender_occupation_reward = 0
        attacker_occupation_reward = 0
        
        # 追踪总奖励（包括成本）
        attacker_total_reward = 0
        defender_total_reward = 0
        
        max_steps = self.config.get('duration', 800)
        
        # 获取占领奖励配置
        occupation_reward_per_step = self.config.get('rew_config', {}).get('occupation_reward', 2.0)
        
        while not done and step < max_steps:
            step += 1
            
            # 记录当前控制方（在动作执行前）
            controller_before = env.current_controller
            
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
                # 无动作或最大可负担单位（上限4）
                att_action = (0, 0) if att_action == 0 else (att_action, 4)
            elif not isinstance(att_action, tuple):
                att_action = (0, 1)
                
            if isinstance(def_action, int):
                def_action = (def_action, 4)
            elif not isinstance(def_action, tuple):
                def_action = (0, 1)
            
            # 构建双方动作格式
            combined_action = (att_action, def_action)
            
            # 执行动作
            try:
                next_obs, reward, done, info = env.step(combined_action)
            except Exception as e:
                print(f"步骤执行错误: {e}")
                break
            
            # 记录当前控制方（动作执行后）
            controller_after = env.current_controller
            
            # 累积占领奖励（关键修正）
            if controller_after == 0:  # 防守方控制
                defender_occupation_reward += occupation_reward_per_step
            else:  # 攻击方控制
                attacker_occupation_reward += occupation_reward_per_step
            
            # 累积总奖励（包括成本，用于参考）
            if 'engagement_result' in info:
                result = info['engagement_result']
                if result.success:  # 攻击成功
                    attacker_total_reward += abs(reward)
                    defender_total_reward -= abs(reward)
                else:  # 防守成功
                    defender_total_reward += abs(reward)
                    attacker_total_reward -= abs(reward)
            else:
                if reward > 0:
                    attacker_total_reward += reward
                    defender_total_reward -= reward
                else:
                    defender_total_reward += abs(reward)
                    attacker_total_reward -= abs(reward)
            
            obs = next_obs
        
        # 判断胜负（基于占领奖励，与DRL实验一致）
        attacker_success = attacker_occupation_reward > defender_occupation_reward
        
        return {
            'attacker_reward': attacker_total_reward,
            'defender_reward': defender_total_reward,
            'attacker_occupation_reward': attacker_occupation_reward,
            'defender_occupation_reward': defender_occupation_reward,
            'attacker_success': attacker_success,
            'total_steps': step
        }
        
    def _generate_complete_results(self) -> Dict:
        """生成完整结果 - 修正版：包含占领奖励统计"""
        
        # 计算统计
        attack_successes = sum(1 for r in self.experiment_results if r['attacker_success'])
        attack_success_rate = attack_successes / len(self.experiment_results)
        
        # 总奖励统计（包括成本）
        avg_attacker_reward = np.mean([r['attacker_reward'] for r in self.experiment_results])
        avg_defender_reward = np.mean([r['defender_reward'] for r in self.experiment_results])
        std_attacker_reward = np.std([r['attacker_reward'] for r in self.experiment_results])
        std_defender_reward = np.std([r['defender_reward'] for r in self.experiment_results])
        
        # 占领奖励统计（与DRL实验一致的度量标准）
        avg_attacker_occupation = np.mean([r['attacker_occupation_reward'] for r in self.experiment_results])
        avg_defender_occupation = np.mean([r['defender_occupation_reward'] for r in self.experiment_results])
        std_attacker_occupation = np.std([r['attacker_occupation_reward'] for r in self.experiment_results])
        std_defender_occupation = np.std([r['defender_occupation_reward'] for r in self.experiment_results])
        
        # 博弈长度统计
        avg_steps = np.mean([r['total_steps'] for r in self.experiment_results])
        std_steps = np.std([r['total_steps'] for r in self.experiment_results])
        
        complete_results = {
            'experiment_info': {
                'experiment_id': self.config.get('experiment_id'),
                'config_path': self.config_path,
                'deception_mode': self.config.get('deception_mode'),
                'num_episodes': len(self.experiment_results),
                'timestamp': datetime.now().isoformat()
            },
            'final_performance': {
                'attacker_success_rate': attack_success_rate,
                # 总奖励（包括成本）
                'avg_attacker_reward': avg_attacker_reward,
                'avg_defender_reward': avg_defender_reward,
                'std_attacker_reward': std_attacker_reward,
                'std_defender_reward': std_defender_reward,
                # 占领奖励（与DRL实验可对比）
                'avg_attacker_occupation_reward': avg_attacker_occupation,
                'avg_defender_occupation_reward': avg_defender_occupation,
                'std_attacker_occupation_reward': std_attacker_occupation,
                'std_defender_occupation_reward': std_defender_occupation,
                # 博弈长度
                'avg_game_length': avg_steps,
                'std_game_length': std_steps
            },
            'detailed_results': self.experiment_results
        }
        
        return complete_results
        
    def _save_all_results(self, results: Dict):
        """保存所有结果"""
        
        # 主要结果文件
        with open(self.results_dir / "complete_training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # 生成摘要报告
        summary_report = self._generate_summary(results)
        with open(self.results_dir / "training_summary.md", 'w', encoding='utf-8') as f:
            f.write(summary_report)
            
    def _generate_summary(self, results: Dict) -> str:
        """生成摘要报告 - 修正版：包含占领奖励统计"""
        
        info = results['experiment_info']
        perf = results['final_performance']
        
        report = f"""# TRC传统算法基线实验报告

## 实验配置
- **实验ID**: {info['experiment_id']}
- **实验轮数**: {info['num_episodes']}
- **欺骗模式**: {info['deception_mode']}
- **完成时间**: {info['timestamp'][:19]}

## 最终性能

### 占领奖励（与DRL实验可对比）
- **攻击方胜率**: {perf['attacker_success_rate']:.1%}
- **攻击方平均占领奖励**: {perf['avg_attacker_occupation_reward']:.2f} ± {perf['std_attacker_occupation_reward']:.2f}
- **防守方平均占领奖励**: {perf['avg_defender_occupation_reward']:.2f} ± {perf['std_defender_occupation_reward']:.2f}
- **占领优势**: {perf['avg_defender_occupation_reward'] - perf['avg_attacker_occupation_reward']:.2f}

### 总奖励（包括成本，仅供参考）
- **攻击方平均总奖励**: {perf['avg_attacker_reward']:.2f} ± {perf['std_attacker_reward']:.2f}
- **防守方平均总奖励**: {perf['avg_defender_reward']:.2f} ± {perf['std_defender_reward']:.2f}

### 博弈复杂度
- **平均博弈长度**: {perf['avg_game_length']:.1f} ± {perf['std_game_length']:.1f} 步

## 科学价值
- 建立传统算法性能基线
- 为DRL算法对比提供参考
- 验证Cheat/FlipIt机制对传统算法的影响
- **关键**：占领奖励是与DRL实验直接可对比的核心指标

---
结果保存在: {self.results_dir}
"""
        
        return report

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='传统算法完整实验')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--episodes', '-n', type=int, default=100, help='实验轮数')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"[ERROR] 配置文件不存在: {args.config}")
        return
        
    # 运行完整实验
    experiment = TraditionalExperiment(args.config)
    results = experiment.run_complete_experiment(num_episodes=args.episodes)
    
    # 打印关键结果
    perf = results['final_performance']
    print(f"\n" + "="*60)
    print("TRC传统算法实验最终结果")
    print("="*60)
    print(f"攻击方胜率: {perf['attacker_success_rate']:.1%}")
    print(f"\n【占领奖励 - 与DRL可对比】")
    print(f"攻击方平均占领奖励: {perf['avg_attacker_occupation_reward']:.2f} ± {perf['std_attacker_occupation_reward']:.2f}")
    print(f"防守方平均占领奖励: {perf['avg_defender_occupation_reward']:.2f} ± {perf['std_defender_occupation_reward']:.2f}")
    print(f"占领优势: {perf['avg_defender_occupation_reward'] - perf['avg_attacker_occupation_reward']:.2f}")
    print(f"\n【总奖励 - 包括成本】")
    print(f"攻击方平均总奖励: {perf['avg_attacker_reward']:.2f} ± {perf['std_attacker_reward']:.2f}")
    print(f"防守方平均总奖励: {perf['avg_defender_reward']:.2f} ± {perf['std_defender_reward']:.2f}")
    print(f"\n平均博弈长度: {perf['avg_game_length']:.1f} ± {perf['std_game_length']:.1f} 步")
    print(f"结果保存在: {experiment.results_dir}")

if __name__ == "__main__":
    main()


