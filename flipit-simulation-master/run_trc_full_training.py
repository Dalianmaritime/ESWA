#!/usr/bin/env python3
"""
TRC论文完整DRL训练实验系统
真正进行充分的DRL训练，符合TRC期刊科研标准
"""

import os
import sys
import json
import yaml
import time
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../gym-flipit-master'))

# 导入环境和策略
import gym_flipit.envs
from gym_flipit.envs.maritime_nontraditional_env import MaritimeNontraditionalEnv

# 导入核心算法
from strategies.rainbow_dqn_standalone import RainbowDQNAgent  
from strategies.MaritimeDeceptiveGreedy import MaritimeDeceptiveGreedy

class TRCFullTrainingExperiment:
    """TRC论文完整DRL训练实验"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = self._setup_results_directory()
        
        # 训练统计
        self.training_history = []
        self.evaluation_history = []
        self.learning_curves = defaultdict(list)
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEVICE] 使用设备: {self.device}")
        
    def _load_config(self) -> Dict:
        """加载配置并设置随机种子"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'random_seed' in config:
            seed = config['random_seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
        return config
        
    def _setup_results_directory(self) -> Path:
        """设置结果目录"""
        experiment_id = self.config.get('experiment_id', 'trc_training_experiment')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = Path("results") / f"{experiment_id}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(results_dir / "config.yml", 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
        return results_dir
        
    def run_full_training_experiment(self) -> Dict:
        """运行完整的DRL训练实验"""
        
        print(f"[START] TRC完整训练实验: {self.config.get('experiment_id')}")
        print(f"[CONFIG] 训练轮数: {self.config.get('drl_config', {}).get('training_episodes', 1500)}")
        print(f"[CONFIG] 欺骗模式: {self.config.get('deception_mode', 'cheat')}")
        
        # 创建环境
        env = MaritimeNontraditionalEnv(self.config_path)
        
        # 自动获取观察空间维度
        if hasattr(env.observation_space, 'shape'):
            actual_obs_dim = env.observation_space.shape[0]
            print(f"[ENV] 检测到观察空间维度: {actual_obs_dim}")
        else:
            actual_obs_dim = 15
            print(f"[ENV] 无法检测观察空间维度，使用默认值: {actual_obs_dim}")
        
        # 创建策略
        attacker = MaritimeDeceptiveGreedy(
            move_cost=15, cheat_cost=5, debug=False
        )
        
        drl_config = self.config.get('drl_config', {})
        dqn_config = drl_config.get('dqn', {})
        
        defender = RainbowDQNAgent(
            obs_dim=actual_obs_dim,
            action_dim=dqn_config.get('action_dim', 5),
            max_units=dqn_config.get('max_units', 8),
            lr=dqn_config.get('lr', 0.0001),
            gamma=dqn_config.get('gamma', 0.98),
            memory_size=dqn_config.get('memory_size', 100000),
            batch_size=dqn_config.get('batch_size', 256),
            device=str(self.device)
        )
        
        print(f"[AGENT] 攻击方: 贪心算法")
        print(f"[AGENT] 防守方: Rainbow DQN")
        
        # 开始训练
        training_episodes = drl_config.get('training_episodes', 2000)
        evaluation_frequency = drl_config.get('evaluation_frequency', 50)
        
        print(f"[TRAINING] 开始DRL训练，共{training_episodes}轮...")
        
        best_defender_performance = float('-inf')
        
        for episode in range(training_episodes):
            episode_start_time = time.time()
            
            # 训练一轮
            episode_results = self._run_training_episode(attacker, defender, env, episode)
            
            self.training_history.append(episode_results)
            
            # 记录学习曲线
            self.learning_curves['episode'].append(episode)
            self.learning_curves['defender_reward'].append(episode_results['defender_reward'])
            self.learning_curves['attacker_reward'].append(episode_results['attacker_reward'])
            self.learning_curves['game_length'].append(episode_results['game_length'])
            self.learning_curves['epsilon'].append(getattr(defender, 'epsilon', 0.0))
            
            # 定期评估
            if episode % evaluation_frequency == 0:
                eval_results = self._evaluate_performance(attacker, defender, env, num_episodes=10)
                
                avg_defender_reward = np.mean([r['defender_occupation_reward'] for r in eval_results])
                avg_attacker_success_rate = np.mean([r['attacker_success'] for r in eval_results])
                
                self.evaluation_history.append({
                    'episode': episode,
                    'avg_defender_reward': avg_defender_reward,
                    'avg_attacker_success_rate': avg_attacker_success_rate,
                    'evaluation_results': eval_results
                })
                
                print(f"[EVAL] Episode {episode:4d}: 防守方奖励={avg_defender_reward:8.1f}, 攻击方胜率={avg_attacker_success_rate:.1%}")
                
                # 保存最佳模型
                if avg_defender_reward > best_defender_performance:
                    best_defender_performance = avg_defender_reward
                    self._save_best_model(defender, episode, avg_defender_reward)
                    
        # 最终评估
        print(f"[FINAL] 进行最终性能评估...")
        final_results = self._evaluate_performance(attacker, defender, env, num_episodes=20)
        
        # 生成完整结果
        complete_results = self._generate_complete_results(final_results)
        
        # 保存所有数据
        self._save_all_results(complete_results)
        
        # 生成可视化
        self._create_training_visualizations()
        
        return complete_results
        
    def _run_training_episode(self, attacker, defender, env, episode: int) -> Dict:
        """运行单个训练回合"""
        
        obs = env.reset()
        done = False
        step = 0
        
        episode_defender_reward = 0
        episode_attacker_reward = 0
        episode_defender_occupation_reward = 0  # 纯净占领奖励跟踪
        episode_attacker_occupation_reward = 0   # 纯净占领奖励跟踪
        
        max_steps = self.config.get('duration', 800)
        last_info: Dict[str, Any] = {}
        
        while not done and step < max_steps:
            step += 1
            
            # 攻击方选择动作
            if hasattr(attacker, 'pre'):
                att_action = attacker.pre(env.tick, obs)
            else:
                att_action = env.action_space.sample()
                
            # 防守方选择动作（训练模式）
            if hasattr(defender, 'select_action'):
                def_action = defender.select_action(obs, training=True)
            else:
                def_action = env.action_space.sample()
            
            # 转换格式
            if isinstance(att_action, int):
                # 兼容旧策略返回int的情况
                att_action = (0, 0) if att_action == 0 else (att_action, 1)
            elif isinstance(att_action, tuple) and len(att_action) == 2:
                # 新策略直接返回 (action_id, units)
                # 需要重新映射动作ID，以匹配环境定义
                # Greedy策略定义：1->Inflatable(Env:0), 2->Boarding(Env:2), 3->Standoff(Env:3)
                greedy_id, units = att_action
                if greedy_id == 0:
                    env_id = 0
                    units = 0
                elif greedy_id == 1:
                    env_id = 0 # Inflatable
                elif greedy_id == 2:
                    env_id = 2 # Boarding
                elif greedy_id == 3:
                    env_id = 3 # Standoff
                else:
                    env_id = 0
                att_action = (env_id, units)
            else:
                att_action = (0, 1)
                
            if not isinstance(def_action, tuple):
                # 解码Rainbow DQN的离散动作ID (0-19) -> (type, units)
                # 假设 action_dim=20, 5种动作 * 4种单位数量(1-4)
                # 注意：这需要与ActionAdapter保持一致，或者在此处硬编码解码逻辑
                max_units = 4 
                type_id = def_action // max_units
                units = (def_action % max_units) + 1
                def_action = (type_id, units)
            
            combined_action = (att_action, def_action)
            
            # 执行动作
            try:
                next_obs, reward, done, info = env.step(combined_action)
                last_info = info
            except Exception as e:
                print(f"[ERROR] 训练步骤错误: {e}")
                break
            
            # 使用纯占领奖励进行学习曲线记录（用于算法对比）
            if 'pure_occupation_reward' in info:
                # 从环境获取纯占领奖励
                pure_reward = info['pure_occupation_reward']
                
                # 根据当前控制者分配纯占领奖励
                if hasattr(env, 'current_controller'):
                    if env.current_controller == 0:  # 防守方控制
                        step_def_occupation = pure_reward
                        step_att_occupation = 0
                    else:  # 攻击方控制
                        step_att_occupation = pure_reward
                        step_def_occupation = 0
                else:
                    # 备用逻辑：基于奖励符号
                    if pure_reward > 0:
                        step_att_occupation = pure_reward
                        step_def_occupation = 0
                    else:
                        step_def_occupation = abs(pure_reward)
                        step_att_occupation = 0
            else:
                # 备用逻辑：估算占领奖励
                if 'engagement_result' in info:
                    result = info['engagement_result']
                    if result.success:  # 攻击成功
                        step_att_occupation = 1.0
                        step_def_occupation = 0
                    else:  # 防守成功
                        step_def_occupation = 1.0
                        step_att_occupation = 0
                else:
                    # 默认防守方获得占领奖励
                    step_def_occupation = 1.0
                    step_att_occupation = 0
            
            # 累计占领奖励（用于统计）
            episode_defender_occupation_reward += step_def_occupation
            episode_attacker_occupation_reward += step_att_occupation
            
            # DQN学习：只使用防守方的训练奖励
            training_reward = info.get('training_reward', reward)
            episode_defender_reward += training_reward  # 防守方累计训练奖励
            episode_attacker_reward += -training_reward  # 攻击方获得相反奖励（仅用于记录）
            
            # DQN学习：防守方使用训练奖励进行学习
            if hasattr(defender, 'store_transition'):
                defender.store_transition(obs, def_action, training_reward, next_obs, done)
                
                # 开始学习
                learning_starts = self.config.get('drl_config', {}).get('dqn', {}).get('learning_starts', 2000)
                if len(defender.memory) > learning_starts:
                    if hasattr(defender, 'update'):
                        loss_info = defender.update()
                        if loss_info and 'loss' in loss_info:
                            self.learning_curves['loss'].append(loss_info['loss'])
            
            obs = next_obs

        winner = last_info.get('winner')
        termination_reason = last_info.get('failure_reason') or last_info.get('winner_determination')

        if winner is None:
            if episode_attacker_occupation_reward > episode_defender_occupation_reward:
                winner = 'attacker'
            elif episode_attacker_occupation_reward < episode_defender_occupation_reward:
                winner = 'defender'
            else:
                winner = 'draw'
            if termination_reason is None:
                termination_reason = 'manual_score_comparison'
            
        return {
            'episode': episode,
            'defender_reward': episode_defender_reward,  # DRL学习用的完整奖励
            'attacker_reward': episode_attacker_reward,  # DRL学习用的完整奖励
            'defender_occupation_reward': episode_defender_occupation_reward,  # 纯净占领奖励（统计用）
            'attacker_occupation_reward': episode_attacker_occupation_reward,  # 纯净占领奖励（统计用）
            'game_length': step,
            'winner': winner,
            'termination_reason': termination_reason,
            'attacker_success': winner == 'attacker'
        }
        
    def _evaluate_performance(self, attacker, defender, env, num_episodes: int) -> List[Dict]:
        """评估性能"""
        results = []
        
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            step = 0
            
            att_total_reward = 0
            def_total_reward = 0
            
            max_steps = self.config.get('duration', 800)
            last_info: Dict[str, Any] = {}
            
            while not done and step < max_steps:
                step += 1
                
                # 攻击方动作
                if hasattr(attacker, 'pre'):
                    att_action = attacker.pre(env.tick, obs)
                else:
                    att_action = env.action_space.sample()
                    
                # 防守方动作（评估模式，不训练）
                if hasattr(defender, 'select_action'):
                    def_action = defender.select_action(obs, training=False)
                else:
                    def_action = env.action_space.sample()
                
                # 格式转换
                if isinstance(att_action, int):
                    att_action = (0, 0) if att_action == 0 else (att_action, 1)
                elif not isinstance(att_action, tuple):
                    att_action = (0, 1)
                    
                if not isinstance(def_action, tuple):
                    def_action = (def_action, 1)
                
                combined_action = (att_action, def_action)
                
                try:
                    next_obs, reward, done, info = env.step(combined_action)
                    last_info = info
                except:
                    break
                
                # 纯净占领奖励统计（仅用于算法对比）
                # 只记录占领奖励，不包含系统给予预算和行动成本
                if 'pure_occupation_reward' in info:
                    # 使用环境提供的纯净占领奖励
                    pure_reward = info['pure_occupation_reward']
                    if env.current_controller == 1:  # 攻击方控制
                        att_total_reward += pure_reward
                    else:  # 防守方控制 (current_controller == 0)
                        def_total_reward += pure_reward
                else:
                    # 备用：基于控制权计算纯净占领奖励
                    if 'engagement_result' in info:
                        result = info['engagement_result']
                        if result.success:
                            att_total_reward += 1.0  # 攻击方占领奖励
                        else:
                            def_total_reward += 1.0  # 防守方占领奖励
                
                obs = next_obs

            winner = last_info.get('winner')
            termination_reason = last_info.get('failure_reason') or last_info.get('winner_determination')

            if winner is None:
                if att_total_reward > def_total_reward:
                    winner = 'attacker'
                elif att_total_reward < def_total_reward:
                    winner = 'defender'
                else:
                    winner = 'draw'
                if termination_reason is None:
                    termination_reason = 'manual_score_comparison'
                
            results.append({
                'attacker_occupation_reward': att_total_reward,  # 攻击方纯净占领奖励
                'defender_occupation_reward': def_total_reward,  # 防守方纯净占领奖励
                'winner': winner,
                'termination_reason': termination_reason,
                'attacker_success': winner == 'attacker',
                'total_steps': step,
                'occupation_advantage': att_total_reward - def_total_reward,  # 占领优势差
                'total_occupation_rewards': att_total_reward + def_total_reward  # 总占领奖励
            })
            
        return results
        
    def _save_best_model(self, defender, episode: int, performance: float):
        """保存最佳模型"""
        model_dir = self.results_dir / "best_model"
        model_dir.mkdir(exist_ok=True)
        
        if hasattr(defender, 'save'):
            model_path = model_dir / f"best_defender_episode_{episode}.pth"
            defender.save(str(model_path))
            
            # 保存性能记录
            perf_record = {
                'episode': episode,
                'performance': performance,
                'model_path': str(model_path),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(model_dir / "performance_record.json", 'w') as f:
                json.dump(perf_record, f, indent=2)
                
    def _generate_complete_results(self, final_evaluation: List[Dict]) -> Dict:
        """生成完整结果 (包含科学统计指标)"""
        
        # 计算最终统计（基于纯净占领奖励）
        n_samples = len(final_evaluation)
        attack_successes = sum(1 for r in final_evaluation if r['attacker_success'])
        attack_success_rate = attack_successes / n_samples
        
        # 基础均值
        avg_attacker_occupation = np.mean([r['attacker_occupation_reward'] for r in final_evaluation])
        avg_defender_occupation = np.mean([r['defender_occupation_reward'] for r in final_evaluation])
        avg_steps = np.mean([r['total_steps'] for r in final_evaluation])
        avg_occupation_advantage = np.mean([r['occupation_advantage'] for r in final_evaluation])
        avg_total_occupation = np.mean([r['total_occupation_rewards'] for r in final_evaluation])
        
        # 科学统计指标：标准差 (稳定性)
        std_attacker_occupation = np.std([r['attacker_occupation_reward'] for r in final_evaluation])
        std_defender_occupation = np.std([r['defender_occupation_reward'] for r in final_evaluation])
        std_occupation_advantage = np.std([r['occupation_advantage'] for r in final_evaluation])
        
        # 科学统计指标：胜率置信区间 (95% CI)
        # SE = sqrt(p(1-p)/n)
        se_win_rate = np.sqrt(attack_success_rate * (1 - attack_success_rate) / n_samples) if n_samples > 1 else 0
        ci95_win_rate = 1.96 * se_win_rate
        
        # 计算每步平均占领奖励效率
        avg_attacker_per_step = avg_attacker_occupation / avg_steps if avg_steps > 0 else 0
        avg_defender_per_step = avg_defender_occupation / avg_steps if avg_steps > 0 else 0
        
        complete_results = {
            'experiment_info': {
                'experiment_id': self.config.get('experiment_id'),
                'config_path': self.config_path,
                'deception_mode': self.config.get('deception_mode'),
                'training_episodes': self.config.get('drl_config', {}).get('training_episodes'),
                'timestamp': datetime.now().isoformat()
            },
            'final_performance': {
                # 核心均值
                'attacker_success_rate': attack_success_rate,
                'avg_attacker_occupation_reward': avg_attacker_occupation,
                'avg_defender_occupation_reward': avg_defender_occupation,
                'avg_game_length': avg_steps,
                'avg_occupation_advantage': avg_occupation_advantage,
                'avg_total_occupation_rewards': avg_total_occupation,
                'avg_attacker_per_step_occupation': avg_attacker_per_step,
                'avg_defender_per_step_occupation': avg_defender_per_step,
                
                # 科学统计量
                'std_attacker_occupation': std_attacker_occupation,
                'std_defender_occupation': std_defender_occupation,
                'std_occupation_advantage': std_occupation_advantage,
                'win_rate_ci95': ci95_win_rate,
                'sample_size': n_samples
            },
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'learning_curves': dict(self.learning_curves),
            'final_evaluation_details': final_evaluation
        }
        
        return complete_results
        
    def _save_all_results(self, results: Dict):
        """保存所有结果"""
        
        # 主要结果文件
        with open(self.results_dir / "complete_training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # 训练历史（处理bool类型）
        with open(self.results_dir / "training_history.json", 'w') as f:
            json.dump(results['training_history'], f, indent=2, default=str)
            
        # 学习曲线数据
        with open(self.results_dir / "learning_curves.json", 'w') as f:
            json.dump(results['learning_curves'], f, indent=2)
            
        # 生成摘要报告
        summary_report = self._generate_training_summary(results)
        with open(self.results_dir / "training_summary.md", 'w', encoding='utf-8') as f:
            f.write(summary_report)
            
    def _generate_training_summary(self, results: Dict) -> str:
        """生成训练摘要报告 (ESWA科研标准)"""
        
        info = results['experiment_info']
        perf = results['final_performance']
        
        # 分析学习趋势
        training_hist = results['training_history']
        if len(training_hist) >= 100:
            early_perf = np.mean([ep['defender_reward'] for ep in training_hist[:50]])
            late_perf = np.mean([ep['defender_reward'] for ep in training_hist[-50:]])
            learning_improvement = late_perf - early_perf
            
            early_att_rate = np.mean([ep['attacker_success'] for ep in training_hist[:50]])
            late_att_rate = np.mean([ep['attacker_success'] for ep in training_hist[-50:]])
        else:
            learning_improvement = 0
            early_att_rate = late_att_rate = 0
            
        # 使用字符串格式化避免f-string问题
        report = """# TRC完整DRL训练实验报告 (ESWA科研标准)

## 📊 实验概况
- **实验ID**: {}
- **欺骗模式**: {}
- **样本量(N)**: {} (最终评估局数)
- **训练轮数**: {}

## 🎯 核心性能指标 (Mean ± Std)

### 1. 占领奖励 (Occupation Reward)
*反映控制权的总体分布*
- **攻击方**: {:.2f} ± {:.2f}
- **防守方**: {:.2f} ± {:.2f}
- **优势差 (Advantage)**: {:.2f} ± {:.2f} (正值代表攻击方优势)

### 2. 胜率分析 (Win Rate Analysis)
- **攻击方胜率**: {:.1%} ± {:.1%} (95% CI)
- **博弈长度**: {:.1f}步

### 3. 效率指标 (Efficiency)
- **攻击方每步收益**: {:.3f}
- **防守方每步收益**: {:.3f}

## 📈 学习动态 (Learning Dynamics)
- **策略改进幅度**: {:.2f} (收敛值 - 初始值)
- **攻击成功率演变**: {:.1%} (Early) -> {:.1%} (Late)

## 💡 科学性解读
- **稳定性分析**: 优势差的标准差为 {:.2f}，体现了对抗过程的波动性。
- **统计显著性**: 胜率置信区间 ±{:.1%} 表明了结果的可靠程度。

---
*报告生成时间: {}*
""".format(
            info.get('experiment_id', 'N/A'),
            info.get('deception_mode', 'N/A'),
            perf.get('sample_size', 0),
            info.get('training_episodes', 'N/A'),
            
            perf['avg_attacker_occupation_reward'], perf['std_attacker_occupation'],
            perf['avg_defender_occupation_reward'], perf['std_defender_occupation'],
            perf['avg_occupation_advantage'], perf['std_occupation_advantage'],
            
            perf['attacker_success_rate'], perf['win_rate_ci95'],
            perf['avg_game_length'],
            
            perf['avg_attacker_per_step_occupation'],
            perf['avg_defender_per_step_occupation'],
            
            learning_improvement,
            early_att_rate,
            late_att_rate,
            
            perf['std_occupation_advantage'],
            perf['win_rate_ci95'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        if learning_improvement > 0:
            report += """
### DRL学习能力验证 [CONFIRMED]
- DRL算法展现了明显的学习改进
- 训练过程中防守策略不断优化
- 证明了深度强化学习的有效性
"""
        
        if 0.2 <= perf['attacker_success_rate'] <= 0.6:
            report += """
### 平衡性验证 [CONFIRMED]
- 攻击方胜率在合理范围内（20%-60%）
- 体现了真实海事安全威胁
- 防守方占优但攻击方有合理成功机会
"""
        
        report += """
## 📁 数据完整性
- **训练历史记录**: {} 轮
- **评估记录**: {} 次
- **学习曲线数据**: 完整保存
- **最终评估**: {} 次测试

结果保存在: {}
""".format(
            len(results['training_history']),
            len(results['evaluation_history']),
            len(results['final_evaluation_details']),
            self.results_dir
        )
        
        return report
        
    def _create_training_visualizations(self):
        """创建训练过程可视化"""
        
        if not self.learning_curves['episode']:
            print("[WARNING] 无学习曲线数据，跳过可视化")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('TRC DRL Training Analysis - Occupation Reward Focus', fontsize=16)

        episodes = self.learning_curves['episode']
        
        # 防守方占领奖励学习曲线
        if 'defender_reward' in self.learning_curves:
            axes[0, 0].plot(episodes, self.learning_curves['defender_reward'], alpha=0.7)
            axes[0, 0].set_title('Defender Occupation Reward Learning')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Defender Occupation Reward')
            axes[0, 0].grid(True)
        
        # 攻击方占领奖励学习曲线
        if 'attacker_reward' in self.learning_curves:
            axes[0, 1].plot(episodes, self.learning_curves['attacker_reward'], alpha=0.7, color='red')
            axes[0, 1].set_title('Attacker Occupation Reward Learning')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Attacker Occupation Reward')
            axes[0, 1].grid(True)
            
        # 3. 博弈长度变化
        if 'game_length' in self.learning_curves:
            axes[1, 0].plot(episodes, self.learning_curves['game_length'], alpha=0.7, color='green')
            axes[1, 0].set_title('Game Length Evolution')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Steps per Game')
            axes[1, 0].grid(True)
            
        # 4. 探索率衰减
        if 'epsilon' in self.learning_curves:
            axes[1, 1].plot(episodes, self.learning_curves['epsilon'], alpha=0.7, color='purple')
            axes[1, 1].set_title('Exploration Rate Decay')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(self.results_dir / "occupation_reward_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / "occupation_reward_analysis.pdf", bbox_inches='tight')

        print(f"[SAVE] 占领奖励可视化已保存")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TRC完整DRL训练实验')
    parser.add_argument('config', help='配置文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"[ERROR] 配置文件不存在: {args.config}")
        return
        
    # 运行完整训练实验
    experiment = TRCFullTrainingExperiment(args.config)
    results = experiment.run_full_training_experiment()
    
    # 打印关键结果
    perf = results['final_performance']
    print(f"\n" + "="*60)
    print("TRC完整训练实验最终结果")
    print("="*60)
    print(f"攻击方胜率: {perf['attacker_success_rate']:.1%}")
    print(f"攻击方平均占领奖励: {perf['avg_attacker_occupation_reward']:.2f}")
    print(f"防守方平均占领奖励: {perf['avg_defender_occupation_reward']:.2f}")
    print(f"平均博弈长度: {perf['avg_game_length']:.1f}步")
    print(f"结果保存在: {experiment.results_dir}")

if __name__ == "__main__":
    main()

