"""
海运非传统安全深度强化学习增强环境

专为DRL训练优化的环境包装器：
- 标准化观察空间
- 优化奖励设计
- 支持连续训练
- 状态表示增强
- 多智能体支持
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque
import copy

from .maritime_nontraditional_env import MaritimeNontraditionalEnv


class MaritimeDRLEnv(gym.Env):
    """
    海运DRL增强环境
    
    对原始环境进行包装和优化，提供：
    - 标准化的观察空间
    - DRL友好的奖励函数
    - 状态历史追踪
    - 自动重置机制
    - 训练统计收集
    """
    
    def __init__(self, config_file: str = None, **kwargs):
        super().__init__()
        
        # 创建基础环境
        self.base_env = MaritimeNontraditionalEnv(config_file=config_file)
        
        # DRL参数
        self.history_length = kwargs.get('history_length', 4)  # 状态历史长度
        self.reward_scale = kwargs.get('reward_scale', 1.0)    # 奖励缩放
        self.normalize_obs = kwargs.get('normalize_obs', True)  # 观察标准化
        self.max_episode_steps = kwargs.get('max_episode_steps', 500)
        
        # 状态历史缓冲区
        self.obs_history = deque(maxlen=self.history_length)
        
        # 统计信息
        self.episode_count = 0
        self.step_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # 观察空间增强
        self._setup_enhanced_observation_space()
        
        # 动作空间（保持与基础环境一致）
        self.action_space = self.base_env.action_space
        
        # 奖励参数
        self.reward_config = {
            'success_reward': 100.0,
            'step_penalty': -0.1,
            'efficiency_bonus': 10.0,
            'exploration_bonus': 5.0,
            'cooperation_bonus': 20.0,
            'early_termination_penalty': -50.0
        }
        
        # 更新奖励配置
        if hasattr(self.base_env, 'config') and self.base_env.config:
            rew_config = self.base_env.config.get('rew_config', {})
            self.reward_config.update(rew_config)
        
        print(f"🌊 初始化海运DRL增强环境")
        print(f"   📏 观察空间: {self.observation_space.shape}")
        print(f"   🎯 动作空间: {self.action_space}")
        print(f"   📚 历史长度: {self.history_length}")
    
    def _setup_enhanced_observation_space(self):
        """设置增强的观察空间"""
        # 基础观察维度
        base_obs_dim = 13  # 原始观察维度
        
        # 增强特征
        enhanced_features = 7  # 额外特征：历史统计、趋势等
        
        # 历史特征：当前观察 + 历史变化
        history_dim = base_obs_dim + 3  # 当前观察 + 3维变化趋势
        
        # 总观察维度
        total_obs_dim = history_dim + enhanced_features
        
        # 观察空间边界
        obs_low = np.full(total_obs_dim, -10.0, dtype=np.float32)
        obs_high = np.full(total_obs_dim, 10.0, dtype=np.float32)
        
        # 设置合理的边界
        # 前13维：基础观察
        obs_low[:13] = [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0]
        obs_high[:13] = [1, 100, 100, 1, 1, 1, 1000, 1.5, 1.5, 1.5, 1.5, 10, 5]
        
        # 变化趋势维度：[-1, 1]
        obs_low[13:16] = -1.0
        obs_high[13:16] = 1.0
        
        # 增强特征维度
        obs_low[16:] = 0.0
        obs_high[16:] = 1.0
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 重置基础环境
        base_obs = self.base_env.reset()
        
        # 重置统计
        if self.step_count > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.step_count)
        
        self.episode_count += 1
        self.step_count = 0
        self.current_episode_reward = 0
        
        # 初始化历史缓冲区
        self.obs_history.clear()
        for _ in range(self.history_length):
            self.obs_history.append(base_obs.copy())
        
        # 生成增强观察
        enhanced_obs = self._enhance_observation(base_obs)
        
        return enhanced_obs.astype(np.float32)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步"""
        # 执行基础环境步骤
        base_obs, base_reward, done, info = self.base_env.step(action)
        
        # 更新统计
        self.step_count += 1
        self.total_steps += 1
        
        # 更新观察历史
        self.obs_history.append(base_obs.copy())
        
        # 生成增强观察
        enhanced_obs = self._enhance_observation(base_obs)
        
        # 计算增强奖励
        enhanced_reward = self._calculate_enhanced_reward(
            base_reward, base_obs, action, done, info
        )
        
        # 缩放奖励
        final_reward = enhanced_reward * self.reward_scale
        self.current_episode_reward += final_reward
        
        # 检查最大步数
        if self.step_count >= self.max_episode_steps:
            done = True
            info['max_steps_reached'] = True
        
        # 更新info
        info.update({
            'base_reward': base_reward,
            'enhanced_reward': enhanced_reward,
            'episode_step': self.step_count,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count
        })
        
        return enhanced_obs.astype(np.float32), final_reward, done, info
    
    def _enhance_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """增强观察"""
        # 确保base_obs是正确的维度
        if len(base_obs) < 13:
            # 填充到13维
            padded_obs = np.zeros(13, dtype=np.float32)
            padded_obs[:len(base_obs)] = base_obs
            base_obs = padded_obs
        
        enhanced_obs = []
        
        # 1. 当前观察（标准化）
        if self.normalize_obs:
            normalized_obs = self._normalize_observation(base_obs[:13])
        else:
            normalized_obs = base_obs[:13]
        enhanced_obs.extend(normalized_obs)
        
        # 2. 历史变化趋势（3维）
        if len(self.obs_history) >= 2:
            current = np.array(self.obs_history[-1][:13])
            previous = np.array(self.obs_history[-2][:13])
            
            # 计算变化率（安全处理除零）
            change = current - previous
            # 预算变化
            budget_change = change[1:3].mean() if len(change) > 2 else 0
            # 成功率变化 
            success_change = change[3:6].mean() if len(change) > 5 else 0
            # 时间进度
            time_progress = current[6] / self.max_episode_steps if len(current) > 6 else 0
            
            enhanced_obs.extend([
                np.tanh(budget_change),      # 预算变化趋势
                np.tanh(success_change),     # 成功率变化趋势
                time_progress                # 时间进度
            ])
        else:
            enhanced_obs.extend([0.0, 0.0, 0.0])
        
        # 3. 增强统计特征（7维）
        # 胜率统计
        recent_wins = self._get_recent_win_rate()
        enhanced_obs.append(recent_wins)
        
        # 效率指标
        efficiency = self._calculate_efficiency()
        enhanced_obs.append(efficiency)
        
        # 探索程度
        exploration_level = self._calculate_exploration_level()
        enhanced_obs.append(exploration_level)
        
        # 资源利用率
        resource_utilization = self._calculate_resource_utilization(base_obs)
        enhanced_obs.append(resource_utilization)
        
        # 环境适应性
        env_adaptation = self._calculate_environmental_adaptation(base_obs)
        enhanced_obs.append(env_adaptation)
        
        # 战术多样性
        tactical_diversity = self._calculate_tactical_diversity()
        enhanced_obs.append(tactical_diversity)
        
        # 时间压力
        time_pressure = self.step_count / self.max_episode_steps
        enhanced_obs.append(time_pressure)
        
        return np.array(enhanced_obs, dtype=np.float32)
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """标准化观察"""
        normalized = obs.copy().astype(np.float32)
        
        # 控制者：0-1
        if len(normalized) > 0:
            normalized[0] = np.clip(normalized[0], 0, 1)
        
        # 预算：0-100 -> 0-1
        if len(normalized) > 2:
            normalized[1:3] = np.clip(normalized[1:3] / 100.0, 0, 1)
        
        # 概率：0-1
        if len(normalized) > 6:
            normalized[3:6] = np.clip(normalized[3:6], 0, 1)
        
        # 时间：0-max_steps -> 0-1
        if len(normalized) > 6:
            normalized[6] = np.clip(normalized[6] / self.max_episode_steps, 0, 1)
        
        # 环境因素：0.5-1.5 -> 0-1
        if len(normalized) > 11:
            normalized[7:11] = np.clip((normalized[7:11] - 0.5) / 1.0, 0, 1)
        
        # 其他因素
        if len(normalized) > 13:
            normalized[11:13] = np.clip(normalized[11:13] / 10.0, 0, 1)
        
        return normalized
    
    def _calculate_enhanced_reward(self, base_reward: float, obs: np.ndarray, 
                                 action, done: bool, info: Dict) -> float:
        """计算增强奖励"""
        enhanced_reward = base_reward
        
        # 步骤惩罚（鼓励更快解决）
        enhanced_reward += self.reward_config['step_penalty']
        
        # 成功奖励
        if done and base_reward > 0:
            enhanced_reward += self.reward_config['success_reward']
            
            # 效率奖励（更快成功获得更多奖励）
            efficiency_bonus = self.reward_config['efficiency_bonus'] * (1 - self.step_count / self.max_episode_steps)
            enhanced_reward += efficiency_bonus
        
        # 早期终止惩罚
        if done and self.step_count < 10:
            enhanced_reward += self.reward_config['early_termination_penalty']
        
        # 探索奖励
        if self._is_exploration_action(action):
            enhanced_reward += self.reward_config['exploration_bonus']
        
        # 合作奖励（多智能体场景）
        if info.get('cooperation_detected', False):
            enhanced_reward += self.reward_config['cooperation_bonus']
        
        # 资源管理奖励
        resource_efficiency = self._calculate_resource_efficiency(obs, info)
        enhanced_reward += resource_efficiency * 10
        
        return enhanced_reward
    
    def _get_recent_win_rate(self, window: int = 50) -> float:
        """获取最近的胜率"""
        if len(self.episode_rewards) < 2:
            return 0.5  # 初始值
        
        recent_rewards = self.episode_rewards[-window:]
        wins = sum(1 for r in recent_rewards if r > 0)
        return wins / len(recent_rewards)
    
    def _calculate_efficiency(self) -> float:
        """计算效率指标"""
        if len(self.episode_lengths) == 0:
            return 0.5
        
        avg_length = np.mean(self.episode_lengths[-10:])  # 最近10回合平均长度
        efficiency = 1.0 - (avg_length / self.max_episode_steps)
        return np.clip(efficiency, 0, 1)
    
    def _calculate_exploration_level(self) -> float:
        """计算探索程度"""
        # 基于动作多样性的简单探索度量
        if not hasattr(self, 'recent_actions'):
            self.recent_actions = deque(maxlen=20)
            return 0.5
        
        if len(self.recent_actions) < 5:
            return 0.8  # 初期鼓励探索
        
        unique_actions = len(set(self.recent_actions))
        exploration = unique_actions / len(self.recent_actions)
        return np.clip(exploration, 0, 1)
    
    def _calculate_resource_utilization(self, obs: np.ndarray) -> float:
        """计算资源利用率"""
        if len(obs) < 3:
            return 0.5
        
        # 假设obs[1]和obs[2]是预算信息
        budget_usage = 1.0 - (obs[1] + obs[2]) / 200.0  # 假设总预算200
        return np.clip(budget_usage, 0, 1)
    
    def _calculate_environmental_adaptation(self, obs: np.ndarray) -> float:
        """计算环境适应性"""
        if len(obs) < 11:
            return 0.5
        
        # 基于环境因素的适应性（obs[7:11]是环境因素）
        env_factors = obs[7:11] if len(obs) >= 11 else [1.0] * 4
        adaptation = np.mean(env_factors) - 1.0  # 中心化到0
        return np.clip(0.5 + adaptation, 0, 1)
    
    def _calculate_tactical_diversity(self) -> float:
        """计算战术多样性"""
        if not hasattr(self, 'tactical_history'):
            self.tactical_history = deque(maxlen=30)
            return 0.5
        
        if len(self.tactical_history) < 5:
            return 0.7
        
        # 基于最近动作的多样性
        unique_tactics = len(set(self.tactical_history))
        diversity = unique_tactics / min(len(self.tactical_history), 10)
        return np.clip(diversity, 0, 1)
    
    def _is_exploration_action(self, action) -> bool:
        """判断是否为探索性动作"""
        # 简化实现：假设某些动作是探索性的
        if isinstance(action, tuple) and len(action) >= 2:
            action_id, n_units = action[0], action[1]
            # 使用较少见的动作类型或单位数量作为探索
            return action_id >= 3 or n_units >= 5
        return False
    
    def _calculate_resource_efficiency(self, obs: np.ndarray, info: Dict) -> float:
        """计算资源效率"""
        # 基于成本效益的效率计算
        success_prob = obs[3:6].mean() if len(obs) >= 6 else 0.5
        cost = info.get('action_cost', 1.0)
        
        if cost > 0:
            efficiency = success_prob / cost
            return np.clip(efficiency - 0.5, -0.5, 0.5)  # 中心化
        return 0.0
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        stats = {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'average_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'win_rate': self._get_recent_win_rate(),
            'recent_performance': self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        }
        return stats
    
    def render(self, mode='human'):
        """渲染环境"""
        return self.base_env.render(mode)
    
    def close(self):
        """关闭环境"""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()


class MultiAgentMaritimeDRLEnv(MaritimeDRLEnv):
    """
    多智能体海运DRL环境
    
    支持多个智能体同时训练的环境包装器
    """
    
    def __init__(self, config_file: str = None, num_agents: int = 4, **kwargs):
        super().__init__(config_file=config_file, **kwargs)
        
        self.num_agents = num_agents
        self.agent_obs_dims = kwargs.get('agent_obs_dims', [self.observation_space.shape[0]] * num_agents)
        
        # 多智能体观察空间
        self.observation_spaces = [
            spaces.Box(
                low=np.full(obs_dim, -10.0),
                high=np.full(obs_dim, 10.0),
                dtype=np.float32
            ) for obs_dim in self.agent_obs_dims
        ]
        
        # 多智能体动作空间
        self.action_spaces = [self.action_space] * num_agents
        
        # 智能体状态
        self.agent_rewards = [0.0] * num_agents
        self.agent_done = [False] * num_agents
        
        print(f"🤝 初始化多智能体海运DRL环境: {num_agents}个智能体")
    
    def reset(self) -> List[np.ndarray]:
        """重置多智能体环境"""
        base_obs = super().reset()
        
        # 为每个智能体生成观察
        agent_observations = self._generate_agent_observations(base_obs)
        
        # 重置智能体状态
        self.agent_rewards = [0.0] * self.num_agents
        self.agent_done = [False] * self.num_agents
        
        return agent_observations
    
    def step(self, actions: List) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
        """多智能体步骤"""
        # 将多智能体动作组合
        combined_action = self._combine_actions(actions)
        
        # 执行环境步骤
        obs, reward, done, info = super().step(combined_action)
        
        # 生成多智能体观察
        agent_observations = self._generate_agent_observations(obs)
        
        # 分配奖励
        agent_rewards = self._distribute_rewards(reward, actions, info)
        
        # 更新智能体状态
        agent_dones = [done] * self.num_agents
        agent_infos = [copy.deepcopy(info) for _ in range(self.num_agents)]
        
        # 添加智能体特定信息
        for i, agent_info in enumerate(agent_infos):
            agent_info['agent_id'] = i
            agent_info['agent_reward'] = agent_rewards[i]
        
        return agent_observations, agent_rewards, agent_dones, agent_infos
    
    def _generate_agent_observations(self, base_obs: np.ndarray) -> List[np.ndarray]:
        """为每个智能体生成观察"""
        agent_observations = []
        
        for i in range(self.num_agents):
            # 基础观察
            agent_obs = base_obs.copy()
            
            # 添加智能体特定信息
            agent_specific = np.array([
                i / self.num_agents,  # 智能体ID标准化
                self.agent_rewards[i] / 100.0,  # 累计奖励标准化
                float(self.agent_done[i]),  # 完成状态
            ])
            
            # 裁剪到所需维度
            target_dim = self.agent_obs_dims[i]
            if len(agent_obs) > target_dim - 3:
                agent_obs = agent_obs[:target_dim-3]
            elif len(agent_obs) < target_dim - 3:
                padding = np.zeros(target_dim - 3 - len(agent_obs))
                agent_obs = np.concatenate([agent_obs, padding])
            
            # 组合观察
            final_obs = np.concatenate([agent_obs, agent_specific])
            agent_observations.append(final_obs.astype(np.float32))
        
        return agent_observations
    
    def _combine_actions(self, actions: List) -> Any:
        """组合多智能体动作"""
        # 简化实现：使用第一个智能体的动作
        # 实际使用中可能需要更复杂的动作组合逻辑
        if actions and len(actions) > 0:
            return actions[0]
        return 0
    
    def _distribute_rewards(self, total_reward: float, actions: List, info: Dict) -> List[float]:
        """分配奖励给各智能体"""
        # 简化实现：均等分配
        base_reward = total_reward / self.num_agents
        
        agent_rewards = []
        for i in range(self.num_agents):
            # 基础奖励
            reward = base_reward
            
            # 智能体特定奖励调整
            if i < 2:  # 攻击方
                if total_reward > 0:
                    reward *= 1.2  # 攻击成功奖励
            else:  # 防守方
                if total_reward < 0:
                    reward = -reward * 1.2  # 防守成功奖励
            
            agent_rewards.append(reward)
            
            # 更新累计奖励
            self.agent_rewards[i] += reward
        
        return agent_rewards


# 环境注册
def register_drl_environments():
    """注册DRL环境"""
    from gym.envs.registration import register
    
    try:
        register(
            id='MaritimeDRL-v0',
            entry_point='gym_flipit.envs:MaritimeDRLEnv',
            max_episode_steps=500,
        )
        
        register(
            id='MultiAgentMaritimeDRL-v0', 
            entry_point='gym_flipit.envs:MultiAgentMaritimeDRLEnv',
            max_episode_steps=500,
        )
        
        print("✅ 注册DRL增强环境成功")
    except Exception as e:
        print(f"⚠️ 环境注册警告: {e}")


if __name__ == "__main__":
    # 测试环境
    print("🧪 测试海运DRL增强环境...")
    
    # 创建环境
    env = MaritimeDRLEnv()
    
    # 测试重置
    obs = env.reset()
    print(f"✅ 重置成功，观察形状: {obs.shape}")
    
    # 测试步骤
    action = (0, 1)  # 简单动作
    obs, reward, done, info = env.step(action)
    print(f"✅ 步骤执行成功")
    print(f"   观察形状: {obs.shape}")
    print(f"   奖励: {reward:.3f}")
    print(f"   完成: {done}")
    print(f"   信息: {list(info.keys())}")
    
    # 获取统计
    stats = env.get_training_stats()
    print(f"✅ 训练统计: {stats}")
    
    env.close()
    print("🎉 环境测试完成！")

