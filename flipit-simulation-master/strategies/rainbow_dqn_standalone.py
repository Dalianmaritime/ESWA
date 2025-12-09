"""
独立的Rainbow DQN算法实现
专为TRC论文海运安全博弈设计，去除外部依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random
from collections import deque
from typing import Dict, List, Tuple, Optional

class NoisyLinear(nn.Module):
    """噪声线性层"""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(input, 
                           self.weight_mu + self.weight_sigma * self.weight_epsilon,
                           self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class RainbowDQNNetwork(nn.Module):
    """Rainbow DQN网络"""
    def __init__(self, obs_dim: int, action_dim: int, atom_size: int = 51):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.atom_size = atom_size
        
        # 特征提取
        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Dueling架构
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, atom_size)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(), 
            NoisyLinear(64, action_dim * atom_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        
        value = self.value_stream(features).view(-1, 1, self.atom_size)
        advantage = self.advantage_stream(features).view(-1, self.action_dim, self.atom_size)
        
        # Dueling合并
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return F.softmax(q_dist, dim=2)
    
    def reset_noise(self):
        """重置噪声"""
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

class PrioritizedReplayBuffer:
    """优先级经验回放池"""
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.obs_dim = obs_dim
        self.pos = 0
        self.size = 0
        
        # 数据存储
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        
        # 优先级存储
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        
        # 新经验使用最高优先级
        self.priorities[self.pos] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int, beta: float = 0.4):
        """采样经验"""
        if self.size == 0:
            return None
            
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # 计算重要性权重
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.LongTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.BoolTensor(self.dones[indices]),
            'indices': indices,
            'is_weights': torch.FloatTensor(weights)
        }
        
        return batch
        
    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return self.size

class RainbowDQNAgent:
    """
    独立的Rainbow DQN代理
    集成所有高级特性但无外部依赖
    """
    
    def __init__(self, obs_dim: int, action_dim: int, max_units: int = 8,
                 lr: float = 0.0001, gamma: float = 0.98, 
                 v_min: float = -50, v_max: float = 50, atom_size: int = 51,
                 memory_size: int = 100000, batch_size: int = 256,
                 device: str = 'cuda', target_update_freq: int = 1000):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_units = max_units
        self.lr = lr
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.batch_size = batch_size
        self.device = device
        
        # 分布式RL支持
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)
        self.delta_z = float(v_max - v_min) / (atom_size - 1)
        
        # 网络
        self.q_network = RainbowDQNNetwork(obs_dim, action_dim * max_units, atom_size).to(device)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.target_update_freq = target_update_freq
        
        # 经验回放
        self.memory = PrioritizedReplayBuffer(memory_size, obs_dim)
        
        # 统计
        self.steps = 0
        self.episodes = 0
        
        print(f"[DRL] 初始化独立Rainbow DQN: obs_dim={obs_dim}, action_dim={action_dim}")
        
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, int]:
        """选择动作"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
            
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            # 获取动作分布
            action_dist = self.q_network(state_tensor)
            
            # 计算Q值期望
            q_values = torch.sum(action_dist * self.support.unsqueeze(0).unsqueeze(0), dim=2)
            
            # 选择最优动作
            best_idx = q_values.argmax(1).item()
            action_id = best_idx // self.max_units
            n_units = (best_idx % self.max_units) + 1
            
            return action_id, n_units
    
    def store_transition(self, state: np.ndarray, action: Tuple[int, int], 
                        reward: float, next_state: np.ndarray, done: bool):
        """存储经验"""
        action_array = np.array([action[0], action[1]], dtype=np.int64)
        self.memory.push(state, action_array, reward, next_state, done)
        
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.memory) < self.batch_size:
            return {}
            
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return {}
            
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        is_weights = batch['is_weights'].to(self.device)
        indices = batch['indices']
        
        # 转换动作为索引
        action_indices = actions[:, 0] * self.max_units + (actions[:, 1] - 1)
        action_indices = action_indices.clamp(0, self.action_dim * self.max_units - 1)
        
        # 当前分布
        current_dist = self.q_network(states)
        current_dist = current_dist[range(self.batch_size), action_indices]
        
        # 目标分布 (Double DQN)
        with torch.no_grad():
            next_action_dist = self.q_network(next_states)
            next_q_values = torch.sum(next_action_dist * self.support.unsqueeze(0).unsqueeze(0), dim=2)
            next_actions = next_q_values.argmax(1)
            
            next_target_dist = self.target_network(next_states)
            next_target_dist = next_target_dist[range(self.batch_size), next_actions]
            
            # 分布式Bellman更新
            target_support = rewards.unsqueeze(1) + self.gamma * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
            target_support = target_support.clamp(self.v_min, self.v_max)
            
            # 投影到支持集
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long().clamp(min=0, max=self.atom_size - 1)
            u = b.ceil().long().clamp(min=0, max=self.atom_size - 1)
            
            target_dist = torch.zeros_like(next_target_dist)
            target_dist.scatter_add_(1, l, next_target_dist * (u.float() - b))
            target_dist.scatter_add_(1, u, next_target_dist * (b - l.float()))
        
        # 计算损失
        loss = -(target_dist * torch.log(current_dist + 1e-8)).sum(1)
        
        # 重要性权重
        loss = (loss * is_weights).mean()
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新优先级
        with torch.no_grad():
            td_errors = loss.detach().cpu().numpy()
            # 转换为标准格式
            td_errors = np.atleast_1d(td_errors)
            if not isinstance(indices, (list, np.ndarray)):
                indices = [indices]
            
            # 确保长度匹配
            if len(td_errors) == 1 and len(indices) > 1:
                td_errors = np.full(len(indices), td_errors[0])
            
            self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # 重置噪声
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        
        self.steps += 1
        
        # 软更新目标网络
        if self.steps % self.target_update_freq == 0:
            self._soft_update_target()
        
        return {'loss': float(loss.item())}
        
    def _soft_update_target(self, tau: float = 0.005):
        """软更新目标网络"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']



