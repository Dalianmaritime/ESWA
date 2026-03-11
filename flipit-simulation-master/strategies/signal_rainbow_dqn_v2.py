"""Discrete-action Rainbow-style DQN for the V2 defender."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_flipit.envs.maritime_cheat_attention_env import OBS_INDEX


def project_distribution(
    support: torch.Tensor,
    next_target_dist: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    v_min: float,
    v_max: float,
) -> torch.Tensor:
    atom_size = support.shape[0]
    delta_z = float((v_max - v_min) / (atom_size - 1))
    target_support = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
    target_support = target_support.clamp(v_min, v_max)

    b = (target_support - v_min) / delta_z
    lower = b.floor().long().clamp(0, atom_size - 1)
    upper = b.ceil().long().clamp(0, atom_size - 1)
    same_bin = (upper == lower).float()

    projected = torch.zeros_like(next_target_dist)
    projected.scatter_add_(1, lower, next_target_dist * (upper.float() - b + same_bin))
    projected.scatter_add_(1, upper, next_target_dist * (b - lower.float()) * (1.0 - same_bin))
    projected = projected / projected.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return projected


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        noise = torch.randn(size)
        return noise.sign().mul_(noise.abs().sqrt_())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(inputs, weight, bias)
        return F.linear(inputs, self.weight_mu, self.bias_mu)


class DistributionalQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, atom_size: int):
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, atom_size),
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim * atom_size),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.feature(observations)
        value = self.value_stream(features).view(-1, 1, self.atom_size)
        advantage = self.adv_stream(features).view(-1, self.action_dim, self.atom_size)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=-1)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


@dataclass
class BufferSample:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray


class PrioritizedReplayBufferV2:
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.alpha = float(alpha)
        self.position = 0
        self.size = 0

        self.states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self.states[self.position] = state
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.priorities[self.position] = self.max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4) -> Optional[BufferSample]:
        if self.size < batch_size:
            return None
        priorities = self.priorities[: self.size]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return BufferSample(
            states=torch.as_tensor(self.states[indices], dtype=torch.float32),
            actions=torch.as_tensor(self.actions[indices], dtype=torch.long),
            rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32),
            next_states=torch.as_tensor(self.next_states[indices], dtype=torch.float32),
            dones=torch.as_tensor(self.dones[indices], dtype=torch.float32),
            weights=torch.as_tensor(weights, dtype=torch.float32),
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for index, priority in zip(indices, priorities):
            adjusted = float(max(priority, 1e-6))
            self.priorities[index] = adjusted
            self.max_priority = max(self.max_priority, adjusted)

    def __len__(self):
        return int(self.size)


class SignalRainbowDQNAgentV2:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        v_min: float = -40.0,
        v_max: float = 40.0,
        atom_size: int = 51,
        memory_size: int = 50000,
        batch_size: int = 128,
        beta_start: float = 0.4,
        beta_frames: int = 50000,
        target_update_freq: int = 250,
        defender_initial_budget: float = 100.0,
        defender_inspect_cost: float = 1.0,
        defender_respond_cost_by_zone: Dict[str, float] | None = None,
        defender_action_floor: float = -6.0,
        use_action_mask: bool = True,
        device: str = "cpu",
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.atom_size = int(atom_size)
        self.batch_size = int(batch_size)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.target_update_freq = int(target_update_freq)
        self.device = torch.device(device)
        self.defender_initial_budget = float(defender_initial_budget)
        self.defender_inspect_cost = float(defender_inspect_cost)
        self.defender_action_floor = float(defender_action_floor)
        self.use_action_mask = bool(use_action_mask)
        respond_costs = defender_respond_cost_by_zone or {"outer": 4.0, "lane": 5.0, "core": 6.0}
        self.defender_respond_costs = [
            float(respond_costs["outer"]),
            float(respond_costs["lane"]),
            float(respond_costs["core"]),
        ]

        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size, device=self.device)
        self.delta_z = float((self.v_max - self.v_min) / (self.atom_size - 1))

        self.online_net = DistributionalQNetwork(self.obs_dim, self.action_dim, self.atom_size).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBufferV2(memory_size, self.obs_dim)

        self.learn_steps = 0
        self.total_steps = 0
        self.beta = self.beta_start

    def select_action(self, observation, training: bool = True) -> int:
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        self.online_net.train(training)
        with torch.no_grad():
            distribution = self.online_net(state)
            q_values = (distribution * self.support.view(1, 1, -1)).sum(dim=-1)
            q_values = q_values.masked_fill(~self._valid_action_mask(state), -1e9)
            action = int(q_values.argmax(dim=1).item())
        if training:
            self.total_steps += 1
        return action

    def store_transition(self, state, action: int, reward: float, next_state, done: bool):
        self.memory.push(state, int(action), float(reward), next_state, bool(done))

    def update(self) -> Dict[str, float]:
        sample = self.memory.sample(self.batch_size, beta=self.beta)
        if sample is None:
            return {}

        self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / max(self.beta_frames, 1))

        states = sample.states.to(self.device)
        actions = sample.actions.to(self.device)
        rewards = sample.rewards.to(self.device)
        next_states = sample.next_states.to(self.device)
        dones = sample.dones.to(self.device)
        weights = sample.weights.to(self.device)

        current_dist = self.online_net(states)
        current_dist = current_dist[torch.arange(self.batch_size, device=self.device), actions]

        with torch.no_grad():
            next_online_dist = self.online_net(next_states)
            next_online_q = (next_online_dist * self.support.view(1, 1, -1)).sum(dim=-1)
            next_online_q = next_online_q.masked_fill(~self._valid_action_mask(next_states), -1e9)
            next_actions = next_online_q.argmax(dim=1)

            next_target_dist = self.target_net(next_states)
            next_target_dist = next_target_dist[torch.arange(self.batch_size, device=self.device), next_actions]
            projected = project_distribution(
                support=self.support,
                next_target_dist=next_target_dist,
                rewards=rewards,
                dones=dones,
                gamma=self.gamma,
                v_min=self.v_min,
                v_max=self.v_max,
            )

        log_prob = torch.log(current_dist + 1e-8)
        elementwise_loss = -(projected * log_prob).sum(dim=1)
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        td_priority = elementwise_loss.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(sample.indices, td_priority)

        self.online_net.reset_noise()
        self.target_net.reset_noise()
        self.learn_steps += 1

        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {"loss": float(loss.item()), "beta": float(self.beta)}

    def _valid_action_mask(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if not self.use_action_mask:
            return torch.ones((observations.shape[0], self.action_dim), dtype=torch.bool, device=observations.device)
        budgets = observations[:, OBS_INDEX["defender_budget_ratio"]] * self.defender_initial_budget
        batch_size = observations.shape[0]
        mask = torch.ones((batch_size, self.action_dim), dtype=torch.bool, device=observations.device)
        inspect_valid = budgets - self.defender_inspect_cost >= self.defender_action_floor
        mask[:, 1:4] = inspect_valid.unsqueeze(1).repeat(1, 3)
        for action_index, cost in enumerate(self.defender_respond_costs, start=4):
            mask[:, action_index] = budgets - cost >= self.defender_action_floor
        return mask

    def save(self, path: str):
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_steps": self.learn_steps,
                "total_steps": self.total_steps,
                "beta": self.beta,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.beta = float(checkpoint.get("beta", self.beta_start))
