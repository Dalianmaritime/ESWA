"""
激进攻击策略 - 用于FlipIt模式基线实验

这个策略专门设计用于FlipIt模式，采用极度激进的攻击风格：
1. 频繁发起攻击
2. 使用更多单位
3. 不考虑成本
4. 持续施压

目的：在FlipIt模式下让攻击方有更高的胜率
"""

import numpy as np
import random

class AggressiveAttacker:
    """激进的攻击策略"""
    
    def __init__(self, move_cost=15, cheat_cost=5, debug=False):
        self.strategy = 'aggressive-attacker'
        self.move_cost = move_cost
        self.cheat_cost = cheat_cost
        self.debug = debug
        
        # 激进参数
        self.aggression_level = 0.9       # 90%概率发起攻击
        self.resource_spending = 1.5      # 不节约资源，多用单位
        self.risk_tolerance = 0.9         # 高风险容忍度
        
        # 状态追踪
        self.last_action = (0, 0)
        self.consecutive_attacks = 0
        self.total_attacks = 0
        
    def pre(self, tick, prev_observation):
        """
        决策函数 - 返回 (action_id, n_units)
        
        激进策略：
        1. 几乎每回合都攻击
        2. 使用较多单位
        3. 优先高威胁动作
        """
        if tick == 0:
            # 初始回合：立即发起攻击
            return (2, 2)  # boarding_assault，2个单位
        
        # 90%概率发起攻击
        if random.random() < self.aggression_level:
            action_id, n_units = self._choose_aggressive_attack(tick, prev_observation)
            self.consecutive_attacks += 1
            self.total_attacks += 1
            return (action_id, n_units)
        else:
            # 10%概率休息（恢复资源）
            self.consecutive_attacks = 0
            return (0, 0)
    
    def _choose_aggressive_attack(self, tick, observation):
        """
        选择激进的攻击动作
        
        策略：
        1. 优先选择高威胁动作
        2. 使用较多单位（2-3个）
        3. 不考虑成本
        """
        # 可用的攻击动作（按威胁从高到低）
        # 0: inflatable_fast_boat (2.0)
        # 1: hard_hull_speedboat (4.0)
        # 2: boarding_assault (5.0) - 最高威胁
        # 3: standoff_attack (3.0)
        
        # 激进策略：70%时间选择高威胁动作
        if random.random() < 0.7:
            # 选择高威胁动作：boarding_assault 或 hard_hull_speedboat
            action_id = random.choice([2, 1])
            # 使用2-3个单位
            if action_id == 2:  # boarding_assault最多2个单位
                n_units = 2
            else:  # hard_hull_speedboat最多3个单位
                n_units = random.choice([2, 3])
        else:
            # 30%时间选择中等威胁动作
            action_id = random.choice([0, 3])  # inflatable或standoff
            n_units = random.choice([2, 3])  # 仍然用多个单位
        
        return (action_id, n_units)
    
    def post(self, tick, prev_observation, observation, reward, action, true_action, info=None):
        """后处理 - 激进策略不学习"""
        pass
    
    def move(self, tick):
        """FlipIt模式的移动决策"""
        if tick == 0:
            return 3  # 初始等待3回合就攻击
        
        # 频繁攻击：间隔3-5回合
        return tick + random.randint(3, 5)
    
    def act(self, observation, training=False):
        """
        DRL环境接口 - 返回动作ID
        
        激进策略：优先选择高威胁动作
        """
        # 70%选择高威胁动作
        if random.random() < 0.7:
            action_id = random.choice([2, 1])  # boarding_assault 或 hard_hull_speedboat
        else:
            action_id = random.choice([0, 3])  # inflatable 或 standoff
        
        return action_id

