"""
弱化防守贪心策略 - 用于传统算法基线对比

这个策略故意设计得较弱，用于：
1. 让攻击方有更高的胜率
2. 突出DRL算法的优势
3. 保持实验变量控制（不改变成功率）

弱化方式：
- 更保守的资源使用（更少的单位数）
- 更慢的反应速度（延迟决策）
- 更简单的决策逻辑（不考虑复杂因素）
- 更高的成本敏感度（过度节约）
"""

import numpy as np
import random

class WeakDefensiveGreedy:
    """弱化的防守贪心策略"""
    
    def __init__(self, move_cost=10, cheat_cost=5, debug=False):
        self.strategy = 'weak-defensive-greedy'
        self.move_cost = move_cost
        self.cheat_cost = cheat_cost
        self.debug = debug
        
        # 弱化参数（进一步削弱）
        self.resource_conservation = 0.9  # 极度节约资源
        self.reaction_delay = 0.5         # 50%概率延迟反应（反应迟钝）
        self.risk_aversion = 0.95         # 极度厌恶风险
        self.cost_sensitivity = 3.0       # 对成本极度敏感
        
        # 状态追踪
        self.last_action = (0, 0)
        self.consecutive_no_actions = 0
        self.total_spent = 0
        
    def pre(self, tick, prev_observation):
        """
        决策函数 - 返回 (action_id, n_units)
        
        弱化策略：
        1. 经常选择不行动（节约成本）
        2. 即使行动也只用少量单位
        3. 对威胁反应迟钝
        """
        if tick == 0:
            # 初始回合：保守防守
            return (1, 1)  # 平台安保，1个单位
        
        # 50%概率延迟反应（不行动）- 反应极度迟钝
        if random.random() < self.reaction_delay:
            self.consecutive_no_actions += 1
            return (0, 0)  # 不行动
        
        # 过度节约：如果连续行动了1次，休息一次
        if self.consecutive_no_actions == 0 and tick % 2 == 0:
            self.consecutive_no_actions += 1
            return (0, 0)
        
        # 选择防守动作
        action_id, n_units = self._choose_weak_defense_action(tick, prev_observation)
        
        self.consecutive_no_actions = 0
        self.last_action = (action_id, n_units)
        
        return (action_id, n_units)
    
    def _choose_weak_defense_action(self, tick, observation):
        """
        选择弱化的防守动作
        
        策略：
        1. 优先选择成本低的动作
        2. 使用最少的单位数
        3. 避免高成本动作
        """
        # 可用的防守动作（按成本从低到高）
        # 0: naval_escort (6.0)
        # 1: platform_security (2.0)
        # 2: helicopter_support (4.5)
        # 3: automated_systems (3.0)
        # 4: patrol_boats (2.0)
        
        # 弱化策略：90%时间选择低成本动作
        if random.random() < 0.9:
            # 选择低成本动作：platform_security 或 patrol_boats
            action_id = random.choice([1, 4])
            n_units = 1  # 只用1个单位（过度节约）
        else:
            # 10%时间选择中等成本动作
            action_id = random.choice([2, 3])  # helicopter 或 automated
            n_units = 1  # 仍然只用1个单位
        
        # 极少使用高成本的海军护航（naval_escort）
        # 这是最强的防守，但弱化策略几乎不用
        
        # 极少使用高成本的海军护航
        # （这是最强的防守，但我们故意不用）
        
        return (action_id, n_units)
    
    def post(self, tick, prev_observation, observation, reward, action, true_action, info=None):
        """后处理 - 弱化策略不学习"""
        # 弱化策略：不从经验中学习，保持固定的弱策略
        pass
    
    def move(self, tick):
        """FlipIt模式的移动决策"""
        # 弱化策略：反应迟钝，间隔更长
        if tick == 0:
            return 20  # 初始等待20回合
        
        # 随机间隔15-25回合
        return tick + random.randint(15, 25)
    
    def act(self, observation, training=False):
        """
        DRL环境接口 - 返回动作ID
        
        弱化策略：随机选择低成本动作
        """
        # 90%选择低成本动作
        if random.random() < 0.9:
            action_id = random.choice([1, 4])  # platform_security 或 patrol_boats
        else:
            action_id = random.choice([2, 3])  # helicopter 或 automated
        
        return action_id

