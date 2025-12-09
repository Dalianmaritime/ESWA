import gym
from gym import error, utils, spaces
from gym.utils import seeding
import random
import numpy as np
import yaml
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

@dataclass
class ActionConfig:
    """动作配置数据类"""
    id: str
    name: str
    unit_label: str
    description: str
    base_success: Dict[str, float]
    nonlinear: Dict[str, float] 
    cost_per_unit: float
    max_units: int

@dataclass  
class EngagementResult:
    """交战结果数据类"""
    att_action: str
    att_units: int
    def_action: str  
    def_units: int
    P_att: float
    P_def: float
    P_final: float
    att_cost: float
    def_cost: float
    success: bool

class MaritimeNontraditionalEnv(gym.Env):
    """
    海运非传统安全对抗环境
    
    支持多单位、非线性成功率、攻防对抗合成的复杂海运安全博弈。
    涵盖海盗袭击、海上平台防御、军舰护卫等真实场景。
    
    核心特性:
    - 多单位动作选择: (action_id, n_units)  
    - 非线性成功率: p_eff(n) = min(cap, 1 - (1-p)^(n^alpha))
    - 攻防对抗合成: P_final = P_att * (1 - P_def)
    - 预算约束和成本计算
    - 环境因素影响
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = None
        self.attacker_actions = {}
        self.defender_actions = {}
        
        # 游戏状态
        self.tick = 0
        self.duration = 1000
        self.current_controller = 0  # 0-防守方控制, 1-攻击方控制
        
        # Cheat-FlipIt 欺骗模式配置
        self.deception_mode = "flipit"  # 默认使用flipit模式
        
        # 预算和成本追踪
        self.attacker_budget = 10.0
        self.defender_budget = 10.0
        self.attacker_spent = 0
        self.defender_spent = 0
        
        # 记录初始预算（用于逐局重置，防止资源无上限累积）
        self.initial_attacker_budget = self.attacker_budget
        self.initial_defender_budget = self.defender_budget
        
        # 交战历史
        self.engagement_history = []
        self.tactical_patterns = defaultdict(int)
        
        # 环境因素
        self.environmental_modifiers = {}
        
        # 性能统计
        self.unit_effectiveness = defaultdict(lambda: {'used': 0, 'successful': 0})
        
        # 占领奖励机制（修正为符合论文设计）
        self.occupation_reward = 1.0  # 每回合控制方获得的额外资源奖励
        self.total_occupation_rewards = [0.0, 0.0]  # [防守方累计, 攻击方累计]

        # 双方每回合资源累积（修正为符合论文设计）
        # 防守方每回合+2，攻击方每回合+1，可通过配置覆盖
        self.attacker_income_per_step = 1.0  # 攻击方每回合基础预算
        self.defender_income_per_step = 2.0  # 防守方每回合基础预算
        
        # 初始化动作列表（避免属性错误）
        self.attacker_action_list = []
        self.defender_action_list = []
        
        # 设置默认动作和观察空间（必须在__init__中设置）
        self._setup_default_spaces()
        
        if config_file:
            self.load_config(config_file)
            self._setup_environment()
    
    def _setup_default_spaces(self):
        """设置默认的动作和观察空间"""
        # 默认动作空间：(action_id, n_units)
        self.action_space = spaces.Tuple((
            spaces.Discrete(6),  # 默认6种动作
            spaces.Discrete(7)   # 0-6个单位
        ))
        
        # 默认观察空间
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0]),
            high=np.array([1, 100, 100, 1, 1, 1, 1000, 1.5, 1.5, 1.5, 1.5, 10, 5]),
            dtype=np.float32
        )
    
    def load_config(self, config_file: str):
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 解析攻击方动作
            for action_data in self.config['attacker_actions']:
                action = ActionConfig(
                    id=action_data['id'],
                    name=action_data['name'],
                    unit_label=action_data['unit_label'],
                    description=action_data['description'],
                    base_success=action_data['base_success'],
                    nonlinear=action_data['nonlinear'],
                    cost_per_unit=action_data['cost_per_unit'],
                    max_units=action_data['max_units']
                )
                self.attacker_actions[action.id] = action
            
            # 解析防守方动作
            for action_data in self.config['defender_actions']:
                action = ActionConfig(
                    id=action_data['id'],
                    name=action_data['name'], 
                    unit_label=action_data['unit_label'],
                    description=action_data['description'],
                    base_success=action_data['base_success'],
                    nonlinear=action_data['nonlinear'],
                    cost_per_unit=action_data['cost_per_unit'],
                    max_units=action_data['max_units']
                )
                self.defender_actions[action.id] = action
                
            print(f"[OK] 成功加载配置: {len(self.attacker_actions)}种攻击动作, {len(self.defender_actions)}种防守动作")
                
        except Exception as e:
            raise ValueError(f"配置文件加载失败: {e}")
    
    def _setup_environment(self):
        """设置环境参数"""
        # 基础参数
        self.duration = self.config.get('duration', 1000)
        
        # Cheat-FlipIt 欺骗模式配置
        self.deception_mode = self.config.get('deception_mode', 'flipit')
        print(f"[OK] 设置欺骗模式: {self.deception_mode}")
        
        # 预算设置
        constraints = self.config.get('constraints', {})
        budgeting = constraints.get('budgeting', {})
        if budgeting.get('enabled', False):
            self.attacker_budget = budgeting.get('attacker_budget', 10.0)
            self.defender_budget = budgeting.get('defender_budget', 12.0)
        else:
            self.attacker_budget = float('inf')
            self.defender_budget = float('inf')

        self.initial_attacker_budget = self.attacker_budget
        self.initial_defender_budget = self.defender_budget
        
        # 环境因素
        env_factors = self.config.get('environmental_factors', {})
        self._setup_environmental_modifiers(env_factors)
        
        # 占领奖励设置
        rew_config = self.config.get('rew_config', {})
        self.occupation_reward = rew_config.get('occupation_reward', 0)

        # 读取每回合资源累积配置
        self.attacker_income_per_step = rew_config.get('attacker_income_per_step', 1.0)
        self.defender_income_per_step = rew_config.get('defender_income_per_step', 2.0)
        
        print(f"[OK] 资源增长配置: 防守方每步+{self.defender_income_per_step}, 攻击方每步+{self.attacker_income_per_step}")
        
        # 设置动作和观察空间
        self._setup_action_spaces()
        self._setup_observation_space()
    
    def _setup_environmental_modifiers(self, env_factors: Dict):
        """设置环境修正因子"""
        self.environmental_modifiers = {
            'weather_modifier': self._get_weather_modifier(env_factors.get('weather_condition', 'moderate')),
            'visibility_modifier': self._get_visibility_modifier(env_factors.get('visibility', 'normal')),
            'sea_state_modifier': self._get_sea_state_modifier(env_factors.get('sea_state', 3)),
            'time_modifier': self._get_time_modifier(env_factors.get('time_of_day', 'mixed'))
        }
    
    def _get_weather_modifier(self, weather: str) -> float:
        """天气修正因子"""
        modifiers = {'poor': 0.85, 'moderate': 1.0, 'good': 1.15}
        return modifiers.get(weather, 1.0)
    
    def _get_visibility_modifier(self, visibility: str) -> float:
        """能见度修正因子"""
        modifiers = {'poor': 0.8, 'normal': 1.0, 'good': 1.2}
        return modifiers.get(visibility, 1.0)
    
    def _get_sea_state_modifier(self, sea_state: int) -> float:
        """海况修正因子"""
        if sea_state <= 2:
            return 1.1  # 平静海面有利于精确作战
        elif sea_state <= 4:
            return 1.0  # 中等海况
        elif sea_state <= 6:
            return 0.9  # 恶劣海况影响作战效能
        else:
            return 0.75  # 极端恶劣海况
    
    def _get_time_modifier(self, time: str) -> float:
        """时间修正因子"""
        modifiers = {'day': 1.0, 'night': 0.9, 'mixed': 0.95}
        return modifiers.get(time, 1.0)
    
    def _setup_action_spaces(self):
        """设置动作空间"""
        # 更新主要动作空间（使用攻击方和防守方的最大值）
        max_att_actions = len(self.attacker_actions) if self.attacker_actions else 6
        max_def_actions = len(self.defender_actions) if self.defender_actions else 6
        max_actions = max(max_att_actions, max_def_actions)
        
        max_att_units = self.config['constraints']['max_units']['attacker'] + 1 if self.config else 7
        max_def_units = self.config['constraints']['max_units']['defender'] + 1 if self.config else 7
        max_units = max(max_att_units, max_def_units)
        
        # 更新主动作空间
        self.action_space = spaces.Tuple((
            spaces.Discrete(max_actions),
            spaces.Discrete(max_units)
        ))
        
        # 攻击方动作空间: (action_id, n_units)
        self.attacker_action_space = spaces.Tuple((
            spaces.Discrete(len(self.attacker_actions)),  # 动作ID
            spaces.Discrete(max_att_units)  # 单位数量
        ))
        
        # 防守方动作空间: (action_id, n_units) 
        self.defender_action_space = spaces.Tuple((
            spaces.Discrete(len(self.defender_actions)),  # 动作ID
            spaces.Discrete(max_def_units)  # 单位数量
        ))
        
        # 创建动作ID到名称的映射
        self.attacker_action_list = list(self.attacker_actions.keys())
        self.defender_action_list = list(self.defender_actions.keys())
    
    def _setup_observation_space(self):
        """设置观察空间"""
        # 复杂观察空间包含：
        # - 当前控制者 (1)
        # - 剩余预算 (2)  
        # - 最近交战结果 (3)
        # - 环境因素 (4)
        # - 威胁评估 (3)
        max_budget = max(self.attacker_budget, self.defender_budget) if self.attacker_budget != float('inf') else 100
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0]),
            high=np.array([1, max_budget, max_budget, 1, 1, 1, self.duration, 1.5, 1.5, 1.5, 1.5, 10, 5]),
            dtype=np.float32
        )
    
    def calculate_nonlinear_effectiveness(self, n_units: int, base_prob: float, 
                                        alpha: float, cap: float) -> float:
        """
        计算非线性单位效能
        
        公式: p_eff(n) = min(cap, 1 - (1-p)^(n^alpha))
        
        参数:
            n_units: 投入单位数量
            base_prob: 单个单位基础成功率
            alpha: 非线性指数
            cap: 成功率上限
        """
        if n_units == 0:
            return 0.0
        
        if base_prob >= 1.0:
            return min(cap, 1.0)
        
        # 应用非线性公式
        effective_power = n_units ** alpha
        prob_failure = (1 - base_prob) ** effective_power
        prob_success = 1 - prob_failure
        
        # 应用成功率上限
        final_prob = min(cap, prob_success)
        
        return final_prob
    
    def resolve_engagement(self, att_action_id: str, att_units: int,
                          def_action_id: str, def_units: int) -> EngagementResult:
        """
        解决攻防对抗
        
        参数:
            att_action_id: 攻击方动作ID
            att_units: 攻击方单位数
            def_action_id: 防守方动作ID  
            def_units: 防守方单位数
            
        返回:
            EngagementResult: 交战结果
        """
        # 获取动作配置
        att_action = self.attacker_actions[att_action_id]
        def_action = self.defender_actions[def_action_id]
        
        # 根据欺骗模式计算基础成功概率
        if self.deception_mode == 'cheat':
            # 使用cheat模式的成功率（攻击方有欺骗优势，防守方效果下降）
            att_base_prob = att_action.base_success.get('cheat', 
                                                       att_action.base_success.get('flipit', 0.5))
            def_base_prob = def_action.base_success.get('cheat', 
                                                      def_action.base_success.get('flipit', 
                                                      def_action.base_success.get('generic', 0.5)))
        else:
            # 使用flipit模式的成功率（正常情况）
            att_base_prob = att_action.base_success.get('flipit', 0.5)
            def_base_prob = def_action.base_success.get('flipit', 
                                                      def_action.base_success.get('generic', 0.5))
        
        # 应用环境修正
        env_modifier = np.mean(list(self.environmental_modifiers.values()))
        att_base_prob *= env_modifier
        def_base_prob *= env_modifier
        
        # 计算非线性效能
        P_att = self.calculate_nonlinear_effectiveness(
            att_units, att_base_prob, 
            att_action.nonlinear['alpha'], 
            att_action.nonlinear['cap']
        )
        
        P_def = self.calculate_nonlinear_effectiveness(
            def_units, def_base_prob,
            def_action.nonlinear['alpha'],
            def_action.nonlinear['cap'] 
        )
        
        # 攻防对抗合成: P_final = P_att * (1 - P_def)
        P_final = P_att * (1 - P_def)
        
        # 计算成本
        att_cost = att_action.cost_per_unit * att_units
        def_cost = def_action.cost_per_unit * def_units
        
        # 判定成功
        success = random.random() < P_final
        
        result = EngagementResult(
            att_action=att_action_id,
            att_units=att_units,
            def_action=def_action_id,
            def_units=def_units,
            P_att=P_att,
            P_def=P_def,
            P_final=P_final,
            att_cost=att_cost,
            def_cost=def_cost,
            success=success
        )
        
        return result
    
    def _generate_defender_action(self) -> Tuple[int, int]:
        """生成简单的防守方动作"""
        if not self.defender_actions:
            return (0, 1)
        
        # 简单的防守策略：根据剩余预算选择动作
        available_actions = []
        for i, (action_id, action_config) in enumerate(self.defender_actions.items()):
            max_units = min(action_config.max_units, 
                          int(self.defender_budget - self.defender_spent) // max(1, int(action_config.cost_per_unit)))
            if max_units > 0:
                available_actions.append((i, random.randint(1, max_units)))
        
        if available_actions:
            return random.choice(available_actions)
        else:
            return (0, 1)  # 默认动作
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步博弈
        
        参数:
            action: 攻击方动作 (action_id, n_units) 或 双方动作 ((att_action, att_units), (def_action, def_units))
            
        返回:
            observation, reward, done, info
        """
        self.tick += 1
        
        # 解析动作格式
        if isinstance(action, tuple) and len(action) == 2:
            if isinstance(action[0], tuple):
                # 双方动作格式: ((att_action, att_units), (def_action, def_units))
                attacker_action, defender_action = action
                att_action_idx, att_units = attacker_action
                def_action_idx, def_units = defender_action
            else:
                # 单方动作格式: (action_id, n_units) - 只有攻击方，需要生成防守方动作
                att_action_idx, att_units = action
                def_action_idx, def_units = self._generate_defender_action()
        else:
            raise ValueError(f"无效的动作格式: {action}")
        
        # 解析动作
        # att_action_idx, att_units = attacker_action
        # def_action_idx, def_units = defender_action
        
        # 安全获取动作ID
        if self.attacker_action_list and att_action_idx < len(self.attacker_action_list):
            att_action_id = self.attacker_action_list[att_action_idx]
        else:
            att_action_id = list(self.attacker_actions.keys())[0] if self.attacker_actions else 'default_attack'
            
        if self.defender_action_list and def_action_idx < len(self.defender_action_list):
            def_action_id = self.defender_action_list[def_action_idx]
        else:
            def_action_id = list(self.defender_actions.keys())[0] if self.defender_actions else 'default_defense'
        
        # 安全获取动作配置
        att_action_config = self.attacker_actions.get(att_action_id)
        def_action_config = self.defender_actions.get(def_action_id)
        
        # 如果动作配置不存在，跳过此步骤
        if not att_action_config or not def_action_config:
            # 返回默认结果
            observation = self._get_observation()
            return observation, 0, False, {'error': 'Invalid action configuration'}
        
        att_cost = att_action_config.cost_per_unit * att_units
        def_cost = def_action_config.cost_per_unit * def_units
        
        # 严格预算约束检查 - 在动作执行前进行
        attacker_available_budget = self.attacker_budget - self.attacker_spent
        defender_available_budget = self.defender_budget - self.defender_spent
        
        # 攻击方预算约束
        if att_cost > attacker_available_budget:
            original_att_cost = att_cost
            original_att_units = att_units
            # 计算在预算内能使用的最大单位数
            max_affordable_att_units = max(0, int(attacker_available_budget / att_action_config.cost_per_unit))
            if max_affordable_att_units == 0:
                # 预算不足，无法执行任何动作
                att_units = 0
                att_cost = 0.0
                print(f"[BUDGET] 攻击方预算不足: 需要{original_att_cost:.2f}, 可用{attacker_available_budget:.2f}, 强制设为0单位")
            else:
                att_units = min(att_units, max_affordable_att_units)
                att_cost = att_action_config.cost_per_unit * att_units
                print(f"[BUDGET] 攻击方预算限制: 原计划{original_att_units}单位(成本{original_att_cost:.2f}), 预算限制为{att_units}单位(成本{att_cost:.2f})")
            
        # 防守方预算约束
        if def_cost > defender_available_budget:
            original_def_cost = def_cost
            original_def_units = def_units
            # 计算在预算内能使用的最大单位数
            max_affordable_def_units = max(0, int(defender_available_budget / def_action_config.cost_per_unit))
            if max_affordable_def_units == 0:
                # 预算不足，无法执行任何动作
                def_units = 0
                def_cost = 0.0
                print(f"[BUDGET] 防守方预算不足: 需要{original_def_cost:.2f}, 可用{defender_available_budget:.2f}, 强制设为0单位")
            else:
                def_units = min(def_units, max_affordable_def_units)
                def_cost = def_action_config.cost_per_unit * def_units
                print(f"[BUDGET] 防守方预算限制: 原计划{original_def_units}单位(成本{original_def_cost:.2f}), 预算限制为{def_units}单位(成本{def_cost:.2f})")
        
        # 单位数约束
        att_units = min(att_units, att_action_config.max_units)
        def_units = min(def_units, def_action_config.max_units)
        
        # 解决交战
        result = self.resolve_engagement(att_action_id, att_units, def_action_id, def_units)
        
        # 更新状态
        self.attacker_spent += result.att_cost
        self.defender_spent += result.def_cost
        
        # 每步资源增长：防守方+2，攻击方+1
        self.defender_budget += self.defender_income_per_step
        self.attacker_budget += self.attacker_income_per_step
        
        if result.success:
            self.current_controller = 1  # 攻击方获得控制
        else:
            self.current_controller = 0  # 防守方保持控制
        
        # 占领奖励加入预算：控制方获得占领奖励作为可用预算
        pure_occupation_reward = self._calculate_pure_occupation_reward()
        if self.current_controller == 0:  # 防守方控制
            self.defender_budget += pure_occupation_reward
        else:  # 攻击方控制
            self.attacker_budget += pure_occupation_reward
        
        # 记录历史
        self.engagement_history.append(result)
        self.tactical_patterns[f"{att_action_id}_{def_action_id}"] += 1
        
        # 更新单位效能统计
        self.unit_effectiveness[att_action_id]['used'] += att_units
        if result.success:
            self.unit_effectiveness[att_action_id]['successful'] += att_units
            
        self.unit_effectiveness[def_action_id]['used'] += def_units
        if not result.success:
            self.unit_effectiveness[def_action_id]['successful'] += def_units
        
        # 计算防守方DQN训练用的奖励
        training_reward = self._calculate_reward(result)

        # 统计用的纯净占领奖励（仅用于算法对比和累计统计）
        pure_occupation_reward = self._calculate_pure_occupation_reward()
        self.total_occupation_rewards[self.current_controller] += pure_occupation_reward
        
        # 返回防守方训练奖励（用于DQN学习）
        reward = training_reward
        
        # 生成观察
        observation = self._get_observation()
        
        # 生成信息
        info = {
            'engagement_result': result,
            'attacker_budget_remaining': self.attacker_budget - self.attacker_spent,
            'defender_budget_remaining': self.defender_budget - self.defender_spent,
            'attacker_available_budget': self.attacker_budget - self.attacker_spent,
            'defender_available_budget': self.defender_budget - self.defender_spent,
            'current_controller': self.current_controller,
            'environmental_modifiers': self.environmental_modifiers,
            'training_reward': training_reward,  # 防守方DQN训练奖励
            'pure_occupation_reward': pure_occupation_reward,  # 统计用纯净占领奖励
            'per_step_income': {
                'defender': float(self.defender_income_per_step),
                'attacker': float(self.attacker_income_per_step)
            },
            'total_occupation_rewards': {
                'defender': self.total_occupation_rewards[0],
                'attacker': self.total_occupation_rewards[1]
            },
            'reward_components': {
                'occupation_reward': pure_occupation_reward if self.current_controller == 0 else -pure_occupation_reward,
                'defender_cost_penalty': -result.def_cost,
                'attacker_cost_bonus': result.att_cost * 0.5,
                'total_training_reward': training_reward
            },
            'budget_changes': {
                'defender_income': self.defender_income_per_step,
                'attacker_income': self.attacker_income_per_step,
                'occupation_bonus': pure_occupation_reward if self.current_controller == 0 else 0,
                'defender_spent': result.def_cost,
                'attacker_spent': result.att_cost
            }
        }
        
        # 检查结束条件
        attacker_budget_bound = (not math.isinf(self.attacker_budget)) and self.attacker_budget > 0
        defender_budget_bound = (not math.isinf(self.defender_budget)) and self.defender_budget > 0

        # 先检查硬终止：预算超界
        done = (self.tick >= self.duration or 
                (attacker_budget_bound and self.attacker_spent > self.attacker_budget) or
                (defender_budget_bound and self.defender_spent > self.defender_budget))

        if attacker_budget_bound and self.attacker_spent > self.attacker_budget:
            info['failure_reason'] = 'attacker_budget_limit'
            info['winner'] = 'defender'
            info['early_termination'] = True
        elif defender_budget_bound and self.defender_spent > self.defender_budget:
            info['failure_reason'] = 'defender_budget_limit'
            info['winner'] = 'attacker'
            info['early_termination'] = True
        
        # 以“可用余额”判定透支：允许小额透支，但超过-30立即判负
        attacker_available = self.attacker_budget - self.attacker_spent
        defender_available = self.defender_budget - self.defender_spent
        
        if attacker_available <= -30:
            done = True
            info['failure_reason'] = 'attacker_budget_overdraft'
            info['winner'] = 'defender'
            info['early_termination'] = True
            info['budget_overdraft'] = attacker_available
            print(f"[BUDGET] 攻击方预算透支{attacker_available:.1f}，游戏结束，防守方获胜")
        elif defender_available <= -30:
            done = True
            info['failure_reason'] = 'defender_budget_overdraft'
            info['winner'] = 'attacker'
            info['early_termination'] = True
            info['budget_overdraft'] = defender_available
            print(f"[BUDGET] 防守方预算透支{defender_available:.1f}，游戏结束，攻击方获胜")
        # 双方都耗尽或接近耗尽预算
        elif attacker_available <= 0 and defender_available <= 0:
            done = True
            info['failure_reason'] = 'both_resources_depleted'
            # 根据占领奖励判断胜负
            if self.total_occupation_rewards[1] > self.total_occupation_rewards[0]:
                info['winner'] = 'attacker'
            else:
                info['winner'] = 'defender'
            info['early_termination'] = True

        if done:
            info = self._finalize_episode_outcome(info)
        
        return observation, reward, done, info
    
    def _calculate_reward(self, result: EngagementResult) -> float:
        """
        计算防守方DQN训练用的奖励函数
        新设计：防守方奖励 = 占领奖励 - 自己成本 + 对方成本
        目标：自己花越小成本，对方花越多成本，同时尽量占领
        """
        rew_config = self.config.get('rew_config', {})
        
        # 1. 占领奖励：防守方控制时获得正奖励，攻击方控制时获得负奖励
        if self.current_controller == 0:  # 防守方控制
            occupation_reward = self.occupation_reward
        else:  # 攻击方控制
            occupation_reward = -self.occupation_reward
        
        # 2. 成本考虑：自己的成本是负面的，对方的成本是正面的
        cost_penalty_factor = rew_config.get('cost_penalty', 1.0)
        defender_cost_penalty = -result.def_cost * cost_penalty_factor  # 自己成本（负面）
        attacker_cost_bonus = result.att_cost * cost_penalty_factor * 0.5  # 对方成本（正面，权重0.5）
        
        # 防守方训练奖励 = 占领奖励 + 成本优势
        training_reward = occupation_reward + defender_cost_penalty + attacker_cost_bonus
        
        return training_reward
    
    def _calculate_pure_occupation_reward(self) -> float:
        """
        计算纯净的占领奖励（仅用于统计对比）
        这是算法优势评估的核心指标
        """
        return self.occupation_reward
    
    def _get_observation(self) -> np.ndarray:
        """生成当前观察"""
        obs = np.zeros(13, dtype=np.float32)
        
        # 当前控制者
        obs[0] = self.current_controller
        
        # 剩余预算 (归一化)
        max_budget = max(self.attacker_budget, self.defender_budget) if self.attacker_budget != float('inf') else 100
        obs[1] = (self.attacker_budget - self.attacker_spent) / max_budget
        obs[2] = (self.defender_budget - self.defender_spent) / max_budget
        
        # 最近交战结果
        if self.engagement_history:
            latest = self.engagement_history[-1]
            obs[3] = latest.P_att
            obs[4] = latest.P_def  
            obs[5] = latest.P_final
        
        # 时间进度
        obs[6] = self.tick / self.duration
        
        # 环境修正因子
        env_values = list(self.environmental_modifiers.values())
        if len(env_values) >= 4:
            obs[7:11] = env_values[:4]
        else:
            # 如果环境修正因子不足4个，用默认值填充
            default_env = [1.0, 1.0, 1.0, 1.0]
            for i, val in enumerate(env_values):
                if i < 4:
                    default_env[i] = val
            obs[7:11] = default_env
        
        # 威胁评估
        recent_successes = sum(1 for r in self.engagement_history[-10:] if r.success)
        obs[11] = recent_successes / min(10, len(self.engagement_history)) if self.engagement_history else 0
        
        # 战术多样性
        obs[12] = len(self.tactical_patterns) / max(1, self.tick / 10)
        
        return obs
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.tick = 0
        self.current_controller = 0
        self._reset_budgets()
        self.engagement_history = []
        self.tactical_patterns = defaultdict(int)
        self.unit_effectiveness = defaultdict(lambda: {'used': 0, 'successful': 0})
        
        # 重置占领奖励统计
        self.total_occupation_rewards = [0.0, 0.0]
        
        # 重新设置环境因素 (添加随机性)
        if 'environmental_factors' in self.config:
            self._setup_environmental_modifiers(self.config['environmental_factors'])
        
        return self._get_observation()

    def _reset_budgets(self):
        """将预算重置为初始值"""
        self.attacker_budget = float(self.initial_attacker_budget)
        self.defender_budget = float(self.initial_defender_budget)
        self.attacker_spent = 0
        self.defender_spent = 0

    def _finalize_episode_outcome(self, info: Dict) -> Dict:
        """统一处理胜负判定和终止原因"""
        if 'winner' in info:
            # 已由预算透支等逻辑确定胜者，只补充缺失信息
            if 'winner_determination' not in info:
                info['winner_determination'] = info.get('failure_reason', 'explicit_condition')
            return info

        defender_score = self.total_occupation_rewards[0]
        attacker_score = self.total_occupation_rewards[1]

        if attacker_score > defender_score:
            info['winner'] = 'attacker'
        elif attacker_score < defender_score:
            info['winner'] = 'defender'
        else:
            info['winner'] = 'draw'

        if self.tick >= self.duration:
            info['winner_determination'] = 'duration_limit'
        else:
            info['winner_determination'] = 'score_comparison'

        return info
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"\n=== 海运非传统安全对抗状态 (回合 {self.tick}) ===")
            print(f"当前控制者: {'攻击方' if self.current_controller == 1 else '防守方'}")
            print(f"攻击方预算: {self.attacker_budget - self.attacker_spent:.1f}/{self.attacker_budget}")
            print(f"防守方预算: {self.defender_budget - self.defender_spent:.1f}/{self.defender_budget}")
            print(f"占领收益: 防守方{self.total_occupation_rewards[0]:.1f}, 攻击方{self.total_occupation_rewards[1]:.1f}")
            
            if self.engagement_history:
                latest = self.engagement_history[-1]
                print(f"最近交战: {latest.att_action}({latest.att_units}) vs {latest.def_action}({latest.def_units})")
                print(f"成功概率: 攻击{latest.P_att:.3f}, 防守{latest.P_def:.3f}, 最终{latest.P_final:.3f}")
                print(f"结果: {'攻击成功' if latest.success else '防守成功'}")
            
            print(f"环境修正: {self.environmental_modifiers}")
            print("-" * 60)
    
    def get_statistics(self) -> Dict:
        """获取详细统计信息"""
        stats = {
            'total_engagements': len(self.engagement_history),
            'attacker_success_rate': sum(1 for r in self.engagement_history if r.success) / max(1, len(self.engagement_history)),
            'average_att_units': np.mean([r.att_units for r in self.engagement_history]) if self.engagement_history else 0,
            'average_def_units': np.mean([r.def_units for r in self.engagement_history]) if self.engagement_history else 0,
            'total_costs': {
                'attacker': self.attacker_spent,
                'defender': self.defender_spent
            },
            'tactical_patterns': dict(self.tactical_patterns),
            'unit_effectiveness': dict(self.unit_effectiveness),
            'environmental_impact': np.mean(list(self.environmental_modifiers.values())),
            'budget_utilization': {
                'attacker': self.attacker_spent / self.attacker_budget if self.attacker_budget != float('inf') else 0,
                'defender': self.defender_spent / self.defender_budget if self.defender_budget != float('inf') else 0
            },
            'occupation_rewards': {
                'defender_total': self.total_occupation_rewards[0],
                'attacker_total': self.total_occupation_rewards[1],
                'per_step_reward': self.occupation_reward,
                'total_resource_extracted': sum(self.total_occupation_rewards)
            }
        }
        
        return stats
