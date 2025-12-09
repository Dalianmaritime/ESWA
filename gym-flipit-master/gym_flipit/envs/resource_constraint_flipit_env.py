import gym
from gym import error, utils, spaces
from gym.utils import seeding
import random
import numpy as np
from gym_flipit.envs.strategies import periodic, exponential, uniform, normal
from gym_flipit.envs.state import reset_state, set_obs_space, set_state
from gym_flipit.envs.rew import calc_rew

class ResourceConstraintFlipitEnv(gym.Env):
    """
    资源约束Cheat-FlipIt: 海运安全资源管理博弈环境
    
    Description:
        在Cheat-FlipIt基础上引入资源约束机制，模拟真实海运安全场景中的资源管理。
        攻击方（海盗）初始资源充足但无法补给，防守方初始资源较少但控制节点时可获得补给。
    
    核心资源类型:
        - 攻击方资源：船只燃料、武器弹药、人员工资、设备维护
        - 防守方资源：安保人员、防护设备、通信系统、后勤补给
    
    Actions:
        Type: Discrete(5)
        Num Action              Resource Cost    Description
        0   Do not play         0               潜伏观察，无资源消耗
        1   Light Attack        3               轻型攻击（小型快艇袭击）
        2   Heavy Attack        8               重型攻击（武装突击）
        3   Cheat Attack        2               欺骗攻击（虚假信号）
        4   Stealth Attack      12              隐蔽攻击（夜间渗透）
    
    海运具体场景映射:
        - Light Attack: 单艘充气快艇侦察/骚扰
        - Heavy Attack: 多艘武装快艇协同攻击
        - Cheat Attack: 发送虚假GPS信号、伪造求救信号
        - Stealth Attack: 夜间潜水接近、无声登台
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, scenario_type='pirate_assault', state_type='composite', 
                 rew_type='resource_maritime', duration=1000):
        """
        初始化资源约束环境
        
        Args:
            scenario_type: 场景类型 ('pirate_assault', 'offshore_defense', 'convoy_escort')
            state_type: 观察空间类型
            rew_type: 奖励类型
            duration: 游戏持续时间
        """
        self.scenario_type = scenario_type
        self.duration = duration
        self.state_type = state_type
        self.rew_type = rew_type
        
        # 设置场景特定参数
        self._setup_scenario_configs()
        
        # 设置观察和动作空间
        self.action_space = spaces.Discrete(5)
        self.set_obs_space()
        
        self.reset()

    def _setup_scenario_configs(self):
        """根据场景类型设置配置参数"""
        scenario_configs = {
            'pirate_assault': {
                'name': '海盗袭击商船',
                'target': '货运商船',
                'attacker_initial_resources': 25,  # 海盗有充足准备
                'defender_initial_resources': 8,   # 商船防护有限
                'resource_gain_rate': 1.2,         # 控制商船的资源获取率
                'action_costs': {
                    0: 0,   # 观察
                    1: 3,   # 小快艇
                    2: 8,   # 武装突击  
                    3: 2,   # 虚假信号
                    4: 12   # 夜间渗透
                },
                'defender_action_costs': {
                    0: 0,   # 正常巡航
                    1: 4,   # 警戒状态
                    2: 9    # 全面防御
                },
                'environmental_factors': {
                    'weather_impact': 0.15,
                    'visibility_factor': 0.8,
                    'sea_conditions': 'moderate'
                }
            },
            'offshore_defense': {
                'name': '海上石油平台防御',
                'target': '石油钻井平台',
                'attacker_initial_resources': 20,
                'defender_initial_resources': 15,  # 平台有更好的防护
                'resource_gain_rate': 2.0,         # 石油平台资源丰富
                'action_costs': {
                    0: 0,   # 观察
                    1: 4,   # 小快艇
                    2: 10,  # 武装突击
                    3: 3,   # 虚假信号  
                    4: 15   # 夜间渗透
                },
                'defender_action_costs': {
                    0: 1,   # 平台运行成本
                    1: 6,   # 安保警戒
                    2: 12   # 全面戒备
                },
                'environmental_factors': {
                    'weather_impact': 0.25,     # 海上天气影响更大
                    'visibility_factor': 0.6,  # 平台灯光影响
                    'sea_conditions': 'rough'
                }
            },
            'convoy_escort': {
                'name': '军舰护卫商船队',
                'target': '商船护航编队',
                'attacker_initial_resources': 18,
                'defender_initial_resources': 22,  # 军舰护卫资源充足
                'resource_gain_rate': 0.8,         # 护航期间资源获取有限
                'action_costs': {
                    0: 0,   # 观察
                    1: 2,   # 小快艇（容易被发现）
                    2: 6,   # 武装突击（风险极高）
                    3: 4,   # 虚假信号（军舰有反制）
                    4: 18   # 夜间渗透（几乎不可能）
                },
                'defender_action_costs': {
                    0: 2,   # 正常护航成本
                    1: 5,   # 提高警戒
                    2: 8    # 战斗准备
                },
                'environmental_factors': {
                    'weather_impact': 0.20,
                    'visibility_factor': 0.9,  # 军舰雷达覆盖
                    'sea_conditions': 'variable'
                }
            }
        }
        
        self.config = scenario_configs[self.scenario_type]
        
        # 设置奖励配置
        self.rew_configs = {
            'base_gain': 100,
            'resource_multiplier': 1.5,
            'cheat_gain': 25,
            'cheat_detection_penalty': 15,
            'resource_depletion_penalty': 50,
            'critical_resource_threshold': 3
        }

    def reset(self):
        """重置环境，包括资源状态"""
        # 初始化资源
        self.attacker_resources = self.config['attacker_initial_resources']
        self.defender_resources = self.config['defender_initial_resources']
        
        # 资源历史记录
        self.attacker_resource_history = [self.attacker_resources]
        self.defender_resource_history = [self.defender_resources]
        
        # 游戏状态
        self.player_moves = [[0], [0]]
        self.player_total_gain = [0, 0]
        self.cheat_moves = [0]
        self.successful_cheats = 0
        self.detected_cheats = 0
        self.resource_depletion_count = [0, 0]  # 资源耗尽次数
        
        # 设置防守方策略
        self.p0 = periodic.Periodic()
        self.p0.config({'delta': 50})
        self.p0_next_move = self.p0.first_move()
        
        self.controller = 0
        self.tick = 0
        self.found_FM = False
        
        # 欺骗相关状态
        self.defender_suspicion = 0.0
        self.cheat_history = []
        
        # 海运环境因素
        self.environmental_modifier = self._calculate_environmental_modifier()
        
        self.reset_state()
        return self.state

    def step(self, action):
        """执行一步，包括资源检查和扣除"""
        assert self.action_space.contains(action), f"动作 {action} 无效"
        
        # 保存原始动作
        original_action = action
        
        # 检查攻击方是否有足够资源
        original_action_cost = self.config['action_costs'][action]
        if self.attacker_resources < original_action_cost:
            # 资源不足，强制选择无成本行动
            action = 0
            self.resource_depletion_count[1] += 1
        
        # 重新计算实际行动成本
        action_cost = self.config['action_costs'][action]
        
        self.tick += 1
        true_action = action
        cheat_detected = False
        cheat_successful = False
        
        # 扣除攻击方行动成本
        if action_cost > 0:
            self.attacker_resources = max(0, self.attacker_resources - action_cost)
        
        # 处理不同类型的攻击
        if action == 1:  # Light Attack - 轻型攻击
            success_rate = 0.4 * self.environmental_modifier
            if random.random() < success_rate:
                true_action = action
            else:
                true_action = 0
                
        elif action == 2:  # Heavy Attack - 重型攻击  
            success_rate = 0.7 * self.environmental_modifier
            if random.random() < success_rate:
                true_action = action
            else:
                true_action = 0
                
        elif action == 3:  # Cheat Attack - 欺骗攻击
            cheat_detected, cheat_successful = self._handle_cheat_action()
            self.cheat_moves.append(self.tick)
            
        elif action == 4:  # Stealth Attack - 隐蔽攻击
            success_rate = 0.8 * self.environmental_modifier
            if random.random() < success_rate:
                true_action = action
            else:
                true_action = 0
        
        # 防守方按策略行动
        if self.tick == self.p0_next_move:
            self._defender_move()
            self.p0_next_move = self.p0.move(self.tick)
        
        # 处理攻击方真实行动
        if true_action in [1, 2, 4]:  # 真实攻击类型
            if self.tick == self.p0_next_move:
                # 防守方同时行动，攻击失败
                true_action = 0
            else:
                self.move(1)  # 攻击方获得控制权
                if self.get_LM(0) > 0:
                    self.found_FM = True
        
        # 资源增长（基于控制权）
        self._update_resources()
        
        # 更新状态
        self.set_state()
        reward = self._calculate_resource_aware_reward(true_action, cheat_detected, cheat_successful, action)
        done = self.tick >= self.duration or self._check_game_end_conditions()
        
        # 记录资源历史
        self.attacker_resource_history.append(self.attacker_resources)
        self.defender_resource_history.append(self.defender_resources)
        
        info = {
            'true_action': true_action,
            'original_action': original_action,
            'action_cost': action_cost,
            'attacker_resources': self.attacker_resources,
            'defender_resources': self.defender_resources,
            'cheat_detected': cheat_detected,
            'cheat_successful': cheat_successful,
            'environmental_modifier': self.environmental_modifier,
            'scenario': self.config['name'],
            'resource_depletion_count': self.resource_depletion_count.copy()
        }
        
        return self.state, reward, done, info

    def _defender_move(self):
        """防守方行动，包括资源消耗"""
        # 根据威胁等级选择防守强度
        threat_level = self._assess_threat_level()
        
        if threat_level < 0.3:
            defend_action = 0  # 正常状态
        elif threat_level < 0.7:
            defend_action = 1  # 提高警戒
        else:
            defend_action = 2  # 全面防御
        
        # 检查防守方资源
        defend_cost = self.config['defender_action_costs'][defend_action]
        if self.defender_resources >= defend_cost:
            self.defender_resources = max(0, self.defender_resources - defend_cost)
            self.move(0)  # 防守方重新获得控制权
        else:
            # 资源不足，降级防御
            self.resource_depletion_count[0] += 1
            if self.defender_resources >= self.config['defender_action_costs'][0]:
                self.defender_resources = max(0, self.defender_resources - self.config['defender_action_costs'][0])
                self.move(0)

    def _update_resources(self):
        """基于控制权更新资源"""
        if self.controller == 0:  # 防守方控制
            # 防守方获得资源补给
            resource_gain = self.config['resource_gain_rate']
            self.defender_resources += resource_gain
        elif self.controller == 1:  # 攻击方控制
            # 攻击方从控制中获得少量资源（掠夺）
            plunder_gain = self.config['resource_gain_rate'] * 0.3
            self.attacker_resources += plunder_gain
        
        # 资源上限
        self.attacker_resources = min(50, self.attacker_resources)
        self.defender_resources = min(60, self.defender_resources)

    def _assess_threat_level(self):
        """评估当前威胁等级"""
        recent_attacks = sum(1 for move in self.player_moves[1][-5:] if move > 0)
        recent_cheats = len([c for c in self.cheat_history[-3:] if c['successful']])
        
        base_threat = (recent_attacks * 0.2 + recent_cheats * 0.15)
        resource_pressure = max(0, (30 - self.defender_resources) / 30 * 0.3)
        
        return min(1.0, base_threat + resource_pressure + self.defender_suspicion * 0.2)

    def _handle_cheat_action(self):
        """处理欺骗攻击"""
        base_detection_prob = 0.25
        adjusted_detection_prob = min(0.85, base_detection_prob + self.defender_suspicion * 0.4)
        
        cheat_detected = random.random() < adjusted_detection_prob
        
        if cheat_detected:
            self.detected_cheats += 1
            self.defender_suspicion = min(1.0, self.defender_suspicion + 0.15)
            cheat_successful = False
        else:
            self.successful_cheats += 1
            self.defender_suspicion = max(0.0, self.defender_suspicion - 0.05)
            cheat_successful = True
        
        self.cheat_history.append({
            'tick': self.tick,
            'detected': cheat_detected,
            'successful': cheat_successful
        })
        
        return cheat_detected, cheat_successful

    def _calculate_environmental_modifier(self):
        """计算环境因素修正值"""
        weather_factor = 1.0 - self.config['environmental_factors']['weather_impact'] * random.random()
        visibility_factor = self.config['environmental_factors']['visibility_factor']
        
        return weather_factor * visibility_factor

    def _calculate_resource_aware_reward(self, true_action, cheat_detected, cheat_successful, original_action):
        """计算考虑资源因素的奖励"""
        base_reward = 0
        
        if true_action == 0:  # 无行动
            base_reward = 0
        elif true_action in [1, 2, 4]:  # 真实攻击成功
            if self.controller == 1:  # 获得控制权
                base_reward = self.rew_configs['base_gain']
                # 资源奖励加成
                if self.attacker_resources > 15:
                    base_reward *= self.rew_configs['resource_multiplier']
            else:
                base_reward = -10  # 攻击失败
        elif true_action == 3:  # 欺骗攻击
            if cheat_detected:
                base_reward = -self.rew_configs['cheat_detection_penalty']
            elif cheat_successful:
                base_reward = self.rew_configs['cheat_gain']
            else:
                base_reward = -5
        
        # 资源枯竭惩罚
        if original_action != 0 and self.attacker_resources <= self.rew_configs['critical_resource_threshold']:
            base_reward -= self.rew_configs['resource_depletion_penalty']
        
        # 环境因素影响
        base_reward *= self.environmental_modifier
        
        return base_reward

    def _check_game_end_conditions(self):
        """检查游戏结束条件"""
        # 双方资源都耗尽
        if self.attacker_resources <= 0 and self.defender_resources <= 0:
            return True
        
        # 一方持续资源枯竭
        if self.resource_depletion_count[1] > 10:  # 攻击方资源枯竭次数过多
            return True
            
        return False

    def set_obs_space(self):
        """设置扩展的观察空间，包含资源信息"""
        if self.state_type == 'composite':
            # (对手LM, 自己LM, 成功欺骗次数, 怀疑度, 攻击方资源, 防守方资源, 环境修正)
            self.observation_space = spaces.Tuple((
                spaces.Discrete(self.duration + 1),  # 对手上次行动时间
                spaces.Discrete(self.duration + 1),  # 自己上次行动时间  
                spaces.Discrete(50),                 # 成功欺骗次数
                spaces.Discrete(11),                 # 防守方怀疑度 (0-10)
                spaces.Discrete(51),                 # 攻击方资源 (0-50)
                spaces.Discrete(61),                 # 防守方资源 (0-60)
                spaces.Discrete(101)                 # 环境修正 (0-100, 表示0.0-1.0)
            ))

    def reset_state(self):
        """重置状态"""
        if self.state_type == 'composite':
            self.state = (0, 0, 0, 0, self.attacker_resources, self.defender_resources, int(self.environmental_modifier * 100))

    def set_state(self):
        """更新当前状态"""
        if self.state_type == 'composite':
            opp_lm = self.get_LM(0) if self.found_FM else 0
            own_lm = self.tick - self.get_LM(1) if len(self.player_moves[1]) > 1 else self.tick
            suspicion_level = int(self.defender_suspicion * 10)
            
            self.state = (
                min(opp_lm, self.duration),
                min(own_lm, self.duration), 
                min(self.successful_cheats, 49),
                min(suspicion_level, 10),
                min(self.attacker_resources, 50),
                min(self.defender_resources, 60),
                int(self.environmental_modifier * 100)
            )

    def get_LM(self, player):
        """获取玩家上次行动时间"""
        if len(self.player_moves[player]) > 0:
            return self.tick - self.player_moves[player][-1]
        return self.tick

    def move(self, player):
        """玩家行动"""
        self.player_moves[player].append(self.tick)
        self.controller = player

    def render(self, mode='human'):
        """渲染资源约束海运博弈状态"""
        if mode == 'human':
            print(f"\n=== {self.config['name']} (回合 {self.tick}) ===")
            print(f"目标: {self.config['target']}")
            print(f"当前控制者: {'攻击方' if self.controller == 1 else '防守方'}")
            print(f"攻击方资源: {self.attacker_resources:.1f}")
            print(f"防守方资源: {self.defender_resources:.1f}")
            print(f"防守方怀疑度: {self.defender_suspicion:.2f}")
            print(f"环境影响系数: {self.environmental_modifier:.2f}")
            print(f"成功欺骗次数: {self.successful_cheats}")
            print(f"资源枯竭次数: 攻击方{self.resource_depletion_count[1]}, 防守方{self.resource_depletion_count[0]}")

    def get_resource_info(self):
        """获取详细的资源信息"""
        return {
            'scenario': self.config['name'],
            'attacker_resources': self.attacker_resources,
            'defender_resources': self.defender_resources,
            'attacker_resource_history': self.attacker_resource_history.copy(),
            'defender_resource_history': self.defender_resource_history.copy(),
            'resource_depletion_count': self.resource_depletion_count.copy(),
            'action_costs': self.config['action_costs'],
            'defender_action_costs': self.config['defender_action_costs'],
            'environmental_modifier': self.environmental_modifier
        }
