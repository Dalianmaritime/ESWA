import gym
from gym import error, utils, spaces
from gym.utils import seeding
import random
import numpy as np
from gym_flipit.legacy.strategies import periodic, exponential, uniform, normal
from gym_flipit.legacy.state import reset_state, set_obs_space, set_state
from gym_flipit.legacy.rew import calc_rew

class CheatFlipitEnv(gym.Env):
    """
    Cheat-FlipIt: 海运非传统安全博弈环境
    
    Description:
        Cheat-FlipIt是基于FLIPIT博弈的扩展模型，专门用于海运非传统安全对抗建模。
        在传统FlipIt基础上引入了欺骗机制，攻击方可以使用cheat动作诱导防守方
        浪费资源，模拟海运环境中针对无人船舶和智能港口的网络攻击场景。
    
    海运安全背景:
        - 攻击方：恶意行为者试图控制无人船舶或智能港口系统
        - 防守方：港口安全系统和船舶网络安全防护
        - 欺骗攻击：攻击方发送虚假信号诱导防守方采取不必要的安全措施
    
    Actions:
        Type: Discrete(3)
        Num Action
        0   Do not play (不行动 - 潜伏状态)
        1   Play (行动 - 真实攻击，试图控制资源)  
        2   Cheat (欺骗 - 发送虚假信号，诱导防守方浪费资源)
    
    Cheat Mechanism:
        - 当攻击方选择cheat动作时，会向防守方发送虚假的攻击信号
        - 防守方如果对虚假信号做出反应（翻转），会浪费防护资源
        - 欺骗成功会给攻击方带来收益，给防守方带来损失
        - 欺骗行为有一定的被发现概率，影响后续决策
    
    Maritime Security Context:
        - 资源控制：船舶导航系统、港口管理系统、货物追踪系统
        - 欺骗攻击：GPS欺骗、AIS信号伪造、虚假网络入侵告警
        - 防护响应：系统重启、安全模式切换、人工介入
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, state_type='composite', rew_type='maritime_security', 
                 rew_configs={'base_gain':0, 'move_penalty':3.0, 'cheat_gain':0, 'cheat_cost':2.5, 'detection_prob':0.3, 'occupation_reward':2.0},
                 p0='periodic', p0_configs={'delta':50}, duration=1000, 
                 p0_move_cost=5, p1_move_cost=15, cheat_detection_decay=0.95):
        self.config(state_type, rew_type, rew_configs, p0, p0_configs, 
                   duration, p0_move_cost, p1_move_cost, cheat_detection_decay)

    def config(self, state_type, rew_type, rew_configs, p0, p0_configs, 
              duration=1000, p0_move_cost=5, p1_move_cost=15, cheat_detection_decay=0.95):
        self.duration = duration
        self.state_type = state_type
        self.set_obs_space()
        
        # 扩展动作空间：0-不行动, 1-真实攻击, 2-欺骗攻击
        self.action_space = spaces.Discrete(3)
        
        self.rew_configs = rew_configs
        self.rew_type = rew_type
        self.p0_configs = p0_configs
        
        # 欺骗机制相关参数
        self.cheat_detection_decay = cheat_detection_decay  # 欺骗检测概率衰减
        self.cheat_history = []  # 欺骗行为历史
        self.defender_suspicion = 0.0  # 防守方怀疑度
        self.false_alarm_count = 0  # 虚假告警次数
        
        # 海运安全特定参数
        self.maritime_threat_level = 0  # 海运威胁等级
        self.port_security_status = 1   # 港口安全状态 (0-正常, 1-警戒, 2-高警戒)
        
        # 占领奖励机制
        self.occupation_reward = rew_configs.get('occupation_reward', 20.0)  # 每回合控制方获得的资源奖励
        self.total_occupation_rewards = [0.0, 0.0]  # [防守方累计, 攻击方累计]
        
        if p0 == 'periodic':
            self.p0 = periodic.Periodic()
        elif p0 == 'exponential':
            self.p0 = exponential.Exponential()
        elif p0 == 'uniform':
            self.p0 = uniform.Uniform()
        elif p0 == 'normal':
            self.p0 = normal.Normal()
        else:
            raise ValueError(f"未知的防守方策略: {p0}")
        
        self.player_move_costs = [p0_move_cost, p1_move_cost]
        self.reset()
    
    def reset(self):
        """重置海运安全博弈环境"""
        self.p0.config(self.p0_configs)
        self.player_moves = [[0], [0]]  # [防守方行动记录, 攻击方行动记录]
        self.cheat_moves = [0]  # 攻击方欺骗行动记录
        self.player_total_gain = [0, 0]
        self.player_total_move_cost = [0, 0]
        self.cheat_total_cost = 0  # 欺骗行为总成本
        self.successful_cheats = 0  # 成功欺骗次数
        self.detected_cheats = 0   # 被发现的欺骗次数
        
        # 防守方下次行动时间
        self.p0_next_move = self.p0.first_move()
        
        self.controller = 0  # 当前控制者 (0-防守方, 1-攻击方)
        self.tick = 0
        self.found_FM = False  # 是否发现对手首次行动
        
        # 重置欺骗相关状态
        self.cheat_history = []
        self.defender_suspicion = 0.0
        self.false_alarm_count = 0
        self.maritime_threat_level = 0
        self.port_security_status = 1
        
        # 重置占领奖励统计
        self.total_occupation_rewards = [0.0, 0.0]
        
        self.reset_state()
        return self.state

    def step(self, action):
        """执行一步博弈行动"""
        assert self.action_space.contains(action), f"动作 {action} 无效"
        
        self.tick += 1
        true_action = action
        cheat_detected = False
        cheat_successful = False
        
        # 处理攻击方的欺骗行动
        if action == 2:  # 欺骗行动
            cheat_detected, cheat_successful = self._handle_cheat_action()
            self.cheat_moves.append(self.tick)
            self.cheat_total_cost += self.rew_configs.get('cheat_cost', 5)
            
            # 如果欺骗成功，可能诱导防守方提前行动
            if cheat_successful and random.random() < 0.6:  # 60%概率诱导防守方行动
                self._induce_defender_response()
        
        # 处理真实攻击行动 
        elif action == 1:  # 真实攻击
            # 如果与防守方同时行动，防守方优先（海运安全中防守方反应迅速）
            if self.tick == self.p0_next_move:
                true_action = 0  # 攻击被阻止
        
        # 防守方按策略行动
        if self.tick == self.p0_next_move:
            self.move(0)
            self.p0_next_move = self.p0.move(self.tick)
        
        # 攻击方真实行动生效
        if true_action == 1:
            self.move(1)
            if self.get_LM(0) > 0:
                self.found_FM = True
        
        # 更新环境状态
        self.set_state()
        base_reward = self.calc_maritime_reward(true_action, cheat_detected, cheat_successful)
        
        # 添加占领奖励：每个时间步，当前控制方获得资源收益
        occupation_bonus = self.occupation_reward
        self.total_occupation_rewards[self.controller] += occupation_bonus
        
        # 总奖励 = 基础奖励 + 占领奖励
        reward = base_reward + occupation_bonus
        done = self.tick >= self.duration
        
        # 添加资源耗尽失败判定机制
        attacker_total_reward = self.total_occupation_rewards[1] - self.player_total_move_cost[1] - self.cheat_total_cost
        defender_total_reward = self.total_occupation_rewards[0] - self.player_total_move_cost[0]
        
        # 更新控制者收益
        self.player_total_gain[self.controller] += 1
        
        # 更新海运威胁等级
        self._update_maritime_threat_level(action, cheat_detected)
        
        info = {
            'true_action': true_action,
            'cheat_detected': cheat_detected,
            'cheat_successful': cheat_successful,
            'defender_suspicion': self.defender_suspicion,
            'maritime_threat_level': self.maritime_threat_level,
            'port_security_status': self.port_security_status,
            'successful_cheats': self.successful_cheats,
            'detected_cheats': self.detected_cheats,
            'base_reward': base_reward,
            'occupation_bonus': occupation_bonus,
            'total_occupation_rewards': {
                'defender': self.total_occupation_rewards[0],
                'attacker': self.total_occupation_rewards[1]
            }
        }
        
        # 检查资源耗尽失败判定
        # 修复：只有在游戏进行一定步数后才启用严格的资源判定
        if self.tick > 5:  # 至少进行5步后才检查
            # 计算总收益（包括占领奖励）
            attacker_total_reward = self.total_occupation_rewards[1] - self.player_total_move_cost[1] - self.cheat_total_cost
            defender_total_reward = self.total_occupation_rewards[0] - self.player_total_move_cost[0]
            
            # 严格的预算透支限制：任意一方透支超过-30，游戏立即结束
            if attacker_total_reward <= -30:
                done = True
                info['failure_reason'] = 'attacker_budget_overdraft'
                info['winner'] = 'defender'
                info['early_termination'] = True
                info['budget_overdraft'] = attacker_total_reward
                print(f"[BUDGET] 攻击方预算透支{attacker_total_reward:.1f}，游戏结束，防守方获胜")
            elif defender_total_reward <= -30:
                done = True
                info['failure_reason'] = 'defender_budget_overdraft'
                info['winner'] = 'attacker'
                info['early_termination'] = True
                info['budget_overdraft'] = defender_total_reward
                print(f"[BUDGET] 防守方预算透支{defender_total_reward:.1f}，游戏结束，攻击方获胜")
        
        return self.state, reward, done, info

    def _handle_cheat_action(self):
        """处理欺骗行动的逻辑"""
        # 计算欺骗被检测的概率（基于历史欺骗行为和防守方怀疑度）
        base_detection_prob = self.rew_configs.get('detection_prob', 0.3)
        adjusted_detection_prob = min(0.9, base_detection_prob + self.defender_suspicion * 0.5)
        
        cheat_detected = random.random() < adjusted_detection_prob
        
        if cheat_detected:
            # 欺骗被发现
            self.detected_cheats += 1
            self.defender_suspicion = min(1.0, self.defender_suspicion + 0.2)
            cheat_successful = False
        else:
            # 欺骗成功
            self.successful_cheats += 1
            self.defender_suspicion = max(0.0, self.defender_suspicion - 0.1)
            cheat_successful = True
        
        # 记录欺骗历史
        self.cheat_history.append({
            'tick': self.tick,
            'detected': cheat_detected,
            'successful': cheat_successful
        })
        
        return cheat_detected, cheat_successful

    def _induce_defender_response(self):
        """诱导防守方提前响应（浪费资源）"""
        if random.random() < 0.7:  # 70%概率诱导成功
            # 防守方提前行动，浪费资源
            self.false_alarm_count += 1
            self.player_total_move_cost[0] += self.player_move_costs[0] * 0.5  # 额外成本
            
            # 更新港口安全状态
            if self.port_security_status < 2:
                self.port_security_status += 1

    def _update_maritime_threat_level(self, action, cheat_detected):
        """更新海运威胁等级"""
        if action == 1:  # 真实攻击
            self.maritime_threat_level += 2
        elif action == 2:  # 欺骗攻击
            if cheat_detected:
                self.maritime_threat_level += 1
            else:
                self.maritime_threat_level += 0.5
        
        # 威胁等级自然衰减
        self.maritime_threat_level = max(0, self.maritime_threat_level - 0.1)

    def calc_maritime_reward(self, action, cheat_detected, cheat_successful):
        """计算海运安全场景下的奖励（只保留成本惩罚，删除成功奖励）"""
        if action == 0:  # 不行动
            return 0
        
        elif action == 1:  # 真实攻击
            move_penalty = self.rew_configs.get('move_penalty', 10)
            # 只有成本惩罚，无成功奖励
            return -move_penalty
        
        elif action == 2:  # 欺骗攻击
            cheat_cost = self.rew_configs.get('cheat_cost', 5)
            
            if cheat_detected:
                # 欺骗被发现，承担额外惩罚
                detection_penalty = cheat_cost * 2
                return -detection_penalty
            else:
                # 欺骗无论成功与否，都只有成本
                return -cheat_cost

    def set_obs_space(self):
        """设置观察空间（扩展以包含欺骗信息）"""
        if self.state_type == 'opp_LM':
            self.observation_space = set_obs_space.opp_LM(self.duration)
        elif self.state_type == 'own_LM':
            self.observation_space = set_obs_space.own_LM(self.duration)
        elif self.state_type == 'composite':
            # 扩展复合观察空间，包含欺骗相关信息
            # (对手上次行动时间, 自己上次行动时间, 欺骗成功次数, 防守方怀疑度*10)
            self.observation_space = spaces.Tuple((
                spaces.Discrete(self.duration + 1),  # 对手上次行动时间
                spaces.Discrete(self.duration + 1),  # 自己上次行动时间  
                spaces.Discrete(100),                # 欺骗成功次数
                spaces.Discrete(11)                  # 防守方怀疑度 (0-10)
            ))
        else:
            raise NotImplementedError(f"未实现的观察类型: {self.state_type}")

    def reset_state(self):
        """重置状态"""
        if self.state_type == 'opp_LM':
            self.state = reset_state.opp_LM()
        elif self.state_type == 'own_LM':
            self.state = reset_state.own_LM()
        elif self.state_type == 'composite':
            self.state = (0, 0, 0, 0)  # (opp_LM, own_LM, successful_cheats, suspicion)
        else:
            raise NotImplementedError

    def set_state(self):
        """设置当前状态"""
        if self.state_type == 'opp_LM':
            self.state = set_state.opp_LM(self.get_LM(0), self.moved(1), 
                                        self.found_FM, self.tick, self.state)
        elif self.state_type == 'own_LM':
            self.state = set_state.own_LM(self.moved(1), self.state)
        elif self.state_type == 'composite':
            opp_lm = self.get_LM(0) if self.found_FM else 0
            own_lm = self.tick - self.get_LM(1) if len(self.player_moves[1]) > 1 else self.tick
            suspicion_level = int(self.defender_suspicion * 10)
            self.state = (
                min(opp_lm, self.duration),
                min(own_lm, self.duration), 
                min(self.successful_cheats, 99),
                min(suspicion_level, 10)
            )
        else:
            raise NotImplementedError

    def get_LM(self, player):
        """获取玩家上次行动时间"""
        if len(self.player_moves[player]) > 0:
            return self.tick - self.player_moves[player][-1]
        return self.tick

    def moved(self, player):
        """检查玩家是否在当前回合行动"""
        return (len(self.player_moves[player]) > 0 and 
                self.tick == self.player_moves[player][-1])

    def move(self, player):
        """玩家行动"""
        self.player_moves[player].append(self.tick)
        self.player_total_move_cost[player] += self.player_move_costs[player]
        self.controller = player

    def calc_benefit(self, player):
        """计算玩家总收益"""
        base_benefit = self.player_total_gain[player] - self.player_total_move_cost[player]
        
        # 添加占领奖励
        occupation_benefit = self.total_occupation_rewards[player]
        
        if player == 1:  # 攻击方
            # 减去欺骗成本，加上欺骗收益
            cheat_benefit = (self.successful_cheats * self.rew_configs.get('cheat_gain', 30) 
                           - self.cheat_total_cost)
            return base_benefit + cheat_benefit + occupation_benefit
        else:  # 防守方
            # 减去因虚假告警产生的额外成本
            false_alarm_penalty = self.false_alarm_count * self.player_move_costs[0] * 0.5
            return base_benefit - false_alarm_penalty + occupation_benefit

    def render(self, mode='human'):
        """渲染海运安全博弈状态"""
        if mode == 'human':
            print(f"\n=== 海运安全博弈状态 (回合 {self.tick}) ===")
            print(f"当前控制者: {'攻击方' if self.controller == 1 else '防守方'}")
            print(f"海运威胁等级: {self.maritime_threat_level:.2f}")
            print(f"港口安全状态: {['正常', '警戒', '高警戒'][self.port_security_status]}")
            print(f"防守方怀疑度: {self.defender_suspicion:.2f}")
            print(f"成功欺骗次数: {self.successful_cheats}")
            print(f"被发现欺骗次数: {self.detected_cheats}")
            print(f"虚假告警次数: {self.false_alarm_count}")
            print(f"占领收益: 防守方{self.total_occupation_rewards[0]:.1f}, 攻击方{self.total_occupation_rewards[1]:.1f}")
            print(f"攻击方收益: {self.calc_benefit(1)}")
            print(f"防守方收益: {self.calc_benefit(0)}")

    def get_info(self):
        """获取环境详细信息"""
        return {
            'total_ticks': self.tick,
            'controller': self.controller,
            'maritime_threat_level': self.maritime_threat_level,
            'port_security_status': self.port_security_status,
            'defender_suspicion': self.defender_suspicion,
            'successful_cheats': self.successful_cheats,
            'detected_cheats': self.detected_cheats,
            'false_alarm_count': self.false_alarm_count,
            'attacker_benefit': self.calc_benefit(1),
            'defender_benefit': self.calc_benefit(0),
            'cheat_success_rate': (self.successful_cheats / max(1, len(self.cheat_history))),
            'cheat_detection_rate': (self.detected_cheats / max(1, len(self.cheat_history))),
            'occupation_rewards': {
                'defender_total': self.total_occupation_rewards[0],
                'attacker_total': self.total_occupation_rewards[1],
                'per_step_reward': self.occupation_reward,
                'total_resource_extracted': sum(self.total_occupation_rewards)
            }
        }
