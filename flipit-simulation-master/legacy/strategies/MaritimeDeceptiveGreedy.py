import scipy as sp
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.optimize as optimize
import numpy as np
import random
import math
from cachetools import cached, TTLCache

class MaritimeDeceptiveGreedy:
    """
    海运欺骗性贪心策略
    
    这个策略类专门为海运安全场景设计，支持欺骗攻击。
    结合了传统贪心策略的数学优化特性和海运环境的欺骗战术。
    
    海运应用场景：
    - 针对无人船舶导航系统的GPS欺骗攻击
    - 对智能港口管理系统的虚假信号注入
    - 结合真实攻击和欺骗攻击的混合策略
    """
    
    def __init__(self, move_cost=15, cheat_cost=5, opponent=None, debug=False, 
                 deception_threshold=0.6, maritime_context=True):
        self.strategy = 'maritime-deceptive-greedy'
        self.move_cost = move_cost        # 真实攻击成本
        self.cheat_cost = cheat_cost      # 欺骗攻击成本
        self.opponent = opponent
        self.debug = debug
        self.dropped = False
        
        # 海运欺骗特定参数
        self.deception_threshold = deception_threshold  # 欺骗动作阈值
        self.maritime_context = maritime_context
        
        # 海运环境参数
        self.vessel_vulnerability = 0.7    # 船舶系统脆弱性
        self.port_security_level = 0.6     # 港口安全等级
        self.ais_deception_success = 0.8   # AIS欺骗成功率
        self.gps_jamming_capability = 0.9  # GPS干扰能力
        
        # 动态学习参数
        self.defender_suspicion_estimate = 0.3  # 估计的防守方怀疑度
        self.cheat_success_history = []         # 欺骗成功历史
        self.environmental_factors = {
            'weather_interference': 0.1,    # 天气干扰（有助于欺骗）
            'maritime_traffic_density': 0.5, # 海上交通密度
            'communication_quality': 0.8     # 通信质量
        }
        
        # 策略状态
        self.last_action = 0
        self.consecutive_cheats = 0
        self.defender_response_pattern = {}

    def pre(self, tick, prev_observation):
        """
        决策函数：选择动作ID和单位数量 (action_id, n_units)
        """
        # 获取当前预算信息（如果可用）
        current_budget = 100.0  # 默认值
        if isinstance(prev_observation, dict):
             # 尝试从字典观察中获取
             pass
        elif hasattr(prev_observation, 'shape') and len(prev_observation.shape) > 0:
             # 从归一化向量中恢复（假设max_budget=100，这是环境默认值）
             # obs[1] 是 attacker_budget / max_budget
             if len(prev_observation) >= 2:
                 current_budget = prev_observation[1] * 100.0
        
        # 确定动作ID
        if tick == 0:
            action_id = self._initial_maritime_strategy()
        else:
            # 计算各动作的预期收益
            no_action_value = self._calculate_no_action_value(tick, prev_observation)
            real_attack_value = self._calculate_real_attack_value(tick, prev_observation)
            cheat_value = self._calculate_cheat_value(tick, prev_observation)
            
            # 新增：考虑远程火力 (Action 3)
            standoff_value = self._calculate_standoff_value(tick, prev_observation)
            
            if self.debug:
                print(f"\n海运策略评估 (回合 {tick}):")
                print(f"不行动价值: {no_action_value:.3f}")
                print(f"真实攻击价值: {real_attack_value:.3f}")
                print(f"欺骗攻击价值: {cheat_value:.3f}")
                print(f"远程压制价值: {standoff_value:.3f}")
            
            # 选择最优动作
            values = [no_action_value, real_attack_value, cheat_value, standoff_value]
            optimal_action = np.argmax(values)
            
            # 应用海运环境的随机性和不确定性
            action_id = self._apply_maritime_uncertainty(optimal_action, values)
            
        # 计算最优单位数量
        action_config = self._get_action_config(action_id)
        n_units = self._calculate_optimal_units(action_id, current_budget, action_config)
        
        # 更新状态
        self._update_strategy_state(action_id)
        
        return (action_id, n_units)

    def _get_action_config(self, action_id):
        """获取动作配置简表"""
        # 这里硬编码了环境配置中的关键参数，为了保持策略独立性
        configs = {
            0: {'cost': 0, 'max': 0},         # No Action
            1: {'cost': 2.0, 'max': 4},       # Inflatable Boat
            2: {'cost': 4.0, 'max': 3},       # Hard Hull Speedboat (原逻辑映射可能有误，这里按config修正)
            3: {'cost': 5.0, 'max': 2},       # Boarding Assault
            4: {'cost': 3.0, 'max': 3}        # Standoff Attack
        }
        # 注意：Greedy策略内部ID与环境ID的映射需要对齐
        # 原始代码假设：0=NoAction, 1=Real, 2=Cheat
        # 环境Action列表：0=inflatable, 1=hard_hull, 2=boarding, 3=standoff
        # 为了兼容，我们需要做一个映射：
        # Greedy 0 -> (0, 0)
        # Greedy 1 (Real) -> Action 1 (Hard Hull) or Action 0 (Inflatable)? 
        # 原代码只用了1和2。
        # 我们重新定义映射以利用所有动作：
        # Greedy 0 -> Action -1 (Wait)
        # Greedy 1 -> Action 0 (Inflatable) - 低成本试探
        # Greedy 2 -> Action 2 (Boarding) - 高风险强攻 (Cheat模式优势大)
        # Greedy 3 -> Action 3 (Standoff) - 中距离压制
        
        # 修正后的映射表
        if action_id == 0: return {'cost': 0, 'max': 0, 'env_id': 0} # 实际上是空动作，Env中处理为(0,0)
        if action_id == 1: return {'cost': 2.0, 'max': 4, 'env_id': 0} # Inflatable
        if action_id == 2: return {'cost': 5.0, 'max': 2, 'env_id': 2} # Boarding (最强)
        if action_id == 3: return {'cost': 3.0, 'max': 3, 'env_id': 3} # Standoff
        
        return {'cost': 2.0, 'max': 1, 'env_id': 0}

    def _calculate_optimal_units(self, action_id, budget, config):
        """计算最优单位投入量"""
        if action_id == 0:
            return 0
            
        cost_per_unit = config['cost']
        max_units = config['max']
        
        if cost_per_unit <= 0:
            return 1
            
        # 1. 预算约束
        max_affordable = int(budget // cost_per_unit)
        if max_affordable == 0:
            return 0 # 买不起
            
        # 2. 策略性投入
        # 资金充裕时(>20)，倾向于投入更多；资金紧张时保守
        if budget > 20:
            desired_units = max_units
        elif budget > 10:
            desired_units = max(1, int(max_units * 0.7))
        else:
            desired_units = 1 # 保守试探
            
        # 3. 取交集
        final_units = min(desired_units, max_affordable, max_units)
        return max(1, final_units) # 至少投入1个，除非买不起(前面已处理)

    def _calculate_standoff_value(self, tick, observation):
        """计算远程压制价值"""
        # 远程压制成本适中，且不易受反击（假设）
        base_value = 18.0
        # 如果最近多次被拦截，远程压制价值提升
        if self.consecutive_cheats < 0: # 表示失败
            base_value += 5.0
        return base_value



    def post(self, tick, prev_observation, observation, reward, action, true_action, info=None):
        """
        学习和更新，基于行动结果调整策略参数
        """
        # 更新防守方怀疑度估计
        if info and 'defender_suspicion' in info:
            actual_suspicion = info['defender_suspicion']
            # 使用指数移动平均更新估计
            self.defender_suspicion_estimate = (0.7 * self.defender_suspicion_estimate + 
                                              0.3 * actual_suspicion)
        
        # 处理欺骗动作的反馈
        if true_action == 2:  # 欺骗动作
            self._process_cheat_feedback(reward, info)
        
        # 学习防守方响应模式
        self._learn_defender_pattern(prev_observation, observation, true_action, info)
        
        # 调整海运环境因子
        self._update_maritime_factors(info)
        
        if self.debug and tick % 500 == 0:
            self._print_maritime_status(tick)

    def _initial_maritime_strategy(self):
        """初始海运策略：通常从侦察开始"""
        if random.random() < 0.7:  # 70%概率选择欺骗（侦察）
            return 2
        elif random.random() < 0.8:  # 20%概率潜伏
            return 0
        else:  # 10%概率直接攻击
            return 1

    def _calculate_no_action_value(self, tick, observation):
        """计算不行动的价值（潜伏收益）"""
        # 潜伏可以降低被发现风险，在海运中很有价值
        stealth_value = 5.0
        
        # 如果连续欺骗过多，潜伏价值增加
        if self.consecutive_cheats > 3:
            stealth_value += self.consecutive_cheats * 2
        
        # 考虑海运环境的隐蔽性
        maritime_stealth_bonus = (self.environmental_factors['weather_interference'] * 10 +
                                 (1 - self.environmental_factors['communication_quality']) * 8)
        
        return stealth_value + maritime_stealth_bonus

    def _calculate_real_attack_value(self, tick, observation):
        """计算真实攻击的预期价值"""
        # 处理复合观察状态
        if isinstance(observation, tuple):
            tau = observation[0]  # 对手上次行动时间
        else:
            tau = observation
        
        if not self.opponent:
            # 简化计算：基于观察时间的启发式评估
            if hasattr(tau, '__len__'):  # 如果是数组，取第一个元素
                tau_val = float(tau[0]) if len(tau) > 0 else 0.0
            else:
                tau_val = float(tau)
            base_gain = max(0.0, tau_val * 2 - self.move_cost)
            return base_gain * self.vessel_vulnerability
        
        try:
            # 基于对手分布的最优化计算（类似传统Greedy策略）
            opp_cdf = self.opponent.cdf(tau) if tau > 0 else 0
            
            if self.opponent.strategy == 'periodic':
                # 对周期性防守者的最优响应
                if tau > 2:
                    z_optimal = max(1, self.opponent.delta - tau)
                else:
                    z_optimal = self.opponent.delta
                
                expected_gain = self._calculate_expected_benefit(z_optimal, tau, opp_cdf)
                # 考虑船舶系统的脆弱性
                expected_gain *= self.vessel_vulnerability
                
            else:
                # 数值优化求解最优时机
                result = optimize.minimize(
                    lambda z: -self._calculate_expected_benefit(z[0], tau, opp_cdf),
                    x0=[max(1, self.move_cost)],
                    bounds=[(1, 1000)],
                    options={"maxiter": 20}
                )
                expected_gain = -result.fun if result.success else 0
            
            return max(0, expected_gain)
            
        except Exception as e:
            if self.debug:
                print(f"真实攻击计算错误: {e}")
            return max(0, tau - self.move_cost)

    def _calculate_cheat_value(self, tick, observation):
        """计算欺骗攻击的预期价值"""
        # 基础欺骗收益
        base_cheat_gain = 25
        
        # 根据防守方怀疑度调整成功概率
        success_probability = max(0.1, 1.0 - self.defender_suspicion_estimate)
        
        # 海运特定的欺骗优势
        maritime_deception_bonus = 0
        
        # AIS欺骗：在高密度交通区域效果更好
        ais_bonus = (self.environmental_factors['maritime_traffic_density'] * 
                    self.ais_deception_success * 15)
        
        # GPS欺骗：在恶劣天气下更容易成功
        gps_bonus = (self.environmental_factors['weather_interference'] * 
                    self.gps_jamming_capability * 20)
        
        # 通信干扰：通信质量差时欺骗更容易
        comm_bonus = ((1 - self.environmental_factors['communication_quality']) * 10)
        
        maritime_deception_bonus = ais_bonus + gps_bonus + comm_bonus
        
        # 考虑连续欺骗的递减效应
        consecutive_penalty = min(10, self.consecutive_cheats * 2)
        
        # 基于历史成功率的调整
        historical_success = np.mean(self.cheat_success_history[-10:]) if self.cheat_success_history else 0.5
        historical_bonus = historical_success * 10
        
        expected_cheat_value = (
            (base_cheat_gain + maritime_deception_bonus + historical_bonus - consecutive_penalty) 
            * success_probability - self.cheat_cost
        )
        
        return max(0, expected_cheat_value)

    @cached(TTLCache(maxsize=100, ttl=300))
    def _calculate_expected_benefit(self, z, tau, opp_cdf):
        """计算期望收益（缓存以提高性能）"""
        if z <= 0 or not self.opponent:
            return -self.move_cost
        
        try:
            # 第一项：积分计算
            first_term = integrate.quad(
                lambda x: x * self._opponent_pdf(x, tau, opp_cdf),
                0, z, limit=50
            )[0]
            
            # 第二项：剩余概率
            remaining_prob = integrate.quad(
                lambda x: self._opponent_pdf(x, tau, opp_cdf),
                z, np.inf, limit=50
            )[0]
            
            expected_benefit = (1/z) * (first_term + z * remaining_prob) - self.move_cost
            return expected_benefit
            
        except Exception as e:
            if self.debug:
                print(f"期望收益计算错误: {e}")
            return -self.move_cost

    def _opponent_pdf(self, x, tau, opp_cdf):
        """对手行动的概率密度函数"""
        if not self.opponent or opp_cdf >= 1:
            return 0.001  # 避免除零
        
        try:
            pdf_value = self.opponent.pdf(tau + x)
            return pdf_value / max(0.001, 1 - opp_cdf)
        except:
            return 0.001

    def _apply_maritime_uncertainty(self, optimal_action, values):
        """应用海运环境的不确定性和随机干扰"""
        # 海运环境存在各种不确定因素
        uncertainty_factor = (
            self.environmental_factors['weather_interference'] * 0.3 +
            (1 - self.environmental_factors['communication_quality']) * 0.2 +
            self.environmental_factors['maritime_traffic_density'] * 0.1
        )
        
        # 有一定概率受环境影响改变决策
        if random.random() < uncertainty_factor:
            # 随机选择次优动作
            sorted_indices = np.argsort(values)[::-1]  # 降序排列
            if len(sorted_indices) > 1 and random.random() < 0.5:
                return sorted_indices[1]  # 次优动作
            elif len(sorted_indices) > 2 and random.random() < 0.3:
                return sorted_indices[2]  # 第三优动作
        
        return optimal_action

    def _update_strategy_state(self, action):
        """更新策略内部状态"""
        if action == 2:  # 欺骗动作
            self.consecutive_cheats += 1
        else:
            self.consecutive_cheats = 0
        
        self.last_action = action

    def _process_cheat_feedback(self, reward, info):
        """处理欺骗动作的反馈"""
        if info:
            success = info.get('cheat_successful', False)
            detected = info.get('cheat_detected', False)
            
            # 记录成功率（用于历史学习）
            self.cheat_success_history.append(1.0 if success else 0.0)
            if len(self.cheat_success_history) > 20:
                self.cheat_success_history.pop(0)
            
            # 根据结果调整海运欺骗能力参数
            if success:
                self.ais_deception_success = min(0.95, self.ais_deception_success + 0.02)
                self.gps_jamming_capability = min(0.95, self.gps_jamming_capability + 0.01)
            elif detected:
                self.ais_deception_success = max(0.3, self.ais_deception_success - 0.05)
                self.gps_jamming_capability = max(0.4, self.gps_jamming_capability - 0.03)

    def _learn_defender_pattern(self, prev_obs, obs, action, info):
        """学习防守方的响应模式"""
        if info and prev_obs not in self.defender_response_pattern:
            self.defender_response_pattern[prev_obs] = {
                'responses': [],
                'suspicion_changes': []
            }
        
        if info:
            # 记录防守方的响应
            response_strength = info.get('port_security_status', 1)
            self.defender_response_pattern[prev_obs]['responses'].append(response_strength)
            
            # 记录怀疑度变化
            suspicion_change = info.get('defender_suspicion', 0.5) - self.defender_suspicion_estimate
            self.defender_response_pattern[prev_obs]['suspicion_changes'].append(suspicion_change)

    def _update_maritime_factors(self, info):
        """更新海运环境因子"""
        if info:
            # 根据威胁等级调整环境参数
            threat_level = info.get('maritime_threat_level', 0)
            if threat_level > 5:
                # 高威胁时通信质量可能下降，有利于欺骗
                self.environmental_factors['communication_quality'] *= 0.98
            
            # 根据港口安全状态调整
            port_status = info.get('port_security_status', 1)
            if port_status > 1:
                # 港口进入警戒状态，欺骗难度增加
                self.port_security_level = min(1.0, self.port_security_level + 0.05)
            else:
                # 港口安全状态正常，可以放松警惕
                self.port_security_level = max(0.3, self.port_security_level - 0.02)

    def _print_maritime_status(self, tick):
        """打印海运策略状态信息"""
        print(f"\n=== 海运欺骗性贪心策略状态 (回合 {tick}) ===")
        print(f"防守方怀疑度估计: {self.defender_suspicion_estimate:.3f}")
        print(f"连续欺骗次数: {self.consecutive_cheats}")
        print(f"AIS欺骗成功率: {self.ais_deception_success:.3f}")
        print(f"GPS干扰能力: {self.gps_jamming_capability:.3f}")
        print(f"港口安全等级: {self.port_security_level:.3f}")
        
        if self.cheat_success_history:
            recent_success = np.mean(self.cheat_success_history[-5:])
            print(f"最近欺骗成功率: {recent_success:.3f}")
        
        print("海运环境因子:")
        for factor, value in self.environmental_factors.items():
            print(f"  {factor}: {value:.3f}")
        print("-" * 50)

    def get_maritime_statistics(self):
        """获取海运策略统计信息"""
        return {
            'defender_suspicion_estimate': self.defender_suspicion_estimate,
            'consecutive_cheats': self.consecutive_cheats,
            'ais_deception_success': self.ais_deception_success,
            'gps_jamming_capability': self.gps_jamming_capability,
            'port_security_level': self.port_security_level,
            'cheat_success_rate': np.mean(self.cheat_success_history) if self.cheat_success_history else 0,
            'environmental_factors': self.environmental_factors.copy()
        }
