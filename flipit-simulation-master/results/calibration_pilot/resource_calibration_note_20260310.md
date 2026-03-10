# V2 资源可持续性机制标定说明

## 1. 说明目的
本说明用于解释 V2 新资源机制的标定逻辑，而不是替代正式论文主实验。

本轮标定只回答三个问题：

1. 新资源机制是否已经从“软预算约束”变成了真正影响攻防节奏的机制。
2. 资源崩溃是否是“有意义但不过度主导”的失败模式。
3. `cheat` 场景是否仍然比 `flipit` 更难，从而符合“欺骗提升防守难度”的研究叙事。

## 2. 标定原则
本轮不是按真实美元成本或真实海事预算做经验拟合，而是采用“结构标定”：

- 用文献和官方指南确定参数关系，而不是直接假装有精确现场数据。
- 先保证相对关系合理，再用 pilot 数据把参数推入一个可发表、可训练、可解释的区间。

本轮采用的四条原则如下。

### 原则 A：控制权必须影响可持续性
关键海上设施一旦被一方持续控制，其后勤、响应和持续作业能力不可能与失控状态相同。

因此预算更新不再使用“双方每步固定回血”的旧逻辑，而是：

`budget_{t+1} = budget_t - action_cost + base_income + control_bonus_if_controller`

这样处理后，控制权不只是分数，还会影响后续行动能力。

### 原则 B：允许短期超耗，但不允许持续失血
海事应急中，短时间超耗是现实存在的；但连续低于最低保障线，应被视为任务体系崩溃。

因此本轮采用：

- `action_floor`：限制一次动作后最多能透支到哪里
- `guarantee_line`：低于该线则记为保障失守
- `guarantee_breach_patience`：连续若干步低于保障线才判负

这比“预算一到 0 立即失败”更符合连续性管理和 contingency planning 逻辑。

### 原则 C：欺骗动作必须明显便宜于真实 takeover
海事欺骗如 AIS spoofing、假航迹、假求救、假目标，不应与真实接近、登临、接管拥有相同成本等级。

因此 `cheat_cost` 必须显著低于 `takeover_cost`。

### 原则 D：防守方可以有主场后勤优势，但不能失真成无限资源
防守方作为设施运营者，理应有轻微资源优势；但如果控制后预算持续失真式增长，资源机制就会退化成只约束攻击者。

因此标定时同时观测：

- `attacker_resource_collapse_rate`
- `defender_resource_collapse_rate`
- `avg_final_defender_budget`

## 3. 参考依据
本轮标定主要依据下列来源。

### 官方与行业指南
- BMP Maritime Security 2025 明确把 threat assessment、contingency planning、AIS spoofing、虚假信号和 incident management 视为海事安全决策的重要组成部分。  
  Link: https://www.maritimeglobalsecurity.org/media/lx4jmieu/bmp-ms-2025-final-hi-res.pdf

- IMO maritime cyber risk guidance 强调 identify / protect / detect / respond / recover 以及 continuity planning，这为“短期超耗可容忍，但持续失血应视为体系失效”提供了管理逻辑支撑。  
  Link: https://www.imo.org/en/ourwork/security/pages/guidance-home.aspx

- MSC-FAL.1/Circ.3/Rev.2 文本可从公开镜像访问，内容同样强调 contingency、response 与恢复。  
  Link: https://www.uscg.mil/Portals/0/MSC-FAL_1-Circ_3-Rev_2%20-%20Guidelines%20On%20Maritime%20Cyber%20Risk%20Management%20%28Secretariat%29.pdf

### 博弈与资源分配依据
- U.S. Coast Guard PROTECT gameboard 说明海事关键基础设施防护天然是一个资源受限、对手自适应的分配问题。  
  Link: https://www.dco.uscg.mil/Our-Organization/Assistant-Commandant-for-C4IT-CG-6/The-Game-Theory-Behind-the-US-Coast-Guards-Protect-Gameboard/

- MISTRAL framework 说明海事安全研究适合使用资源受限、安全博弈和巡逻/护航类结构，而不是无限资源假设。  
  Link: https://doi.org/10.3389/fmars.2024.1296854

## 4. 小测试设计
本轮没有直接跑正式主实验，而是做三层 pilot。

### Pilot A：环境/规则扫描
目标：

- 找出不会让资源崩溃主导全局的 `guarantee_line` 与 `patience`
- 找出合理的初始预算区间

扫描维度：

- `attacker_initial_budget / defender_initial_budget`
- `attacker_guarantee_line / defender_guarantee_line`
- `guarantee_breach_patience`

评价指标：

- `attacker_resource_collapse_rate`
- `defender_resource_collapse_rate`
- `avg_episode_length`
- `avg_defender_return`

### Pilot B：欺骗成本扫描
目标：

- 检查在新资源机制下，`cheat_cost` 是否把欺骗动作定得过贵

扫描维度：

- `attacker_cheat_cost = 0.5 / 0.8 / 1.0 / 1.2 / 1.5`

评价指标：

- `attacker_success_rate`
- `attacker_resource_collapse_rate`
- `avg_defender_return`
- `avg_episode_length`

### Pilot C：短 DRL 验证
目标：

- 验证所选参数不是只让 baseline 好看
- 检查 `cheat` 是否比 `flipit` 更难

设置：

- DRL 训练 `80` episodes
- 中间评估 `8` episodes
- 最终评估 `20` episodes

说明：

- 该 pilot 只用于标定，不用于论文显著性结论

## 5. Pilot 结果
### 5.1 过紧保障线会让崩溃主导博弈
在第一轮扫描里，`guarantee=-2` 或偏紧的早期组合，会把 attacker collapse 拉高到大约 `0.825-0.95`，同时把平均 episode 长度压到 `24-33` 步。

这说明该区间不适合作为论文主设定，因为：

- 失败主要由资源断裂触发
- 而不是由 inspection-response 节奏和控制权争夺触发

因此排除。

### 5.2 `guarantee=-4, patience=2` 或 `guarantee=-3, patience=3` 更合理
这两类区间把 attacker collapse 大致压回 `0.10-0.40`，episode 长度回到 `50-58`。

这意味着：

- 资源压力是“有感”的
- 但不会替代博弈本身

在两者中，我最终更偏向 `-4, 2`，原因是：

- 它给一次高成本应急响应保留了更真实的短期透支空间
- 同时仍然会惩罚连续超耗

### 5.3 欺骗成本若保持 1.5，会偏高
在固定资源参数后，`cheat_cost` 的 baseline 扫描结果如下：

- `0.5`：`attacker_success_rate ≈ 0.30`，`attacker_collapse ≈ 0.15`
- `0.8`：`attacker_success_rate ≈ 0.25`，`attacker_collapse ≈ 0.20`
- `1.0`：`attacker_success_rate ≈ 0.275`，但 `attacker_collapse ≈ 0.375`
- `1.2`：攻击难度开始回落
- `1.5`：欺骗场景对 attacker 的预算压力偏大

综合来看，`0.8` 是更稳的折中：

- 比 takeover 便宜很多
- 不会让欺骗本身变成“先耗死自己”
- 仍然保留可观的 defender 压力

### 5.4 降 defender 收入虽然能压预算增长，但会破坏场景层次
我还测试过更严格的 defender 收入版本，例如：

- `defender_base_income = 1.0`
- `defender_control_bonus = 2.0`

它的优点是 defender 最终预算不会涨得太快；但短 DRL pilot 里出现了不理想结果：

- `cheat` 攻击成功率约 `50%`
- `flipit` 攻击成功率约 `70%`

也就是 `flipit` 反而比 `cheat` 更难。

这不符合研究叙事，因此这组更“紧”的 defender 收入被排除。

## 6. 最终采用的标定值
最终保留的 V2 主链路标定如下。

### 资源参数
- `attacker_initial_budget = 32.0`
- `defender_initial_budget = 34.0`
- `attacker_base_income_per_step = 1.0`
- `defender_base_income_per_step = 1.2`
- `attacker_control_bonus_per_step = 2.2`
- `defender_control_bonus_per_step = 2.3`
- `attacker_action_floor = -6.0`
- `defender_action_floor = -6.0`
- `attacker_guarantee_line = -4.0`
- `defender_guarantee_line = -4.0`
- `guarantee_breach_patience = 2`

### 欺骗动作成本
- `attacker_cheat_cost = 0.8`

## 7. 这组值为什么最终被接受
### 原因 1：结构上最符合海事安全叙事
它保留了：

- defender 的轻微主场后勤优势
- 一次高成本应急动作的短时透支空间
- 持续低保障线时的体系崩溃

### 原因 2：baseline 下不至于 collapse 统治全局
在 30-episode baseline pilot 下，采用最终标定后的代表性结果为：

#### Cheat
- `attacker_success_rate ≈ 0.233`
- `avg_defender_return ≈ -4.4`
- `attacker_resource_collapse_rate ≈ 0.20`
- `defender_resource_collapse_rate = 0.0`
- `avg_episode_length ≈ 56.37`

#### FlipIt
- `attacker_success_rate ≈ 0.167`
- `avg_defender_return ≈ 44.633`
- `attacker_resource_collapse_rate = 0.0`
- `defender_resource_collapse_rate ≈ 0.033`
- `avg_episode_length ≈ 58.53`

含义是：

- `cheat` 比 `flipit` 更难
- 但两者都不是“全靠 collapse 决定胜负”

### 原因 3：短 DRL pilot 仍能体现 `cheat > flipit`
在 80-episode DRL pilot 下，代表性结果如下：

#### Cheat
- `attacker_success_rate = 45%`
- `avg_defender_return = 26.6`
- `attacker_resource_collapse_rate = 5%`
- `avg_episode_length ≈ 58.95`

#### FlipIt
- `attacker_success_rate = 25%`
- `avg_defender_return = 43.75`
- `attacker_resource_collapse_rate = 0%`
- `avg_episode_length = 60.0`

这说明在短训练下：

- `cheat` 仍然比 `flipit` 更难
- collapse 不是主导性终止原因
- 新资源机制不会把欺骗场景误写成“攻击者先自损”

## 8. 当前版本的边界
虽然本轮标定已能作为正式 pilot 起点，但仍有两个边界需要记录。

### 边界 A：防守方预算后程仍有上升趋势
在 `flipit` 下 defender 若长时间控场，最终预算仍可能偏高。

这不是当前阶段的致命问题，因为：

- 资源机制已经对 attacker 形成真实约束
- defender 也不再完全无约束

但在正式主实验中，仍应继续盯：

- `avg_final_defender_budget`

如果它在多种子下持续过高，我建议下一步只微调：

- `defender_control_bonus_per_step`

而不要重改 `guarantee_line`。

### 边界 B：短 DRL pilot 不能替代正式收敛结论
本说明中的 DRL 结果只用于标定，不用于论文显著性表述。

正式论文仍应以：

- 更多训练轮数
- 多随机种子
- 完整 main / ablation / robustness

作为结论基础。

## 9. 本轮标定结论
本轮最终结论是：

1. 新 V2 资源机制可以保留。
2. `action_floor=-6, guarantee_line=-4, patience=2` 是目前更合适的资源崩溃区间。
3. `attacker_cheat_cost` 应下调到 `0.8`，以保证欺骗动作仍是低成本扰动，而不是攻击者自损动作。
4. `32/34` 的初始预算与 `2.2/2.3` 的控制加成，能在当前实现下给出一个“有约束、可训练、叙事正确”的论文初标定。

## 10. 对应结果目录
本轮用到的代表性 pilot 结果在：

- `results/calibration_pilot/calib_cheat_drl_pilot_20260309_183802`
- `results/calibration_pilot/calib_flipit_drl_pilot_20260309_183802`
- `results/calibration_pilot/calib_cheat_lowcost_drl_pilot_20260309_184049`
- `results/calibration_pilot/calib_final_cheat_drl_pilot_20260309_184412`
- `results/calibration_pilot/calib_final_flipit_drl_pilot_20260309_184412`

其中真正支撑最终采用值的，是：

- 资源扫描结果
- `cheat_cost=0.8` 的 baseline 与短 DRL 表现
- “更紧 defender 收入版本”被排除的反例结果
