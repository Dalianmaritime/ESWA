# 项目快速总览（审稿人/导师提要）

本项目聚焦海运非传统安全场景，将 **Cheat-FlipIt 博弈** 与 **深度强化学习防御** 相结合，构建可复现的对抗模拟、算法实现与实验评估。以下要点可使读者在数分钟内把握创新性与技术实现。

## 1. 场景与博弈建模
- **参与方与资源**：攻击方（海盗/恐怖行为）与防守方（护航与平台防御），持续争夺离岸平台控制权。每步获得增量预算（防守 +2、攻击 +1），并受行动成本与总预算硬约束。
- **动作与非对称性**：
  - 攻击方：充气快艇、硬壳快艇、武装登船、远程火力、以及 **欺骗(cheat)** 行动；各自有成功率、成本、最大编队规模。
  - 防守方：海军护航、平台安保、直升机、自动化系统、巡逻艇；成本/成功率与非线性协同上限（alpha/cap）独立设定。
  - **Cheat-FlipIt 创新**：在经典 FlipIt 隐蔽接管基础上加入欺骗通道，改变攻防信息结构与成功率分布，使得防守需在资源分配与欺骗识别间权衡。
- **状态与终止**：30 步上限或任一方预算耗尽即终止；控制权决定占领奖励的累积；终局胜负由占领奖励与耗尽条件统一裁决。
- **预算与约束**：每回合动作成本必须小于当前可用预算，行动数受 max_units 约束；预算在 episode 级别重置，避免跨局累积。

## 2. 奖励与动力学（核心公式文字化）
- **占领奖励**：控制方每步获得 occupation_reward（默认 2.0 防守 / 1.0 攻击增益叠加占领收益），用于度量控制时长与资源利用。
- **动作成本惩罚**：即时奖励 = 占领奖励 − cost_penalty × 行动成本；无额外成功/失败奖励，突出资源效率与控制持续性。
- **成功率修正**：基础成功率结合非线性协同 p_eff(n) = min(cap, 1 − (1 − p)^(n^alpha))，并随 cheat / flipit 模式切换。
- **终局判定**：若预算先耗尽则劣势；若达到步长上限，比较累计占领奖励决定胜负。

## 3. 求解算法（Rainbow DQN 防守方）
- **结构**：Dueling Double DQN + Prioritized Replay；目标网络软更新；多头输出覆盖防守动作空间。
- **损失与更新**：基于 TD 误差的 Huber/均方损失，含重要性采样权重；epsilon 从 1.0 衰减至 0.1，以平衡探索与收敛。
- **训练配置**：500 轮 × 30 步；warmup 与固定频率评估（每 25 轮记录 evaluation_history）；全流程日志含 training_history / learning_curves / complete_training_results。
- **对照基线**：传统贪心防守与贪心攻击，无学习；同样遵守预算与协同约束，用作性能下界。

## 4. 实验与主要发现（最新批次 20251128）
- **DRL + Cheat**：防守占领奖励 48.5，攻击 11.5，攻击胜率 0%，平均 30 步。
- **DRL + FlipIt**：防守占领奖励 56.4，攻击 3.6，攻击胜率 0%，平均 30 步。
- **传统 + Cheat**：防守 30.33，攻击 29.67，攻击胜率 38.5%，30 步。
- **传统 + FlipIt**：防守 44.45，攻击 15.55，攻击胜率 0%，30 步。
- **收敛与稳定性**：100% 评估胜率贯穿 20 次验证；loss 对数下降，策略方差在 200+ 轮后明显收敛。
- **战术洞察**：DRL 在低预算优先巡逻艇、在中高预算转向护航/平台安保；传统策略分布分散，预算效率显著劣化。

## 5. 复现与产出物
- **运行指令**（PowerShell）：
  - 训练与对照：`$env:PYTHONPATH="E:\1031\gym-flipit-master"; cd E:\1031\flipit-simulation-master; python run_trc_full_training.py configs/trc_balanced_realistic.yml`
  - 传统基线：`python run_traditional_experiment.py configs/trc_traditional_baseline.yml --episodes 200`；`python run_traditional_experiment.py configs/trc_traditional_flipit_baseline.yml --episodes 200`
  - 全量分析与制图：`python analysis/trc_drl_defense_analysis.py results`
- **关键文件**：
  - 环境与逻辑：`gym_flipit/envs/maritime_nontraditional_env.py`
  - 训练脚本：`run_trc_full_training.py`（DRL）、`run_traditional_experiment.py`（基线）
  - 配置：`configs/trc_balanced_realistic*.yml`
  - 可视化与报告：`analysis/trc_drl_defense_analysis.py`；输出位于 `results/`（单图单文件 PNG/PDF）及 `trc_drl_defense_analysis_report_*.md`

## 6. 创新点摘要
1) **Cheat-FlipIt 扩展**：在隐蔽接管博弈中显式建模欺骗通道与非对称动作成功率，贴合海运威胁情景。  
2) **预算与协同一体化**：硬预算、每步增量、动作成本与协同上限统一进入奖励与约束，保证策略可执行性与可解释性。  
3) **Rainbow DQN 防御策略**：针对高维、非对称动作空间的端到端学习，显著压制欺骗态势（攻击胜率由 38.5% 降为 0%）。  
4) **全链路可复现**：配置-运行-可视化-报告一体化脚本，所有图表独立文件便于论文插图。

