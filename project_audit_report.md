# 海运 Cheat-FlipIt 项目审计式汇报

**汇报对象**: 导师 / 内部评审  
**汇报日期**: 2026-03-09  
**汇报口径**: 问题导向审计  
**结论定位**: 本项目当前更适合界定为“具有代码、结果与初步研究价值的研究原型”，尚不宜直接包装为“概念闭环、实验合规、论文级可复现”的完成态成果。

**本报告锁定的主证据批次**为 2025-12-18 生成的四组结果目录：

1. `flipit-simulation-master/results/trc_balanced_realistic_drl_defense_vs_greedy_attack_20251218_095303`
2. `flipit-simulation-master/results/trc_balanced_realistic_drl_defense_vs_greedy_attack_flipit_20251218_095630`
3. `flipit-simulation-master/results/trc_traditional_baseline_vs_greedy_cheat_20251218_095805`
4. `flipit-simulation-master/results/trc_traditional_flipit_baseline_20251218_095813`

---

## 1. 项目定位与当前结论边界

### 1.1 项目目标

本项目试图将海运非传统安全场景、Cheat-FlipIt 博弈扩展与深度强化学习防御结合起来，形成一套从环境建模、对抗训练、基线对比到图表分析的完整实验链路。

从仓库现状看，这一目标已经形成了以下基础能力：

- 有可运行的海事对抗环境与训练脚本；
- 有 DRL 防守与传统策略对比结果；
- 有按实验目录保存的配置、模型和结果 JSON；
- 有自动生成摘要报告和分析图表的脚本。

### 1.2 当前结论边界

截至本次审计，项目**可以被支持的结论**与**暂时不能被支持的结论**必须明确分开：

- 可以支持：在当前 `maritime_nontraditional_env.py` 的实现和 2025-12-18 这四组结果下，DRL 防守在“占领奖励”指标上优于当前传统 baseline。
- 可以支持：在当前实现里，`Cheat` 模式相对 `FlipIt` 模式会降低防守方占领奖励，即当前参数化的“欺骗模式”确实提高了防守难度。
- 暂时不能支持：海事环境已经完整落地了“显式 cheat 动作 + 检测概率 + 怀疑度 + 部分可观测信息结构”的 Cheat-FlipIt 扩展。
- 暂时不能支持：当前 DRL 与 baseline 对比是严格公平的控制变量实验。
- 暂时不能支持：当前结果已经达到论文投稿级别的可追溯性与统计合规性。
- 暂时不能支持：当前场景已经实现“攻击方具有 20%-35% 合理威胁胜率”的平衡现实设定；相反，四组权威结果中攻击方胜率均为 `0.0%`。

### 1.3 术语口径统一

为避免后续继续混用概念，本报告采用以下统一口径：

- **Cheat 机制（当前实现）**：`deception_mode` 触发的攻防基础成功率切换。
- **Cheat 机制（目标定义）**：显式欺骗动作、检测概率、怀疑度或部分可观测状态、欺骗收益/惩罚通道共同构成的信息结构扩展。
- **可比较主指标**：`avg_attacker_occupation_reward` / `avg_defender_occupation_reward`。
- **训练奖励**：仅供 DRL 学习使用；当前实现中混入了防守成本惩罚和攻击成本 bonus，不能直接当作横向主结论。
- **样本量**：DRL 使用最终评估 `N=20`；传统 baseline 使用 `100` 个 episode；配置中的 `num_experiments: 15` 不视为已执行样本量。

---

## 2. 仓库结构与实验流水线现状

### 2.1 当前关键组件

| 组件 | 当前作用 | 审计结论 |
| --- | --- | --- |
| `gym-flipit-master/gym_flipit/envs/maritime_nontraditional_env.py` | 海事多单位攻防环境 | 是当前主实验的真实环境实现 |
| `gym-flipit-master/gym_flipit/envs/cheat_flipit_env.py` | 早期显式 cheat 环境 | 体现了显式欺骗信息结构，但不是当前海事主实验所用环境 |
| `flipit-simulation-master/run_trc_full_training.py` | DRL 训练与最终评估 | 是 DRL 主结果的直接来源 |
| `flipit-simulation-master/run_traditional_experiment.py` | 传统 baseline 运行 | 是传统结果的直接来源，但存在硬编码弱化 baseline 的问题 |
| `flipit-simulation-master/analysis/trc_drl_defense_analysis.py` | 汇总分析和出图 | 有真实数据读取能力，但也包含 simulated fallback 与硬编码图表数据 |
| `summary.md`、`section_5_experiments.md`、`section_5_experimental_results_eswa.md` | 摘要与论文式叙述 | 与当前权威结果存在口径漂移，需要整改 |

### 2.2 当前实验流水线

当前最可信的实验链路如下：

1. 通过 `configs/trc_*.yml` 定义场景、预算、动作、奖励与训练参数。
2. `run_trc_full_training.py` 调用 `MaritimeNontraditionalEnv` 与 `RainbowDQNAgent` 生成 DRL 结果。
3. `run_traditional_experiment.py` 生成传统 baseline 结果。
4. 每次实验在 `results/实验ID_时间戳/` 下保存：
   - `config.yml`
   - `complete_training_results.json`
   - `training_summary.md`
   - `training_history.json` / `learning_curves.json` / 模型文件（DRL）
5. `analysis/trc_drl_defense_analysis.py` 会读取 `results/` 目录并输出图表和分析报告。

### 2.3 当前流水线的主要状态判断

- **优点**：结果目录化保存已经具备基本审计入口。
- **不足**：配置、实现、结果、论文叙述之间还没有形成唯一一致口径。
- **关键风险**：仓库里同时存在旧结果、旧摘要、自动分析报告和论文章节草稿，容易把不同时间、不同口径、甚至不同数据来源的内容混写在一起。

---

## 3. 四类核心问题审计

### 3.1 概念一致性问题

当前最大的问题不是“没有 Cheat”，而是**海事主实验里的 Cheat 与文档叙述中的 Cheat 不是同一层次的概念**。

现状如下：

- 在 `maritime_nontraditional_env.py` 中，`Cheat` 的落地方式是 `deception_mode == "cheat"` 时切换攻防双方的 `base_success` 参数。
- 该环境没有显式的 `cheat action`、`cheat_detected`、`cheat_successful`、`defender_suspicion` 这些状态变量进入当前海事主实验。
- 与之相对，`cheat_flipit_env.py` 中确实实现了显式 `cheat` 动作、检测概率、怀疑度、虚假告警和部分可观测状态。

因此，当前项目必须区分：

- `cheat_flipit_env.py` 代表的是“显式欺骗机制原型”；
- `maritime_nontraditional_env.py` 代表的是“海事场景中的参数化欺骗模式”。

如果继续把后者表述为“完整 Cheat-FlipIt 信息结构已在海事环境中落地”，会造成概念错配。

另外，奖励与预算口径也存在偏差：

- 文档常写“即时奖励 = 占领奖励 - cost_penalty × 行动成本”；
- 但 `maritime_nontraditional_env.py` 的 DRL 训练奖励实际为：
  - 占领奖励；
  - 减去防守方自身成本；
  - 加上 `0.5 * 攻击方成本` bonus。
- 同时，环境每步还会给双方增加预算，并把控制方的 `occupation_reward` 直接加回预算。

这意味着：**训练奖励、预算动态、统计主指标是三个不同层面的量**，不能继续用一句简化公式概括全部机制。

### 3.2 实验合规性问题

当前实验设计的主要问题不是“没有对照”，而是**对照未真正做到严格控制变量**。

第一，`Cheat` 与 `FlipIt` 的 DRL 配置并不一致：

- `trc_balanced_realistic.yml` 中 DRL 训练轮数为 `500`；
- `trc_balanced_realistic_flipit.yml` 中为 `250`；
- 两者还在 `gamma`、`memory_size`、`batch_size`、`learning_starts`、`polyak_tau`、`evaluation_frequency` 等参数上存在差异。

因此，“只改变 deception_mode，其余完全一致”的说法当前不成立。

第二，传统 baseline 并非中性、公平的“按配置选择策略”：

- `run_traditional_experiment.py` 中，防守方被固定为 `WeakDefensiveGreedy`；
- 在 `flipit` 模式下，攻击方被切换为 `AggressiveAttacker`；
- 这些选择并非由配置文件完整驱动，而是脚本内部硬编码。

这意味着当前 baseline 更像“人为塑造过的对照组”，而不是严格中立的传统方法。

第三，配置中的多项参数并未真正形成可验证实验协议：

- `num_experiments: 15` 仅出现在配置中，代码没有实际消费这一字段；
- DRL 配置里声明了大量 Rainbow 特性开关，但真实 agent 并不是“按开关启停”，而是固定实现；
- 同一份配置里还同时出现 `drl_config.evaluation_frequency` 和 `evaluation_config.evaluation_frequency` 两种口径，增加了执行歧义。

### 3.3 可追溯性问题

当前仓库已经具备“保存结果目录”的基本意识，但距离论文级可追溯性仍有明显差距。

主要缺口如下：

- 结果目录没有记录 git commit hash；
- 没有记录精确依赖锁定文件，仅有宽范围 `requirements_drl.txt`；
- 没有记录硬件、Python 版本、CUDA 状态、命令行参数快照等环境元数据；
- 当前所有主实验都使用 `random_seed: 42`，但没有形成多 seed 重复设计；
- 运行日志中已经出现 `gym` 过时警告，说明依赖环境本身存在稳定性风险。

这会导致两个直接后果：

- 后续即使重新运行同一配置，也不能保证环境完全一致；
- 导师或审稿人无法把“某一条结论”唯一追溯到“某一次代码状态 + 某一组依赖状态”。

### 3.4 分析可信度问题

当前分析脚本与论文式文本中，存在多处**超出真实结果证据边界**的表述。

最典型的问题包括：

- `analysis/trc_drl_defense_analysis.py` 在缺少传统结果时可自动生成 simulated data；
- 多个图表使用硬编码的动作分布、热力图偏好、场景成功率、响应时间等数值；
- 自动分析报告会把这些内容写成“已被实验验证”的学术结论；
- `summary.md` 仍写着“最新批次 20251128”，与当前权威结果批次 `2025-12-18` 不一致；
- `section_5_experiments.md` 和 `section_5_experimental_results_eswa.md` 写成 “200 次 Monte Carlo 模拟”，但当前权威结果实际是 DRL 最终评估 `N=20`、传统 baseline `N=100`；
- 自动分析报告甚至写出了“攻防双方预算相等”，而实际配置是攻击方预算 `25`、防守方预算 `30`。

因此，当前分析产物不能整体当作“正式证据”；必须区分：

- **真实数据驱动的结论**；
- **脚本扩写、示意图或模拟数据支持的描述**。

### 3.5 “说法 - 实现 - 证据 - 风险 - 整改”矩阵

| 说法 | 当前实现 | 证据 | 主要风险 | 必要整改 |
| --- | --- | --- | --- | --- |
| `Cheat` 已是显式动作与信息结构 | 海事主环境中只是 `deception_mode` 触发成功率切换；显式 cheat 只存在于另一套环境 | `maritime_nontraditional_env.py`、`cheat_flipit_env.py` | 概念错配，创新点被夸大 | 文档明确区分“当前实现”与“目标定义”；若要保留强 claim，需把显式 cheat 机制真正并入海事主环境 |
| 奖励函数只含占领收益减成本 | DRL 训练奖励还加入了攻击方成本 bonus，预算也会随每步收入和占领收益变化 | `maritime_nontraditional_env.py` | 训练目标与论文公式不一致 | 分离“训练奖励”“统计指标”“预算更新”三套口径，重写奖励描述 |
| 预算耗尽即立即终止 | 当前动作会先按预算裁剪；环境还允许透支到 `-30` 才强制结束 | `maritime_nontraditional_env.py` | 终止条件被错误描述，影响机制解释 | 用真实终止逻辑重写文档；如要改成硬预算终止，需要重做环境与实验 |
| Cheat / FlipIt 只改一个变量 | 两个 DRL 配置在训练轮数和多项超参数上不同 | `configs/trc_balanced_realistic.yml`、`configs/trc_balanced_realistic_flipit.yml` | 消融实验不构成严格控制变量 | 统一配置模板，只允许 `deception_mode` 差异 |
| baseline 是公平传统对照 | 传统实验脚本固定使用弱化防守和激进攻击硬编码 | `run_traditional_experiment.py`、`WeakDefensiveGreedy.py`、`AggressiveAttacker.py` | DRL 优势可能被放大 | baseline 必须改为配置驱动；弱化/激进策略只能作为单独消融 |
| 200 次 Monte Carlo / 15 次重复已执行 | 当前权威结果是 DRL `N=20`、传统 `N=100`；`num_experiments: 15` 未被消费 | 结果目录 `complete_training_results.json`、`training_summary.md`、`rg "num_experiments"` | 论文叙述与真实执行口径不一致 | 在所有文档中改成真实样本量；后续按协议重跑 |
| 图表和结论都来自真实数据 | 分析脚本存在 simulated fallback 与硬编码图表数组 | `analysis/trc_drl_defense_analysis.py` | 非证据内容被误当成证据 | 删除、隔离或显著标注“示意图/非证据图” |

---

## 4. 经核验的真实结果表

以下表格只引用 2025-12-18 四组权威结果目录中的 `complete_training_results.json` 与 `training_summary.md`，不引用后续自动分析报告的扩写结论。

| 组别 | 权威结果目录 | 样本量口径 | 攻击方胜率 | 攻击方占领奖励 | 防守方占领奖励 | 平均博弈长度 |
| --- | --- | --- | --- | --- | --- | --- |
| DRL + Cheat | `trc_balanced_realistic_drl_defense_vs_greedy_attack_20251218_095303` | DRL 最终评估 `N=20` | `0.0%` | `5.90` | `54.10` | `30.0` |
| DRL + FlipIt | `trc_balanced_realistic_drl_defense_vs_greedy_attack_flipit_20251218_095630` | DRL 最终评估 `N=20` | `0.0%` | `1.00` | `59.00` | `30.0` |
| Traditional + Cheat | `trc_traditional_baseline_vs_greedy_cheat_20251218_095805` | baseline episode `N=100` | `0.0%` | `10.18` | `49.82` | `30.0` |
| Traditional + FlipIt | `trc_traditional_flipit_baseline_20251218_095813` | baseline episode `N=100` | `0.0%` | `7.02` | `52.98` | `30.0` |

### 4.1 基于真实结果，目前只能稳妥得出的结论

1. 在当前实现和当前 baseline 下，DRL 防守在两种模式下都优于传统 baseline：
   - Cheat 模式：`54.10 - 49.82 = 4.28`
   - FlipIt 模式：`59.00 - 52.98 = 6.02`
2. 在当前实现中，`Cheat` 模式相对 `FlipIt` 模式确实增加了防守难度：
   - DRL 防守奖励下降 `4.90`
   - 传统 baseline 防守奖励下降 `3.16`
3. 当前四组权威结果中攻击方胜率均为 `0.0%`，因此不能再把项目表述为“已实现平衡现实威胁场景”。

### 4.2 基于真实结果，目前不能继续沿用的旧说法

- 不能继续沿用“最新批次 20251128”的口径；当前权威批次是 **2025-12-18**。
- 不能继续沿用“200 次 Monte Carlo 模拟”的口径；当前权威结果并非这一执行规模。
- 不能继续沿用“传统 + Cheat 攻击方胜率 38.5%”这类旧数字，除非能给出与当前代码状态一致的权威结果目录。

---

## 5. 可发表前必须完成的整改路线

### P0：文档口径立即止损

目标：先把“说错的话”全部收回到当前证据边界内。

- 新增一份统一口径的项目审计报告，明确项目当前是研究原型。
- 冻结 2025-12-18 四组结果为当前权威批次，其他批次只能作为历史记录。
- 重写 `summary.md`、`section_5_experiments.md`、`section_5_experimental_results_eswa.md` 中不再成立的叙述。
- 在所有分析图和自动分析报告中，删除或显著标注 simulated fallback、硬编码图表和示意数据。
- 统一术语：区分“当前实现的参数化 Cheat 模式”与“目标定义的显式 Cheat 信息结构”。

### P1：重做实验协议与公平对照

目标：让“对比”和“消融”真正具备实验合规性。

- 建立单一实验协议模板；Cheat / FlipIt 之间只能更改 `deception_mode`。
- 删除 `run_traditional_experiment.py` 中的弱化/激进硬编码，改为完全配置驱动。
- 明确 baseline 类型：
  - 中性传统 baseline；
  - 弱化 baseline；
  - 激进攻击 baseline；
  三者必须分开命名，不能混在同一“公平对照”口径下。
- 明确样本量协议：
  - 训练轮数；
  - 最终评估 episode 数；
  - 多 seed 数；
  - 是否 bootstrap / 置信区间。
- 清理未接线或多口径参数，确保配置文件里出现的关键超参数都能在代码中唯一生效。

### P2：补全真正的 Cheat 信息结构与论文级可追溯性

目标：把项目从“参数化研究原型”升级为“可答辩、可投稿的实验系统”。

- 将显式 `cheat action`、检测概率、怀疑度、欺骗收益/惩罚、部分可观测状态引入海事主环境。
- 重新定义训练奖励、统计指标和预算更新三者的关系，保证可解释性。
- 记录完整元数据：
  - git commit hash；
  - 依赖锁定；
  - Python / Torch / CUDA / Gym 版本；
  - 硬件信息；
  - 启动命令；
  - 随机种子集合。
- 建立论文级统计输出：
  - 多 seed 重复；
  - 置信区间；
  - 显著性检验或 bootstrap；
  - 图表全部由真实结果数据自动生成。

---

## 6. 最终判断：当前能 claim 什么、不能 claim 什么

### 6.1 当前能 claim 的内容

- 项目已经实现了一套海事攻防环境、DRL 训练脚本、传统基线脚本和结果目录化保存机制。
- 当前海事主环境包含非线性成功率、预算约束、动作成本和占领奖励等核心机制。
- 在 2025-12-18 的四组权威结果中，DRL 防守在占领奖励指标上优于当前传统 baseline。
- 当前参数化的 `Cheat` 模式确实会降低防守方表现，因此“欺骗提高防守难度”这一方向性结论成立。

### 6.2 当前不能 claim 的内容

- 不能 claim：海事主环境已经完整实现了显式 Cheat-FlipIt 信息结构。
- 不能 claim：当前 DRL 与 `FlipIt` / `Cheat` 的消融是严格控制变量实验。
- 不能 claim：当前 baseline 对照是严格公平且中性的。
- 不能 claim：当前结果已经达到论文级可复现与统计合规标准。
- 不能 claim：当前场景已经构造出“真实且平衡的海运威胁强度”；因为权威结果中攻击方胜率仍是 `0.0%`。

### 6.3 审计结论

综合判断，本项目当前最准确的定位是：

> **一个具备研究价值、代码和结果基础较完整，但概念一致性、实验合规性与证据口径仍需系统整改的研究原型。**

因此，当前最合适的汇报方式不是“直接宣称论文结果已经成立”，而是：

- 先如实交代现有实现与结果；
- 再明确指出四类核心问题；
- 最后给出分层整改路线。

这样既能保住项目已有技术积累，也能避免在导师评审或后续论文写作中因口径失真而被整体否定。
