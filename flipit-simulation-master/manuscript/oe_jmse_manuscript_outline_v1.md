# Ocean Engineering / JMSE 风格论文大纲（修订版）

## 1. 文章整体定位

这篇文章应当写成一篇围绕 `offshore critical infrastructure defense` 展开的工程化研究，而不是一篇以算法为主角的强化学习论文，也不是一篇以形式推导为核心的博弈论文。全文的主线应当始终围绕这样一个问题展开：在远海关键能源设施面临欺骗诱导、隐蔽接近和资源约束的海运非传统安全威胁时，运营方如何通过区域化检查与响应调度维持设施控制连续性与服务连续性。

因此，章节设计需要体现三点：

1. 工程问题先于方法；
2. `Cheat-FlipIt` 是刻画问题的建模工具，而不是文章的展示目的；
3. DRL 是求解和决策支持手段，必须服务于工程问题的解释与验证。

从期刊气质上看，这样的结构更接近 `Ocean Engineering` 常见的“工程问题—模型—数值试验—工程含义”写法，同时又能兼容 `JMSE` 对海事系统安全、关键基础设施韧性和数字化风险的叙事偏好。

推荐标题先围绕工程问题，而不是围绕算法命名。建议保留以下两个方向：

- `Adaptive Inspection-Response Scheduling for the Protection of Remote Offshore Critical Energy Infrastructure under Deceptive Intrusion Threats`
- `Resource-Sustainable Defensive Control of Deceptive Takeover Attempts against Offshore Critical Energy Infrastructure`

## 2. 建议正文结构

### 2.1 Introduction

第一章建议不再分成 1.1、1.2、1.3 这种细分小节，而是作为一个完整的引言来写。写作顺序建议是四个自然段落，而不是四个小标题。

第一段写行业与工程背景。建议从海上风电升压站、能源岛、海底电缆接入节点等远海设施切入，突出这些设施的共同特征：高价值、低驻守、强联通、慢增援。这里的任务是让审稿人迅速意识到这是一个典型的海洋工程设施运行安全问题，而不是抽象安全博弈。

第二段写海运非传统安全问题本身。重点不在于列出所有威胁，而在于把问题机制说清楚：守方面临的不是单次、明确、可验证的威胁，而是由可疑靠近、身份伪装、AIS/GNSS 欺骗、虚假告警、低速徘徊以及后续真实 takeover 交织构成的长期扰动。要把“先通过欺骗信号诱导守方耗散资源，再在关键时机发起真实接管”这一逻辑讲清楚。

第三段写现有研究的不足。建议用一句话概括海洋工程文献的主要不足，再用一句话概括 `FlipIt` 类研究的不足。前者可以概括为“偏重预警、识别和局部拦截，缺少长期控制连续性的决策框架”；后者可以概括为“适合描述持续控制争夺，但对海上设施中的欺骗信号、区域结构和资源可持续性刻画不足”。

第四段写本文做了什么、为什么有意义。这里不要写成项目汇报，而是自然地说：本文提出了一个面向远海关键能源设施防护的区域化 `Cheat-FlipIt` 框架，并基于自适应策略学习构建检查-响应决策支持方法，以系统分析欺骗诱导、资源约束和区域防护之间的耦合关系。最后用三句左右概括贡献，并在结尾交代全文结构。

引言的整体语气要稳，不要一上来强调“先进算法”或“首次提出”。主词应当是 `offshore critical infrastructure`, `service continuity`, `control continuity`, `inspection-response scheduling`, `resource sustainability`，而不是 `Rainbow DQN`。

### 2.2 Literature Review

第二章可以明确写成文献综述，但也不宜拆得过细。建议控制在四个子节，逻辑上从工程问题走向建模工具，再落回文章定位。

#### 2.2.1 Offshore critical infrastructure security in marine environments

这一节回顾与 offshore wind farm、offshore platform、critical energy infrastructure protection、risk warning、intrusion monitoring 有关的 `Ocean Engineering` 和 `JMSE` 研究。重点不是简单列文献，而是指出：现有工作已经意识到远海设施的脆弱性与高价值，但多数仍集中于预警、监测、状态识别和局部风险评估，较少把设施控制连续性作为一个动态决策问题来处理。

结尾应明确收束到本文问题：现有工程研究为远海设施安全监测提供了基础，但尚未充分回答在欺骗性威胁和有限资源条件下如何维持长期防护节奏与控制稳定性。

#### 2.2.2 Maritime non-traditional security, deception, and adaptive defense

这一节将海运非传统安全、网络-物理复合威胁和自适应防护调度放在一起讨论。写作重点是：海上设施风险越来越表现为信息操纵与物理接近的耦合，传统规则式巡检和静态资源配置难以应对由欺骗性信号引发的连续资源消耗。这里可以顺带回顾 inspection、interception、response scheduling 相关工作，但不要展开成独立大综述，始终服务于“为什么需要一个能处理 deception 的动态决策框架”。

#### 2.2.3 FlipIt and persistent control games

这一节专门引入 `FlipIt` 文献线，但分量要适度。建议先概括其核心思想：`FlipIt` 描述的是在不完全可见条件下围绕控制权进行持续争夺。然后说明它为何适合迁移到远海设施防护：海上关键设施的控制状态并不总是完全透明，攻守双方围绕控制收益和占用时间长期互动。随后指出经典 `FlipIt` 的局限：通常缺少显式欺骗信号层、缺少区域化检查-响应结构，也缺少对资源可持续性和操作成本的工程化刻画。由此自然引出本文采用的是面向海洋工程场景扩展后的 `Cheat-FlipIt`，而不是网络安全语境下的直接照搬。

#### 2.2.4 Research gap and article positioning

这一节应写得干净、有力，最好三到四句话结束。建议明确三点：

- 现有海洋工程安全研究对欺骗诱导下的长期控制问题讨论不足；
- 现有 `FlipIt` 研究对海上关键设施中的区域检查-响应调度与预算可持续性刻画不足；
- 因此，本文定位于海上关键基础设施持续控制的工程决策研究，方法和算法只是服务于这一工程问题的求解工具。

这样写有助于弱化“方法论痕迹太重”的感觉。

### 2.3 Problem Formulation

第三章建议统一命名为 `Problem Formulation`，不要再把“Problem Description + Game Formulation + Theoretical Basis”全部摆在标题里。理论部分可以保留，但要收敛成支持性分析，而不是章节主轴。

#### 2.3.1 Operational scenario and modeling assumptions

这一节先用工程语言交代场景，再逐步抽象成模型。需要交代设施对象、攻守双方、三层区域 `outer-lane-core`、守方动作 `hold / inspect / respond`、攻方动作 `wait / cheat / takeover`，以及守方只能基于信号、检查结果和预算状态做决策这一信息结构。这里建议尽量把区域与动作对应到海上实际防护逻辑，例如外围警戒区、通达航路/运维通道和核心设施区，以及低成本核查与强响应之间的差异。

#### 2.3.2 Regional Cheat-FlipIt game formulation

这一节再进入形式化定义。建议按状态、动作、信号、转移、收益和终止条件来组织，而不是一开始就铺大量符号。重点是说明：

- `Cheat` 并不直接改变控制权，而是改变守方 belief 和后续调度行为；
- `Takeover` 则对应真实控制权争夺；
- `Inspect` 与 `Respond` 对 takeover 的抑制效应具有区域性和成本差异；
- 控制权收益来自长期占用而非单次事件。

这一节的写作目标是让审稿人明白：模型的复杂性来自工程问题本身，而不是人为加复杂度。

#### 2.3.3 Belief evolution and resource sustainability

这一节合并原来单列的 belief 更新和资源演化内容。建议一半写信号如何改变 breach belief 与区域信念，一半写预算如何通过 base income、control bonus、action cost、action floor、guarantee line 和 collapse 规则影响策略空间。这样处理会比单独拆成两个很细的小节更凝练，也更像期刊正文。

这里应明确强调两点：

- 信号不等于真实威胁，守方是在 belief 驱动下决策；
- 预算不是背景参数，而是决定可行动作集合与长期控制能力的核心状态。

#### 2.3.4 Supporting theoretical remarks

这一节不建议再高调命名成 “Theoretical Basis” 或 “Basic propositions and lemmas”，而是更自然地处理成支持性理论说明。正文中可以放两到三个小结论，作用是增强模型可信度，而不是把文章推向重理论风格。

推荐保留以下三类结论：

1. **Feasible action monotonicity**  
   在动作成本和 action floor 给定的条件下，守方预算越高，其可行动作集合单调不减。  
   这说明资源状态直接决定策略空间，是预算变量必要性的最直接理论支持。

2. **Myopic threshold structure**  
   在固定 belief、固定成本和单步收益设定下，守方单步最优响应呈现阈值结构：低风险偏向 hold，中等风险偏向 inspect，高风险偏向 respond。  
   这为 threshold baseline 提供理论合理性，也让基线不显得武断。

3. **Deception-induced resource drain**  
   在其他条件不变时，欺骗信号频率上升会增加守方检查与误响应负担，并削弱其预算可持续性。  
   这一点可以作为说明性命题，直接支撑为什么 `Cheat` 不能被当作普通噪声。

如果篇幅有限，可以只在正文保留结论与简要证明思路，把详细推导放附录。这样既保留理论支撑，也不会让第三章显得“开题味”太重。

### 2.4 Adaptive Defense Framework

第四章建议命名为 `Adaptive Defense Framework` 或 `Defense Solution Framework`，而不要让章节名显得像算法文档。这样方法仍然是完整的，但在结构上是从属于工程问题的。

#### 2.4.1 Rule-based benchmark

先写阈值式基线，说明其对应现实中的规则式检查-响应逻辑。这样全文在进入 DRL 之前，先让读者看到一个具有操作意义的比较对象。

#### 2.4.2 Learning-based defense policy

这一节写 DRL，但语气应当是“为解决前述动态决策问题而采用学习方法”。建议按 observation、action、training reward、episode 的顺序写，再自然引出 Rainbow DQN。不要把 Noisy Nets、PER、Dueling 一上来列成主角，而是放在后半段简述为何选择该类结构。

#### 2.4.3 Operationally constrained policy selection

这一节保留当前 V2 链路里的一个关键亮点，但表述必须工程化。重点是说明：为了避免单一训练分数导致策略在安全性或资源表现上偏离工程要求，本文采用 baseline-aware 的 `constrained_operational` checkpoint 选择逻辑，以优先保证控制率、攻击成功率和原始收益的操作可接受性。这里的方法重点是“工程可用的策略筛选”，而不是“发明了一套炫技选择器”。

这一章整体只需三节，不必再单列 “training reward versus reporting metrics” 和 “implementation flow”，这些内容可以自然并入 4.2 和 4.3。

### 2.5 Numerical Experiments and Discussion

第五章建议合并实验设计、结果与讨论，统一命名为 `Numerical Experiments and Discussion`。这会明显减弱项目设计稿的感觉，也更接近 `Ocean Engineering` 和 `JMSE` 常见正文结构。

#### 2.5.1 Experimental settings

先集中交代场景参数、区域结构、`cheat` 与 `flipit` 设定、预算参数、baseline 配置、DRL 训练配置、随机种子与评价指标。写法以表格为主、文字为辅，不要让实验设置本身占过多篇幅。

#### 2.5.2 Main comparative results

这一节给主实验结果，建议先写 `cheat`，再写 `flipit`。理由要明确：`cheat` 是主要科学问题场景，`flipit` 是对照场景。图表可以先给总体比较，再给 learning dynamics。讨论重点不是“谁赢了”，而是 DRL 是否在攻击抑制、控制保持和运营效率上形成了更平衡的优势。

#### 2.5.3 Ablation analysis

这一节集中讨论 cheat-only ablation。建议按“关键机制的重要性”来组织，而不是按配置文件顺序念一遍。写作上要回答：性能改进主要来自哪里？是资源可持续性机制、约束型 checkpoint 选择、信号相关特征，还是奖励引导与预算感知动作约束？

#### 2.5.4 Robustness analysis

这一节按三个 family 组织：budget stress、deception intensity、attack strength。每一部分都围绕一个问题展开：当预算更紧、欺骗更强或 takeover 更强时，所提方法是否仍能保持相对优势？写法要强调稳健性是工程部署可行性的组成部分，而不是额外装饰。

#### 2.5.5 Policy interpretation and engineering insights

这一节承接解释性分析结果。可用动作分布、预算轨迹、collapse timing 和 response lag 等结果说明，学习到的策略并不是简单“更激进”，而是形成了更稳定的检查-响应节奏和更合理的预算使用模式。这里是把 `policy-pattern analysis` 转化为论文中的工程解释，而不是单纯展示更多图。

#### 2.5.6 Discussion of implications and limitations

在这一小节里把原先独立的 `Discussion` 收回来。前半部分写工程与管理含义，面向运营方强调：

- 防护体系应从事件处置转向控制连续性管理；
- 预警系统和响应调度应联合设计；
- 不应只看拦截率，还要看误漏响应和单位成本控制效率。

后半部分主动写限制：

- 仿真环境尚未完整纳入真实海况、传感器失效和多船协同；
- 攻击者策略仍属有限行为族；
- 当前 ablation 与 robustness 主要围绕 `cheat` 展开。

这样处理后，讨论不会显得过长，也不会把全文结构拆得太碎。

### 2.6 Conclusions

结论章建议保持简洁。第一段回收问题与方法，第二段回收主要发现，第三段写未来工作，如多设施协同、真实 AIS / 告警数据校准、多船威胁扩展，以及与数字孪生或海事态势系统的联动。

## 3. 这版结构相对上一版的核心收缩

这版修改有四个明确方向：

1. **引言不再拆成开题式分节**  
   第一章回到成熟期刊常见的连贯引言写法。

2. **方法章节降级为服务章节**  
   第四章不再围绕算法组件展开，而是围绕“如何形成可操作的防守决策框架”展开。

3. **实验、结果、讨论合并**  
   第五章统一为 `Numerical Experiments and Discussion`，整体结构更凝练，更接近期刊正文。

4. **理论保留但弱化姿态**  
   理论内容保留在 3.4，但改成 supporting theoretical remarks，而不是单列成很重的 “Theoretical Basis”。

## 4. 与当前 V2 项目的对应关系

这份修订版大纲与当前项目实现保持一致：

- `Problem Formulation` 可直接对应当前 V2 环境中的 `outer-lane-core`、`cheat / takeover`、`hold / inspect / respond`、belief 更新和资源可持续性规则；
- `Adaptive Defense Framework` 可直接对应 baseline、Rainbow DQN、training reward / reported raw return 分离和 `constrained_operational` selection；
- `Numerical Experiments and Discussion` 可直接承接当前主链路中的：
  - `run_paper_main_v2.py`
  - `run_paper_ablation_v2.py`
  - `run_paper_robustness_v2.py`
  - `analysis/analyze_policy_patterns_v2.py`

因此，这版结构不是空泛的写作建议，而是可以直接落回你现有实验链路的正文骨架。

## 5. 风格参考口径

这版大纲主要参考了近年 `Ocean Engineering` 和 `JMSE` 中与 offshore critical infrastructure、intrusion interception、offshore platform risk warning、maritime cyber-physical security 相关论文的章节组织与叙事口径，同时结合 `FlipIt` 的文献线做了适配。参考的主要方向包括：

- offshore wind farm asset protection
- offshore platform risk-based warning
- maritime cyber-physical security
- maritime critical energy infrastructure digital risk
- FlipIt and persistent control games

在写法上，建议始终让工程问题而不是方法名称占据标题、章节名和段首句的主语位置。
