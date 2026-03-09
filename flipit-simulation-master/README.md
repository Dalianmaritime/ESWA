# 🚢 海运非传统安全对抗仿真系统

基于 Cheat-FlipIt 博弈的海运非传统安全对抗建模与分析系统，专注于海盗袭击、海上平台防御、军舰护卫等复杂对抗场景。

## ⚡ 快速开始

### 安装

```bash
# 安装依赖
pip install gym numpy pyyaml pandas matplotlib

# 安装环境
cd gym-flipit-master && pip install -e .
```

## Legacy vs V2

- `legacy`：原始 `MaritimeNontraditionalEnv` 链路，现已整体收进 [legacy](/e:/1031/flipit-simulation-master/legacy) 目录。
- `V2`：新的信号-区域 Maritime Cheat-FlipIt 建模，攻击方动作为 `wait / cheat(zone) / takeover(zone)`，防守方动作为 `hold / inspect(zone) / respond(zone)`。
- 顶层运行入口和配置现在默认只保留 V2；历史脚本、历史策略、历史配置与历史分析均在 `legacy/` 下归档。
- 依赖边界：V2 使用 `Gymnasium`；`legacy/` 继续依赖原始 `gym`。

### 运行仿真

```bash
# 海运非传统安全仿真（推荐）
python run_maritime_nontraditional.py configs/maritime_nontraditional_extended.yml

# 资源约束系统
python run_resource_constraint.py configs/pirate_assault_scenario.yml

# Cheat-FlipIt系统  
python run_cheat_flipit.py configs/maritime_security_scenario.yml
```

### 运行 V2 信号-区域实验

```bash
# DRL 防守方训练
python run_trc_full_training_v2.py configs/trc_signal_cheat_v2.yml
python run_trc_full_training_v2.py configs/trc_signal_flipit_v2.yml

# 传统 baseline
python run_traditional_experiment_v2.py configs/trc_signal_baseline_cheat_v2.yml --episodes 100
python run_traditional_experiment_v2.py configs/trc_signal_baseline_flipit_v2.yml --episodes 100

# 一键运行四组 V2 实验并分析
python run_all_trc_experiments_v2.py

# 快速 smoke test
python run_all_trc_experiments_v2.py --smoke
```

### 测试系统

```bash
python test_maritime_nontraditional.py
python test_resource_constraint.py
python test_cheat_flipit.py
```

## 🌟 核心特性

- **🎯 多场景支持**：海盗袭击、海上平台防御、军舰护卫
- **⚔️ 多单位作战**：支持多艘船只、多个小队协同作战  
- **🧠 智能策略**：基于Q学习的策略优化
- **💰 资源约束**：有限预算下的策略决策
- **🎭 欺骗机制**：攻击方可使用欺骗策略
- **📊 详细分析**：完整的仿真结果分析和可视化

## 📁 项目结构

```
flipit-simulation-master/          # 主仿真模块
├── configs/                       # 场景配置文件
├── strategies/                    # 智能策略算法
├── analysis/                      # 结果分析脚本
├── run_*.py                       # 仿真运行脚本
├── test_*.py                      # 系统测试脚本
└── results/                       # 仿真结果输出

gym-flipit-master/                 # 环境模块
└── gym_flipit/envs/              # 游戏环境实现
```

## 📖 详细文档

查看 [PROJECT_GUIDE.md](PROJECT_GUIDE.md) 获取完整的使用指南和自定义修改方法。

## 🧹 维护工具

```bash
# 清理输出文件
python clean_outputs.py
```

## 📊 仿真结果

运行后会生成：
- **JSON文件**：详细交战数据
- **CSV文件**：交战历史记录
- **YAML文件**：汇总分析报告

## 🔧 自定义配置

修改 `configs/*.yml` 文件来：
- 调整攻防动作和参数
- 设置预算和约束条件
- 配置环境因素影响
- 自定义仿真时长和次数

## 🚀 核心算法

- **非线性成功率**：`p_eff(n) = min(cap, 1-(1-p)^(n^α))`
- **对抗合成机制**：`P_final = P_att × (1 - P_def)`
- **Q学习策略**：基于状态-动作价值函数的智能决策

---

**版本**: 2.0 | **最后更新**: 2024年12月
