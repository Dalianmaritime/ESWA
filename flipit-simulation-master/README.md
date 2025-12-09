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

### 运行仿真

```bash
# 海运非传统安全仿真（推荐）
python run_maritime_nontraditional.py configs/maritime_nontraditional_extended.yml

# 资源约束系统
python run_resource_constraint.py configs/pirate_assault_scenario.yml

# Cheat-FlipIt系统  
python run_cheat_flipit.py configs/maritime_security_scenario.yml
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