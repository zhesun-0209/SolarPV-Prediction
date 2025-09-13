# Project1140 光伏发电预测消融实验

## 🎯 项目概述

本项目实现了对Project1140光伏发电站的Capacity Factor预测，采用严格的消融实验设计，系统地评估不同输入特征、模型架构和超参数对预测性能的影响。

## 📊 实验设计

### 数据信息
- **数据范围**: 2022-01-01 到 2024-09-28
- **记录数量**: 24,028条
- **目标变量**: Capacity Factor
- **天气特征**: 11个高相关性特征

### 消融实验矩阵
- **输入特征类别**: 6种
  - PV: 仅历史PV功率
  - PV+NWP: 历史PV + 目标日NWP
  - PV+NWP+: 历史PV + 理想NWP
  - PV+HW: 历史PV + 历史HW
  - NWP: 仅目标日NWP
  - NWP+: 仅理想NWP

- **实验参数**: 
  - 回看窗口: 24小时, 72小时
  - 时间编码: 无TE, 有TE
  - 模型复杂度: Low, High
  - 模型类型: 8种 (RF, XGB, LGBM, LSTM, GRU, TCN, Transformer, LSR)

- **总配置数**: 360个实验配置

## 🚀 快速开始

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 验证配置
```bash
python scripts/test_ablation_configs.py
```

### 3. 运行实验

#### 运行单个实验
```bash
python main.py --config config/ablation/LSR_baseline_PV_24h_noTE.yaml
```

#### 运行所有消融实验
```bash
python scripts/run_ablation_experiments.py
```

#### 运行特定模型
```bash
python scripts/run_ablation_experiments.py --model-filter LSR,Transformer
```

#### 测试运行
```bash
python scripts/run_ablation_experiments.py --max-configs 5 --dry-run
```

## 📁 项目结构

详见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 📚 文档

- [消融实验设计](docs/ABLATION_STUDY_DESIGN.md) - 详细的实验设计说明
- [实验总结](docs/PROJECT1140_ABLATION_EXPERIMENT_SUMMARY.md) - 实验结果总结

## 🔬 研究问题

本消融实验旨在回答以下6个关键研究问题：

1. **PV-only sufficiency**: 历史PV单独预测是否足够准确？
2. **Marginal value of NWP**: NWP在多大程度上提升准确性？
3. **Upper bound of NWP**: 理想NWP的最大收益是多少？
4. **Incremental value of HW**: HW在PV基础上是否增加准确性？
5. **NWP-only sufficiency**: NWP单独预测是否足够准确？
6. **Ideal NWP-only sufficiency**: 理想NWP单独预测是否足够准确？

## 📈 预期输出

- 360个实验配置的完整性能评估
- 各输入特征的边际贡献分析
- 模型复杂度对性能的影响评估
- 时间编码的效用分析
- 不同回看窗口的效果比较

## 🛠️ 技术栈

- **Python 3.8+**
- **深度学习框架**: PyTorch
- **机器学习库**: scikit-learn, XGBoost, LightGBM
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn

## 📝 许可证

本项目仅供学术研究使用。

## 🤝 贡献

如有问题或建议，请提交Issue或Pull Request。