# Project1140 消融实验总结

## 实验概述

本项目为Project1140光伏发电预测设计了完整的消融实验，严格按照3.7 Controlled ablation study要求进行。

## 实验设计

### 数据特征
- **数据文件**: `Project1140.csv` (24,028条记录，2022-2024年)
- **数据范围**: 2022-01-01 到 2024-09-28 (训练样本)
- **目标变量**: `Capacity Factor` (容量因子)
- **天气特征**: 11个特征 (与_pred后缀对应的历史天气特征)
  - global_tilted_irradiance, vapour_pressure_deficit, relative_humidity_2m
  - temperature_2m, wind_gusts_10m, cloud_cover_low, wind_speed_100m
  - snow_depth, dew_point_2m, precipitation, surface_pressure

### 实验矩阵

#### 输入特征类别 (6种)
1. **PV**: 仅历史PV功率
2. **PV+NWP**: 历史PV + 目标日NWP
3. **PV+NWP+**: 历史PV + 理想NWP (目标日HW)
4. **PV+HW**: 历史PV + 历史HW
5. **NWP**: 仅目标日NWP
6. **NWP+**: 仅理想NWP

#### 实验参数
- **回看窗口**: 24小时, 72小时
- **时间编码**: 无TE, 有TE (正弦/余弦编码)
- **模型复杂度**: Low, High
- **机器学习方法**: 8种 (RF, XGB, LGBM, LSTM, GRU, TCN, Transformer, LSR)

### 配置总数
- **主要模型**: 6 × 2 × 2 × 2 × 7 = 336个配置
- **基线模型**: 6 × 2 × 2 × 1 = 24个配置
- **总计**: 360个配置

## 生成的文件

### 配置文件
- **目录**: `config/ablation/`
- **数量**: 360个YAML配置文件
- **索引**: `config/ablation/config_index.yaml`

### 实验脚本
- **配置生成**: `generate_ablation_configs.py`
- **实验运行**: `run_ablation_experiments.py`
- **配置测试**: `test_ablation_configs.py`

### 文档
- **实验设计**: `ABLATION_STUDY_DESIGN.md`
- **实验总结**: `PROJECT1140_ABLATION_EXPERIMENT_SUMMARY.md`

## 运行实验

### 测试运行
```bash
# 测试配置完整性
python test_ablation_configs.py

# 干运行 (只列出配置)
python run_ablation_experiments.py --dry-run

# 测试运行少量配置
python run_ablation_experiments.py --max-configs 5
```

### 完整实验
```bash
# 运行所有实验
python run_ablation_experiments.py

# 运行特定模型
python run_ablation_experiments.py --model-filter LSR,Transformer

# 并行运行 (推荐)
python run_ablation_experiments.py --max-workers 8
```

### 结果输出
- **结果目录**: `results/ablation/`
- **摘要文件**: `experiment_summary.csv`
- **统计报告**: `statistics_report.md`
- **失败配置**: `failed_configs.csv` (如有)

## 研究问题

实验设计回答以下6个关键问题：

1. **PV-only sufficiency**: 历史PV单独预测是否足够准确？
2. **Marginal value of NWP**: NWP在多大程度上提升准确性？
3. **Upper bound of NWP**: 理想NWP的最大收益是多少？
4. **Incremental value of HW**: HW在PV基础上是否增加准确性？
5. **NWP-only sufficiency**: NWP单独预测是否足够准确？
6. **Ideal NWP-only sufficiency**: 理想NWP单独预测是否足够准确？

## 模型复杂度设置

### Low复杂度
- **训练轮数**: 15 epochs
- **树模型**: n_estimators=50, max_depth=5, learning_rate=0.1
- **深度学习**: d_model=64, num_heads=4, num_layers=6, hidden_dim=32, dropout=0.1

### High复杂度
- **训练轮数**: 50 epochs
- **树模型**: n_estimators=200, max_depth=12, learning_rate=0.01
- **深度学习**: d_model=256, num_heads=16, num_layers=18, hidden_dim=128, dropout=0.3

## 实验控制

- **数据分割**: 80%训练, 10%验证, 10%测试
- **预处理**: 统一标准化流程
- **评估指标**: MAE, RMSE, R², MAPE
- **训练调度**: 标准化训练过程

## 预期结果

实验将生成：
1. 360个配置的完整性能评估
2. 各输入特征的边际贡献分析
3. 模型复杂度对性能的影响
4. 时间编码的效用分析
5. 不同回看窗口的效果比较

## 注意事项

1. **计算资源**: 完整实验预计需要大量计算时间和资源
2. **并行运行**: 建议使用多核并行运行提高效率
3. **结果监控**: 实验过程中监控失败配置和性能指标
4. **数据备份**: 确保原始数据文件安全

## 清理状态

已清理与Project1140消融实验无关的文件：
- 相关性分析相关文件
- Project1033相关分析文件
- 冗余的Jupyter notebook文件
- 旧的图片和数据文件

项目现在专注于Project1140的Capacity Factor预测消融实验。
