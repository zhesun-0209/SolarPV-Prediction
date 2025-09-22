# SolarPV 敏感性分析实验

## 概述

本敏感性分析实验旨在研究不同因素对太阳能发电预测模型性能的影响。实验在前20个厂上进行，结果保存到Google Drive的`Solar PV electricity/sensitivity analysis`目录下。

## 实验设计

### 1. Weather Feature Adoption（天气特征采用）
- **SI (Solar Irradiance)**: 仅使用太阳辐射
- **H (High)**: 太阳辐射 + 水汽压差 + 相对湿度
- **M (Medium)**: H + 温度 + 阵风 + 云量 + 风速
- **L (Low)**: H + M + 雪深 + 露点 + 表面气压 + 降水

### 2. Lookback Window Length（回看窗口长度）
- 24小时
- 72小时
- 120小时
- 168小时

**样本数量控制**：
- 使用固定样本数量策略，确保所有回看窗口使用相同的训练样本数量
- 以最大回看窗口（168小时）为基准，保证实验的公平性

### 3. Model Complexity（模型复杂度）
- **Level 1**: 最简单设置（新增）
- **Level 2**: 原low设置
- **Level 3**: 中等设置（新增）
- **Level 4**: 原high设置

### 4. Training Dataset Scale（训练数据集规模）
- **Low**: 20+10+10（训练+验证+测试）
- **Medium**: 40+10+10
- **High**: 60+10+10
- **Full**: 80+10+10

## 模型类型

### 非LSR模型（受所有因素影响）
- RF (Random Forest)
- XGB (XGBoost)
- LGBM (LightGBM)
- LSTM
- GRU
- TCN
- Transformer

### LSR模型（受天气特征和数据集规模影响）
- LSR (Linear Regression) - 参与Weather feature adoption和Training dataset scale实验

## 实验数量计算

每个plant的实验数量（单项实验设计）：
- **Weather feature adoption实验**：4个档 × 8个模型 = 32个实验
- **Lookback window length实验**：4个档 × 7个模型 = 28个实验（LSR不受影响）
- **Model complexity实验**：4个档 × 7个模型 = 28个实验（LSR不受影响）
- **Training dataset scale实验**：4个档 × 8个模型 = 32个实验
- **总计：120个实验/plant**

## 文件结构

```
sensitivity_analysis/
├── configs/                    # 配置文件目录
│   ├── {project_id}/          # 每个项目的配置
│   │   ├── *.yaml            # 实验配置文件
│   │   └── sensitivity_index.yaml
│   └── sensitivity_global_index.yaml
├── scripts/                   # 脚本目录
│   ├── generate_sensitivity_configs.py  # 配置生成器
│   └── run_sensitivity_experiments.py   # 实验运行器
├── results/                   # 结果目录
└── README.md                  # 说明文档
```

## 使用方法

### 1. 生成配置文件
```bash
python sensitivity_analysis/scripts/generate_sensitivity_configs.py
```

### 2. 运行敏感性分析实验
```bash
python sensitivity_analysis/scripts/run_sensitivity_experiments.py
```

## 结果保存

实验结果将保存到Google Drive的以下位置：
- 路径：`/content/drive/MyDrive/Solar PV electricity/sensitivity analysis/`
- 格式：每个项目一个CSV文件（`{project_id}_sensitivity_results.csv`）
- 内容：包含所有实验参数和性能指标

## 实验参数说明

### 默认配置
- 24小时回看窗口
- 低复杂度
- 无时间编码
- PV+NWP输入

### 模型复杂度设置

#### Level 1（最简单）
- 深度学习模型：d_model=32, num_heads=2, num_layers=3
- 机器学习模型：n_estimators=20, max_depth=3
- 训练轮数：30

#### Level 2（原low）
- 深度学习模型：d_model=64, num_heads=4, num_layers=6
- 机器学习模型：n_estimators=50, max_depth=5
- 训练轮数：50

#### Level 3（中等）
- 深度学习模型：d_model=128, num_heads=8, num_layers=12
- 机器学习模型：n_estimators=100, max_depth=8
- 训练轮数：65

#### Level 4（原high）
- 深度学习模型：d_model=256, num_heads=16, num_layers=18
- 机器学习模型：n_estimators=200, max_depth=12
- 训练轮数：80

## 注意事项

1. LSR模型只受数据集规模影响，其他因素固定
2. 实验支持断点续传，已完成的实验会被跳过
3. 每个实验最多运行60分钟，超时会被标记为失败
4. 结果实时保存到Google Drive，避免数据丢失
5. **样本数量控制**：Lookback window length实验使用固定样本数量策略，确保不同回看窗口使用相同的训练样本数量，保证实验结果的公平性

## 重要说明

### 样本数量问题
不同的回看窗口长度会导致训练样本数量不一致：
- 24小时回看：N-24个样本
- 72小时回看：N-72个样本
- 120小时回看：N-120个样本
- 168小时回看：N-168个样本

为了解决这个问题，我们采用**固定样本数量策略**：
- 以最大回看窗口（168小时）为基准
- 所有回看窗口实验使用相同的有效数据范围
- 确保实验结果的公平性和可比较性

详细说明请参考：`LOOKBACK_SAMPLE_ISSUE.md`
