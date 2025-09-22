# SolarPV 敏感性分析实验总结

## 实验概述

本敏感性分析实验旨在研究不同因素对太阳能发电预测模型性能的影响。实验采用**单项实验设计**，即每次只改变一个变量，其他变量保持默认配置。

## 默认配置

- **回看窗口**：24小时
- **模型复杂度**：Level 2（原low设置）
- **时间编码**：无时间编码（noTE）
- **输入特征**：PV+NWP（所有天气特征）
- **训练集比例**：80%

## 实验设计

### 1. Weather Feature Adoption（天气特征采用）
**目标**：研究不同天气特征组合对模型性能的影响

**实验设置**：
- 只改变天气特征组合
- 其他参数保持默认配置

**特征组合**：
- **SI (Solar Irradiance)**: 仅使用太阳辐射
- **H (High)**: 太阳辐射 + 水汽压差 + 相对湿度
- **M (Medium)**: H + 温度 + 阵风 + 云量 + 风速
- **L (Low)**: H + M + 雪深 + 露点 + 表面气压 + 降水

**实验数量**：4个档 × 8个模型 = 32个实验

### 2. Lookback Window Length（回看窗口长度）
**目标**：研究不同回看窗口长度对模型性能的影响

**实验设置**：
- 只改变回看窗口长度
- 其他参数保持默认配置

**回看窗口**：
- 24小时
- 72小时
- 120小时
- 168小时

**实验数量**：4个档 × 7个模型 = 28个实验（LSR不受影响）

### 3. Model Complexity（模型复杂度）
**目标**：研究不同模型复杂度对性能的影响

**实验设置**：
- 只改变模型复杂度级别
- 其他参数保持默认配置

**复杂度级别**：
- **Level 1**: 最简单设置
- **Level 2**: 原low设置（默认）
- **Level 3**: 中等设置
- **Level 4**: 原high设置

**实验数量**：4个档 × 7个模型 = 28个实验（LSR不受影响）

### 4. Training Dataset Scale（训练数据集规模）
**目标**：研究不同训练数据量对模型性能的影响

**实验设置**：
- 只改变训练集比例
- 其他参数保持默认配置

**数据集规模**：
- **Low**: 20+10+10（训练+验证+测试）
- **Medium**: 40+10+10
- **High**: 60+10+10
- **Full**: 80+10+10

**实验数量**：4个档 × 8个模型 = 32个实验

## 模型类型

### 非LSR模型（参与所有实验）
- RF (Random Forest)
- XGB (XGBoost)
- LGBM (LightGBM)
- LSTM
- GRU
- TCN
- Transformer

### LSR模型（仅参与部分实验）
- LSR (Linear Regression)
  - 参与：Weather feature adoption实验
  - 参与：Training dataset scale实验
  - 不参与：Lookback window length实验（不受影响）
  - 不参与：Model complexity实验（不受影响）

## 实验数量统计

**每个plant的实验数量**：
- Weather feature adoption: 32个实验
- Lookback window length: 28个实验
- Model complexity: 28个实验
- Training dataset scale: 32个实验
- **总计：120个实验/plant**

**20个厂的总实验数量**：
- 120 × 20 = 2,400个实验

## 结果保存

实验结果将保存到Google Drive：
- 路径：`/content/drive/MyDrive/Solar PV electricity/sensitivity analysis/`
- 格式：每个项目一个CSV文件（`{project_id}_sensitivity_results.csv`）
- 内容：包含所有实验参数和性能指标

## 实验优势

1. **单项实验设计**：每次只改变一个变量，便于分析各因素的影响
2. **控制变量**：其他参数保持默认配置，确保实验的公平性
3. **全面覆盖**：涵盖天气特征、回看窗口、模型复杂度、数据集规模四个关键维度
4. **模型适配**：针对不同模型类型设计合适的实验方案

## 使用方法

1. 生成配置文件：
```bash
python sensitivity_analysis/scripts/generate_sensitivity_configs.py
```

2. 运行敏感性分析实验：
```bash
python sensitivity_analysis/scripts/run_sensitivity_experiments.py
```

3. 在Colab上运行：
```bash
python sensitivity_analysis/run_sensitivity_colab.py
```
