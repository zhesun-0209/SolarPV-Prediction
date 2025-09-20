# 340个组合结果展示指南

## 概述
本项目包含340个实验组合，涵盖6种情况 × 8个模型 × 多种参数组合。本文档说明如何合理展示这些结果。

## 实验组合结构

### 6种情况 (Scenarios)
1. **PV**: 仅历史PV功率
2. **PV+NWP**: 历史PV + 目标日NWP
3. **PV+NWP+**: 历史PV + 理想NWP
4. **PV+HW**: 历史PV + 历史天气
5. **NWP**: 仅目标日NWP
6. **NWP+**: 仅理想NWP

### 8个模型 (Models)
1. **LSTM**: 长短期记忆网络
2. **GRU**: 门控循环单元
3. **TCN**: 时间卷积网络
4. **Transformer**: 注意力机制模型
5. **RF**: 随机森林
6. **XGB**: XGBoost
7. **LGBM**: LightGBM
8. **Linear**: 线性回归

### 参数组合
- **Lookback**: 24h, 72h
- **Time Encoding**: True, False
- **Complexity**: low, high

**总组合数**: 6 × 8 × 2 × 2 × 2 = 384个（实际340个，因为LSR只有NWP和NWP+）

## 展示方式

### 1. 场景对比图 (`plot_340_combinations.py`)
- **6张图片**，每张对应一个场景
- 每张图片：2×4子图布局，显示8个模型
- 每个子图：显示该模型在该场景下的所有参数组合结果
- 文件名：`scenario_{场景名}_comparison.png`

### 2. 模型对比图 (`plot_340_combinations.py`)
- **8张图片**，每张对应一个模型
- 每张图片：2×3子图布局，显示6个场景
- 每个子图：显示该模型在该场景下的所有参数组合结果
- 文件名：`model_{模型名}_comparison.png`

### 3. 结果汇总图 (`plot_340_summary.py`)
- **3张汇总图片**：
  - `best_results_summary.png`: 最佳结果汇总（按场景）
  - `model_performance_heatmap.png`: 模型性能热力图
  - `parameter_analysis.png`: 参数分析图

## 使用方法

### 在Colab中运行

1. **场景和模型对比**：
   ```bash
   python plot_340_combinations.py
   ```
   - 生成14张详细对比图
   - 每张图显示所有参数组合的结果

2. **结果汇总分析**：
   ```bash
   python plot_340_summary.py
   ```
   - 生成3张汇总分析图
   - 提供整体性能概览

## 输出文件结构

```
340_combinations_plots/          # 详细对比图
├── scenario_PV_comparison.png
├── scenario_PV+NWP_comparison.png
├── scenario_PV+NWP+_comparison.png
├── scenario_PV+HW_comparison.png
├── scenario_NWP_comparison.png
├── scenario_NWP+_comparison.png
├── model_LSTM_comparison.png
├── model_GRU_comparison.png
├── model_TCN_comparison.png
├── model_Transformer_comparison.png
├── model_RF_comparison.png
├── model_XGB_comparison.png
├── model_LGBM_comparison.png
└── model_Linear_comparison.png

340_combinations_summary/        # 汇总分析图
├── best_results_summary.png
├── model_performance_heatmap.png
└── parameter_analysis.png
```

## 图片说明

### 详细对比图
- **X轴**: 时间步（168小时，7天）
- **Y轴**: Capacity Factor (%)
- **线条**:
  - 灰色粗线: Ground Truth
  - 彩色细线: 不同参数组合的预测结果
- **图例**: 显示参数组合（如"72h-TE-high"）

### 汇总分析图
- **柱状图**: 显示最佳MSE结果
- **热力图**: 显示模型-场景性能矩阵
- **参数分析**: 显示不同参数对性能的影响

## 注意事项

1. **LSR模型**: 只有NWP和NWP+两种情况
2. **训练时间**: 340个组合需要较长时间训练
3. **内存使用**: 建议使用GPU加速训练
4. **结果保存**: 图片自动保存到指定目录

## 扩展使用

如需自定义展示方式，可以修改脚本中的以下参数：
- 子图布局
- 颜色方案
- 指标选择
- 图片尺寸
- 保存格式
