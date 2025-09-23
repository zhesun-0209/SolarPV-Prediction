# Season and Hour Analysis 实验总结

## 实验概述

本实验旨在对100个太阳能发电厂进行季节和小时分析，每个厂进行8个不同的模型实验，总共800个实验。

## 实验设计

### 实验规模
- **厂数量**: 100个厂（Project ID: 前100个）
- **每个厂实验数**: 8个实验
- **总实验数**: 800个实验

### 实验配置

#### 1. Linear/LSR 实验
- **模型**: Linear
- **回看窗口**: 24小时
- **时间编码**: 否 (noTE)
- **数据集**: 80%训练
- **天气特征**: 仅NWP预测特征

#### 2. 其他7个模型实验 (RF, XGB, LGBM, LSTM, GRU, TCN, Transformer)
- **回看窗口**: 24小时
- **复杂度**: 低复杂度 (level2)
- **时间编码**: 否 (noTE)
- **数据集**: 80%训练
- **天气特征**: PV + NWP预测特征

## 文件结构

```
season_and_hour_analysis/
├── scripts/
│   ├── generate_season_hour_configs.py    # 配置生成脚本
│   └── run_season_hour_experiments.py     # 实验运行脚本
├── run_season_hour_colab.py               # Colab运行脚本
├── test_config_generation.py              # 配置测试脚本
├── configs/                               # 生成的配置文件
│   ├── 171/
│   │   ├── season_hour_linear.yaml
│   │   ├── season_hour_rf.yaml
│   │   ├── season_hour_xgb.yaml
│   │   ├── season_hour_lgbm.yaml
│   │   ├── season_hour_lstm.yaml
│   │   ├── season_hour_gru.yaml
│   │   ├── season_hour_tcn.yaml
│   │   ├── season_hour_transformer.yaml
│   │   └── season_hour_index.yaml
│   ├── 172/
│   └── ...
├── results/                               # 实验结果（本地）
└── README.md                              # 说明文档
```

## 输出文件

### 保存位置
Google Drive: `/content/drive/MyDrive/Solar PV electricity/hour and season analysis/`

### 文件格式

#### 1. 汇总文件: `{project_id}_summary.csv`
包含每个厂所有8个实验的汇总结果，字段包括：
- 模型配置信息（模型名称、复杂度、参数等）
- 性能指标（训练时间、推理时间、参数数量等）
- 评估指标（MSE、RMSE、MAE、R²等）
- 实验标识（配置文件名称等）

#### 2. 预测文件: `{project_id}_prediction.csv`
包含测试集的详细预测结果，字段包括：
- 日期时间信息
- 真实值和预测值
- 模型和项目标识
- 窗口和小时信息

## 使用方法

### 快速开始
```bash
# 1. 生成配置文件
python season_and_hour_analysis/scripts/generate_season_hour_configs.py

# 2. 运行实验
python season_and_hour_analysis/scripts/run_season_hour_experiments.py

# 或者使用shell脚本
./run_season_hour_analysis.sh
```

### Colab运行
```bash
python season_and_hour_analysis/run_season_hour_colab.py
```

## 技术实现

### 核心修改
1. **新增保存函数**: `save_season_hour_results()` 在 `eval/eval_utils.py` 中
2. **修改主程序**: `main.py` 支持根据实验类型选择保存模式
3. **配置生成**: 自动生成800个实验配置文件
4. **结果解析**: 从实验输出中提取指标和预测结果

### 关键特性
- **自动日期提取**: 从测试集数据中提取日期信息
- **增量保存**: 支持中断后继续运行
- **详细日志**: 提供完整的运行状态信息
- **错误处理**: 单个实验失败不影响整体进度

## 预期结果

### 数据量
- **汇总文件**: 100个文件，每个8行（8个实验）
- **预测文件**: 100个文件，每个包含测试集的所有预测结果
- **总数据量**: 约800个实验的完整结果

### 分析价值
1. **模型比较**: 8种不同模型在同一数据集上的性能对比
2. **季节分析**: 通过日期信息分析不同季节的预测性能
3. **小时分析**: 通过小时信息分析不同时段的预测性能
4. **厂间差异**: 100个厂之间的性能差异分析

## 注意事项

### 运行要求
- 需要GPU支持深度学习模型
- 需要足够的存储空间（Google Drive）
- 建议在服务器上运行（长时间运行）

### 时间估算
- 每个实验: 5-15分钟
- 总运行时间: 67-200小时
- 建议分批运行或使用并行处理

### 监控建议
- 定期检查Google Drive存储空间
- 监控GPU内存使用情况
- 保存运行日志以便调试

## 后续分析

### 建议分析方向
1. **模型性能排名**: 基于MSE、RMSE等指标
2. **季节效应分析**: 按月份/季节分组分析
3. **小时效应分析**: 按小时分组分析
4. **天气特征重要性**: 分析不同天气特征的影响
5. **厂间差异分析**: 识别高性能和低性能厂的特征

### 数据使用
- 使用pandas读取CSV文件进行分析
- 使用matplotlib/seaborn进行可视化
- 使用scikit-learn进行统计分析
