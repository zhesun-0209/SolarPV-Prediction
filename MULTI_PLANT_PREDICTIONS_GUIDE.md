# 多Plant预测结果保存指南

## 概述

本指南说明如何为Plant 171、172、186生成配置文件并保存预测结果到Google Drive。

## 文件说明

### 1. 配置文件生成
- **`generate_multi_plant_configs.py`**: 为多个Plant生成配置文件
- 为每个Plant生成384个配置文件（8个模型 × 6个场景 × 2个lookback × 2个TE × 2个复杂度）

### 2. 预测结果保存
- **`save_multi_plant_predictions.py`**: 本地环境下的多Plant预测结果保存
- **`colab_multi_plant_predictions.py`**: Colab环境下的多Plant预测结果保存

## 支持的Plant

- **Plant 171**: 使用 `data/Project171.csv`
- **Plant 172**: 使用 `data/Project172.csv`  
- **Plant 186**: 使用 `data/Project186.csv`

## 支持的模型

1. **深度学习模型**:
   - LSTM
   - GRU
   - TCN
   - Transformer

2. **机器学习模型**:
   - Random Forest (RF)
   - XGBoost (XGB)
   - LightGBM (LGBM)
   - Linear Regression (Linear)

## 支持的场景

1. **PV**: 仅使用PV数据
2. **PV+NWP**: PV + 数值天气预报
3. **PV+NWP+**: PV + 数值天气预报 + 理想NWP
4. **PV+HW**: PV + 历史天气
5. **NWP**: 仅使用数值天气预报
6. **NWP+**: 数值天气预报 + 理想NWP

## 参数组合

- **Lookback**: 24小时, 72小时
- **Time Encoding**: True, False
- **Complexity**: low, high

## 存储结构

每个Plant的结果将保存在Google Drive的以下结构中：

```
/content/drive/MyDrive/Solar PV electricity/plot/
├── Plant_171/
│   ├── Plant_171_all_predictions_summary.csv
│   ├── Plant_171_experiment_configs.csv
│   ├── by_scenario/
│   │   ├── PV_predictions.csv
│   │   ├── PV+NWP_predictions.csv
│   │   ├── PV+NWP+_predictions.csv
│   │   ├── PV+HW_predictions.csv
│   │   ├── NWP_predictions.csv
│   │   └── NWP+_predictions.csv
│   └── by_model/
│       ├── LSTM_predictions.csv
│       ├── GRU_predictions.csv
│       ├── TCN_predictions.csv
│       ├── Transformer_predictions.csv
│       ├── RF_predictions.csv
│       ├── XGB_predictions.csv
│       ├── LGBM_predictions.csv
│       └── Linear_predictions.csv
├── Plant_172/
│   └── ... (相同结构)
└── Plant_186/
    └── ... (相同结构)
```

## 使用方法

### 在Colab中运行

1. **上传数据文件**:
   - 确保 `data/Project171.csv`, `data/Project172.csv`, `data/Project186.csv` 存在

2. **运行Colab脚本**:
   ```python
   # 在Colab中运行
   !python colab_multi_plant_predictions.py
   ```

3. **查看结果**:
   - 结果将自动保存到Google Drive的指定路径
   - 每个Plant一个文件夹，包含所有预测结果

### 在本地运行

1. **生成配置文件**:
   ```bash
   python generate_multi_plant_configs.py
   ```

2. **运行预测保存**:
   ```bash
   python save_multi_plant_predictions.py
   ```

## 数据格式

### 汇总文件格式
每个Plant的汇总文件包含以下列：
- `experiment_id`: 实验ID
- `model`: 模型名称
- `scenario`: 场景名称
- `lookback`: 回看小时数
- `te`: 是否使用时间编码
- `complexity`: 模型复杂度
- `timestep`: 时间步
- `ground_truth`: 真实值
- `prediction`: 预测值
- `config_file`: 配置文件名称

### 数据量估算
- **每个Plant**: 384个实验 × 168个时间步 = 64,512行
- **总计**: 3个Plant × 64,512行 = 193,536行

## 注意事项

1. **数据文件**: 确保所有Plant的数据文件存在且格式正确
2. **存储空间**: Google Drive需要有足够的存储空间
3. **运行时间**: 完整运行可能需要数小时，建议在Colab Pro中运行
4. **内存使用**: 建议使用高内存的Colab实例

## 故障排除

### 常见问题

1. **数据文件不存在**:
   - 检查数据文件路径是否正确
   - 确保文件已上传到正确位置

2. **内存不足**:
   - 使用Colab Pro的高内存实例
   - 考虑分批处理

3. **存储空间不足**:
   - 清理Google Drive空间
   - 考虑压缩结果文件

### 调试建议

1. **单Plant测试**: 先测试单个Plant确保流程正常
2. **日志检查**: 查看控制台输出了解进度
3. **中间结果**: 检查中间保存的文件是否正确

## 扩展功能

### 添加新Plant
1. 在 `plant_ids` 列表中添加新的Plant ID
2. 确保对应的数据文件存在
3. 重新运行脚本

### 添加新模型
1. 在 `get_model_configs()` 函数中添加新模型配置
2. 在训练函数中添加对应的训练逻辑
3. 更新模型列表

### 添加新场景
1. 在 `get_scenario_configs()` 函数中添加新场景配置
2. 更新场景列表
3. 重新生成配置文件
