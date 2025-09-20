# 340个组合预测结果存储指南

## 存储方案

### 推荐存储结构
```
340_predictions_results/
├── all_predictions_summary.csv      # 汇总文件（所有结果）
├── experiment_configs.csv           # 实验配置汇总
├── PV_predictions.csv              # PV场景结果
├── PV+NWP_predictions.csv          # PV+NWP场景结果
├── PV+NWP+_predictions.csv         # PV+NWP+场景结果
├── PV+HW_predictions.csv           # PV+HW场景结果
├── NWP_predictions.csv             # NWP场景结果
├── NWP+_predictions.csv            # NWP+场景结果
├── LSTM_predictions.csv            # LSTM模型结果
├── GRU_predictions.csv             # GRU模型结果
├── TCN_predictions.csv             # TCN模型结果
├── Transformer_predictions.csv     # Transformer模型结果
├── RF_predictions.csv              # RF模型结果
├── XGB_predictions.csv             # XGB模型结果
├── LGBM_predictions.csv            # LGBM模型结果
└── Linear_predictions.csv          # Linear模型结果（包含LSR）
```

## 文件格式说明

### 1. 汇总文件 (all_predictions_summary.csv)
包含所有实验的预测结果，每行代表一个时间步的预测：

| 列名 | 类型 | 说明 |
|------|------|------|
| experiment_id | int | 实验ID (1-340) |
| model | str | 模型名称 |
| scenario | str | 场景名称 |
| lookback | int | 回看窗口(24/72) |
| te | bool | 是否使用时间编码 |
| complexity | str | 模型复杂度(low/high) |
| timestep | int | 时间步(0-167) |
| ground_truth | float | 真实值 |
| prediction | float | 预测值 |
| config_file | str | 配置文件名称 |

### 2. 场景文件 (PV_predictions.csv等)
按场景分组的预测结果，格式与汇总文件相同，但只包含特定场景的数据。

### 3. 模型文件 (LSTM_predictions.csv等)
按模型分组的预测结果，格式与汇总文件相同，但只包含特定模型的数据。

### 4. 配置汇总文件 (experiment_configs.csv)
包含所有实验的配置信息：

| 列名 | 类型 | 说明 |
|------|------|------|
| experiment_id | int | 实验ID |
| model | str | 模型名称 |
| scenario | str | 场景名称 |
| lookback | int | 回看窗口 |
| te | bool | 时间编码 |
| complexity | str | 复杂度 |
| config_file | str | 配置文件 |
| data_points | int | 数据点数量 |

## 数据特点

### 时间序列长度
- 每个实验包含168个时间步（7天 × 24小时）
- 每个时间步包含ground_truth和prediction两个值

### 数据量估算
- **汇总文件**: 340个实验 × 168个时间步 = 57,120行
- **场景文件**: 平均每个场景约9,520行
- **模型文件**: 平均每个模型约7,140行（8个模型）

## 使用方法

### 1. 运行脚本保存结果
```bash
python save_340_predictions.py
```

### 2. 加载特定场景数据
```python
import pandas as pd

# 加载PV场景的所有结果
pv_data = pd.read_csv('340_predictions_results/PV_predictions.csv')

# 查看特定模型的PV场景结果
lstm_pv = pv_data[pv_data['model'] == 'LSTM']
```

### 3. 加载特定模型数据
```python
# 加载LSTM模型的所有结果
lstm_data = pd.read_csv('340_predictions_results/LSTM_predictions.csv')

# 查看LSTM在PV场景的结果
lstm_pv = lstm_data[lstm_data['scenario'] == 'PV']
```

### 4. 分析最佳结果
```python
# 加载汇总数据
all_data = pd.read_csv('340_predictions_results/all_predictions_summary.csv')

# 计算每个实验的MSE
mse_results = all_data.groupby('experiment_id').apply(
    lambda x: ((x['ground_truth'] - x['prediction']) ** 2).mean()
).reset_index()
mse_results.columns = ['experiment_id', 'mse']

# 找到最佳结果
best_experiment = mse_results.loc[mse_results['mse'].idxmin()]
```

## 优势

1. **结构化存储**: 按场景和模型分组，便于分析
2. **时间序列完整**: 保留完整的168小时预测序列
3. **配置信息完整**: 包含所有实验参数
4. **易于查询**: 支持按不同维度筛选数据
5. **可扩展性**: 便于添加新的分析维度

## 注意事项

1. **文件大小**: 汇总文件可能较大，建议使用pandas的chunksize参数
2. **内存使用**: 加载所有数据时注意内存使用
3. **数据完整性**: 确保所有实验都成功完成
4. **备份**: 建议定期备份结果文件
