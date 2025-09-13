# Project1140 消融实验结果保存结构

## 📁 结果保存位置

消融实验的结果保存在以下目录结构中：

```
results/
└── ablation/                          # 消融实验主目录
    ├── experiment_summary.csv         # 所有实验结果摘要
    ├── failed_configs.csv            # 失败的配置列表（如有）
    ├── statistics_report.md          # 统计分析报告
    ├── ablation_experiments.log      # 实验运行日志
    │
    ├── LSR_baseline_PV_24h_noTE/     # 单个实验结果目录
    │   ├── config.yaml               # 实验配置
    │   ├── results.xlsx              # 详细结果（Excel格式）
    │   ├── predictions.csv           # 预测结果
    │   ├── training_log.csv          # 训练日志（仅深度学习模型）
    │   └── model.pth                 # 训练好的模型文件
    │
    ├── Transformer_high_PV_plus_NWP_72h_TE/
    │   ├── config.yaml
    │   ├── results.xlsx
    │   ├── predictions.csv
    │   ├── training_log.csv
    │   └── model.pth
    │
    └── [其他358个实验目录...]
```

## 📊 结果文件详细说明

### 1. 实验摘要文件 (`experiment_summary.csv`)

包含所有360个实验的摘要信息：

| 列名 | 描述 |
|------|------|
| `config_name` | 配置名称 |
| `status` | 实验状态 (completed/failed/timeout/error) |
| `duration` | 实验耗时（秒） |
| `timestamp` | 实验完成时间 |
| `mae` | 平均绝对误差 |
| `rmse` | 均方根误差 |
| `r2` | R²决定系数 |
| `mape` | 平均绝对百分比误差 |

### 2. 失败配置文件 (`failed_configs.csv`)

记录失败的实验配置：

| 列名 | 描述 |
|------|------|
| `config_name` | 配置名称 |
| `status` | 失败类型 (failed/timeout/error/exception) |
| `error` | 错误信息 |
| `duration` | 运行时间 |

### 3. 统计分析报告 (`statistics_report.md`)

包含以下内容：
- **总体统计**: 成功/失败数量、成功率
- **性能指标统计**: MAE、RMSE、R²、MAPE的统计信息
- **模型性能比较**: 按模型类型分组的性能对比
- **失败配置分析**: 失败类型分布

### 4. 单个实验结果目录

每个实验都有独立的目录，包含：

#### 4.1 配置文件 (`config.yaml`)
- 实验的完整配置信息
- 模型参数、训练参数、数据设置等

#### 4.2 详细结果文件 (`results.xlsx`)
包含25列的详细实验结果：

**实验配置列 (14列)**:
- `model`: 模型类型
- `use_pv`: 是否使用PV输入
- `use_hist_weather`: 是否使用历史天气
- `use_forecast`: 是否使用预测天气
- `weather_category`: 天气特征类别
- `use_time_encoding`: 是否使用时间编码
- `past_days`: 历史天数
- `model_complexity`: 模型复杂度
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率

**性能指标列 (6列)**:
- `train_time_sec`: 训练时间（秒）
- `inference_time_sec`: 推理时间（秒）
- `param_count`: 参数数量
- `samples_count`: 测试样本数量
- `best_epoch`: 最佳训练轮数
- `final_lr`: 最终学习率

**评估指标列 (5列)**:
- `mse`: 均方误差
- `rmse`: 均方根误差
- `mae`: 平均绝对误差
- `nrmse`: 标准化均方根误差
- `r_square`: R²决定系数
- `smape`: 对称平均绝对百分比误差
- `gpu_memory_used`: GPU内存使用量

#### 4.3 预测结果文件 (`predictions.csv`)
包含每个预测点的详细信息：

| 列名 | 描述 |
|------|------|
| `window_index` | 时间窗口索引 |
| `forecast_datetime` | 预测时间点 |
| `hour` | 小时 |
| `y_true` | 真实值 |
| `y_pred` | 预测值 |

#### 4.4 训练日志文件 (`training_log.csv`)
仅深度学习模型生成，包含：

| 列名 | 描述 |
|------|------|
| `epoch` | 训练轮数 |
| `train_loss` | 训练损失 |
| `val_loss` | 验证损失 |
| `learning_rate` | 学习率 |

#### 4.5 模型文件 (`model.pth`)
训练好的模型权重文件，可用于后续预测或分析。

## 🔍 结果分析建议

### 1. 快速查看总体结果
```bash
# 查看实验摘要
cat results/ablation/experiment_summary.csv

# 查看统计报告
cat results/ablation/statistics_report.md
```

### 2. 分析特定模型性能
```bash
# 查看Transformer模型结果
grep "Transformer" results/ablation/experiment_summary.csv
```

### 3. 比较不同输入特征的效果
```bash
# 比较PV vs PV+NWP的效果
grep -E "(PV_24h_noTE|PV_plus_NWP_24h_noTE)" results/ablation/experiment_summary.csv
```

### 4. 分析失败原因
```bash
# 查看失败的配置
cat results/ablation/failed_configs.csv
```

## 📈 结果可视化

可以使用以下Python代码进行结果分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载结果摘要
df = pd.read_csv('results/ablation/experiment_summary.csv')

# 按模型类型比较性能
model_performance = df.groupby('model')['mae'].mean().sort_values()
print("模型性能排序（按MAE）:")
print(model_performance)

# 绘制性能分布图
plt.figure(figsize=(12, 6))
df.boxplot(column='mae', by='model', ax=plt.gca())
plt.title('不同模型的MAE分布')
plt.show()
```

## ⚠️ 注意事项

1. **存储空间**: 360个实验可能产生大量文件，建议预留足够的存储空间
2. **运行时间**: 完整实验可能需要数小时到数天，建议使用并行运行
3. **结果备份**: 建议定期备份重要的实验结果
4. **错误处理**: 关注失败配置文件，及时处理实验错误
