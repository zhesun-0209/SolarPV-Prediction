# Season and Hour Analysis 使用指南

## 概述

本指南介绍如何使用Season and Hour Analysis实验脚本进行100个厂的季节和小时分析实验。

## 实验设计

### 实验配置
- **厂数量**: 100个厂（前100个Project ID）
- **每个厂实验数**: 8个实验
- **总实验数**: 800个实验

### 8个实验详情

1. **Linear/LSR**: 24 hour look back, noTE, 80%dataset, NWP实验
2. **RF**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
3. **XGB**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
4. **LGBM**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
5. **LSTM**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
6. **GRU**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
7. **TCN**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
8. **Transformer**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP

## 使用步骤

### 1. 环境准备

确保以下文件存在：
- `main.py`
- `data/` 目录包含 `Project*.csv` 文件
- 所有必要的Python依赖包已安装

### 2. 生成配置文件

```bash
cd /path/to/SolarPV-Prediction
python season_and_hour_analysis/scripts/generate_season_hour_configs.py
```

这将为100个厂生成800个配置文件，保存在 `season_and_hour_analysis/configs/` 目录下。

### 3. 运行实验

#### 在Colab中运行
```bash
python season_and_hour_analysis/run_season_hour_colab.py
```

#### 在服务器上运行
```bash
python season_and_hour_analysis/scripts/run_season_hour_experiments.py
```

### 4. 监控进度

实验运行时会显示：
- 当前处理的厂ID
- 每个实验的完成状态
- 实时保存的结果信息

## 输出文件

### 保存位置
所有结果保存在Google Drive：
```
/content/drive/MyDrive/Solar PV electricity/hour and season analysis/
```

### 文件格式

每个厂生成两个文件：

#### 1. `{project_id}_summary.csv`
包含所有8个实验的汇总结果，字段包括：
- model: 模型名称
- weather_level: 天气特征级别
- lookback_hours: 回看小时数
- complexity_level: 复杂度级别
- dataset_scale: 数据集规模
- use_pv: 是否使用PV数据
- use_hist_weather: 是否使用历史天气
- use_forecast: 是否使用预测天气
- use_time_encoding: 是否使用时间编码
- past_days: 回看天数
- use_ideal_nwp: 是否使用理想NWP
- selected_weather_features: 选择的天气特征
- epochs: 训练轮数
- batch_size: 批次大小
- learning_rate: 学习率
- train_time_sec: 训练时间(秒)
- inference_time_sec: 推理时间(秒)
- param_count: 参数数量
- samples_count: 样本数量
- best_epoch: 最佳轮数
- final_lr: 最终学习率
- mse: 均方误差
- rmse: 均方根误差
- mae: 平均绝对误差
- nrmse: 标准化均方根误差
- r_square: 决定系数
- smape: 对称平均绝对百分比误差
- gpu_memory_used: GPU内存使用量
- config_file: 配置文件名称

#### 2. `{project_id}_prediction.csv`
包含测试集的预测结果和日期信息，字段包括：
- date: 日期时间
- ground_truth: 真实值
- prediction: 预测值
- model: 模型名称
- project_id: 项目ID
- window_index: 窗口索引
- hour: 小时

## 故障排除

### 常见问题

1. **配置文件生成失败**
   - 检查data目录是否存在
   - 确保有Project*.csv文件

2. **实验运行失败**
   - 检查Google Drive是否已挂载
   - 确保有足够的存储空间
   - 检查GPU内存是否足够

3. **结果保存失败**
   - 检查Drive路径权限
   - 确保有写入权限

### 调试模式

启用详细日志输出：
```bash
export PYTHONPATH=/path/to/SolarPV-Prediction:$PYTHONPATH
python -u season_and_hour_analysis/scripts/run_season_hour_experiments.py
```

## 性能优化

### 建议设置
- 使用GPU加速深度学习模型训练
- 确保有足够的内存处理大数据集
- 考虑并行运行多个实验（需要修改脚本）

### 时间估算
- 每个实验约需5-15分钟
- 800个实验总计约需67-200小时
- 建议在服务器上运行，可以中断后继续

## 结果分析

### 数据使用
1. 使用 `summary.csv` 进行模型性能比较
2. 使用 `prediction.csv` 进行时间序列分析
3. 可以按季节、小时等维度进行分组分析

### 分析建议
- 比较不同模型在同一厂的表现
- 分析不同厂之间的性能差异
- 研究季节和小时对预测性能的影响
- 评估天气特征的重要性

## 联系支持

如有问题，请检查：
1. 日志输出中的错误信息
2. 配置文件格式是否正确
3. 数据文件是否完整
4. 系统资源是否充足
