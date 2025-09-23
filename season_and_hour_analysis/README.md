# Season and Hour Analysis实验

本目录包含用于进行季节和小时分析的实验脚本和配置。

## 实验概述

本实验在100个厂上进行，每个厂进行8个实验：

1. **Linear/LSR**: 24 hour look back, noTE, 80%dataset, NWP实验
2. **RF**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
3. **XGB**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
4. **LGBM**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
5. **LSTM**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
6. **GRU**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
7. **TCN**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP
8. **Transformer**: 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP

## 文件结构

```
season_and_hour_analysis/
├── scripts/
│   ├── generate_season_hour_configs.py    # 生成实验配置
│   └── run_season_hour_experiments.py     # 运行实验
├── run_season_hour_colab.py               # Colab运行脚本
├── configs/                               # 生成的配置文件
│   ├── 171/
│   ├── 172/
│   └── ...
└── results/                               # 实验结果
    ├── 171/
    ├── 172/
    └── ...
```

## 使用方法

### 1. 生成配置

```bash
python season_and_hour_analysis/scripts/generate_season_hour_configs.py
```

### 2. 运行实验

```bash
python season_and_hour_analysis/scripts/run_season_hour_experiments.py
```

### 3. Colab运行

```bash
python season_and_hour_analysis/run_season_hour_colab.py
```

## 实验结果

实验结果将保存到Google Drive的以下位置：

- **路径**: `/content/drive/MyDrive/Solar PV electricity/hour and season analysis/`
- **格式**: 每个项目两个文件
  - `{project_id}_summary.csv` - 包含所有实验的汇总结果
  - `{project_id}_prediction.csv` - 包含测试集的预测结果和日期信息

## 实验配置说明

### Linear/LSR配置
- 模型: Linear
- 回看窗口: 24小时
- 时间编码: 否
- 数据集: 80%训练
- 天气特征: 仅NWP预测特征

### 其他模型配置
- 模型: RF, XGB, LGBM, LSTM, GRU, TCN, Transformer
- 回看窗口: 24小时
- 复杂度: 低复杂度 (level2)
- 时间编码: 否
- 数据集: 80%训练
- 天气特征: PV + NWP预测特征

## 输出文件格式

### summary.csv字段
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

### prediction.csv字段
- date: 日期
- ground_truth: 真实值
- prediction: 预测值
- model: 模型名称
- project_id: 项目ID

## 注意事项

1. 确保Google Drive已挂载
2. 确保data目录下有Project*.csv文件
3. 实验可能需要较长时间，建议在服务器上运行
4. 结果会自动保存到指定的Drive路径
