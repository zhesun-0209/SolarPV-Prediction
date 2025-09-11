# Project1033单厂测试指南

## 📋 概述

本指南用于在Project1033上进行单厂测试，运行所有300个实验组合，结果保存到Drive的`result_new`文件夹。

## 🎯 测试目标

- **模型**: 8种 (Transformer, LSTM, GRU, TCN, RF, XGB, LGBM, Linear)
- **特征组合**: 20种
- **复杂度**: 2种 (Low, High)
- **总实验数**: 300个
- **预计时间**: 2.5小时

## 📁 文件说明

### 配置文件
- `config/project1033_test.yaml`: Project1033测试专用配置
  - 保存路径: `/content/drive/MyDrive/Solar PV electricity/result_new`
  - 不保存预测结果文件，只保存Excel结果

### 测试脚本
- `test_project1033.py`: 完整测试脚本 (300个实验)
- `quick_test_project1033.py`: 快速测试脚本 (3个关键实验)

## 🚀 使用方法

### 方法1: 快速测试 (推荐先运行)

```bash
python quick_test_project1033.py
```

**测试内容**:
1. Transformer + 仅历史PV + 时间编码 + 1天 + Low
2. RF + 历史PV+历史天气 + 全部天气 + 无时间编码 + 3天 + High  
3. Linear + 仅预测天气 + 太阳辐射 + 时间编码 + 无历史数据

**用途**: 验证配置是否正确，预计5-10分钟

### 方法2: 完整测试

```bash
python test_project1033.py
```

**测试内容**: 所有300个实验组合
**用途**: 完整测试，预计2.5小时

### 方法3: 手动运行单个实验

```bash
python main.py --config config/project1033_test.yaml --model Transformer --use_pv true --use_hist_weather false --use_forecast false --weather_category irradiance --use_time_encoding true --past_days 1 --model_complexity low --data_path data/Project1033.csv --plant_id Project1033 --save_dir /content/drive/MyDrive/Solar PV electricity/result_new/Project1033
```

## 📊 结果文件

### Excel结果文件
- **位置**: `/content/drive/MyDrive/Solar PV electricity/result_new/Project1033/Project1033_results.xlsx`
- **内容**: 25列，包含所有实验的配置和性能指标
- **行数**: 300行 (每个实验一行)

### 列结构
- **配置列 (14列)**: model, use_pv, use_hist_weather, use_forecast, weather_category, use_time_encoding, past_days, model_complexity, epochs, batch_size, learning_rate
- **性能列 (6列)**: train_time_sec, inference_time_sec, param_count, samples_count, best_epoch, final_lr
- **评估列 (5列)**: mse, rmse, mae, nrmse, r_square, smape, gpu_memory_used

## 🔧 配置说明

### 特征组合 (20种)
1. **仅历史PV (4种)**: 1天/3天 × 时间编码/无时间编码
2. **历史PV+历史天气 (4种)**: 太阳辐射/全部天气 × 1天/3天
3. **历史PV+预测天气 (4种)**: 太阳辐射/全部天气 × 1天/3天
4. **历史PV+历史+预测天气 (4种)**: 太阳辐射/全部天气 × 1天/3天
5. **仅预测天气 (4种)**: 太阳辐射/全部天气 × 时间编码/无时间编码

### 模型配置
- **DL模型**: 使用优化后的配置，dropout=0.2，学习率=5e-4
- **ML模型**: 使用优化后的参数，更好的性能
- **Linear模型**: 无复杂度区分

## ⚠️ 注意事项

1. **数据文件**: 确保`data/Project1033.csv`存在
2. **Drive挂载**: 确保Google Drive已正确挂载
3. **权限**: 确保有写入`result_new`文件夹的权限
4. **超时**: 完整测试设置了5小时超时
5. **中断恢复**: 支持断点续传，已完成的实验会跳过

## 🐛 故障排除

### 常见问题
1. **数据文件不存在**: 检查`data/Project1033.csv`是否存在
2. **Drive未挂载**: 运行`from google.colab import drive; drive.mount('/content/drive')`
3. **权限不足**: 检查Drive文件夹权限
4. **内存不足**: 减少batch_size或使用更简单的模型

### 调试方法
1. 先运行快速测试验证配置
2. 检查错误日志和返回码
3. 验证单个实验是否成功
4. 检查结果文件是否正确生成

## 📈 预期结果

- **成功实验数**: 300个
- **平均每实验时间**: 30秒
- **总耗时**: 2.5小时
- **结果文件大小**: 约50-100KB
- **最佳模型**: 根据RMSE/MAE指标判断

## 🎉 完成检查

测试完成后，请检查:
1. ✅ Excel文件是否生成
2. ✅ 实验数量是否为300个
3. ✅ 所有列是否完整
4. ✅ 性能指标是否合理
5. ✅ 无错误或异常值
