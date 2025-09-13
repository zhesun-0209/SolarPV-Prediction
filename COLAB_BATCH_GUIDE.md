# 🚀 Colab批量实验指南

在Google Colab上运行100个项目的全量实验，每个项目340个实验，结果保存到Google Drive。

## 📁 文件说明

### 1. `colab_batch_experiments.py` - 批量实验脚本
- 自动扫描data/目录下的所有Project*.csv文件
- 为每个项目运行340个实验
- 结果保存到Google Drive
- 生成详细的实验报告

### 2. `utils/drive_utils.py` - Google Drive工具
- 自动挂载Google Drive
- 保存结果文件到Drive
- 管理Drive文件夹结构

## 🎯 使用方法

### 步骤1: 准备数据
确保data/目录下有以下文件：
```
data/
├── Project1140.csv
├── Project1141.csv
├── Project1142.csv
├── ...
└── Project1199.csv
```

### 步骤2: 运行批量实验
```python
# 在Colab中运行
!python colab_batch_experiments.py
```

### 步骤3: 查看结果
结果将保存到Google Drive的以下位置：
```
/content/drive/MyDrive/SolarPV_Results/
├── Project_1140/
│   └── 1140_results.csv
├── Project_1141/
│   └── 1141_results.csv
├── ...
└── experiment_report.csv
```

## 📊 实验配置

每个项目将运行以下340个实验：
- **深度学习模型**: LSTM, GRU, Transformer, TCN
- **机器学习模型**: RF, XGB, LGBM, LSR
- **复杂度**: low, high
- **特征组合**: 各种PV、天气、时间编码组合
- **预测时长**: 24h, 72h

## 🔧 自定义设置

### 修改实验范围
```python
# 只运行特定项目
projects = ['1140', '1141', '1142']  # 修改get_available_projects函数

# 只运行特定模型
config_files = [f for f in config_files if 'LSTM' in f]  # 只运行LSTM实验
```

### 修改保存位置
```python
# 修改Drive保存位置
drive_folder = "/content/drive/MyDrive/MyCustomFolder"
```

## 📈 监控进度

脚本会实时显示：
- 当前项目进度
- 实验成功/失败统计
- 预计剩余时间
- 实时结果指标

## ⚠️ 注意事项

1. **运行时间**: 100个项目 × 340个实验 ≈ 大量时间，建议分批运行
2. **存储空间**: 确保Google Drive有足够空间
3. **超时设置**: 每个实验30分钟超时，可根据需要调整
4. **错误处理**: 失败的实验会记录错误信息，不影响其他实验

## 🛠️ 故障排除

### 常见问题
1. **Drive挂载失败**: 检查Colab权限设置
2. **内存不足**: 减少并发实验数量
3. **超时错误**: 增加超时时间或检查模型复杂度

### 恢复实验
如果实验中断，可以：
1. 检查已完成的项目
2. 从未完成的项目继续
3. 重新运行失败的实验

## 📊 结果分析

实验完成后，可以：
1. 下载CSV结果文件进行分析
2. 使用pandas读取结果数据
3. 生成性能对比图表
4. 识别最佳模型配置
