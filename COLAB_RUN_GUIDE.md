# 🚀 SolarPV-Prediction Colab运行指南

## 📋 准备工作

### 1. 挂载Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. 克隆项目
```python
!git clone https://github.com/zhesun-0209/SolarPV-Prediction.git
import os
os.chdir('SolarPV-Prediction')
```

### 3. 安装依赖
```python
!pip install -r requirements.txt
```

## 🎯 运行方式

### 方式1: 快速模型对比（推荐）
```python
# 运行4个模型的快速对比
!python colab_run.py
```

### 方式2: 全参数组合实验
```python
# 运行所有96个实验组合
!python colab_full_experiments.py
```

### 方式3: 单个模型测试
```python
# 测试单个模型
!python main.py --model Transformer --use_hist_weather true --use_forecast true --model_complexity medium
```

## 📊 结果保存

### 默认保存位置
- 本地: `./result/`
- 建议修改为: `/content/drive/MyDrive/Solar PV electricity/results/`

### 修改保存路径
```python
import yaml

# 读取配置
with open('config/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 修改保存路径
config['save_dir'] = '/content/drive/MyDrive/Solar PV electricity/results'

# 保存配置
with open('config/default.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("✅ 保存路径已修改为Google Drive")
```

## 🔧 配置说明

### 模型参数
- **模型**: Transformer, LSTM, GRU, TCN, RF, GBR, XGB, LGBM
- **特征组合**: 4种 (hist_only, fcst_only, both, none)
- **复杂度**: 3种 (low, medium, high)
- **总实验数**: 8 × 4 × 3 = 96个

### 时间窗口
- **历史天数**: 3天 (72小时)
- **预测小时**: 24小时
- **样本数**: 约1000-2000个

## 📈 结果分析

### 输出文件
- `summary.csv`: 每个实验的详细指标
- `all_experiments_results.csv`: 所有实验的合并结果
- `comprehensive_analysis.png`: 可视化分析图表

### 指标说明
- **主要指标**: test_loss, rmse, mae (overall方式)
- **辅助指标**: hourly_rmse, daily_rmse, sample_rmse
- **性能指标**: train_time_sec, inference_time_sec
- **标准化指标**: norm_rmse, norm_mae

## ⚡ 性能优化

### A100优化设置
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(0.9)
```

### 内存管理
```python
# 清理内存
import gc
gc.collect()
torch.cuda.empty_cache()
```

## 🚨 注意事项

### 1. 运行时间
- **快速对比**: 10-30分钟
- **全参数实验**: 2-8小时
- **Colab限制**: 最长12小时

### 2. 数据检查
```python
# 检查数据文件
import pandas as pd
df = pd.read_csv('data/Project1033.csv')
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
```

### 3. 结果验证
```python
# 检查结果文件
import glob
result_files = glob.glob('result/**/*.csv', recursive=True)
print(f"找到 {len(result_files)} 个结果文件")
```

## 🔍 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size或使用CPU
2. **依赖安装失败**: 使用conda或手动安装
3. **数据文件缺失**: 检查文件路径和权限

### 调试命令
```python
# 检查环境
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 检查数据
import os
print(f"数据文件存在: {os.path.exists('data/Project1033.csv')}")
print(f"结果目录存在: {os.path.exists('result')}")
```

## 📞 支持

如果遇到问题，请检查：
1. 数据文件是否正确
2. 依赖是否完整安装
3. 内存是否充足
4. 运行时间是否超限

---

**祝您实验顺利！** 🎉
