# Colab A100 运行指南

## 🚀 快速开始

### 1. 环境准备
```python
# 检查GPU
!nvidia-smi

# 检查CUDA版本
!nvcc --version
```

### 2. 克隆项目
```bash
!git clone https://github.com/zhesun-0209/SolarPV-Prediction.git
!cd SolarPV-Prediction
```

### 3. 安装依赖
```bash
!cd SolarPV-Prediction && pip install -r requirements.txt
```

## ⚙️ 配置A100优化

### 1. 启用混合精度训练
在 `config/default.yaml` 中添加：
```yaml
# A100优化配置
use_amp: true              # 自动混合精度
batch_size: 64             # 增大批次大小
num_workers: 4             # 数据加载器工作进程
pin_memory: true           # 固定内存
```

### 2. 修改训练参数
```yaml
train_params:
  batch_size: 64           # A100可以处理更大批次
  epochs: 100              # 增加训练轮数
  learning_rate: 2e-3      # 稍微提高学习率
  early_stop_patience: 15  # 增加早停耐心
```

## 🎯 推荐运行命令

### 1. 基础测试
```bash
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model Transformer --use_hist_weather true --use_forecast true
```

### 2. 模型对比实验
```bash
# Transformer
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model Transformer --use_hist_weather true --use_forecast true --model_complexity medium

# LSTM
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model LSTM --use_hist_weather true --use_forecast true --model_complexity medium

# TCN
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model TCN --use_hist_weather true --use_forecast true --model_complexity medium

# XGBoost
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model XGB --use_hist_weather true --use_forecast true --model_complexity medium
```

### 3. 消融实验
```bash
# 特征消融
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model Transformer --use_hist_weather false --use_forecast true --model_complexity medium

# 时间窗口测试
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model Transformer --past_days 7 --use_hist_weather true --use_forecast true --model_complexity medium

# 复杂度测试
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model Transformer --use_hist_weather true --use_forecast true --model_complexity high
```

## 📊 监控和调试

### 1. 实时监控
```python
# 监控GPU使用率
!watch -n 1 nvidia-smi

# 监控训练进度
import matplotlib.pyplot as plt
import pandas as pd

# 读取训练日志
def plot_training_progress(log_file):
    df = pd.read_csv(log_file)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['epoch_time'], label='Epoch Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.title('Training Speed')
    
    plt.tight_layout()
    plt.show()

# 使用示例
# plot_training_progress('outputs/Project_1033/dl/transformer/featTrue_fcstTrue_days3_compmedium/training_log.csv')
```

### 2. 结果分析
```python
# 分析结果
def analyze_results(summary_file):
    df = pd.read_csv(summary_file)
    print("=== 模型性能对比 ===")
    print(df[['model', 'test_loss', 'rmse', 'mae', 'train_time_sec', 'inference_time_sec']].round(4))
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(df['model'], df['rmse'])
    plt.title('RMSE Comparison')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.bar(df['model'], df['train_time_sec'])
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    plt.bar(df['model'], df['inference_time_sec'])
    plt.title('Inference Time Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# 使用示例
# analyze_results('outputs/Project_1033/dl/transformer/featTrue_fcstTrue_days3_compmedium/summary.csv')
```

## 🔧 性能优化建议

### 1. A100特定优化
```python
# 在训练脚本中添加
import torch

# 启用TensorFloat-32 (TF32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 启用cuDNN基准测试
torch.backends.cudnn.benchmark = True

# 设置内存分配策略
torch.cuda.set_per_process_memory_fraction(0.9)
```

### 2. 数据加载优化
```python
# 在data_utils.py中优化
def create_dataloader(X, y, batch_size=64, shuffle=True, num_workers=4):
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
```

## 📁 结果保存

### 1. 保存到Google Drive
```python
# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 复制结果到Drive
!cp -r outputs/ /content/drive/MyDrive/SolarPV-Results/
```

### 2. 下载结果
```python
# 压缩结果
!tar -czf results.tar.gz outputs/

# 下载
from google.colab import files
files.download('results.tar.gz')
```

## ⚠️ 注意事项

### 1. Colab限制
- **运行时间**: 最长12小时
- **内存限制**: 约25GB RAM
- **存储限制**: 约100GB临时存储

### 2. 最佳实践
- 定期保存检查点
- 监控GPU使用率
- 使用混合精度训练
- 合理设置批次大小

### 3. 故障排除
```python
# 检查CUDA可用性
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# 检查内存使用
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## 🎯 推荐实验流程

### 阶段1: 快速验证 (30分钟)
```bash
# 测试基本功能
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model TCN --use_hist_weather true --use_forecast false --model_complexity low
```

### 阶段2: 模型对比 (2-3小时)
```bash
# 运行所有模型
for model in Transformer LSTM GRU TCN RF GBR XGB LGBM; do
  echo "Running $model..."
  !cd SolarPV-Prediction && python main.py --config config/default.yaml --model $model --use_hist_weather true --use_forecast true --model_complexity medium
done
```

### 阶段3: 消融实验 (1-2小时)
```bash
# 特征消融
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model Transformer --use_hist_weather false --use_forecast true --model_complexity medium
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model Transformer --use_hist_weather true --use_forecast false --model_complexity medium
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model Transformer --use_hist_weather false --use_forecast false --model_complexity medium
```

### 阶段4: 深度优化 (剩余时间)
```bash
# 时间窗口和复杂度测试
!cd SolarPV-Prediction && python main.py --config config/default.yaml --model Transformer --past_days 7 --use_hist_weather true --use_forecast true --model_complexity high
```

## 📞 技术支持

如果遇到问题，请检查：
1. GPU是否正常分配
2. 依赖包是否正确安装
3. 数据文件是否存在
4. 配置文件是否正确

祝您实验顺利！🚀
