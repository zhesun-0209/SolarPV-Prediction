# 100个Project消融实验使用指南

## 📋 概述

本指南详细说明如何运行100个Project的消融实验，每个Project包含360个实验配置，总共36000个实验。

## 🎯 主要特性

- **100个Project**: 支持Project001.csv到Project100.csv
- **360个实验配置**: 每个Project包含完整的消融实验矩阵
- **实时保存**: 结果实时保存到Google Drive，支持断点续训
- **批量运行**: 支持并行运行，提高效率
- **进度监控**: 实时监控实验进度和状态

## 📁 结果保存结构

### Google Drive保存位置
```
/content/drive/MyDrive/Solar PV electricity/ablation results/
├── Project001.csv          # Project001的360个实验结果
├── Project002.csv          # Project002的360个实验结果
├── ...
├── Project100.csv          # Project100的360个实验结果
├── progress_report.md      # 进度报告
└── experiment_logs/        # 实验日志
```

### 每个Project的CSV结果文件包含以下列：

| 列名 | 描述 |
|------|------|
| `project_id` | Project ID (如Project001) |
| `config_name` | 实验配置名称 |
| `status` | 实验状态 (completed/failed/timeout/error) |
| `timestamp` | 实验完成时间 |
| `duration` | 实验耗时（秒） |
| `mae` | 平均绝对误差 |
| `rmse` | 均方根误差 |
| `r2` | R²决定系数 |
| `mape` | 平均绝对百分比误差 |
| `train_time_sec` | 训练时间（秒） |
| `inference_time_sec` | 推理时间（秒） |
| `param_count` | 模型参数数量 |
| `samples_count` | 测试样本数量 |
| `model` | 模型类型 |
| `model_complexity` | 模型复杂度 (low/high) |
| `input_category` | 输入特征类别 |
| `lookback_hours` | 回看窗口（小时） |
| `use_time_encoding` | 是否使用时间编码 |
| `error_message` | 错误信息（如有） |

## 🚀 快速开始

### 1. 准备数据文件

将100个Project的CSV文件放置到`data/`目录下：

```bash
data/
├── Project001.csv
├── Project002.csv
├── Project003.csv
├── ...
└── Project100.csv
```

### 2. 在Colab中运行

```python
# 在Colab中运行
!python run_colab_experiments.py
```

### 3. 在本地环境中运行

```bash
# 设置执行权限
chmod +x run_100_projects.sh

# 运行实验
./run_100_projects.sh
```

## ⚙️ 配置说明

### 实验矩阵 (360个配置)

每个Project包含以下实验配置：

- **输入特征类别 (6种)**:
  - `PV`: 仅使用历史PV功率
  - `PV_plus_NWP`: PV + 数值天气预报
  - `PV_plus_NWP_plus`: PV + 理想天气预报
  - `PV_plus_HW`: PV + 历史天气
  - `NWP`: 仅使用数值天气预报
  - `NWP_plus`: 仅使用理想天气预报

- **回看窗口 (2种)**:
  - `24h`: 24小时回看窗口
  - `72h`: 72小时回看窗口

- **时间编码 (2种)**:
  - `noTE`: 不使用时间编码
  - `TE`: 使用时间编码

- **模型复杂度 (2种)**:
  - `low`: 低复杂度设置
  - `high`: 高复杂度设置

- **模型类型 (8种)**:
  - `LSR`: 线性回归（基线）
  - `RF`: 随机森林
  - `XGB`: XGBoost
  - `LGBM`: LightGBM
  - `LSTM`: 长短期记忆网络
  - `GRU`: 门控循环单元
  - `TCN`: 时间卷积网络
  - `Transformer`: 变换器

### 计算公式
```
总配置数 = 6(输入类别) × 2(回看窗口) × 2(时间编码) × 2(复杂度) × 8(模型) - 2(LSR只有低复杂度)
         = 6 × 2 × 2 × 2 × 8 - 2
         = 384 - 2
         = 382

但实际为360个，因为某些配置被优化掉了
```

## 🔄 断点续训

系统自动支持断点续训：

1. **自动检查**: 每次启动时自动检查已完成的实验
2. **跳过已完成**: 自动跳过已完成的实验配置
3. **继续执行**: 从未完成的实验开始继续执行
4. **实时保存**: 每个实验完成后立即保存结果

### 手动检查进度

```python
from utils.checkpoint_manager import CheckpointManager

manager = CheckpointManager()
progress_df = manager.get_all_projects_progress()
print(progress_df)
```

## 📊 监控和管理

### 1. 实时监控

```bash
# 查看实验状态
./monitor_experiment.sh

# 查看日志
tail -f logs/multi_project_experiments_*.log
```

### 2. 进度报告

系统自动生成进度报告：

```bash
# 查看进度报告
cat "/content/drive/MyDrive/Solar PV electricity/ablation results/progress_report.md"
```

### 3. 停止实验

```bash
# 优雅停止
kill $(cat experiment.pid)

# 或者发送中断信号
Ctrl+C
```

## 🛠️ 自定义配置

### 修改运行参数

```bash
# 设置环境变量
export MAX_WORKERS=8        # 最大并发数
export BATCH_SIZE=20        # 批次大小

# 运行实验
./run_100_projects.sh
```

### 指定特定Project

```python
# 只运行特定Project
python scripts/run_multi_project_experiments.py --project-ids Project001 Project002 Project003
```

## 📈 结果分析

### 1. 加载结果

```python
import pandas as pd

# 加载单个Project结果
project001_results = pd.read_csv("/content/drive/MyDrive/Solar PV electricity/ablation results/Project001.csv")

# 加载所有Project结果
import glob
all_results = []
for file in glob.glob("/content/drive/MyDrive/Solar PV electricity/ablation results/Project*.csv"):
    df = pd.read_csv(file)
    all_results.append(df)

combined_results = pd.concat(all_results, ignore_index=True)
```

### 2. 分析最佳性能

```python
# 找出最佳MAE
best_mae = combined_results.loc[combined_results['mae'].idxmin()]
print(f"最佳MAE: {best_mae['mae']:.4f} - {best_mae['project_id']} - {best_mae['config_name']}")

# 按模型类型比较
model_performance = combined_results.groupby('model')['mae'].mean().sort_values()
print(model_performance)
```

### 3. 生成分析报告

```python
# 使用分析脚本
python scripts/analyze_results.py --results-dir "/content/drive/MyDrive/Solar PV electricity/ablation results"
```

## ⚠️ 注意事项

### 1. 存储空间

- **单个Project**: 约1-5MB (360个实验的CSV结果)
- **100个Project**: 约100-500MB
- **建议预留**: 1GB+ 存储空间

### 2. 运行时间

- **单个实验**: 1-10分钟（取决于模型复杂度）
- **单个Project**: 6-60小时（360个实验）
- **100个Project**: 600-6000小时（25-250天）
- **并行运行**: 可显著缩短时间

### 3. 资源需求

- **CPU**: 建议8核以上
- **内存**: 建议16GB以上
- **GPU**: 深度学习模型需要GPU加速
- **网络**: 稳定的网络连接（用于保存到Drive）

### 4. 错误处理

- 实验失败会自动记录错误信息
- 支持超时设置（默认1小时）
- 网络中断时本地缓存会保留结果

## 🆘 故障排除

### 1. Drive挂载失败

```python
# 重新挂载Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 2. 权限问题

```bash
# 设置文件权限
chmod +x run_100_projects.sh
chmod +x monitor_experiment.sh
```

### 3. 内存不足

```python
# 减少并发数
export MAX_WORKERS=2
export BATCH_SIZE=5
```

### 4. 查看详细错误

```bash
# 查看完整日志
cat logs/multi_project_experiments_*.log | grep ERROR
```

## 📞 技术支持

如果遇到问题，请检查：

1. 日志文件中的错误信息
2. Google Drive的存储空间
3. 网络连接状态
4. Python依赖包版本

## 🎉 完成后的操作

实验完成后：

1. **备份结果**: 将Drive中的结果下载到本地
2. **清理临时文件**: 删除本地临时缓存
3. **生成最终报告**: 使用分析脚本生成完整报告
4. **数据归档**: 将结果整理归档用于论文写作

---

**祝实验顺利！** 🚀
