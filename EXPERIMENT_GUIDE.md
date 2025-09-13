# 100个Project消融实验运行指南

## 🚀 快速开始

### 方法1: 一键启动 (推荐)
```bash
# 设置执行权限
chmod +x start_experiments.sh

# 一键启动实验
./start_experiments.sh
```

### 方法2: Python脚本启动
```bash
# 运行快速启动脚本
python quick_start_experiments.py
```

### 方法3: Colab环境
```python
# 在Colab中运行
exec(open('colab_run_experiments.py').read())
```

## 📋 实验准备

### 1. 数据准备
将100个Project的CSV文件放入 `data/` 目录：
```
data/
├── Project001.csv
├── Project002.csv
├── Project003.csv
├── ...
└── Project100.csv
```

### 2. 环境要求
- **Python**: 3.8+
- **GPU**: 推荐A100或RTX 4090/3090
- **内存**: 建议32GB+
- **存储**: 建议100GB+可用空间

### 3. 依赖安装
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xgboost[gpu] lightgbm[gpu]
pip install pandas numpy scikit-learn pyyaml
```

## 🎯 运行策略选择

### 策略1: GPU专用版 (推荐)
- **特点**: 所有模型都使用GPU版本
- **加速比**: 10倍
- **预计时间**: 25天
- **适用**: A100 GPU

```bash
./run_gpu_only.sh
```

### 策略2: 高性能版
- **特点**: GPU+CPU混合并行
- **加速比**: 5倍
- **预计时间**: 50天
- **适用**: 高端GPU

```bash
./run_a100_optimized.sh
```

### 策略3: 标准版
- **特点**: 传统CPU+少量GPU
- **加速比**: 1倍
- **预计时间**: 250天
- **适用**: 任何环境

```bash
./run_100_projects.sh
```

## 📊 结果保存

### 保存位置
```
/content/drive/MyDrive/Solar PV electricity/ablation results/
├── Project001.csv    # Project001的360个实验结果
├── Project002.csv    # Project002的360个实验结果
├── ...
├── Project100.csv    # Project100的360个实验结果
└── progress_report.md # 进度报告
```

### 结果文件格式
每个CSV文件包含以下列：
- `project_id`: Project ID
- `config_name`: 实验配置名称
- `status`: 实验状态 (completed/failed/timeout/error)
- `timestamp`: 实验完成时间
- `duration`: 实验耗时（秒）
- `mae`, `rmse`, `r2`, `mape`: 性能指标
- `model`: 模型类型
- `model_complexity`: 模型复杂度
- `input_category`: 输入特征类别
- `lookback_hours`: 回看窗口
- `use_time_encoding`: 是否使用时间编码

## 📈 进度监控

### 实时监控
```bash
# 查看实验状态
./monitor_experiments.sh

# 查看日志
tail -f logs/experiments_*.log

# 查看GPU使用情况
nvidia-smi
```

### 进度查询
```python
from utils.checkpoint_manager import CheckpointManager

manager = CheckpointManager()
progress_df = manager.get_all_projects_progress()
print(progress_df)
```

## 🔧 自定义配置

### 修改并行数
```bash
# GPU专用版
python scripts/run_gpu_only_experiments.py \
    --max-gpu-experiments 40 \
    --batch-size 50

# 高性能版
python scripts/run_high_performance_experiments.py \
    --max-gpu-experiments 20 \
    --max-cpu-experiments 40 \
    --batch-size 30
```

### 指定特定Project
```bash
python scripts/run_gpu_only_experiments.py \
    --project-ids Project001 Project002 Project003
```

### 修改结果保存路径
```bash
python scripts/run_gpu_only_experiments.py \
    --drive-path "/your/custom/path"
```

## 🚨 故障排除

### 常见问题

#### 1. GPU内存不足
```bash
# 减少GPU并行数
export MAX_GPU_EXPERIMENTS=16

# 减少批处理大小
# 在配置中设置较小的batch_size
```

#### 2. Google Drive未挂载
```python
# 在Colab中运行
from google.colab import drive
drive.mount('/content/drive')
```

#### 3. 依赖包缺失
```bash
pip install -r requirements.txt
```

#### 4. 实验中断
```bash
# 重新运行，会自动断点续训
./start_experiments.sh
```

### 查看详细错误
```bash
# 查看完整日志
cat logs/experiments_*.log | grep ERROR

# 查看最新日志
tail -n 100 logs/experiments_*.log
```

## 📋 实验管理

### 停止实验
```bash
# 优雅停止
kill $(cat experiment.pid)

# 强制停止
pkill -f "run_gpu_only_experiments"
```

### 清理临时文件
```bash
# 清理临时结果
rm -rf temp_results/*
rm -rf temp_drive_cache/*

# 清理日志
rm -rf logs/*.log
```

### 备份结果
```bash
# 备份到本地
cp -r "/content/drive/MyDrive/Solar PV electricity/ablation results" ./backup_results

# 压缩备份
zip -r ablation_results_backup.zip backup_results/
```

## 🎉 实验完成

### 结果分析
```python
import pandas as pd
import glob

# 加载所有结果
all_results = []
for file in glob.glob("/content/drive/MyDrive/Solar PV electricity/ablation results/Project*.csv"):
    df = pd.read_csv(file)
    all_results.append(df)

combined_results = pd.concat(all_results, ignore_index=True)

# 分析最佳性能
best_mae = combined_results.loc[combined_results['mae'].idxmin()]
print(f"最佳MAE: {best_mae['mae']:.4f} - {best_mae['project_id']} - {best_mae['config_name']}")
```

### 生成报告
```bash
# 生成性能报告
python scripts/gpu_performance_estimator.py

# 生成分析报告
python scripts/analyze_results.py
```

## 📞 技术支持

如果遇到问题：
1. 查看日志文件中的错误信息
2. 检查系统资源使用情况
3. 验证数据文件完整性
4. 确认GPU和依赖包版本

---

**开始你的大规模消融实验之旅吧！** 🚀
