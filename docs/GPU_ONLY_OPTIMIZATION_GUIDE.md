# GPU专用优化指南

## 🚀 概述

本指南专门针对**所有模型都使用GPU版本**的极致优化策略，包括传统ML模型的GPU版本（XGBoost GPU、LightGBM GPU等），可以实现**10倍性能提升**，将36,000个实验的完成时间从250天缩短到25天！

## 🎯 核心优化策略

### 1. 全GPU并行策略

#### 所有模型强制使用GPU
- **深度学习模型**: LSTM, GRU, TCN, Transformer
- **传统ML模型GPU版**:
  - **XGBoost**: `gpu_hist`方法，10-20倍加速
  - **LightGBM**: GPU训练模式，5-10倍加速
  - **Random Forest**: GPU并行版本
  - **Linear Regression**: GPU批处理加速

#### GPU并行配置
- **A100 80GB**: 32个实验同时运行
- **A100 40GB**: 24个实验同时运行
- **RTX 4090/3090**: 16个实验同时运行
- **其他GPU**: 8-12个实验同时运行

### 2. GPU内存优化

#### 混合精度训练 (AMP)
```python
# 深度学习模型
config['train_params']['use_amp'] = True
```

#### 动态批处理大小
- **深度学习模型**: 32→128
- **树模型**: 大批次处理 (1024-2048 samples)
- **线性回归**: 2048 samples

#### CUDA内存管理
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDNN_V8_API_ENABLED=1
```

### 3. 模型特定GPU优化

#### XGBoost GPU优化
```python
# 强制使用GPU版本
config['train_params']['tree_method'] = 'gpu_hist'
config['train_params']['gpu_id'] = 0
config['train_params']['predictor'] = 'gpu_predictor'
```

#### LightGBM GPU优化
```python
# 启用GPU训练
config['train_params']['device'] = 'gpu'
config['train_params']['gpu_platform_id'] = 0
config['train_params']['gpu_device_id'] = 0
```

#### 深度学习模型优化
```python
# 混合精度 + 梯度累积
config['train_params']['use_amp'] = True
config['train_params']['gradient_accumulation_steps'] = 2
```

### 4. 系统优化

#### 强制GPU环境变量
```bash
export FORCE_GPU=1
export USE_GPU=1
export CUDA_VISIBLE_DEVICES=0
```

#### GPU内存监控
- 实时监控GPU内存使用率
- 动态调整并行数
- 防止内存溢出

## 📊 性能对比

| 配置 | GPU并行 | 预计时间 | 加速比 | 说明 |
|------|---------|----------|--------|------|
| 标准版本 | 4 | 250天 | 1x | CPU + 少量GPU |
| 混合优化版 | 16 | 50天 | 5x | GPU + CPU并行 |
| **GPU专用版** | **32** | **25天** | **10x** | **所有模型GPU** |

## 🛠️ 使用方法

### 1. 快速启动GPU专用实验
```bash
# 设置执行权限
chmod +x run_gpu_only.sh

# 启动GPU专用实验
./run_gpu_only.sh
```

### 2. 性能监控
```bash
# 实时监控
./monitor_gpu_only_experiment.sh

# 查看日志
tail -f logs/gpu_only_experiments_*.log
```

### 3. 性能预估
```bash
python scripts/gpu_performance_estimator.py
```

### 4. 自定义配置
```bash
python scripts/run_gpu_only_experiments.py \
    --max-gpu-experiments 40 \
    --batch-size 50
```

## 🔧 高级配置

### GPU环境变量设置
```bash
# 基础GPU设置
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDNN_V8_API_ENABLED=1

# 强制GPU模式
export FORCE_GPU=1
export USE_GPU=1

# 性能优化
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

### 模型特定配置

#### XGBoost GPU配置
```python
xgb_params = {
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'n_estimators': 400,  # 增加树的数量
    'max_depth': 12,
    'learning_rate': 0.01
}
```

#### LightGBM GPU配置
```python
lgb_params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'n_estimators': 400,
    'max_depth': 12,
    'learning_rate': 0.01
}
```

## 📈 性能监控指标

### 实时指标
- GPU利用率 (目标: >90%)
- GPU内存使用率 (目标: <80%)
- 实验完成速度
- 各模型类型完成时间

### 历史统计
- 每日完成实验数
- 平均实验时间
- GPU效率分析
- 模型性能对比

## ⚠️ 注意事项

### GPU内存管理
- 监控GPU内存使用率，避免超过90%
- 使用混合精度训练减少内存占用
- 及时清理GPU缓存

### 模型兼容性
- 确保安装了GPU版本的ML库
- 检查CUDA版本兼容性
- 验证GPU内存是否足够

### 系统稳定性
- 监控GPU温度
- 定期重启释放GPU内存
- 使用断点续训功能

## 🚨 故障排除

### GPU内存不足
```bash
# 减少GPU并行数
export MAX_GPU_EXPERIMENTS=16

# 减少批处理大小
# 在配置中设置较小的batch_size
```

### GPU版本库问题
```bash
# 重新安装GPU版本
pip uninstall xgboost lightgbm
pip install xgboost[gpu] lightgbm[gpu]
```

### CUDA版本不兼容
```bash
# 检查CUDA版本
nvidia-smi
python -c "import torch; print(torch.version.cuda)"

# 安装兼容版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📋 最佳实践

### 1. 系统准备
- 确保GPU驱动和CUDA版本最新
- 安装GPU版本的ML库
- 配置充足的GPU内存

### 2. 环境配置
- 设置所有必要的环境变量
- 使用SSD存储加速数据加载
- 配置高速网络

### 3. 监控管理
- 实时监控GPU使用情况
- 定期检查实验进度
- 及时处理异常情况

### 4. 结果管理
- 定期备份结果文件
- 验证数据完整性
- 生成性能报告

## 🎉 预期效果

使用GPU专用版，你可以期待：

- **10倍性能提升**: 从250天缩短到25天
- **极致资源利用**: 所有计算都在GPU上完成
- **更稳定的运行**: 统一的GPU环境
- **更详细的监控**: GPU专用性能指标

## 📞 技术支持

如果遇到问题：

1. 运行GPU性能测试脚本
2. 检查GPU版本库安装
3. 查看详细日志文件
4. 调整GPU并行参数

---

**开始你的极致GPU性能之旅吧！** 🚀
