#!/usr/bin/env python3
"""
测试GPU检测逻辑
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ml_models import LGB_GPU_AVAILABLE, XGB_GPU_AVAILABLE, GPU_AVAILABLE
import torch

print("=== GPU检测测试 ===")
print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")
print(f"cuML RandomForest GPU可用: {GPU_AVAILABLE}")
print(f"XGBoost GPU可用: {XGB_GPU_AVAILABLE}")
print(f"LightGBM GPU可用: {LGB_GPU_AVAILABLE}")

if torch.cuda.is_available():
    print(f"GPU设备数量: {torch.cuda.device_count()}")
    print(f"当前GPU设备: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name()}")

print("\n=== 测试ML模型训练 ===")
import numpy as np
from models.ml_models import train_rf, train_xgb, train_lgbm

# 创建测试数据
X = np.random.rand(100, 10)
y = np.random.rand(100, 24)

# 测试参数
params = {
    'n_estimators': 10,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42
}

print("\n--- 测试RandomForest ---")
try:
    model = train_rf(X, y, params)
    print("✅ RandomForest GPU训练成功")
except Exception as e:
    print(f"❌ RandomForest训练失败: {e}")

print("\n--- 测试XGBoost ---")
try:
    model = train_xgb(X, y, params)
    print("✅ XGBoost GPU训练成功")
except Exception as e:
    print(f"❌ XGBoost训练失败: {e}")

print("\n--- 测试LightGBM ---")
try:
    model = train_lgbm(X, y, params)
    print("✅ LightGBM GPU训练成功")
except Exception as e:
    print(f"❌ LightGBM训练失败: {e}")
