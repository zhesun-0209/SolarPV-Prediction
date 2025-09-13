#!/usr/bin/env python3
"""
测试配置生成脚本
"""

import sys
import os
sys.path.append('scripts')

from generate_dynamic_project_configs import create_model_config

def test_config_generation():
    """测试配置生成"""
    print("🧪 测试配置生成脚本")
    print("=" * 50)
    
    # 测试DL模型配置
    models = ['GRU', 'LSTM', 'Transformer', 'TCN']
    complexities = ['low', 'high']
    
    for model in models:
        for complexity in complexities:
            print(f"\n📋 测试 {model} {complexity}:")
            config = create_model_config(model, complexity)
            
            print(f"  model: {config.get('model')}")
            print(f"  model_complexity: {config.get('model_complexity')}")
            print(f"  train_params: {config.get('train_params')}")
            print(f"  model_params keys: {list(config.get('model_params', {}).keys())}")
            
            # 检查train_params是否正确
            train_params = config.get('train_params', {})
            if 'batch_size' in train_params and 'learning_rate' in train_params:
                print(f"  ✅ train_params正确: batch_size={train_params['batch_size']}, learning_rate={train_params['learning_rate']}")
            else:
                print(f"  ❌ train_params错误: {train_params}")
    
    # 测试ML模型配置
    print(f"\n📋 测试ML模型:")
    ml_models = ['RF', 'XGB', 'LGBM', 'LSR']
    for model in ml_models:
        for complexity in complexities:
            print(f"\n📋 测试 {model} {complexity}:")
            config = create_model_config(model, complexity)
            
            print(f"  model: {config.get('model')}")
            print(f"  model_complexity: {config.get('model_complexity')}")
            print(f"  train_params: {config.get('train_params')}")
            print(f"  model_params keys: {list(config.get('model_params', {}).keys())}")

if __name__ == "__main__":
    test_config_generation()
