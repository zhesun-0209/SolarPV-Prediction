#!/usr/bin/env python3
"""
快速运行脚本
运行基础模型对比实验
"""

import os
import subprocess
import sys

def run_experiment(model, description):
    """运行单个实验"""
    print(f"\n🚀 运行 {description}")
    print("-" * 50)
    
    cmd = [
        sys.executable, 'main.py',
        '--config', 'config/default.yaml',
        '--model', model,
        '--use_hist_weather', 'true',
        '--use_forecast', 'true',
        '--model_complexity', 'medium',
        '--past_days', '3'
    ]
    
    print(f"命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"✅ {description} 完成")
    else:
        print(f"❌ {description} 失败")
    
    return result.returncode == 0

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction 快速运行")
    print("=" * 50)
    
    # 检查数据文件
    if not os.path.exists('data/Project1033.csv'):
        print("❌ 数据文件不存在: data/Project1033.csv")
        return
    
    # 创建结果目录
    if not os.path.exists('result'):
        os.makedirs('result')
        print("✅ 创建result目录")
    
    print("✅ 数据文件检查通过")
    
    # 运行实验
    models = [
        ('TCN', 'TCN模型'),
        ('LSTM', 'LSTM模型'),
        ('Transformer', 'Transformer模型'),
        ('XGB', 'XGBoost模型')
    ]
    
    success_count = 0
    for model, description in models:
        if run_experiment(model, description):
            success_count += 1
    
    print(f"\n🎉 实验完成！")
    print(f"📊 成功: {success_count}/{len(models)} 个模型")
    print(f"📁 结果保存在: result/ 目录")

if __name__ == "__main__":
    main()
