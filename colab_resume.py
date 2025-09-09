#!/usr/bin/env python3
"""
Colab断点续传脚本
检查现有结果，从未完成的地方继续运行
"""

import os
import subprocess
import sys
import time
import glob
import pandas as pd
from datetime import datetime

def check_environment():
    """检查环境"""
    print("🔍 检查环境...")
    
    # 检查CUDA
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not available")
    
    # 检查数据文件
    if os.path.exists('data/Project1033.csv'):
        print("✅ 数据文件存在")
    else:
        print("❌ 数据文件不存在")
        return False
    
    return True

def setup_gpu_environment():
    """设置GPU环境"""
    print("🔧 设置GPU环境...")
    
    # 安装cuML
    print("📦 检查cuML...")
    try:
        import cuml
        print(f"✅ cuML已安装，版本: {cuml.__version__}")
    except ImportError:
        print("📥 安装cuML...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'cuml-cu11', '--extra-index-url=https://pypi.nvidia.com'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ cuML安装成功")
        else:
            print("❌ cuML安装失败，将使用CPU版本")
    
    # 设置A100优化
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.9)
        print("✅ A100优化设置完成")
    except ImportError:
        print("⚠️ PyTorch未安装，跳过A100优化")

def check_existing_results():
    """检查现有结果"""
    print("🔍 检查现有结果...")
    
    # 查找所有summary.csv文件
    summary_files = glob.glob('result/**/summary.csv', recursive=True)
    
    if not summary_files:
        print("📝 未找到现有结果，将从头开始")
        return set()
    
    # 读取现有结果
    existing_experiments = set()
    for file in summary_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                # 创建实验标识
                exp_id = f"{df.iloc[0]['model']}_{df.iloc[0]['use_hist_weather']}_{df.iloc[0]['use_forecast']}_{df.iloc[0].get('model_complexity', 'medium')}"
                existing_experiments.add(exp_id)
        except Exception as e:
            print(f"⚠️ 读取结果文件失败 {file}: {e}")
    
    print(f"📊 找到 {len(existing_experiments)} 个已完成实验")
    return existing_experiments

def get_experiment_id(model, hist_weather, forecast, complexity):
    """生成实验标识"""
    return f"{model}_{hist_weather}_{forecast}_{complexity}"

def run_experiment(model, hist_weather, forecast, complexity, description):
    """运行单个实验"""
    print(f"\n🚀 {description}")
    print("-" * 60)
    
    cmd = [
        sys.executable, 'main.py',
        '--config', 'config/default.yaml',
        '--model', model,
        '--use_hist_weather', hist_weather,
        '--use_forecast', forecast,
        '--model_complexity', complexity,
        '--past_days', '3'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} 完成 (耗时: {end_time - start_time:.2f}秒)")
        return True
    else:
        print(f"❌ {description} 失败")
        print("错误:", result.stderr)
        return False

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction 断点续传实验")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 检查环境
    if not check_environment():
        return
    
    # 设置GPU环境
    setup_gpu_environment()
    
    # 创建结果目录
    if not os.path.exists('result'):
        os.makedirs('result')
        print("✅ 创建result目录")
    
    # 检查现有结果
    existing_experiments = check_existing_results()
    
    # 模型列表
    models = ['Transformer', 'LSTM', 'GRU', 'TCN', 'RF', 'GBR', 'XGB', 'LGBM']
    
    # 特征组合
    feature_configs = [
        ('hist_only', 'true', 'false'),
        ('fcst_only', 'false', 'true'),
        ('both', 'true', 'true'),
        ('none', 'false', 'false')
    ]
    
    # 复杂度组合
    complexities = ['low', 'medium', 'high']
    
    # 统计信息
    total_experiments = len(models) * len(feature_configs) * len(complexities)
    completed = 0
    failed = 0
    skipped = 0
    
    print(f"📊 总实验数: {total_experiments}")
    print(f"   已完成: {len(existing_experiments)}")
    print(f"   待完成: {total_experiments - len(existing_experiments)}")
    
    start_time = time.time()
    
    for model in models:
        print(f"\n🎯 开始 {model} 模型实验")
        print("-" * 60)
        
        for feat_desc, hist_weather, forecast in feature_configs:
            print(f"\n📋 特征组合: {feat_desc}")
            
            for complexity in complexities:
                exp_id = get_experiment_id(model, hist_weather, forecast, complexity)
                description = f"{model} - {feat_desc} - {complexity}"
                
                if exp_id in existing_experiments:
                    print(f"⏭️ {description} - 跳过 (已完成)")
                    skipped += 1
                    continue
                
                if run_experiment(model, hist_weather, forecast, complexity, description):
                    completed += 1
                else:
                    failed += 1
                
                # 显示进度
                total_done = completed + failed + skipped
                progress = total_done / total_experiments * 100
                elapsed = time.time() - start_time
                eta = elapsed / total_done * (total_experiments - total_done) if total_done > 0 else 0
                
                print(f"📈 进度: {total_done}/{total_experiments} ({progress:.1f}%)")
                print(f"⏱️  已用时间: {elapsed/60:.1f}分钟, 预计剩余: {eta/60:.1f}分钟")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 断点续传实验完成！")
    print(f"📊 成功: {completed}")
    print(f"❌ 失败: {failed}")
    print(f"⏭️ 跳过: {skipped}")
    print(f"⏱️  总耗时: {total_time/60:.1f}分钟")

if __name__ == "__main__":
    main()
