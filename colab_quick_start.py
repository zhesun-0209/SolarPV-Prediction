#!/usr/bin/env python3
"""
Colab快速启动脚本
在Colab中运行此脚本来快速设置和测试项目
"""

# ===== 环境检查和设置 =====
import torch
import os
import subprocess
import sys

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # 检查Python版本
    print(f"Python version: {sys.version}")
    
    # 检查PyTorch版本
    print(f"PyTorch version: {torch.__version__}")
    
    # 检查内存
    if torch.cuda.is_available():
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def setup_a100_optimization():
    """设置A100优化"""
    print("\n⚡ 设置A100优化...")
    
    # 启用TensorFloat-32 (TF32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ 启用TF32")
    
    # 启用cuDNN基准测试
    torch.backends.cudnn.benchmark = True
    print("✅ 启用cuDNN基准测试")
    
    # 设置内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.9)
    print("✅ 设置内存分配策略")

def clone_and_setup():
    """克隆项目并设置"""
    print("\n📥 克隆项目...")
    
    # 检查是否已存在
    if os.path.exists('SolarPV-Prediction'):
        print("✅ 项目已存在，跳过克隆")
        os.chdir('SolarPV-Prediction')
    else:
        # 克隆项目
        result = subprocess.run(['git', 'clone', 'https://github.com/zhesun-0209/SolarPV-Prediction.git'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 项目克隆成功")
            os.chdir('SolarPV-Prediction')
        else:
            print(f"❌ 项目克隆失败: {result.stderr}")
            return False
    
    # 安装依赖
    print("\n📦 安装依赖...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ 依赖安装成功")
    else:
        print(f"❌ 依赖安装失败: {result.stderr}")
        return False
    
    return True

def run_quick_test():
    """运行快速测试"""
    print("\n🧪 运行快速测试...")
    
    # 运行TCN模型快速测试
    cmd = [
        sys.executable, 'main.py',
        '--config', 'config/default.yaml',
        '--model', 'TCN',
        '--use_hist_weather', 'true',
        '--use_forecast', 'false',
        '--model_complexity', 'low',
        '--past_days', '1'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 快速测试成功")
        print("输出:")
        print(result.stdout)
    else:
        print(f"❌ 快速测试失败: {result.stderr}")
        return False
    
    return True

def run_model_comparison():
    """运行模型对比"""
    print("\n🔄 运行模型对比...")
    
    models = ['TCN', 'LSTM', 'Transformer']
    
    for model in models:
        print(f"\n--- 运行 {model} 模型 ---")
        cmd = [
            sys.executable, 'main.py',
            '--config', 'config/default.yaml',
            '--model', model,
            '--use_hist_weather', 'true',
            '--use_forecast', 'true',
            '--model_complexity', 'medium',
            '--past_days', '3'
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {model} 模型运行成功")
        else:
            print(f"❌ {model} 模型运行失败: {result.stderr}")

def analyze_results():
    """分析结果"""
    print("\n📊 分析结果...")
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 查找结果文件
    results_dir = 'outputs'
    if os.path.exists(results_dir):
        print("✅ 找到结果目录")
        
        # 列出所有结果
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file == 'summary.csv':
                    summary_path = os.path.join(root, file)
                    print(f"📄 找到结果文件: {summary_path}")
                    
                    # 读取并显示结果
                    try:
                        df = pd.read_csv(summary_path)
                        print("\n=== 模型性能对比 ===")
                        print(df[['model', 'test_loss', 'rmse', 'mae', 'train_time_sec']].round(4))
                    except Exception as e:
                        print(f"❌ 读取结果失败: {e}")
    else:
        print("❌ 未找到结果目录")

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction Colab 快速启动")
    print("=" * 50)
    
    # 1. 检查环境
    check_environment()
    
    # 2. 设置A100优化
    setup_a100_optimization()
    
    # 3. 克隆和设置项目
    if not clone_and_setup():
        print("❌ 项目设置失败，退出")
        return
    
    # 4. 运行快速测试
    if not run_quick_test():
        print("❌ 快速测试失败，退出")
        return
    
    # 5. 运行模型对比
    run_model_comparison()
    
    # 6. 分析结果
    analyze_results()
    
    print("\n🎉 所有任务完成！")
    print("📁 结果保存在 outputs/ 目录中")
    print("📊 可以查看 summary.csv 文件了解模型性能")

if __name__ == "__main__":
    main()
