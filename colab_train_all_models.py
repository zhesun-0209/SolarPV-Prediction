#!/usr/bin/env python3
"""
Colab训练脚本 - 在Project1140上训练所有模型
支持GPU训练，实时输出训练过程
"""

import os
import sys
import time
import subprocess
import yaml
import glob
from datetime import datetime

def setup_colab_environment():
    """设置Colab环境"""
    print("🚀 设置Colab环境...")
    
    # 安装必要的包
    packages = [
        "torch",
        "torchvision", 
        "xgboost",
        "lightgbm",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "tqdm",
        "openpyxl",
        "xlsxwriter"
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"📦 安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # 检查GPU
    import torch
    if torch.cuda.is_available():
        print(f"🎮 GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"🎮 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ 未检测到GPU，将使用CPU训练")
    
    print("✅ 环境设置完成\n")

def get_all_configs():
    """获取所有配置文件"""
    config_dir = "config/projects/1140"
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    
    # 过滤掉config_index.yaml
    config_files = [f for f in yaml_files if not f.endswith("config_index.yaml")]
    
    print(f"📁 找到 {len(config_files)} 个配置文件")
    return sorted(config_files)

def run_single_experiment(config_path, gpu_id=0):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"🔥 开始训练: {os.path.basename(config_path)}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # 设置GPU环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # 运行训练
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, "main.py", "--config", config_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=3600  # 1小时超时
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 训练成功完成!")
            print(f"⏱️ 训练时间: {duration:.2f} 秒")
            
            # 输出关键信息
            if "mse=" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "mse=" in line and "rmse=" in line and "mae=" in line:
                        print(f"📊 结果: {line.strip()}")
                        break
        else:
            print(f"❌ 训练失败!")
            print(f"返回码: {result.returncode}")
            print(f"错误输出:\n{result.stderr}")
            
        return result.returncode == 0, duration
        
    except subprocess.TimeoutExpired:
        print(f"⏰ 训练超时 (1小时)")
        return False, 3600
    except Exception as e:
        print(f"💥 训练异常: {str(e)}")
        return False, 0

def main():
    """主函数"""
    print("🌟 SolarPV项目 - Colab全模型训练")
    print("=" * 80)
    
    # 设置环境
    setup_colab_environment()
    
    # 获取所有配置文件
    config_files = get_all_configs()
    
    # 统计信息
    total_experiments = len(config_files)
    successful_experiments = 0
    failed_experiments = 0
    total_time = 0
    
    print(f"\n🎯 开始训练 {total_experiments} 个模型...")
    print(f"⏰ 预计总时间: {total_experiments * 10 / 60:.1f} 分钟 (假设每个模型10分钟)")
    
    # 按模型类型分组
    model_groups = {}
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        model = config.get('model', 'Unknown')
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(config_file)
    
    print(f"\n📊 模型分组统计:")
    for model, files in model_groups.items():
        print(f"  {model}: {len(files)} 个配置")
    
    # 开始训练
    start_time = time.time()
    
    for i, config_file in enumerate(config_files, 1):
        print(f"\n🔄 进度: {i}/{total_experiments}")
        
        # 运行实验
        success, duration = run_single_experiment(config_file)
        total_time += duration
        
        if success:
            successful_experiments += 1
        else:
            failed_experiments += 1
        
        # 显示进度
        progress = i / total_experiments * 100
        elapsed = time.time() - start_time
        eta = elapsed / i * (total_experiments - i) if i > 0 else 0
        
        print(f"📈 进度: {progress:.1f}% | 成功: {successful_experiments} | 失败: {failed_experiments}")
        print(f"⏱️ 已用时间: {elapsed/60:.1f}分钟 | 预计剩余: {eta/60:.1f}分钟")
        
        # 每10个实验后显示详细统计
        if i % 10 == 0:
            print(f"\n📊 中间统计 (已完成 {i}/{total_experiments}):")
            print(f"  成功率: {successful_experiments/i*100:.1f}%")
            print(f"  平均时间: {total_time/i:.1f}秒/模型")
            print(f"  预计总时间: {elapsed/i*total_experiments/60:.1f}分钟")
    
    # 最终统计
    total_elapsed = time.time() - start_time
    print(f"\n🎉 训练完成!")
    print("=" * 80)
    print(f"📊 最终统计:")
    print(f"  总实验数: {total_experiments}")
    print(f"  成功: {successful_experiments} ({successful_experiments/total_experiments*100:.1f}%)")
    print(f"  失败: {failed_experiments} ({failed_experiments/total_experiments*100:.1f}%)")
    print(f"  总用时: {total_elapsed/60:.1f} 分钟")
    print(f"  平均用时: {total_elapsed/total_experiments:.1f} 秒/模型")
    
    # 按模型类型统计
    print(f"\n📈 按模型类型统计:")
    for model, files in model_groups.items():
        model_success = 0
        for config_file in files:
            if config_file in [f for f in config_files[:successful_experiments]]:
                model_success += 1
        print(f"  {model}: {model_success}/{len(files)} 成功")
    
    print(f"\n✅ 所有训练完成! 结果保存在 temp_results/ 目录中")

if __name__ == "__main__":
    main()
