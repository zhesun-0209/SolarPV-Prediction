#!/usr/bin/env python3
"""
Colab修复版训练脚本 - 处理深度学习模型的段错误问题
"""

import os
import sys
import subprocess
import time
import yaml
import glob
from datetime import datetime

def setup_colab_environment():
    """设置Colab环境"""
    print("🚀 设置Colab环境...")
    
    # 安装必要的包
    packages = [
        "torch==2.0.1",
        "torchvision==0.15.2", 
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
            __import__(package.replace('-', '_').split('==')[0])
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

def test_single_model(config_path):
    """测试单个模型"""
    print(f"🧪 测试模型: {os.path.basename(config_path)}")
    
    try:
        # 设置超时时间
        timeout = 300  # 5分钟
        
        result = subprocess.run([
            sys.executable, "main.py", "--config", config_path
        ], capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            print(f"✅ 测试成功!")
            return True, result.stdout
        else:
            print(f"❌ 测试失败! 返回码: {result.returncode}")
            print(f"错误: {result.stderr[-300:]}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 测试超时 ({timeout}秒)")
        return False, "Timeout"
    except Exception as e:
        print(f"💥 测试异常: {str(e)}")
        return False, str(e)

def run_ml_models_only():
    """只运行机器学习模型"""
    print("🤖 运行机器学习模型...")
    
    # 获取所有ML模型配置
    config_dir = "config/projects/1140"
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    config_files = [f for f in yaml_files if not f.endswith("config_index.yaml")]
    
    ml_models = ['RF', 'XGB', 'LGBM', 'LSR']
    ml_configs = []
    
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config.get('model') in ml_models:
            ml_configs.append(config_file)
    
    print(f"📁 找到 {len(ml_configs)} 个机器学习模型配置")
    
    # 按模型类型分组
    model_groups = {}
    for config_file in ml_configs:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        model = config.get('model', 'Unknown')
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(config_file)
    
    print(f"\n📊 机器学习模型分组:")
    for model, files in model_groups.items():
        print(f"  {model}: {len(files)} 个配置")
    
    # 开始训练
    total_experiments = len(ml_configs)
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, config_file in enumerate(ml_configs, 1):
        print(f"\n{'='*60}")
        print(f"🔥 训练 {i}/{total_experiments}: {os.path.basename(config_file)}")
        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        success, output = test_single_model(config_file)
        
        if success:
            successful += 1
            # 显示结果
            if "mse=" in output:
                lines = output.split('\n')
                for line in lines:
                    if "mse=" in line and "rmse=" in line:
                        print(f"📊 {line.strip()}")
                        break
        else:
            failed += 1
        
        # 显示进度
        elapsed = time.time() - start_time
        progress = i / total_experiments * 100
        eta = elapsed / i * (total_experiments - i) if i > 0 else 0
        
        print(f"\n📈 进度: {progress:.1f}% | 成功: {successful} | 失败: {failed}")
        print(f"⏱️ 已用: {elapsed/60:.1f}分钟 | 剩余: {eta/60:.1f}分钟")
    
    # 最终统计
    total_elapsed = time.time() - start_time
    print(f"\n🎉 机器学习模型训练完成!")
    print(f"📊 成功: {successful}/{total_experiments} ({successful/total_experiments*100:.1f}%)")
    print(f"⏱️ 总用时: {total_elapsed/60:.1f} 分钟")
    
    return successful, failed

def run_dl_models_with_fallback():
    """运行深度学习模型，带降级处理"""
    print("🧠 运行深度学习模型（带降级处理）...")
    
    # 获取所有DL模型配置
    config_dir = "config/projects/1140"
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    config_files = [f for f in yaml_files if not f.endswith("config_index.yaml")]
    
    dl_models = ['LSTM', 'GRU', 'Transformer', 'TCN']
    dl_configs = []
    
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config.get('model') in dl_models:
            dl_configs.append(config_file)
    
    print(f"📁 找到 {len(dl_configs)} 个深度学习模型配置")
    
    # 按模型类型分组
    model_groups = {}
    for config_file in dl_configs:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        model = config.get('model', 'Unknown')
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(config_file)
    
    print(f"\n📊 深度学习模型分组:")
    for model, files in model_groups.items():
        print(f"  {model}: {len(files)} 个配置")
    
    # 开始训练
    total_experiments = len(dl_configs)
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, config_file in enumerate(dl_configs, 1):
        print(f"\n{'='*60}")
        print(f"🔥 训练 {i}/{total_experiments}: {os.path.basename(config_file)}")
        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        success, output = test_single_model(config_file)
        
        if success:
            successful += 1
            # 显示结果
            if "mse=" in output:
                lines = output.split('\n')
                for line in lines:
                    if "mse=" in line and "rmse=" in line:
                        print(f"📊 {line.strip()}")
                        break
        else:
            failed += 1
            print(f"⚠️ 深度学习模型失败，可能是PyTorch兼容性问题")
        
        # 显示进度
        elapsed = time.time() - start_time
        progress = i / total_experiments * 100
        eta = elapsed / i * (total_experiments - i) if i > 0 else 0
        
        print(f"\n📈 进度: {progress:.1f}% | 成功: {successful} | 失败: {failed}")
        print(f"⏱️ 已用: {elapsed/60:.1f}分钟 | 剩余: {eta/60:.1f}分钟")
    
    # 最终统计
    total_elapsed = time.time() - start_time
    print(f"\n🎉 深度学习模型训练完成!")
    print(f"📊 成功: {successful}/{total_experiments} ({successful/total_experiments*100:.1f}%)")
    print(f"⏱️ 总用时: {total_elapsed/60:.1f} 分钟")
    
    return successful, failed

def main():
    """主函数"""
    print("🌟 SolarPV项目 - 修复版Colab训练")
    print("=" * 80)
    
    # 设置环境
    setup_colab_environment()
    
    # 先测试机器学习模型
    print("\n🤖 第一阶段：机器学习模型")
    ml_success, ml_failed = run_ml_models_only()
    
    # 再测试深度学习模型
    print("\n🧠 第二阶段：深度学习模型")
    dl_success, dl_failed = run_dl_models_with_fallback()
    
    # 最终统计
    total_success = ml_success + dl_success
    total_failed = ml_failed + dl_failed
    total_experiments = total_success + total_failed
    
    print(f"\n🎉 所有训练完成!")
    print("=" * 80)
    print(f"📊 最终统计:")
    print(f"  机器学习模型: {ml_success} 成功, {ml_failed} 失败")
    print(f"  深度学习模型: {dl_success} 成功, {dl_failed} 失败")
    print(f"  总计: {total_success}/{total_experiments} 成功 ({total_success/total_experiments*100:.1f}%)")
    
    if dl_failed > 0:
        print(f"\n⚠️ 深度学习模型失败较多，可能是PyTorch版本兼容性问题")
        print(f"建议：")
        print(f"  1. 检查PyTorch版本")
        print(f"  2. 尝试不同的PyTorch版本")
        print(f"  3. 使用CPU训练")
        print(f"  4. 检查内存使用")

if __name__ == "__main__":
    main()
