#!/usr/bin/env python3
"""
Colab专用实验运行代码
一键运行100个Project消融实验并保存到Google Drive
"""

# =============================================================================
# 1. 环境设置和依赖安装
# =============================================================================

print("🔧 设置Colab环境...")

# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 安装依赖
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q xgboost[gpu] lightgbm[gpu]
!pip install -q pandas numpy scikit-learn pyyaml

# 设置环境变量
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['PYTORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['FORCE_GPU'] = '1'
os.environ['USE_GPU'] = '1'

print("✅ 环境设置完成")

# =============================================================================
# 2. 检查GPU和系统信息
# =============================================================================

import torch
import psutil

print("🎯 系统信息:")
print(f"   GPU数量: {torch.cuda.device_count()}")
print(f"   GPU名称: {torch.cuda.get_device_properties(0).name}")
print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
print(f"   CPU核心: {psutil.cpu_count()}")
print(f"   系统内存: {psutil.virtual_memory().total / 1024**3:.1f}GB")

# =============================================================================
# 3. 准备数据文件
# =============================================================================

print("📁 准备数据文件...")

# 创建data目录
!mkdir -p data

# 检查是否有数据文件
import glob
csv_files = glob.glob("data/Project*.csv")
print(f"📊 发现 {len(csv_files)} 个Project数据文件")

if len(csv_files) == 0:
    print("⚠️ 未找到Project数据文件")
    print("请将100个CSV文件上传到data/目录")
    print("文件命名格式: Project001.csv, Project002.csv, ..., Project100.csv")
else:
    print("✅ 数据文件准备完成")

# =============================================================================
# 4. 生成实验配置
# =============================================================================

print("📝 生成实验配置...")

# 运行配置生成脚本
!python scripts/generate_multi_project_configs.py

print("✅ 配置生成完成")

# =============================================================================
# 5. 选择运行策略
# =============================================================================

print("\n选择运行策略:")
print("1. GPU专用版 (推荐) - 所有模型都使用GPU，10倍加速")
print("2. 高性能版 - GPU+CPU混合，5倍加速") 
print("3. 标准版 - 传统方式")

# 默认选择GPU专用版
choice = "1"  # 可以修改这个值来选择不同策略
print(f"🎯 选择策略: {choice}")

# =============================================================================
# 6. 运行实验
# =============================================================================

print("🚀 开始运行实验...")

# 创建结果目录
drive_results_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
!mkdir -p "$drive_results_dir"

# 根据选择运行不同的实验
if choice == "1":
    # GPU专用版
    print("🎯 启动GPU专用版实验...")
    !python scripts/run_gpu_only_experiments.py \
        --drive-path "$drive_results_dir" \
        --max-gpu-experiments 24 \
        --batch-size 30
elif choice == "2":
    # 高性能版
    print("🚀 启动高性能版实验...")
    !python scripts/run_high_performance_experiments.py \
        --drive-path "$drive_results_dir" \
        --max-gpu-experiments 16 \
        --max-cpu-experiments 32 \
        --batch-size 25
else:
    # 标准版
    print("📊 启动标准版实验...")
    !python scripts/run_multi_project_experiments.py \
        --drive-path "$drive_results_dir" \
        --max-workers 8 \
        --batch-size 20

# =============================================================================
# 7. 查看结果
# =============================================================================

print("📊 查看实验结果...")

# 检查结果文件
result_files = glob.glob(f"{drive_results_dir}/Project*.csv")
print(f"📁 结果文件数量: {len(result_files)}")

if result_files:
    print("📋 结果文件列表 (前10个):")
    for i, file in enumerate(sorted(result_files)[:10]):
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"   {i+1}. {os.path.basename(file)} ({file_size:.1f}KB)")
    
    if len(result_files) > 10:
        print(f"   ... 还有 {len(result_files) - 10} 个文件")
    
    # 显示进度统计
    try:
        import pandas as pd
        from utils.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager()
        progress_df = manager.get_all_projects_progress()
        
        if not progress_df.empty:
            completed = len(progress_df[progress_df['is_complete'] == True])
            total = len(progress_df)
            print(f"\n📈 完成进度: {completed}/{total} ({completed/total*100:.1f}%)")
        else:
            print("📊 暂无进度信息")
    except Exception as e:
        print(f"⚠️ 进度查询失败: {e}")
else:
    print("⚠️ 暂无结果文件")

print("\n🎉 实验完成!")
print(f"📊 结果保存在: {drive_results_dir}")
print("每个Project一个CSV文件: Project001.csv, Project002.csv, ...")

# =============================================================================
# 8. 下载结果 (可选)
# =============================================================================

print("\n📥 可选: 下载结果到本地")
print("如果需要下载结果到本地，请运行以下代码:")

download_code = '''
# 创建下载目录
!mkdir -p downloaded_results

# 下载所有结果文件
import shutil
for file in glob.glob(f"{drive_results_dir}/Project*.csv"):
    shutil.copy(file, "downloaded_results/")
    print(f"下载: {os.path.basename(file)}")

# 压缩结果文件
!zip -r ablation_results.zip downloaded_results/

print("✅ 结果已下载并压缩为 ablation_results.zip")
'''

print(download_code)
