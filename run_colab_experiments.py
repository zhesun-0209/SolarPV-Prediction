#!/usr/bin/env python3
"""
Colab环境下的100个Project消融实验启动脚本
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import subprocess

def setup_colab_environment():
    """设置Colab环境"""
    print("🔧 设置Colab环境...")
    
    # 挂载Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive已挂载")
    except ImportError:
        print("⚠️ 不在Colab环境中，跳过Drive挂载")
    except Exception as e:
        print(f"❌ Drive挂载失败: {e}")
        return False
    
    # 检查结果目录
    drive_results_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(drive_results_dir, exist_ok=True)
    print(f"📁 结果目录: {drive_results_dir}")
    
    return True

def check_data_files():
    """检查数据文件"""
    print("🔍 检查数据文件...")
    
    data_dir = Path("./data")
    if not data_dir.exists():
        print("❌ data目录不存在")
        return False
    
    csv_files = list(data_dir.glob("Project*.csv"))
    print(f"📊 发现 {len(csv_files)} 个Project数据文件")
    
    if len(csv_files) == 0:
        print("❌ 未找到Project数据文件")
        print("   请确保文件命名为Project001.csv, Project002.csv, ...")
        return False
    
    # 显示前几个文件
    for i, file in enumerate(sorted(csv_files)[:5]):
        print(f"   {file.name}")
    
    if len(csv_files) > 5:
        print(f"   ... 还有 {len(csv_files) - 5} 个文件")
    
    return True

def generate_configs():
    """生成配置"""
    print("📝 生成配置...")
    
    config_dir = Path("./config/projects")
    if config_dir.exists():
        print("✅ 配置已存在，跳过生成")
        return True
    
    try:
        result = subprocess.run([
            sys.executable, "scripts/generate_multi_project_configs.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 配置生成成功")
            return True
        else:
            print(f"❌ 配置生成失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 配置生成异常: {e}")
        return False

def check_progress():
    """检查当前进度"""
    print("📊 检查当前进度...")
    
    try:
        # 导入检查点管理器
        sys.path.append(str(Path.cwd()))
        from utils.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager()
        progress_df = manager.get_all_projects_progress()
        
        if not progress_df.empty:
            completed = len(progress_df[progress_df['is_complete'] == True])
            total = len(progress_df)
            print(f"   当前进度: {completed}/{total} 个Project已完成 ({completed/total*100:.1f}%)")
            
            if completed > 0:
                print("🔄 将进行断点续训")
                
                # 显示最近完成的Project
                recent_completed = progress_df[progress_df['is_complete'] == True].tail(3)
                if len(recent_completed) > 0:
                    print("   最近完成:")
                    for _, row in recent_completed.iterrows():
                        print(f"     - {row['project_id']}")
            else:
                print("🆕 首次运行，将开始全新实验")
        else:
            print("🆕 首次运行，将开始全新实验")
        
        return True
    except Exception as e:
        print(f"⚠️ 进度检查失败: {e}")
        return True  # 继续运行

def run_experiments():
    """运行实验"""
    print("🚀 开始运行实验...")
    
    # 设置运行参数
    max_workers = 4  # Colab环境建议使用4个worker
    batch_size = 10
    
    print(f"⚙️ 运行参数:")
    print(f"   最大并发数: {max_workers}")
    print(f"   批次大小: {batch_size}")
    
    try:
        # 运行实验
        result = subprocess.run([
            sys.executable, "scripts/run_multi_project_experiments.py",
            "--drive-path", "/content/drive/MyDrive/Solar PV electricity/ablation results",
            "--max-workers", str(max_workers),
            "--batch-size", str(batch_size)
        ])
        
        if result.returncode == 0:
            print("🎉 所有实验完成!")
        else:
            print(f"⚠️ 实验结束，返回码: {result.returncode}")
        
        return True
    except KeyboardInterrupt:
        print("🛑 用户中断实验")
        return False
    except Exception as e:
        print(f"💥 实验异常: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Project1140 100个Project消融实验 - Colab版本")
    print("=" * 60)
    
    # 设置环境
    if not setup_colab_environment():
        print("❌ 环境设置失败")
        return
    
    # 检查数据文件
    if not check_data_files():
        print("❌ 数据文件检查失败")
        return
    
    # 生成配置
    if not generate_configs():
        print("❌ 配置生成失败")
        return
    
    # 检查进度
    if not check_progress():
        print("❌ 进度检查失败")
        return
    
    print("\n" + "=" * 60)
    print("🎯 准备开始实验!")
    print("📊 结果将实时保存到Google Drive")
    print("🔄 支持断点续训")
    print("⏰ 预计耗时: 数小时到数天")
    print("=" * 60)
    
    # 用户确认
    try:
        response = input("\n是否开始实验? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("🛑 用户取消实验")
            return
    except KeyboardInterrupt:
        print("\n🛑 用户取消实验")
        return
    
    # 运行实验
    start_time = time.time()
    success = run_experiments()
    end_time = time.time()
    
    # 显示最终统计
    duration = end_time - start_time
    print(f"\n⏱️ 总运行时间: {duration/3600:.1f} 小时")
    
    if success:
        print("🎉 实验完成!")
    else:
        print("⚠️ 实验未完全完成")
    
    print("\n📊 查看结果:")
    print("   Google Drive: /content/drive/MyDrive/Solar PV electricity/ablation results")
    print("   每个Project一个CSV文件: Project001.csv, Project002.csv, ...")

if __name__ == "__main__":
    main()
