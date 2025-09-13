#!/usr/bin/env python3
"""
Colab修复版批量实验脚本
确保配置生成和实验运行都正确
"""

import os
import sys
import yaml
import time
import subprocess
import pandas as pd
import glob
from pathlib import Path

def clear_old_configs():
    """清除旧的配置文件，确保重新生成"""
    print("🧹 清除旧的配置文件...")
    
    # 删除所有非1140的配置文件
    config_dirs = glob.glob("config/projects/*")
    for config_dir in config_dirs:
        if not config_dir.endswith("1140"):
            print(f"   删除: {config_dir}")
            import shutil
            shutil.rmtree(config_dir, ignore_errors=True)
    
    print("✅ 旧配置文件已清除")

def generate_all_configs():
    """生成所有项目的配置文件"""
    print("🔧 生成所有项目的配置文件...")
    
    try:
        result = subprocess.run([
            sys.executable, "scripts/generate_dynamic_project_configs.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 配置文件生成成功")
            return True
        else:
            print(f"❌ 配置文件生成失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 配置文件生成异常: {e}")
        return False

def test_single_experiment():
    """测试单个实验"""
    print("🧪 测试单个实验...")
    
    # 找到第一个GRU配置文件
    config_files = glob.glob("config/projects/*/GRU_high_NWP_24h_TE.yaml")
    if not config_files:
        print("❌ 未找到GRU配置文件")
        return False
    
    config_file = config_files[0]
    print(f"📁 使用配置文件: {config_file}")
    
    # 检查配置文件内容
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"🔍 配置文件内容:")
    print(f"  model: {config.get('model')}")
    print(f"  train_params: {config.get('train_params')}")
    print(f"  model_params keys: {list(config.get('model_params', {}).keys())}")
    
    # 运行实验
    print("🚀 运行实验...")
    try:
        result = subprocess.run([
            sys.executable, "main.py", "--config", config_file
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ 实验成功!")
            if "CSV结果已更新" in result.stdout:
                print("✅ CSV结果已保存")
            else:
                print("⚠️ 未看到CSV保存信息")
            return True
        else:
            print(f"❌ 实验失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 实验异常: {e}")
        return False

def main():
    """主函数"""
    print("🌟 SolarPV项目 - 修复版批量实验脚本")
    print("=" * 60)
    
    # 检查Google Drive
    drive_mounted = os.path.exists("/content/drive/MyDrive")
    if drive_mounted:
        print("✅ Google Drive已挂载")
    else:
        print("⚠️ Google Drive未挂载，将跳过Drive保存")
    
    # 清除旧配置
    clear_old_configs()
    
    # 生成新配置
    if not generate_all_configs():
        print("❌ 配置生成失败，退出")
        return
    
    # 测试单个实验
    if not test_single_experiment():
        print("❌ 实验测试失败，退出")
        return
    
    print("✅ 所有测试通过，可以开始批量实验!")
    print("💡 现在可以运行: !python colab_batch_experiments.py")

if __name__ == "__main__":
    main()
