#!/usr/bin/env python3
"""
敏感性分析实验 - Colab运行脚本
在Google Colab上运行敏感性分析实验
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """设置环境"""
    print("🔧 设置敏感性分析实验环境")
    print("=" * 50)
    
    # 检查必要文件
    required_files = [
        'main.py',
        'data',
        'sensitivity_analysis/scripts/generate_sensitivity_configs.py',
        'sensitivity_analysis/scripts/run_sensitivity_experiments.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    
    print("✅ 环境检查通过")
    return True

def generate_configs():
    """生成敏感性分析配置"""
    print("\n📝 生成敏感性分析配置")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            'python', 'sensitivity_analysis/scripts/generate_sensitivity_configs.py'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 配置生成成功")
            print(result.stdout)
            return True
        else:
            print(f"❌ 配置生成失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 配置生成异常: {e}")
        return False

def run_experiments():
    """运行敏感性分析实验"""
    print("\n🚀 运行敏感性分析实验")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            'python', 'sensitivity_analysis/scripts/run_sensitivity_experiments.py'
        ], capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        if result.returncode == 0:
            print("✅ 实验运行成功")
            print(result.stdout)
            return True
        else:
            print(f"❌ 实验运行失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 实验运行异常: {e}")
        return False

def main():
    """主函数"""
    print("🌟 SolarPV 敏感性分析实验 - Colab版本")
    print("=" * 60)
    
    # 设置环境
    if not setup_environment():
        print("❌ 环境设置失败，退出")
        return
    
    # 生成配置
    if not generate_configs():
        print("❌ 配置生成失败，退出")
        return
    
    # 运行实验
    if not run_experiments():
        print("❌ 实验运行失败，退出")
        return
    
    print("\n🎉 敏感性分析实验完成！")
    print("📁 结果保存在: /content/drive/MyDrive/Solar PV electricity/sensitivity analysis/")

if __name__ == "__main__":
    main()
