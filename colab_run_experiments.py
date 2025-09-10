#!/usr/bin/env python3
"""
Colab完整运行脚本
设置环境并运行132个厂实验
"""

import os
import subprocess
import time

def setup_environment():
    """设置环境"""
    print("🔧 设置环境...")
    
    # 1. 安装依赖
    print("1️⃣ 安装依赖...")
    try:
        result = subprocess.run(['pip', 'install', '-r', 'requirements.txt'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ 依赖安装成功")
        else:
            print("❌ 依赖安装失败")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 依赖安装异常: {e}")
        return False
    
    # 2. 设置Drive路径
    print("\n2️⃣ 设置Drive路径...")
    try:
        result = subprocess.run(['python', 'setup_drive_paths.py'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Drive路径设置成功")
        else:
            print("❌ Drive路径设置失败")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Drive路径设置异常: {e}")
        return False
    
    return True

def run_experiments():
    """运行实验"""
    print("\n🚀 开始运行实验...")
    
    # 检查数据文件
    data_files = []
    for file in os.listdir('data'):
        if file.endswith('.csv'):
            data_files.append(file)
    
    if not data_files:
        print("❌ 未找到数据文件")
        return False
    
    print(f"✅ 找到 {len(data_files)} 个数据文件")
    
    # 运行实验
    try:
        result = subprocess.run(['python', 'run_132_plants.py'], 
                              capture_output=True, text=True, timeout=3600*24)  # 24小时超时
        
        if result.returncode == 0:
            print("✅ 所有实验完成!")
            print("输出:")
            print(result.stdout)
            return True
        else:
            print("❌ 实验失败")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 实验超时 (24小时)")
        return False
    except Exception as e:
        print(f"❌ 实验异常: {e}")
        return False

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction Colab运行器")
    print("=" * 80)
    
    # 设置环境
    if not setup_environment():
        print("❌ 环境设置失败，退出")
        return
    
    # 运行实验
    if not run_experiments():
        print("❌ 实验运行失败")
        return
    
    print("\n🎉 所有实验完成!")
    print("结果已保存到: /content/drive/MyDrive/Solar PV electricity/results")

if __name__ == "__main__":
    main()
