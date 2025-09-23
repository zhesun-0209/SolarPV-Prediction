#!/usr/bin/env python3
"""
调试season and hour analysis实验问题
"""

import os
import yaml
import pandas as pd

def debug_season_hour_issue():
    """调试season and hour analysis问题"""
    print("🔍 调试Season and Hour Analysis问题")
    print("=" * 50)
    
    # 1. 检查Drive路径
    drive_path = "/content/drive/MyDrive/Solar PV electricity/hour and season analysis"
    print(f"📁 检查Drive路径: {drive_path}")
    print(f"   路径存在: {os.path.exists(drive_path)}")
    
    if os.path.exists(drive_path):
        print(f"   路径可写: {os.access(drive_path, os.W_OK)}")
        files = os.listdir(drive_path)
        print(f"   目录内容: {files}")
    else:
        print("   ❌ Drive路径不存在，尝试创建...")
        try:
            os.makedirs(drive_path, exist_ok=True)
            print("   ✅ 路径创建成功")
        except Exception as e:
            print(f"   ❌ 路径创建失败: {e}")
    
    # 2. 检查配置文件
    print(f"\n📝 检查配置文件...")
    config_file = "season_and_hour_analysis/configs/1140/season_hour_linear.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   配置文件存在: {config_file}")
        print(f"   experiment_type: {config.get('experiment_type', 'NOT_SET')}")
        print(f"   weather_category: {config.get('weather_category', 'NOT_SET')}")
        print(f"   model: {config.get('model', 'NOT_SET')}")
    else:
        print(f"   ❌ 配置文件不存在: {config_file}")
    
    # 3. 检查是否有已保存的文件
    print(f"\n📊 检查已保存的文件...")
    if os.path.exists(drive_path):
        summary_files = [f for f in os.listdir(drive_path) if f.endswith('_summary.csv')]
        pred_files = [f for f in os.listdir(drive_path) if f.endswith('_prediction.csv')]
        print(f"   Summary文件: {summary_files}")
        print(f"   Prediction文件: {pred_files}")
        
        # 检查文件内容
        for file in summary_files:
            file_path = os.path.join(drive_path, file)
            try:
                df = pd.read_csv(file_path)
                print(f"   {file}: {len(df)} 行")
            except Exception as e:
                print(f"   {file}: 读取失败 - {e}")
    
    # 4. 测试文件创建
    print(f"\n🧪 测试文件创建...")
    test_file = os.path.join(drive_path, "test_file.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        print(f"   ✅ 测试文件创建成功: {test_file}")
        os.remove(test_file)
        print(f"   ✅ 测试文件删除成功")
    except Exception as e:
        print(f"   ❌ 测试文件创建失败: {e}")

if __name__ == "__main__":
    debug_season_hour_issue()
