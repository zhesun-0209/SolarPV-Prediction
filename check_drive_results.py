#!/usr/bin/env python3
"""
检查Drive中的现有结果
"""

import os
import glob
import pandas as pd

def check_drive_results():
    """检查Drive中的结果"""
    print("🔍 检查Drive中的现有结果...")
    
    # 检查多个可能的结果目录
    result_dirs = [
        'result/',  # 本地目录
        '/content/drive/MyDrive/Solar PV electricity/results/',  # Drive目录
        '/content/drive/MyDrive/Solar PV electricity/results',   # 不带斜杠
    ]
    
    all_files = []
    for result_dir in result_dirs:
        if os.path.exists(result_dir):
            print(f"\n📁 检查目录: {result_dir}")
            files = glob.glob(os.path.join(result_dir, '**/summary.csv'), recursive=True)
            all_files.extend(files)
            print(f"   找到 {len(files)} 个summary.csv文件")
            
            # 显示前几个文件
            for i, file in enumerate(files[:5]):
                print(f"   {i+1}. {file}")
            if len(files) > 5:
                print(f"   ... 还有 {len(files) - 5} 个文件")
        else:
            print(f"❌ 目录不存在: {result_dir}")
    
    if not all_files:
        print("\n📝 未找到任何结果文件")
        return
    
    print(f"\n📊 总共找到 {len(all_files)} 个结果文件")
    
    # 读取并分析结果
    existing_experiments = set()
    for file in all_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                # 创建实验标识
                exp_id = f"{df.iloc[0]['model']}_{df.iloc[0]['use_hist_weather']}_{df.iloc[0]['use_forecast']}_{df.iloc[0].get('model_complexity', 'medium')}"
                existing_experiments.add(exp_id)
                print(f"   ✅ {exp_id}")
        except Exception as e:
            print(f"   ❌ 读取失败 {file}: {e}")
    
    print(f"\n📊 找到 {len(existing_experiments)} 个已完成实验")
    
    # 显示实验列表
    if existing_experiments:
        print("\n已完成实验:")
        for i, exp_id in enumerate(sorted(existing_experiments), 1):
            print(f"   {i:2d}. {exp_id}")
    
    return existing_experiments

def main():
    """主函数"""
    print("🚀 检查Drive中的现有结果")
    print("=" * 50)
    
    existing_experiments = check_drive_results()
    
    if existing_experiments:
        print(f"\n✅ 找到 {len(existing_experiments)} 个已完成实验")
        print("💡 运行 colab_resume.py 将从这些实验之后继续")
    else:
        print("\n📝 未找到现有结果")
        print("💡 将从头开始运行所有实验")

if __name__ == "__main__":
    main()
