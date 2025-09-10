#!/usr/bin/env python3
"""
调试Project1033的结果路径
"""

import os
import glob

def debug_project1033_paths():
    """调试Project1033的结果路径"""
    
    print("🔍 调试Project1033的结果路径")
    print("=" * 60)
    
    # 检查Drive和本地结果
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    local_dir = 'result'
    
    result_dirs = []
    if os.path.exists(drive_dir):
        result_dirs.append(drive_dir)
        print(f"✅ Drive目录存在: {drive_dir}")
    else:
        print(f"❌ Drive目录不存在: {drive_dir}")
    
    if os.path.exists(local_dir):
        result_dirs.append(local_dir)
        print(f"✅ 本地目录存在: {local_dir}")
    else:
        print(f"❌ 本地目录不存在: {local_dir}")
    
    print(f"\n📁 检查的目录: {result_dirs}")
    
    # 检查Project1033的结果
    plant_id = 'Project1033'
    
    for result_dir in result_dirs:
        print(f"\n🔍 检查目录: {result_dir}")
        
        # 检查厂目录
        plant_result_dir = os.path.join(result_dir, plant_id)
        print(f"   厂目录: {plant_result_dir}")
        print(f"   存在: {os.path.exists(plant_result_dir)}")
        
        if os.path.exists(plant_result_dir):
            # 列出厂目录下的所有内容
            print(f"   内容:")
            for item in os.listdir(plant_result_dir):
                item_path = os.path.join(plant_result_dir, item)
                print(f"     {item} ({'目录' if os.path.isdir(item_path) else '文件'})")
            
            # 查找summary.csv文件
            summary_files = glob.glob(os.path.join(plant_result_dir, '**/summary.csv'), recursive=True)
            print(f"   summary.csv文件数: {len(summary_files)}")
            
            for file in summary_files:
                print(f"     {file}")
                
                # 从文件路径提取实验ID
                path_parts = file.split(os.sep)
                print(f"       路径部分: {path_parts}")
                if len(path_parts) >= 2:
                    exp_id = path_parts[-2]
                    print(f"       实验ID: {exp_id}")
        else:
            print(f"   厂目录不存在")
    
    # 检查data目录
    print(f"\n📊 检查data目录:")
    data_dir = 'data'
    if os.path.exists(data_dir):
        csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
        print(f"   CSV文件数: {len(csv_files)}")
        for file in csv_files:
            filename = os.path.basename(file)
            print(f"     {filename}")
    else:
        print(f"   data目录不存在")

if __name__ == "__main__":
    debug_project1033_paths()
