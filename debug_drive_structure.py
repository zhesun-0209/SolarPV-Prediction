#!/usr/bin/env python3
"""
调试Drive目录结构
"""

import os
import glob

def debug_drive_structure():
    """调试Drive目录结构"""
    
    print("🔍 调试Drive目录结构")
    print("=" * 60)
    
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    
    if not os.path.exists(drive_dir):
        print(f"❌ Drive目录不存在: {drive_dir}")
        return
    
    print(f"✅ Drive目录存在: {drive_dir}")
    
    # 列出Drive目录下的所有内容
    print(f"\n📁 Drive目录内容:")
    try:
        items = os.listdir(drive_dir)
        for item in sorted(items):
            item_path = os.path.join(drive_dir, item)
            item_type = "目录" if os.path.isdir(item_path) else "文件"
            print(f"   {item} ({item_type})")
    except Exception as e:
        print(f"❌ 无法列出目录内容: {e}")
        return
    
    # 查找所有可能的Project1033相关目录
    print(f"\n🔍 查找Project1033相关目录:")
    project_dirs = []
    
    # 递归查找包含Project1033的目录
    for root, dirs, files in os.walk(drive_dir):
        for dir_name in dirs:
            if 'Project1033' in dir_name or '1033' in dir_name:
                project_dirs.append(os.path.join(root, dir_name))
    
    if project_dirs:
        print(f"   找到 {len(project_dirs)} 个相关目录:")
        for dir_path in project_dirs:
            print(f"     {dir_path}")
            
            # 检查每个目录下的内容
            try:
                sub_items = os.listdir(dir_path)
                print(f"       内容: {len(sub_items)} 项")
                for sub_item in sub_items[:10]:  # 只显示前10项
                    sub_path = os.path.join(dir_path, sub_item)
                    sub_type = "目录" if os.path.isdir(sub_path) else "文件"
                    print(f"         {sub_item} ({sub_type})")
                if len(sub_items) > 10:
                    print(f"         ... 还有 {len(sub_items) - 10} 项")
            except Exception as e:
                print(f"       无法访问: {e}")
    else:
        print(f"   ❌ 未找到Project1033相关目录")
    
    # 查找所有summary.csv文件
    print(f"\n📊 查找所有summary.csv文件:")
    summary_files = glob.glob(os.path.join(drive_dir, '**/summary.csv'), recursive=True)
    print(f"   找到 {len(summary_files)} 个summary.csv文件")
    
    if summary_files:
        # 按目录分组
        summary_by_dir = {}
        for file in summary_files:
            dir_path = os.path.dirname(file)
            if dir_path not in summary_by_dir:
                summary_by_dir[dir_path] = []
            summary_by_dir[dir_path].append(file)
        
        print(f"   按目录分组:")
        for dir_path, files in summary_by_dir.items():
            print(f"     {dir_path}: {len(files)} 个文件")
            # 显示前几个文件的路径结构
            for file in files[:3]:
                path_parts = file.split(os.sep)
                print(f"       {file}")
                print(f"         路径部分: {path_parts}")
                if len(path_parts) >= 2:
                    exp_id = path_parts[-2]
                    print(f"         实验ID: {exp_id}")
            if len(files) > 3:
                print(f"       ... 还有 {len(files) - 3} 个文件")
    
    # 查找可能的厂ID模式
    print(f"\n🔍 查找可能的厂ID模式:")
    all_dirs = []
    for root, dirs, files in os.walk(drive_dir):
        all_dirs.extend(dirs)
    
    # 统计目录名模式
    dir_patterns = {}
    for dir_name in all_dirs:
        if 'feat' in dir_name and 'fcst' in dir_name:
            # 这可能是实验目录
            parts = dir_name.split('_')
            if len(parts) >= 2:
                model = parts[0]
                if model not in dir_patterns:
                    dir_patterns[model] = 0
                dir_patterns[model] += 1
    
    if dir_patterns:
        print(f"   实验目录模式:")
        for model, count in sorted(dir_patterns.items()):
            print(f"     {model}: {count} 个")
    else:
        print(f"   ❌ 未找到实验目录模式")

if __name__ == "__main__":
    debug_drive_structure()
