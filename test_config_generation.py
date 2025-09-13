#!/usr/bin/env python3
"""
测试配置生成器是否正确工作
"""

import os
import yaml
from pathlib import Path

def test_config_generation():
    """测试配置生成器"""
    print("🧪 测试配置生成器")
    print("=" * 50)
    
    # 检查data目录
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data目录不存在")
        return
    
    # 列出所有Project文件
    project_files = list(data_dir.glob("Project*.csv"))
    print(f"📊 发现 {len(project_files)} 个Project文件:")
    for file in project_files:
        print(f"   - {file.name}")
    
    if not project_files:
        print("❌ 没有找到Project*.csv文件")
        return
    
    # 检查配置目录
    config_dir = Path("config/projects")
    if not config_dir.exists():
        print("❌ config/projects目录不存在")
        return
    
    # 列出所有配置目录
    project_config_dirs = [d for d in config_dir.iterdir() if d.is_dir() and d.name != '.git']
    print(f"📁 发现 {len(project_config_dirs)} 个配置目录:")
    for dir in project_config_dirs:
        print(f"   - {dir.name}")
    
    # 检查第一个项目的配置文件
    if project_config_dirs:
        first_project = project_config_dirs[0]
        config_files = list(first_project.glob("*.yaml"))
        if config_files:
            first_config = config_files[0]
            print(f"🔍 检查配置文件: {first_config}")
            
            with open(first_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            data_path = config.get('data_path', '')
            print(f"📄 数据路径: {data_path}")
            
            # 检查文件是否存在
            if os.path.exists(data_path):
                print(f"✅ 数据文件存在: {data_path}")
            else:
                print(f"❌ 数据文件不存在: {data_path}")
                
                # 尝试找到正确的文件
                project_id = first_project.name
                expected_path = f"data/Project{project_id}.csv"
                if os.path.exists(expected_path):
                    print(f"✅ 找到正确文件: {expected_path}")
                else:
                    print(f"❌ 正确文件也不存在: {expected_path}")

if __name__ == "__main__":
    test_config_generation()
