#!/usr/bin/env python3
"""
修复配置文件中的路径问题
"""

import os
import yaml
from pathlib import Path

def fix_config_paths():
    """修复配置文件中的路径"""
    print("🔧 修复配置文件路径")
    print("=" * 50)
    
    config_dir = Path("config/projects")
    if not config_dir.exists():
        print("❌ config/projects目录不存在")
        return
    
    fixed_count = 0
    total_count = 0
    
    # 遍历所有项目配置目录
    for project_dir in config_dir.iterdir():
        if not project_dir.is_dir() or project_dir.name.startswith('.'):
            continue
            
        project_id = project_dir.name
        print(f"🔍 检查项目 {project_id}")
        
        # 遍历所有配置文件
        for config_file in project_dir.glob("*.yaml"):
            total_count += 1
            
            # 读取配置
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查并修复路径
            current_path = config.get('data_path', '')
            expected_path = f"data/Project{project_id}.csv"
            
            if current_path != expected_path:
                print(f"   🔧 修复: {current_path} -> {expected_path}")
                config['data_path'] = expected_path
                
                # 写回文件
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                fixed_count += 1
    
    print("=" * 50)
    print(f"✅ 修复完成: {fixed_count}/{total_count} 个配置文件")

if __name__ == "__main__":
    fix_config_paths()
