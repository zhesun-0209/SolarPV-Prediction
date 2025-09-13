#!/usr/bin/env python3
"""
修复现有配置文件中的LSR模型问题
"""

import os
import yaml
from pathlib import Path
import re

def fix_config_file(config_file):
    """修复单个配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查是否需要修复
        if config.get('model') == 'LSR':
            print(f"🔧 修复配置文件: {config_file}")
            
            # 修复模型名称
            config['model'] = 'Linear'
            
            # 确保有必需的配置参数
            if 'past_hours' not in config:
                config['past_hours'] = 24
            if 'future_hours' not in config:
                config['future_hours'] = 24
            
            # 保存修复后的配置
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ 修复配置文件失败 {config_file}: {e}")
        return False

def fix_all_configs():
    """修复所有配置文件"""
    print("🚀 开始修复配置文件")
    print("=" * 60)
    
    config_dir = Path("config/projects")
    if not config_dir.exists():
        print("❌ 配置目录不存在: config/projects")
        return
    
    fixed_count = 0
    total_count = 0
    
    # 遍历所有项目目录
    for project_dir in config_dir.iterdir():
        if project_dir.is_dir():
            print(f"\n📁 处理项目: {project_dir.name}")
            
            # 遍历项目中的所有配置文件
            for config_file in project_dir.glob("*.yaml"):
                if config_file.name != "config_index.yaml":
                    total_count += 1
                    if fix_config_file(config_file):
                        fixed_count += 1
    
    print(f"\n✅ 修复完成!")
    print(f"   总配置文件: {total_count}")
    print(f"   修复文件数: {fixed_count}")
    print(f"   无需修复: {total_count - fixed_count}")

def verify_configs():
    """验证配置文件修复结果"""
    print("\n🔍 验证配置文件修复结果")
    print("=" * 60)
    
    config_dir = Path("config/projects")
    lsr_count = 0
    linear_count = 0
    
    for project_dir in config_dir.iterdir():
        if project_dir.is_dir():
            for config_file in project_dir.glob("*.yaml"):
                if config_file.name != "config_index.yaml":
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                        
                        model = config.get('model', '')
                        if model == 'LSR':
                            lsr_count += 1
                        elif model == 'Linear':
                            linear_count += 1
                    except:
                        pass
    
    print(f"📊 验证结果:")
    print(f"   LSR配置: {lsr_count}")
    print(f"   Linear配置: {linear_count}")
    
    if lsr_count == 0:
        print("✅ 所有LSR配置已成功修复为Linear")
    else:
        print("⚠️  仍有LSR配置需要修复")

def main():
    """主函数"""
    print("🔧 配置文件修复工具")
    print("=" * 80)
    
    # 修复所有配置文件
    fix_all_configs()
    
    # 验证修复结果
    verify_configs()
    
    print("\n🎯 修复完成！现在可以重新运行实验了")

if __name__ == "__main__":
    main()
