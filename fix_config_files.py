#!/usr/bin/env python3
"""
修复配置文件脚本
删除错误的配置文件，重新生成正确的340个配置文件
"""

import os
import shutil
import subprocess
from pathlib import Path

def clean_project_configs(project_id):
    """清理项目的配置文件"""
    project_dir = Path(f"config/projects/{project_id}")
    if project_dir.exists():
        print(f"🗑️ 删除项目 {project_id} 的现有配置文件...")
        shutil.rmtree(project_dir)
        print(f"✅ 已删除项目 {project_id} 的配置文件")
    else:
        print(f"ℹ️ 项目 {project_id} 没有现有配置文件")

def regenerate_project_configs(project_id):
    """重新生成项目的配置文件"""
    print(f"🔧 重新生成项目 {project_id} 的配置文件...")
    
    # 使用正确的配置文件生成脚本
    result = subprocess.run([
        "python", "scripts/generate_dynamic_project_configs.py"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ 项目 {project_id} 配置文件生成成功")
        print(result.stdout)
    else:
        print(f"❌ 项目 {project_id} 配置文件生成失败")
        print(result.stderr)
        return False
    
    return True

def verify_project_configs(project_id):
    """验证项目的配置文件数量"""
    project_dir = Path(f"config/projects/{project_id}")
    if not project_dir.exists():
        print(f"❌ 项目 {project_id} 配置文件目录不存在")
        return False
    
    # 统计配置文件（排除config_index.yaml）
    yaml_files = [f for f in project_dir.glob("*.yaml") if f.name != "config_index.yaml"]
    total_configs = len(yaml_files)
    
    print(f"📊 项目 {project_id} 配置文件统计:")
    print(f"   总数量: {total_configs}")
    
    if total_configs != 340:
        print(f"❌ 配置文件数量不正确，期望340个，实际{total_configs}个")
        return False
    
    # 按输入类别统计
    input_categories = ['PV', 'PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW', 'NWP', 'NWP_plus']
    for category in input_categories:
        if category == 'PV':
            # 只统计PV，不包括PV_plus_*
            count = len([f for f in yaml_files if f"_{category}_" in f.name and "PV_plus" not in f.name])
        elif category == 'PV_plus_NWP':
            # 只统计PV_plus_NWP，不包括PV_plus_NWP_plus
            count = len([f for f in yaml_files if f"_{category}_" in f.name and "PV_plus_NWP_plus" not in f.name])
        elif category == 'PV_plus_NWP_plus':
            # 只统计PV_plus_NWP_plus
            count = len([f for f in yaml_files if f"_{category}_" in f.name])
        elif category == 'PV_plus_HW':
            # 只统计PV_plus_HW
            count = len([f for f in yaml_files if f"_{category}_" in f.name])
        elif category == 'NWP':
            # 只统计NWP，不包括NWP_plus和PV_plus_NWP
            count = len([f for f in yaml_files if f"_{category}_" in f.name and "NWP_plus" not in f.name and "PV_plus_NWP" not in f.name])
        elif category == 'NWP_plus':
            # 只统计NWP_plus，不包括PV_plus_NWP_plus
            count = len([f for f in yaml_files if f"_{category}_" in f.name and "PV_plus_NWP_plus" not in f.name])
        print(f"   {category}: {count}个")
    
    print(f"✅ 项目 {project_id} 配置文件验证通过")
    return True

def main():
    """主函数"""
    print("🔧 修复配置文件脚本")
    print("=" * 50)
    
    # 获取所有项目
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ data目录不存在")
        return
    
    csv_files = list(data_dir.glob("Project*.csv"))
    if not csv_files:
        print("❌ 没有找到Project*.csv文件")
        return
    
    # 提取项目ID
    project_ids = []
    for csv_file in csv_files:
        project_id = csv_file.stem.replace("Project", "")
        project_ids.append(project_id)
    
    print(f"📊 找到 {len(project_ids)} 个项目: {project_ids}")
    
    # 处理每个项目
    success_count = 0
    for project_id in project_ids:
        print(f"\n{'='*60}")
        print(f"🔧 处理项目 {project_id}")
        print(f"{'='*60}")
        
        # 清理现有配置文件
        clean_project_configs(project_id)
        
        # 重新生成配置文件
        if regenerate_project_configs(project_id):
            # 验证配置文件
            if verify_project_configs(project_id):
                success_count += 1
            else:
                print(f"❌ 项目 {project_id} 验证失败")
        else:
            print(f"❌ 项目 {project_id} 生成失败")
    
    print(f"\n{'='*60}")
    print(f"🎉 修复完成！成功处理 {success_count}/{len(project_ids)} 个项目")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
