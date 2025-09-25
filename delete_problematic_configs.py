#!/usr/bin/env python3
"""
删除有问题的配置文件
删除所有包含时间编码的NWP和NWP_plus配置，因为时间编码特征在预测特征中缺失
"""

import os
import glob
from pathlib import Path

def delete_problematic_configs():
    """删除有问题的配置文件"""
    
    # 需要删除的配置模式
    patterns_to_delete = [
        "*_NWP_*_TE.yaml",           # 普通NWP + 时间编码
        "*_NWP_plus_*_TE.yaml",      # 理想NWP + 时间编码
        "*_PV_plus_NWP_*_TE.yaml",   # PV + 普通NWP + 时间编码
        "*_PV_plus_NWP_plus_*_TE.yaml"  # PV + 理想NWP + 时间编码
    ]
    
    # 项目目录
    project_dirs = ["1140", "171", "172", "186"]
    
    total_deleted = 0
    
    for project_dir in project_dirs:
        config_dir = f"config/projects/{project_dir}"
        if not os.path.exists(config_dir):
            print(f"⚠️ 项目目录不存在: {config_dir}")
            continue
            
        project_deleted = 0
        print(f"\n🗂️ 处理项目 {project_dir}:")
        
        for pattern in patterns_to_delete:
            # 查找匹配的配置文件
            search_pattern = os.path.join(config_dir, pattern)
            matching_files = glob.glob(search_pattern)
            
            if matching_files:
                print(f"  📁 模式 {pattern}: 找到 {len(matching_files)} 个文件")
                
                for file_path in matching_files:
                    try:
                        os.remove(file_path)
                        project_deleted += 1
                        print(f"    ❌ 已删除: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"    ⚠️ 删除失败: {os.path.basename(file_path)} - {e}")
            else:
                print(f"  📁 模式 {pattern}: 未找到匹配文件")
        
        print(f"  📊 项目 {project_dir} 总计删除: {project_deleted} 个配置文件")
        total_deleted += project_deleted
    
    print(f"\n🎯 总计删除: {total_deleted} 个配置文件")
    return total_deleted

def verify_deletion():
    """验证删除结果"""
    print("\n🔍 验证删除结果:")
    
    # 检查剩余的文件数量
    project_dirs = ["1140", "171", "172", "186"]
    
    for project_dir in project_dirs:
        config_dir = f"config/projects/{project_dir}"
        if os.path.exists(config_dir):
            all_files = glob.glob(os.path.join(config_dir, "*.yaml"))
            te_files = glob.glob(os.path.join(config_dir, "*_TE.yaml"))
            nwp_te_files = glob.glob(os.path.join(config_dir, "*_NWP_*_TE.yaml"))
            nwp_plus_te_files = glob.glob(os.path.join(config_dir, "*_NWP_plus_*_TE.yaml"))
            pv_nwp_te_files = glob.glob(os.path.join(config_dir, "*_PV_plus_NWP_*_TE.yaml"))
            pv_nwp_plus_te_files = glob.glob(os.path.join(config_dir, "*_PV_plus_NWP_plus_*_TE.yaml"))
            
            print(f"  📁 项目 {project_dir}:")
            print(f"    总配置文件: {len(all_files)}")
            print(f"    TE配置文件: {len(te_files)}")
            print(f"    NWP TE文件: {len(nwp_te_files)}")
            print(f"    NWP+ TE文件: {len(nwp_plus_te_files)}")
            print(f"    PV+NWP TE文件: {len(pv_nwp_te_files)}")
            print(f"    PV+NWP+ TE文件: {len(pv_nwp_plus_te_files)}")

if __name__ == "__main__":
    print("🗑️ 开始删除有问题的配置文件...")
    print("=" * 60)
    
    # 确认删除
    response = input("⚠️ 确认要删除所有有问题的配置文件吗？(y/N): ")
    if response.lower() != 'y':
        print("❌ 操作已取消")
        exit(0)
    
    # 执行删除
    deleted_count = delete_problematic_configs()
    
    # 验证结果
    verify_deletion()
    
    print(f"\n✅ 删除完成！共删除 {deleted_count} 个配置文件")
