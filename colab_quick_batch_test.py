#!/usr/bin/env python3
"""
Colab快速批量测试脚本 - 测试少量项目验证功能
"""

import os
import sys
import subprocess
import time
import yaml
import glob
import pandas as pd
from datetime import datetime
from utils.drive_utils import mount_drive, save_project_results_to_drive

def quick_batch_test(max_projects: int = 3, max_experiments_per_project: int = 5):
    """
    快速批量测试
    
    Args:
        max_projects: 最大测试项目数
        max_experiments_per_project: 每个项目最大实验数
    """
    print("🧪 SolarPV项目 - 快速批量测试")
    print("=" * 60)
    
    # 检查Google Drive是否已挂载
    print("🔗 检查Google Drive...")
    drive_mounted = os.path.exists("/content/drive/MyDrive")
    if drive_mounted:
        print("✅ Google Drive已挂载")
    else:
        print("⚠️ Google Drive未挂载，将跳过Drive保存")
    
    # 获取可用项目（限制数量）
    print("📁 扫描数据文件...")
    csv_files = glob.glob("data/Project*.csv")
    projects = []
    
    for csv_file in csv_files[:max_projects]:  # 限制项目数量
        filename = os.path.basename(csv_file)
        if filename.startswith("Project") and filename.endswith(".csv"):
            project_id = filename[7:-4]
            projects.append(project_id)
    
    print(f"📊 找到 {len(projects)} 个项目进行测试: {projects}")
    
    # 获取配置文件（限制数量）
    print("📁 扫描配置文件...")
    config_files = glob.glob("config/projects/1140/*.yaml")
    config_files = [f for f in config_files if not f.endswith("config_index.yaml")]
    config_files = config_files[:max_experiments_per_project]  # 限制实验数量
    
    print(f"📊 找到 {len(config_files)} 个配置文件进行测试")
    
    if not projects:
        print("❌ 未找到任何项目数据文件")
        return
    
    if not config_files:
        print("❌ 未找到任何配置文件")
        return
    
    # 创建结果目录
    results_dir = "temp_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 运行测试
    all_stats = []
    
    for i, project_id in enumerate(projects, 1):
        print(f"\n🔄 测试项目 {i}/{len(projects)}: {project_id}")
        
        # 检查数据文件
        data_file = f"data/Project{project_id}.csv"
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            continue
        
        # 创建项目结果目录
        project_results_dir = os.path.join(results_dir, project_id)
        os.makedirs(project_results_dir, exist_ok=True)
        
        # 运行实验
        stats = {
            'project_id': project_id,
            'total_experiments': len(config_files),
            'successful': 0,
            'failed': 0,
            'start_time': time.time(),
            'errors': []
        }
        
        for j, config_file in enumerate(config_files, 1):
            print(f"  🧪 实验 {j}/{len(config_files)}: {os.path.basename(config_file)}")
            
            try:
                # 修改配置文件
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                config['data_path'] = data_file
                config['save_dir'] = project_results_dir
                
                # 保存临时配置
                temp_config = os.path.join(project_results_dir, f"temp_{os.path.basename(config_file)}")
                with open(temp_config, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                
                # 运行实验
                result = subprocess.run([
                    sys.executable, "main.py", "--config", temp_config
                ], capture_output=True, text=True, timeout=300)  # 5分钟超时
                
                if result.returncode == 0:
                    stats['successful'] += 1
                    print(f"    ✅ 成功")
                    
                    # 提取结果
                    if "mse=" in result.stdout:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if "mse=" in line and "rmse=" in line:
                                print(f"    📊 {line.strip()}")
                                break
                else:
                    stats['failed'] += 1
                    print(f"    ❌ 失败: {result.stderr[-100:]}")
                
                # 清理
                if os.path.exists(temp_config):
                    os.remove(temp_config)
                    
            except subprocess.TimeoutExpired:
                stats['failed'] += 1
                print(f"    ⏰ 超时")
            except Exception as e:
                stats['failed'] += 1
                print(f"    💥 异常: {str(e)}")
        
        stats['total_time'] = time.time() - stats['start_time']
        stats['success'] = stats['successful'] > 0
        all_stats.append(stats)
        
        print(f"  📊 项目 {project_id} 完成: 成功 {stats['successful']}/{stats['total_experiments']}")
        
        # 保存到Drive
        if drive_mounted and stats['success']:
            print(f"  💾 保存到Drive...")
            save_project_results_to_drive(project_id, project_results_dir)
    
    # 最终统计
    print(f"\n🎉 快速测试完成!")
    print("=" * 60)
    
    total_projects = len(all_stats)
    successful_projects = sum(1 for s in all_stats if s['success'])
    total_experiments = sum(s['total_experiments'] for s in all_stats)
    total_successful = sum(s['successful'] for s in all_stats)
    
    print(f"📊 项目统计: {successful_projects}/{total_projects} 成功")
    print(f"📊 实验统计: {total_successful}/{total_experiments} 成功")
    
    # 显示结果文件
    print(f"\n📁 结果文件:")
    for stats in all_stats:
        if stats['success']:
            project_dir = os.path.join(results_dir, stats['project_id'])
            csv_files = [f for f in os.listdir(project_dir) if f.endswith('.csv')]
            print(f"  📁 Project {stats['project_id']}: {len(csv_files)} 个CSV文件")
    
    print(f"\n💡 如果测试成功，可以运行完整批量实验:")
    print(f"   !python colab_batch_experiments.py")

if __name__ == "__main__":
    # 可以调整测试参数
    quick_batch_test(max_projects=2, max_experiments_per_project=3)
