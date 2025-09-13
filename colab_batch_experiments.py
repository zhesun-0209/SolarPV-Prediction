#!/usr/bin/env python3
"""
Colab批量实验脚本 - 支持100个项目的全量实验
每个项目运行340个实验，结果保存到Google Drive
"""

import os
import sys
import subprocess
import time
import yaml
import glob
import pandas as pd
from datetime import datetime
from utils.drive_utils import mount_drive, save_project_results_to_drive, list_drive_results

def get_available_projects(data_dir: str = "data") -> list:
    """获取可用的项目列表"""
    csv_files = glob.glob(os.path.join(data_dir, "Project*.csv"))
    projects = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # 提取项目ID，例如 Project1140.csv -> 1140
        if filename.startswith("Project") and filename.endswith(".csv"):
            project_id = filename[7:-4]  # 去掉"Project"和".csv"
            projects.append(project_id)
    
    return sorted(projects)

def get_config_files(config_dir: str = "config/projects/1140") -> list:
    """获取所有配置文件"""
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    config_files = [f for f in yaml_files if not f.endswith("config_index.yaml")]
    return sorted(config_files)

def run_project_experiments(project_id: str, config_files: list, data_dir: str = "data", 
                          results_dir: str = "temp_results", save_to_drive: bool = True) -> dict:
    """
    运行单个项目的所有实验
    
    Args:
        project_id: 项目ID
        config_files: 配置文件列表
        data_dir: 数据目录
        results_dir: 结果目录
        save_to_drive: 是否保存到Drive
        
    Returns:
        实验结果统计
    """
    print(f"\n{'='*80}")
    print(f"🚀 开始项目 {project_id} 的实验")
    print(f"{'='*80}")
    
    # 检查数据文件是否存在
    data_file = os.path.join(data_dir, f"Project{project_id}.csv")
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return {'success': False, 'error': 'Data file not found'}
    
    # 创建项目结果目录
    project_results_dir = os.path.join(results_dir, project_id)
    os.makedirs(project_results_dir, exist_ok=True)
    
    # 统计信息
    stats = {
        'project_id': project_id,
        'total_experiments': len(config_files),
        'successful': 0,
        'failed': 0,
        'start_time': time.time(),
        'errors': []
    }
    
    print(f"📊 项目 {project_id}: 将运行 {len(config_files)} 个实验")
    print(f"📁 结果保存到: {project_results_dir}")
    
    # 运行每个实验
    for i, config_file in enumerate(config_files, 1):
        print(f"\n🔄 进度: {i}/{len(config_files)} - {os.path.basename(config_file)}")
        
        try:
            # 修改配置文件中的数据路径
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # 更新数据路径和保存目录
            config['data_path'] = data_file
            config['save_dir'] = project_results_dir
            
            # 保存临时配置文件
            temp_config_file = os.path.join(project_results_dir, f"temp_{os.path.basename(config_file)}")
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 运行实验
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "main.py", "--config", temp_config_file
            ], capture_output=True, text=True, timeout=1800)  # 30分钟超时
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                stats['successful'] += 1
                print(f"✅ 实验成功! 用时: {duration:.1f}秒")
                
                # 提取结果指标
                if "mse=" in result.stdout:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "mse=" in line and "rmse=" in line and "mae=" in line:
                            print(f"📊 结果: {line.strip()}")
                            break
            else:
                stats['failed'] += 1
                error_msg = f"返回码: {result.returncode}, 错误: {result.stderr[-200:]}"
                stats['errors'].append(error_msg)
                print(f"❌ 实验失败! {error_msg}")
            
            # 清理临时配置文件
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
                
        except subprocess.TimeoutExpired:
            stats['failed'] += 1
            error_msg = "实验超时 (30分钟)"
            stats['errors'].append(error_msg)
            print(f"⏰ 实验超时: {error_msg}")
        except Exception as e:
            stats['failed'] += 1
            error_msg = f"实验异常: {str(e)}"
            stats['errors'].append(error_msg)
            print(f"💥 实验异常: {error_msg}")
    
    # 计算总用时
    stats['total_time'] = time.time() - stats['start_time']
    stats['success'] = stats['successful'] > 0
    
    # 显示项目统计
    print(f"\n📊 项目 {project_id} 完成!")
    print(f"   总实验: {stats['total_experiments']}")
    print(f"   成功: {stats['successful']} ({stats['successful']/stats['total_experiments']*100:.1f}%)")
    print(f"   失败: {stats['failed']} ({stats['failed']/stats['total_experiments']*100:.1f}%)")
    print(f"   总用时: {stats['total_time']/60:.1f} 分钟")
    
    # 保存到Google Drive
    if save_to_drive and stats['success']:
        print(f"💾 保存项目 {project_id} 结果到Google Drive...")
        drive_success = save_project_results_to_drive(project_id, project_results_dir)
        if drive_success:
            print(f"✅ 项目 {project_id} 结果已保存到Drive")
        else:
            print(f"❌ 项目 {project_id} 结果保存到Drive失败")
    
    return stats

def main():
    """主函数"""
    print("🌟 SolarPV项目 - 批量实验脚本")
    print("=" * 80)
    
    # 挂载Google Drive
    print("🔗 挂载Google Drive...")
    drive_mounted = mount_drive()
    
    # 获取可用项目
    print("📁 扫描数据文件...")
    projects = get_available_projects()
    print(f"📊 找到 {len(projects)} 个项目: {projects[:10]}{'...' if len(projects) > 10 else ''}")
    
    # 获取配置文件
    print("📁 扫描配置文件...")
    config_files = get_config_files()
    print(f"📊 找到 {len(config_files)} 个配置文件")
    
    if not projects:
        print("❌ 未找到任何项目数据文件")
        return
    
    if not config_files:
        print("❌ 未找到任何配置文件")
        return
    
    # 批量实验设置
    results_dir = "temp_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 运行所有项目
    all_stats = []
    total_projects = len(projects)
    successful_projects = 0
    
    print(f"\n🚀 开始批量实验!")
    print(f"📊 总项目数: {total_projects}")
    print(f"📊 每项目实验数: {len(config_files)}")
    print(f"📊 总实验数: {total_projects * len(config_files)}")
    
    for i, project_id in enumerate(projects, 1):
        print(f"\n🔄 项目进度: {i}/{total_projects}")
        
        stats = run_project_experiments(
            project_id=project_id,
            config_files=config_files,
            data_dir="data",
            results_dir=results_dir,
            save_to_drive=drive_mounted
        )
        
        all_stats.append(stats)
        if stats['success']:
            successful_projects += 1
        
        # 显示当前统计
        print(f"📈 当前统计: 成功项目 {successful_projects}/{i}")
    
    # 最终统计
    print(f"\n🎉 批量实验完成!")
    print("=" * 80)
    print(f"📊 最终统计:")
    print(f"  总项目数: {total_projects}")
    print(f"  成功项目: {successful_projects} ({successful_projects/total_projects*100:.1f}%)")
    print(f"  失败项目: {total_projects - successful_projects}")
    
    # 计算总实验统计
    total_experiments = sum(s['total_experiments'] for s in all_stats)
    total_successful = sum(s['successful'] for s in all_stats)
    total_failed = sum(s['failed'] for s in all_stats)
    
    print(f"\n📊 实验统计:")
    print(f"  总实验数: {total_experiments}")
    print(f"  成功实验: {total_successful} ({total_successful/total_experiments*100:.1f}%)")
    print(f"  失败实验: {total_failed} ({total_failed/total_experiments*100:.1f}%)")
    
    # 显示Drive结果
    if drive_mounted:
        print(f"\n📁 Google Drive结果:")
        drive_results = list_drive_results()
        for result in drive_results[:10]:  # 显示前10个
            if result['type'] == 'folder':
                print(f"  📁 {result['project']}: {result['csv_count']} 个CSV文件")
            else:
                print(f"  📄 {result['project']}: {result['size']} bytes")
        
        if len(drive_results) > 10:
            print(f"  ... 还有 {len(drive_results) - 10} 个项目")
    
    # 保存实验报告
    report_file = os.path.join(results_dir, "experiment_report.csv")
    report_data = []
    for stats in all_stats:
        report_data.append({
            'project_id': stats['project_id'],
            'total_experiments': stats['total_experiments'],
            'successful': stats['successful'],
            'failed': stats['failed'],
            'success_rate': stats['successful'] / stats['total_experiments'] * 100,
            'total_time_minutes': stats['total_time'] / 60,
            'success': stats['success']
        })
    
    pd.DataFrame(report_data).to_csv(report_file, index=False)
    print(f"\n📊 实验报告已保存: {report_file}")
    
    if drive_mounted:
        save_to_drive(report_file)
        print(f"📊 实验报告已保存到Drive")

if __name__ == "__main__":
    main()
