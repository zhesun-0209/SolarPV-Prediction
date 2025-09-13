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

def get_config_files(config_dir: str = "config/projects") -> list:
    """获取所有配置文件"""
    # 扫描所有项目目录下的配置文件
    all_config_files = []
    
    # 获取所有项目目录
    project_dirs = glob.glob(os.path.join(config_dir, "*"))
    project_dirs = [d for d in project_dirs if os.path.isdir(d)]
    
    for project_dir in project_dirs:
        yaml_files = glob.glob(os.path.join(project_dir, "*.yaml"))
        config_files = [f for f in yaml_files if not f.endswith("config_index.yaml")]
        all_config_files.extend(config_files)
    
    return sorted(all_config_files)

def run_project_experiments(project_id: str, all_config_files: list, data_dir: str = "data", 
                          save_to_drive: bool = True) -> dict:
    """
    运行单个项目的所有实验
    
    Args:
        project_id: 项目ID
        all_config_files: 所有配置文件列表
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
    
    # 筛选出当前项目的配置文件
    project_config_files = []
    for config_file in all_config_files:
        if f"/{project_id}/" in config_file or f"\\{project_id}\\" in config_file:
            project_config_files.append(config_file)
    
    # 如果没有找到项目专用配置，使用通用配置（如1140的配置）
    if not project_config_files:
        print(f"⚠️ 项目 {project_id} 没有专用配置文件，使用通用配置")
        # 使用1140的配置作为模板
        template_configs = [f for f in all_config_files if "/1140/" in f or "\\1140\\" in f]
        project_config_files = template_configs
    
    # 硬编码Drive保存目录，删除本地保存
    drive_save_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(drive_save_dir, exist_ok=True)
    
    # 为项目创建初始CSV文件
    csv_file_path = os.path.join(drive_save_dir, f"{project_id}_results.csv")
    if not os.path.exists(csv_file_path):
        print(f"📄 创建项目CSV文件: {csv_file_path}")
        # 创建空的CSV文件，包含列头
        import pandas as pd
        empty_df = pd.DataFrame(columns=[
            'model', 'use_pv', 'use_hist_weather', 'use_forecast', 'weather_category',
            'use_time_encoding', 'past_days', 'model_complexity', 'epochs', 'batch_size',
            'learning_rate', 'train_time_sec', 'inference_time_sec', 'param_count',
            'samples_count', 'best_epoch', 'final_lr', 'mse', 'rmse', 'mae', 'nrmse',
            'r_square', 'smape', 'gpu_memory_used'
        ])
        empty_df.to_csv(csv_file_path, index=False)
        print(f"✅ 项目CSV文件已创建")
    else:
        print(f"📄 项目CSV文件已存在: {csv_file_path}")
    
    # 统计信息
    stats = {
        'project_id': project_id,
        'total_experiments': len(project_config_files),
        'successful': 0,
        'failed': 0,
        'start_time': time.time(),
        'errors': []
    }
    
    print(f"📊 项目 {project_id}: 将运行 {len(project_config_files)} 个实验")
    print(f"📁 结果保存到: {drive_save_dir}")
    
    # 运行每个实验
    for i, config_file in enumerate(project_config_files, 1):
        print(f"\n🔄 进度: {i}/{len(project_config_files)} - {os.path.basename(config_file)}")
        
        try:
            # 修改配置文件中的数据路径
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"🔍 调试: 原始配置文件加载完成")
            print(f"🔍 调试: 原始config['train_params'] = {config.get('train_params', 'NOT_FOUND')}")
            
            # 更新数据路径和plant_id（save_dir已在eval_utils中硬编码）
            config['data_path'] = data_file
            config['plant_id'] = project_id  # 设置plant_id
            
            print(f"🔍 调试: 修改后config['train_params'] = {config.get('train_params', 'NOT_FOUND')}")
            print(f"🔍 调试: 修改后config['model'] = {config.get('model', 'NOT_FOUND')}")
            print(f"🔍 调试: 修改后config['model_params'] = {config.get('model_params', 'NOT_FOUND')}")
            
            # 保存临时配置文件到临时目录
            temp_dir = "/tmp/solarpv_configs"
            os.makedirs(temp_dir, exist_ok=True)
            temp_config_file = os.path.join(temp_dir, f"temp_{project_id}_{os.path.basename(config_file)}")
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 运行实验
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "main.py", "--config", temp_config_file
            ], capture_output=True, text=True, timeout=1800)  # 30分钟超时
            
            # 显示调试信息
            if "🔍 调试" in result.stdout:
                print("🔍 实验调试信息:")
                for line in result.stdout.split('\n'):
                    if "🔍 调试" in line:
                        print(f"   {line}")
            
            # 显示完整的标准输出（用于调试）
            if "CSV结果已更新" not in result.stdout and "🔍 调试" not in result.stdout:
                print("🔍 完整标准输出:")
                print(result.stdout[-1000:])  # 显示最后1000个字符
            
            # 显示错误输出（如果有）
            if result.stderr:
                print("🔍 错误输出:")
                print(result.stderr[-500:])  # 显示最后500个字符
            
            duration = time.time() - start_time
            
            # 检查是否有错误信息（即使返回码为0）
            has_error = "[ERROR]" in result.stdout or result.returncode != 0
            
            if not has_error and result.returncode == 0:
                stats['successful'] += 1
                print(f"✅ 实验成功! 用时: {duration:.1f}秒")
                
                # 提取结果指标
                if "mse=" in result.stdout:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "mse=" in line and "rmse=" in line and "mae=" in line:
                            print(f"📊 结果: {line.strip()}")
                            break
                
                # 检查是否有CSV保存信息
                if "CSV结果已更新" in result.stdout:
                    print("✅ CSV结果已保存")
                else:
                    print("⚠️ 未看到CSV保存信息")
                
                # 硬编码保存结果到CSV文件
                csv_file_path = os.path.join(drive_save_dir, f"{project_id}_results.csv")
                print(f"🔧 硬编码保存结果到: {csv_file_path}")
                
                # 从实验输出中提取结果
                result_line = None
                for line in result.stdout.split('\n'):
                    if "mse=" in line and "rmse=" in line and "mae=" in line and "r_square=" in line:
                        result_line = line
                        break
                
                if result_line:
                    # 解析结果
                    import re
                    mse_match = re.search(r'mse=([0-9.]+)', result_line)
                    rmse_match = re.search(r'rmse=([0-9.]+)', result_line)
                    mae_match = re.search(r'mae=([0-9.]+)', result_line)
                    r_square_match = re.search(r'r_square=([0-9.]+)', result_line)
                    
                    if mse_match and rmse_match and mae_match and r_square_match:
                        # 从配置文件名中提取更多信息
                        config_filename = os.path.basename(config_file)
                        parts = config_filename.replace('.yaml', '').split('_')
                        
                        # 解析配置文件名: GRU_high_NWP_24h_TE.yaml
                        model_name = parts[0] if len(parts) > 0 else config.get('model', 'Unknown')
                        complexity = parts[1] if len(parts) > 1 else config.get('model_complexity', 'low')
                        input_category = parts[2] if len(parts) > 2 else 'unknown'
                        lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                        time_encoding = parts[4] == 'TE' if len(parts) > 4 else config.get('use_time_encoding', False)
                        
                        # 根据input_category确定其他参数
                        use_pv = input_category in ['PV', 'PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW']
                        use_hist_weather = input_category in ['PV_plus_HW']
                        use_forecast = input_category in ['PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW', 'NWP', 'NWP_plus']
                        
                        # 创建结果行
                        result_row = {
                            'model': model_name,
                            'use_pv': use_pv,
                            'use_hist_weather': use_hist_weather,
                            'use_forecast': use_forecast,
                            'weather_category': config.get('weather_category', 'all_weather'),
                            'use_time_encoding': time_encoding,
                            'past_days': config.get('past_days', 1),
                            'model_complexity': complexity,
                            'epochs': config.get('epochs', 50 if complexity == 'high' else 15),
                            'batch_size': config.get('train_params', {}).get('batch_size', 32),
                            'learning_rate': config.get('train_params', {}).get('learning_rate', 0.001),
                            'train_time_sec': round(duration, 4),
                            'inference_time_sec': 0.0,
                            'param_count': 0,
                            'samples_count': 0,
                            'best_epoch': 0,
                            'final_lr': 0.0,
                            'mse': float(mse_match.group(1)),
                            'rmse': float(rmse_match.group(1)),
                            'mae': float(mae_match.group(1)),
                            'nrmse': 0.0,
                            'r_square': float(r_square_match.group(1)),
                            'smape': 0.0,
                            'gpu_memory_used': 0
                        }
                        
                        # 读取现有CSV文件
                        import pandas as pd
                        if os.path.exists(csv_file_path):
                            df = pd.read_csv(csv_file_path)
                        else:
                            df = pd.DataFrame()
                        
                        # 添加新行
                        new_row_df = pd.DataFrame([result_row])
                        df = pd.concat([df, new_row_df], ignore_index=True)
                        
                        # 保存CSV文件
                        df.to_csv(csv_file_path, index=False)
                        print(f"✅ 结果已硬编码保存到CSV文件")
                        print(f"📊 CSV文件当前行数: {len(df)}")
                        print(f"📊 最新实验: {result_row['model']} - {result_row['mse']:.4f}")
                        print(f"🔍 解析的配置信息:")
                        print(f"   模型: {result_row['model']}, 复杂度: {result_row['model_complexity']}")
                        print(f"   输入类别: {input_category}, 时间编码: {result_row['use_time_encoding']}")
                        print(f"   PV: {result_row['use_pv']}, 历史天气: {result_row['use_hist_weather']}, 预测天气: {result_row['use_forecast']}")
                    else:
                        print(f"❌ 无法解析实验结果: {result_line}")
                else:
                    print(f"❌ 未找到实验结果行")
            else:
                stats['failed'] += 1
                error_msg = f"返回码: {result.returncode}, 错误: {result.stderr[-200:]}"
                if "[ERROR]" in result.stdout:
                    # 提取错误信息
                    error_lines = [line for line in result.stdout.split('\n') if "[ERROR]" in line]
                    if error_lines:
                        error_msg = f"实验错误: {error_lines[-1]}"
                stats['errors'].append(error_msg)
                print(f"❌ 实验失败! {error_msg}")
                print(f"   标准输出: {result.stdout[-500:]}")
                print(f"   错误输出: {result.stderr[-500:]}")
            
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
    
    # 检查Drive中的结果文件
    if save_to_drive and stats['success']:
        drive_csv_file = os.path.join(drive_save_dir, f"{project_id}_results.csv")
        if os.path.exists(drive_csv_file):
            print(f"✅ 项目 {project_id} 结果已保存到Drive: {drive_csv_file}")
        else:
            print(f"⚠️ 项目 {project_id} 结果文件未找到: {drive_csv_file}")
    
    return stats

def main():
    """主函数"""
    print("🌟 SolarPV项目 - 批量实验脚本")
    print("=" * 80)
    
    # 检查Google Drive是否已挂载
    print("🔗 检查Google Drive...")
    drive_mounted = os.path.exists("/content/drive/MyDrive")
    if drive_mounted:
        print("✅ Google Drive已挂载")
    else:
        print("⚠️ Google Drive未挂载，将跳过Drive保存")
    
    # 获取可用项目
    print("📁 扫描数据文件...")
    projects = get_available_projects()
    print(f"📊 找到 {len(projects)} 个项目: {projects[:10]}{'...' if len(projects) > 10 else ''}")
    
    # 检查是否需要生成配置文件
    print("📁 检查配置文件...")
    config_files = get_config_files()
    print(f"📊 找到 {len(config_files)} 个配置文件")
    
    if not projects:
        print("❌ 未找到任何项目数据文件")
        return
    
    # 检查配置文件是否足够
    if len(config_files) < len(projects) * 100:  # 每个项目至少需要100个配置
        print("⚠️ 配置文件数量不足，需要生成配置文件")
        print("🔧 正在生成配置文件...")
        
        try:
            # 运行配置文件生成脚本
            result = subprocess.run([
                sys.executable, "scripts/generate_dynamic_project_configs.py"
            ], capture_output=True, text=True, timeout=300)  # 5分钟超时
            
            if result.returncode == 0:
                print("✅ 配置文件生成成功")
                # 重新扫描配置文件
                config_files = get_config_files()
                print(f"📊 重新扫描到 {len(config_files)} 个配置文件")
            else:
                print(f"❌ 配置文件生成失败: {result.stderr}")
                return
                
        except subprocess.TimeoutExpired:
            print("⏰ 配置文件生成超时")
            return
        except Exception as e:
            print(f"💥 配置文件生成异常: {str(e)}")
            return
    
    if not config_files:
        print("❌ 未找到任何配置文件")
        return
    
    # 批量实验设置
    # 硬编码Drive路径，删除本地结果目录
    drive_save_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(drive_save_dir, exist_ok=True)
    
    # 运行所有项目
    all_stats = []
    total_projects = len(projects)
    successful_projects = 0
    
    print(f"\n🚀 开始批量实验!")
    print(f"📊 总项目数: {total_projects}")
    # 计算总实验数（每个项目使用340个配置）
    experiments_per_project = 340
    total_experiments = total_projects * experiments_per_project
    print(f"📊 每项目实验数: {experiments_per_project}")
    print(f"📊 总实验数: {total_experiments}")
    
    for i, project_id in enumerate(projects, 1):
        print(f"\n🔄 项目进度: {i}/{total_projects}")
        
        stats = run_project_experiments(
            project_id=project_id,
            all_config_files=config_files,
            data_dir="data",
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
        if isinstance(drive_results, dict):
            print(f"  📊 总CSV文件数: {drive_results['total_csv_files']}")
            print(f"  📊 总项目数: {drive_results['total_projects']}")
            
            # 显示前10个CSV文件
            print(f"  📄 CSV文件列表:")
            for csv_file in drive_results['csv_files'][:10]:
                print(f"    📄 {csv_file['filename']} ({csv_file['size']} bytes)")
            
            if drive_results['total_csv_files'] > 10:
                print(f"    ... 还有 {drive_results['total_csv_files'] - 10} 个CSV文件")
            
            # 显示项目统计
            print(f"  📊 项目统计:")
            for project_id, stats in list(drive_results['project_stats'].items())[:10]:
                print(f"    📁 Project {project_id}: {stats['count']} 个CSV文件")
            
            if drive_results['total_projects'] > 10:
                print(f"    ... 还有 {drive_results['total_projects'] - 10} 个项目")
        else:
            print(f"  ❌ 无法读取Drive结果")
    
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
