#!/usr/bin/env python3
"""
SolarPV项目 - 批量实验脚本
在Colab上运行100个项目的完整实验，保存结果到Google Drive
"""

import os
import sys
import time
import subprocess
import yaml
import pandas as pd
from pathlib import Path
import re
from utils.checkpoint_manager import CheckpointManager
from utils.drive_results_saver import DriveResultsSaver

def check_drive_mount():
    """检查Google Drive是否已挂载"""
    drive_path = "/content/drive/MyDrive"
    if os.path.exists(drive_path):
        print("✅ Google Drive已挂载")
        return True
    else:
        print("❌ Google Drive未挂载，请先挂载Drive")
        return False

def get_data_files():
    """扫描data目录，获取所有项目CSV文件"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return []
    
    csv_files = []
    for file in os.listdir(data_dir):
        if file.startswith("Project") and file.endswith(".csv"):
            project_id = file.replace("Project", "").replace(".csv", "")
            csv_files.append((project_id, os.path.join(data_dir, file)))
    
    csv_files.sort(key=lambda x: int(x[0]))
    return csv_files

def get_config_files():
    """获取所有配置文件"""
    config_dir = "config/projects"
    all_config_files = []
    
    if os.path.exists(config_dir):
        for project_dir in os.listdir(config_dir):
            project_path = os.path.join(config_dir, project_dir)
            if os.path.isdir(project_path):
                for file in os.listdir(project_path):
                    if file.endswith('.yaml') and file != 'config_index.yaml':
                        all_config_files.append(os.path.join(project_path, file))
    
    return all_config_files

def run_project_experiments(project_id, data_file, all_config_files, drive_save_dir):
    """运行单个项目的所有实验"""
    print(f"\n{'='*80}")
    print(f"🚀 开始项目 {project_id} 的实验")
    print(f"{'='*80}")
    
    # 创建项目CSV文件
    csv_file_path = os.path.join(drive_save_dir, f"{project_id}_results.csv")
    if not os.path.exists(csv_file_path):
        print(f"📄 创建项目CSV文件: {csv_file_path}")
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
    
    # 过滤出当前项目的配置文件
    project_config_files = [f for f in all_config_files if f"Project{project_id}" in f or f"1140" in f]
    
    if not project_config_files:
        print(f"⚠️ 未找到项目 {project_id} 的配置文件，使用Project1140的配置作为模板")
        project_config_files = [f for f in all_config_files if "1140" in f]
    
    print(f"📊 项目 {project_id}: 将运行 {len(project_config_files)} 个实验")
    print(f"📁 结果保存到: {drive_save_dir}")
    
    stats = {
        'success': 0,
        'failed': 0,
        'errors': []
    }
    
    start_time = time.time()
    
    # 运行每个实验
    for i, config_file in enumerate(project_config_files, 1):
        print(f"\n🔄 进度: {i}/{len(project_config_files)} - {os.path.basename(config_file)}")
        
        try:
            # 修改配置文件中的数据路径
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"🔍 调试: 原始配置文件加载完成")
            print(f"🔍 调试: 原始config['train_params'] = {config.get('train_params', 'NOT_FOUND')}")
            
            # 更新数据路径和plant_id
            config['data_path'] = data_file
            config['plant_id'] = project_id
            
            # 对于ML模型，移除不应该有的DL参数，但保留ML特有的参数
            if config.get('model') in ['LGBM', 'RF', 'XGB', 'Linear']:
                # ML模型不应该有batch_size等DL参数，但可以有learning_rate（XGB、LGBM）
                if 'train_params' in config:
                    ml_train_params = {}
                    for key, value in config['train_params'].items():
                        # 保留ML模型特有的参数
                        if key in ['learning_rate', 'max_depth', 'n_estimators', 'random_state', 'verbosity']:
                            ml_train_params[key] = value
                    config['train_params'] = ml_train_params
            
            print(f"🔍 调试: 修改后config['train_params'] = {config.get('train_params', 'NOT_FOUND')}")
            print(f"🔍 调试: 修改后config['model'] = {config.get('model', 'NOT_FOUND')}")
            print(f"🔍 调试: 修改后config['model_params'] = {config.get('model_params', 'NOT_FOUND')}")
            
            # 创建临时配置文件
            temp_config_file = f"temp_config_{project_id}_{i}.yaml"
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f)
            
            # 运行实验
            start_exp_time = time.time()
            result = subprocess.run(
                ['python', 'main.py', '--config', temp_config_file],
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            duration = time.time() - start_exp_time
            
            # 检查结果
            has_error = "[ERROR]" in result.stdout or result.returncode != 0
            if not has_error and result.returncode == 0:
                stats['success'] += 1
                print(f"✅ 实验成功! 用时: {duration:.1f}秒")
                
                # 显示结果
                if "🔍 调试" in result.stdout:
                    print("🔍 实验调试信息:")
                    for line in result.stdout.split('\n'):
                        if "🔍 调试" in line:
                            print(f"   {line}")
                
                # 显示实验结果
                if "CSV结果已更新" not in result.stdout and "🔍 调试" not in result.stdout:
                    print("🔍 完整标准输出:")
                    print(result.stdout[-1000:])
                
                if result.stderr:
                    print("🔍 错误输出:")
                    print(result.stderr[-500:])
                
                if "CSV结果已更新" in result.stdout:
                    print("✅ CSV结果已保存")
                else:
                    print("⚠️ 未看到CSV保存信息")
                
                # 硬编码保存结果到CSV文件（独立于save_excel_results设置）
                print(f"🔧 硬编码保存结果到: {csv_file_path}")
                
                # 从实验输出中提取结果
                result_line = None
                inference_time = 0.0
                param_count = 0
                samples_count = 0
                best_epoch = 0
                final_lr = 0.0
                nrmse = 0.0
                smape = 0.0
                gpu_memory_used = 0
                
                # 调试：显示所有输出行
                print("🔍 调试：检查实验输出中的METRICS行")
                for line in result.stdout.split('\n'):
                    if "[METRICS]" in line:
                        print(f"   找到METRICS行: {line}")
                
                for line in result.stdout.split('\n'):
                    if "mse=" in line and "rmse=" in line and "mae=" in line and "r_square=" in line:
                        result_line = line
                    elif "[METRICS]" in line:
                        # 使用正则表达式提取所有键值对
                        metrics_in_line = re.findall(r'(\w+)=([0-9.-]+)', line)
                        for key, value_str in metrics_in_line:
                            try:
                                if key == 'inference_time':
                                    inference_time = float(value_str)
                                    print(f"🔍 调试：提取inference_time={inference_time}")
                                elif key == 'param_count':
                                    param_count = int(float(value_str))
                                    print(f"🔍 调试：提取param_count={param_count}")
                                elif key == 'samples_count':
                                    samples_count = int(float(value_str))
                                    print(f"🔍 调试：提取samples_count={samples_count}")
                                elif key == 'best_epoch':
                                    if value_str.lower() == 'nan':
                                        best_epoch = 0
                                    else:
                                        best_epoch = int(float(value_str))
                                    print(f"🔍 调试：提取best_epoch={best_epoch}")
                                elif key == 'final_lr':
                                    if value_str.lower() == 'nan':
                                        final_lr = 0.0
                                    else:
                                        final_lr = float(value_str)
                                    print(f"🔍 调试：提取final_lr={final_lr}")
                                elif key == 'nrmse':
                                    nrmse = float(value_str)
                                    print(f"🔍 调试：提取nrmse={nrmse}")
                                elif key == 'smape':
                                    smape = float(value_str)
                                    print(f"🔍 调试：提取smape={smape}")
                                elif key == 'gpu_memory_used':
                                    gpu_memory_used = int(float(value_str))
                                    print(f"🔍 调试：提取gpu_memory_used={gpu_memory_used}")
                            except Exception as e:
                                print(f"🔍 调试：{key}提取失败: {e}")
                
                if result_line:
                    # 解析结果
                    mse_match = re.search(r'mse=([0-9.]+)', result_line)
                    rmse_match = re.search(r'rmse=([0-9.]+)', result_line)
                    mae_match = re.search(r'mae=([0-9.]+)', result_line)
                    r_square_match = re.search(r'r_square=([0-9.]+)', result_line)
                    
                    if mse_match and rmse_match and mae_match and r_square_match:
                        # 从配置文件名中提取更多信息
                        config_filename = os.path.basename(config_file)
                        parts = config_filename.replace('.yaml', '').split('_')
                        
                        # 解析配置文件名: LGBM_low_NWP_72h_TE.yaml 或 RF_low_NWP_plus_24h_TE.yaml
                        model_name = parts[0] if len(parts) > 0 else config.get('model', 'Unknown')
                        complexity = parts[1] if len(parts) > 1 else config.get('model_complexity', 'low')
                        
                        # 处理input_category（可能是NWP或NWP_plus）
                        if len(parts) > 2:
                            if parts[2] == 'NWP' and len(parts) > 3 and parts[3] == 'plus':
                                input_category = 'NWP_plus'
                                lookback_hours = parts[4].replace('h', '') if len(parts) > 4 else '24'
                                time_encoding = parts[5] == 'TE' if len(parts) > 5 else config.get('use_time_encoding', False)
                            else:
                                input_category = parts[2]
                                lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                                time_encoding = parts[4] == 'TE' if len(parts) > 4 else config.get('use_time_encoding', False)
                        else:
                            input_category = 'unknown'
                            lookback_hours = '24'
                            time_encoding = config.get('use_time_encoding', False)
                        
                        # 调试信息
                        print(f"🔍 配置文件名解析: {config_filename}")
                        print(f"   解析结果: model={model_name}, complexity={complexity}, input_category={input_category}")
                        print(f"   lookback_hours={lookback_hours}, time_encoding={time_encoding}")
                        
                        # 根据input_category确定其他参数
                        use_pv = input_category in ['PV', 'PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW']
                        use_hist_weather = input_category in ['PV_plus_HW']
                        use_forecast = input_category in ['PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW', 'NWP', 'NWP_plus']
                        
                        # 计算past_days（基于lookback_hours）
                        past_days = int(int(lookback_hours) / 24) if lookback_hours.isdigit() else 1
                        
                        # 判断模型类型
                        is_dl_model = model_name in ['Transformer', 'LSTM', 'GRU', 'TCN']
                        has_learning_rate = model_name in ['XGB', 'LGBM']  # 只有XGB和LGBM有learning_rate
                        
                        # 创建结果行
                        result_row = {
                            'model': model_name,
                            'use_pv': use_pv,
                            'use_hist_weather': use_hist_weather,
                            'use_forecast': use_forecast,
                            'weather_category': config.get('weather_category', 'all_weather'),
                            'use_time_encoding': time_encoding,
                            'past_days': past_days,
                            'model_complexity': complexity,
                            'epochs': config.get('epochs', 50 if complexity == 'high' else 15) if is_dl_model else 0,
                            'batch_size': config.get('train_params', {}).get('batch_size', 32) if is_dl_model else 0,
                            'learning_rate': config.get('train_params', {}).get('learning_rate', 0.001) if has_learning_rate else 0.0,
                            'train_time_sec': round(duration, 4),
                            'inference_time_sec': inference_time,
                            'param_count': param_count,
                            'samples_count': samples_count,
                            'best_epoch': best_epoch if is_dl_model else 0,
                            'final_lr': final_lr if is_dl_model else 0.0,
                            'mse': float(mse_match.group(1)),
                            'rmse': float(rmse_match.group(1)),
                            'mae': float(mae_match.group(1)),
                            'nrmse': nrmse,
                            'r_square': float(r_square_match.group(1)),
                            'smape': smape,
                            'gpu_memory_used': gpu_memory_used
                        }
                        
                        # 读取现有CSV文件
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
                        print(f"🔍 提取的额外字段:")
                        print(f"   推理时间: {inference_time}s, 参数数量: {param_count}, 样本数量: {samples_count}")
                        print(f"   最佳轮次: {best_epoch}, 最终学习率: {final_lr}")
                        print(f"   NRMSE: {nrmse}, SMAPE: {smape}, GPU内存: {gpu_memory_used}MB")
                        print(f"🔍 最终结果行字段:")
                        print(f"   param_count: {result_row['param_count']}, samples_count: {result_row['samples_count']}")
                        print(f"   best_epoch: {result_row['best_epoch']}, final_lr: {result_row['final_lr']}")
                        print(f"   smape: {result_row['smape']}, gpu_memory_used: {result_row['gpu_memory_used']}")
                        print(f"   是否为DL模型: {is_dl_model}")
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
    total_time = time.time() - start_time
    print(f"\n📊 项目 {project_id} 完成!")
    print(f"✅ 成功: {stats['success']}")
    print(f"❌ 失败: {stats['failed']}")
    print(f"⏱️ 总用时: {total_time:.1f}秒")
    
    return stats

def run_project_experiments_with_checkpoint(project_id, data_file, all_config_files, drive_save_dir, checkpoint_manager, drive_saver):
    """运行单个项目的所有实验（支持断点续训）"""
    print(f"\n{'='*80}")
    print(f"🚀 开始项目 {project_id} 的实验 (断点续训模式)")
    print(f"{'='*80}")
    
    # 获取项目进度
    project_progress = checkpoint_manager.get_project_progress(project_id)
    print(f"📊 项目 {project_id} 当前进度: {project_progress['completed_experiments']}/{project_progress['total_experiments']} ({project_progress['completion_rate']:.1f}%)")
    
    # 获取待执行的实验
    pending_configs = checkpoint_manager.get_pending_experiments(project_id)
    if not pending_configs:
        print(f"✅ 项目 {project_id} 所有实验已完成!")
        return {'success': 0, 'failed': 0, 'errors': []}
    
    print(f"📋 待执行实验: {len(pending_configs)} 个")
    print(f"📁 结果保存到: {drive_save_dir}")
    
    stats = {
        'success': 0,
        'failed': 0,
        'errors': []
    }
    
    start_time = time.time()
    
    # 运行每个待执行的实验
    for i, config_info in enumerate(pending_configs, 1):
        config_name = config_info['name']
        config = config_info['config']
        config_file = config_info['file_path']
        
        print(f"\n🔄 进度: {i}/{len(pending_configs)} - {config_name}")
        
        try:
            # 修改配置文件中的数据路径
            config['data_path'] = data_file
            config['plant_id'] = project_id
            
            # 对于ML模型，移除不应该有的DL参数，但保留ML特有的参数
            if config.get('model') in ['LGBM', 'RF', 'XGB', 'Linear']:
                # ML模型不应该有batch_size等DL参数，但可以有learning_rate（XGB、LGBM）
                if 'train_params' in config:
                    ml_train_params = {}
                    for key, value in config['train_params'].items():
                        # 保留ML模型特有的参数
                        if key in ['learning_rate', 'max_depth', 'n_estimators', 'random_state', 'verbosity']:
                            ml_train_params[key] = value
                    config['train_params'] = ml_train_params
            
            # 创建临时配置文件
            temp_config_file = f"temp_config_{project_id}_{i}.yaml"
            with open(temp_config_file, 'w') as f:
                yaml.dump(config, f)
            
            # 运行实验
            start_exp_time = time.time()
            result = subprocess.run(
                ['python', 'main.py', '--config', temp_config_file],
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )
            duration = time.time() - start_exp_time
            
            # 检查结果
            has_error = "[ERROR]" in result.stdout or result.returncode != 0
            if not has_error and result.returncode == 0:
                stats['success'] += 1
                print(f"✅ 实验成功! 用时: {duration:.1f}秒")
                
                # 解析结果并保存到断点续训系统
                result_data = parse_and_save_experiment_result(
                    project_id, config_name, config, result, duration, drive_saver
                )
                
                # 标记实验为已完成
                checkpoint_manager.mark_experiment_completed(project_id, config_name, result_data)
                
            else:
                stats['failed'] += 1
                error_msg = f"实验失败: 返回码 {result.returncode}"
                stats['errors'].append(error_msg)
                print(f"❌ 实验失败! 返回码: {result.returncode}")
                
                # 保存失败结果
                result_data = {
                    'config_name': config_name,
                    'status': 'failed',
                    'duration': duration,
                    'error_message': result.stderr
                }
                checkpoint_manager.mark_experiment_completed(project_id, config_name, result_data)
                
        except subprocess.TimeoutExpired:
            stats['failed'] += 1
            error_msg = "实验超时 (30分钟)"
            stats['errors'].append(error_msg)
            print(f"⏰ 实验超时: {error_msg}")
            
            # 保存超时结果
            result_data = {
                'config_name': config_name,
                'status': 'timeout',
                'duration': 1800,
                'error_message': 'Experiment timeout (30 minutes)'
            }
            checkpoint_manager.mark_experiment_completed(project_id, config_name, result_data)
            
        except Exception as e:
            stats['failed'] += 1
            error_msg = f"实验异常: {str(e)}"
            stats['errors'].append(error_msg)
            print(f"💥 实验异常: {error_msg}")
            
            # 保存异常结果
            result_data = {
                'config_name': config_name,
                'status': 'error',
                'duration': 0,
                'error_message': str(e)
            }
            checkpoint_manager.mark_experiment_completed(project_id, config_name, result_data)
            
        finally:
            # 清理临时配置文件
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
    
    # 计算总用时
    total_time = time.time() - start_time
    print(f"\n📊 项目 {project_id} 完成!")
    print(f"✅ 成功: {stats['success']}")
    print(f"❌ 失败: {stats['failed']}")
    print(f"⏱️ 总用时: {total_time:.1f}秒")
    
    return stats

def parse_and_save_experiment_result(project_id, config_name, config, result, duration, drive_saver):
    """解析实验结果并保存到断点续训系统"""
    # 从实验输出中提取结果
    result_line = None
    inference_time = 0.0
    param_count = 0
    samples_count = 0
    best_epoch = 0
    final_lr = 0.0
    nrmse = 0.0
    smape = 0.0
    gpu_memory_used = 0
    
    # 解析METRICS行
    for line in result.stdout.split('\n'):
        if "mse=" in line and "rmse=" in line and "mae=" in line and "r_square=" in line:
            result_line = line
        elif "[METRICS]" in line:
            # 使用正则表达式提取所有键值对
            metrics_in_line = re.findall(r'(\w+)=([0-9.-]+)', line)
            for key, value_str in metrics_in_line:
                try:
                    if key == 'inference_time':
                        inference_time = float(value_str)
                    elif key == 'param_count':
                        param_count = int(float(value_str))
                    elif key == 'samples_count':
                        samples_count = int(float(value_str))
                    elif key == 'best_epoch':
                        if value_str.lower() == 'nan':
                            best_epoch = 0
                        else:
                            best_epoch = int(float(value_str))
                    elif key == 'final_lr':
                        if value_str.lower() == 'nan':
                            final_lr = 0.0
                        else:
                            final_lr = float(value_str)
                    elif key == 'nrmse':
                        nrmse = float(value_str)
                    elif key == 'smape':
                        smape = float(value_str)
                    elif key == 'gpu_memory_used':
                        gpu_memory_used = int(float(value_str))
                except Exception as e:
                    print(f"🔍 调试：{key}提取失败: {e}")
    
    # 解析基本指标
    mse = rmse = mae = r_square = 0.0
    if result_line:
        mse_match = re.search(r'mse=([0-9.]+)', result_line)
        rmse_match = re.search(r'rmse=([0-9.]+)', result_line)
        mae_match = re.search(r'mae=([0-9.]+)', result_line)
        r_square_match = re.search(r'r_square=([0-9.]+)', result_line)
        
        if mse_match and rmse_match and mae_match and r_square_match:
            mse = float(mse_match.group(1))
            rmse = float(rmse_match.group(1))
            mae = float(mae_match.group(1))
            r_square = float(r_square_match.group(1))
    
    # 构建结果数据
    result_data = {
        'config_name': config_name,
        'status': 'completed',
        'duration': duration,
        'mae': mae,
        'rmse': rmse,
        'r2': r_square,
        'mape': smape,
        'train_time_sec': duration,
        'inference_time_sec': inference_time,
        'param_count': param_count,
        'samples_count': samples_count,
        'model': config.get('model', ''),
        'model_complexity': config.get('model_complexity', ''),
        'input_category': extract_input_category_from_config_name(config_name),
        'lookback_hours': config.get('past_hours', 24),
        'use_time_encoding': config.get('use_time_encoding', False)
    }
    
    return result_data

def extract_input_category_from_config_name(config_name):
    """从配置名称中提取输入特征类别"""
    if 'PV_plus_NWP_plus' in config_name:
        return 'PV_plus_NWP_plus'
    elif 'PV_plus_NWP' in config_name:
        return 'PV_plus_NWP'
    elif 'PV_plus_HW' in config_name:
        return 'PV_plus_HW'
    elif 'NWP_plus' in config_name and 'PV' not in config_name:
        return 'NWP_plus'
    elif 'NWP' in config_name and 'PV' not in config_name:
        return 'NWP'
    elif 'PV' in config_name and 'plus' not in config_name:
        return 'PV'
    else:
        return 'Unknown'

def main():
    """主函数 - 支持断点续训"""
    print("🌟 SolarPV项目 - 批量实验脚本 (支持断点续训)")
    print("=" * 60)
    
    # 检查Drive挂载
    if not check_drive_mount():
        return
    
    # 硬编码Drive保存路径
    drive_save_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(drive_save_dir, exist_ok=True)
    
    # 初始化断点续训管理器
    print("🔍 初始化断点续训管理器...")
    checkpoint_manager = CheckpointManager(drive_save_dir)
    drive_saver = DriveResultsSaver(drive_save_dir)
    
    # 检查缺失实验状态
    print("📊 检查缺失实验状态...")
    missing_df = checkpoint_manager.get_all_missing_experiments()
    
    if not missing_df.empty:
        total_projects = len(missing_df)
        completed_projects = len(missing_df[missing_df['is_complete'] == True])
        total_expected = missing_df['expected_experiments'].sum()
        total_completed = missing_df['completed_experiments'].sum()
        total_missing = missing_df['missing_experiments'].sum()
        
        print(f"📊 当前状态:")
        print(f"   总项目数: {total_projects}")
        print(f"   已完成项目: {completed_projects}")
        print(f"   总实验数: {total_expected}")
        print(f"   已完成实验: {total_completed}")
        print(f"   缺失实验: {total_missing}")
        print(f"   总体完成率: {total_completed/total_expected*100:.1f}%")
        
        if total_missing > 0:
            print("🔄 将补全缺失的实验")
        else:
            print("🎉 所有实验已完成!")
            return
    else:
        print("🆕 首次运行，将开始全新实验")
    
    # 获取有缺失实验的项目
    projects_with_missing = checkpoint_manager.get_projects_with_missing_experiments()
    if not projects_with_missing:
        print("🎉 所有项目实验已完成!")
        return
    
    print(f"📋 有缺失实验的项目: {len(projects_with_missing)} 个")
    print(f"   项目列表: {projects_with_missing[:10]}{'...' if len(projects_with_missing) > 10 else ''}")
    
    # 显示详细的缺失实验信息
    print("\n📊 详细缺失实验信息:")
    for _, row in missing_df.iterrows():
        if row['missing_experiments'] > 0:
            print(f"   {row['project_id']}: {row['completed_experiments']}/{row['expected_experiments']} ({row['completion_rate']:.1f}%) - 缺失 {row['missing_experiments']} 个")
    
    # 扫描数据文件
    print("\n📁 扫描数据文件...")
    data_files = get_data_files()
    if not data_files:
        print("❌ 未找到任何数据文件")
        return
    
    # 过滤出有缺失实验的项目
    missing_data_files = [(pid, data_file) for pid, data_file in data_files if pid in projects_with_missing]
    print(f"📊 找到 {len(missing_data_files)} 个有缺失实验项目的数据文件")
    
    # 检查配置文件
    print("📁 检查配置文件...")
    all_config_files = get_config_files()
    print(f"📊 找到 {len(all_config_files)} 个配置文件")
    
    # 检查是否需要生成配置文件
    if len(all_config_files) < len(data_files) * 100:
        print("🔧 配置文件不足，正在生成...")
        try:
            result = subprocess.run([
                'python', 'scripts/generate_dynamic_project_configs.py'
            ], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ 配置文件生成完成")
                all_config_files = get_config_files()
                print(f"📊 现在有 {len(all_config_files)} 个配置文件")
            else:
                print(f"❌ 配置文件生成失败: {result.stderr}")
                return
        except Exception as e:
            print(f"❌ 配置文件生成异常: {e}")
            return
    
    print(f"\n🚀 开始补全缺失实验!")
    print(f"📊 有缺失实验的项目数: {len(missing_data_files)}")
    print(f"📊 每项目应有实验数: 340")
    print(f"📊 预计需要补全的实验数: {total_missing}")
    
    # 运行有缺失实验的项目
    total_stats = {'success': 0, 'failed': 0, 'errors': []}
    
    for i, (project_id, data_file) in enumerate(missing_data_files, 1):
        print(f"\n🔄 项目进度: {i}/{len(missing_data_files)}")
        
        # 检查项目缺失实验状态
        missing_info = checkpoint_manager.check_missing_experiments(project_id)
        print(f"📊 项目 {project_id} 状态: {missing_info['completed_experiments']}/{missing_info['expected_experiments']} ({missing_info['completion_rate']:.1f}%) - 缺失 {missing_info['missing_experiments']} 个")
        
        project_stats = run_project_experiments_with_checkpoint(project_id, data_file, all_config_files, drive_save_dir, checkpoint_manager, drive_saver)
        
        # 累计统计
        total_stats['success'] += project_stats['success']
        total_stats['failed'] += project_stats['failed']
        total_stats['errors'].extend(project_stats['errors'])
    
    # 最终统计
    print(f"\n🎉 所有实验完成!")
    print(f"✅ 总成功: {total_stats['success']}")
    print(f"❌ 总失败: {total_stats['failed']}")
    print(f"📁 结果保存在: {drive_save_dir}")
    
    # 生成最终进度报告
    print("\n📊 生成最终进度报告...")
    report_path = checkpoint_manager.save_progress_report()
    print(f"📋 进度报告已保存: {report_path}")
    
    # 显示Drive结果
    if os.path.exists(drive_save_dir):
        csv_files = [f for f in os.listdir(drive_save_dir) if f.endswith('_results.csv')]
        print(f"📊 生成了 {len(csv_files)} 个结果文件")
        for csv_file in csv_files[:5]:  # 显示前5个文件
            file_path = os.path.join(drive_save_dir, csv_file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"   {csv_file}: {len(df)} 行结果")

if __name__ == "__main__":
    main()
