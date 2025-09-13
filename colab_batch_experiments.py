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
                    elif "[METRICS] inference_time=" in line:
                        try:
                            inference_time = float(line.split("inference_time=")[1].split(",")[0])
                            print(f"🔍 调试：提取inference_time={inference_time}")
                        except Exception as e:
                            print(f"🔍 调试：inference_time提取失败: {e}")
                    elif "[METRICS]" in line and "param_count=" in line:
                        try:
                            param_count = int(line.split("param_count=")[1].split(",")[0])
                            print(f"🔍 调试：提取param_count={param_count}")
                        except Exception as e:
                            print(f"🔍 调试：param_count提取失败: {e}")
                    elif "[METRICS]" in line and "samples_count=" in line:
                        try:
                            samples_count = int(line.split("samples_count=")[1].split()[0])
                            print(f"🔍 调试：提取samples_count={samples_count}")
                        except Exception as e:
                            print(f"🔍 调试：samples_count提取失败: {e}")
                    elif "[METRICS] best_epoch=" in line:
                        try:
                            best_epoch = int(line.split("best_epoch=")[1].split(",")[0])
                            print(f"🔍 调试：提取best_epoch={best_epoch}")
                        except Exception as e:
                            print(f"🔍 调试：best_epoch提取失败: {e}")
                    elif "[METRICS]" in line and "final_lr=" in line:
                        try:
                            final_lr = float(line.split("final_lr=")[1].split()[0])
                            print(f"🔍 调试：提取final_lr={final_lr}")
                        except Exception as e:
                            print(f"🔍 调试：final_lr提取失败: {e}")
                    elif "[METRICS] nrmse=" in line:
                        try:
                            nrmse = float(line.split("nrmse=")[1].split(",")[0])
                            print(f"🔍 调试：提取nrmse={nrmse}")
                        except Exception as e:
                            print(f"🔍 调试：nrmse提取失败: {e}")
                    elif "[METRICS]" in line and "smape=" in line:
                        try:
                            smape = float(line.split("smape=")[1].split(",")[0])
                            print(f"🔍 调试：提取smape={smape}")
                        except Exception as e:
                            print(f"🔍 调试：smape提取失败: {e}")
                    elif "[METRICS]" in line and "gpu_memory_used=" in line:
                        try:
                            gpu_memory_used = int(line.split("gpu_memory_used=")[1].split()[0])
                            print(f"🔍 调试：提取gpu_memory_used={gpu_memory_used}")
                        except Exception as e:
                            print(f"🔍 调试：gpu_memory_used提取失败: {e}")
                
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
                            'epochs': config.get('epochs', 50 if complexity == 'high' else 15),
                            'batch_size': config.get('train_params', {}).get('batch_size', 32),
                            'learning_rate': config.get('train_params', {}).get('learning_rate', 0.001),
                            'train_time_sec': round(duration, 4),
                            'inference_time_sec': inference_time,
                            'param_count': param_count,
                            'samples_count': samples_count,
                            'best_epoch': best_epoch,
                            'final_lr': final_lr,
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

def main():
    """主函数"""
    print("🌟 SolarPV项目 - 批量实验脚本")
    print("=" * 50)
    
    # 检查Drive挂载
    if not check_drive_mount():
        return
    
    # 扫描数据文件
    print("📁 扫描数据文件...")
    data_files = get_data_files()
    if not data_files:
        print("❌ 未找到任何数据文件")
        return
    
    print(f"📊 找到 {len(data_files)} 个项目: {[pid for pid, _ in data_files[:10]]}...")
    
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
    
    # 硬编码Drive保存路径
    drive_save_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(drive_save_dir, exist_ok=True)
    
    print(f"\n🚀 开始批量实验!")
    print(f"📊 总项目数: {len(data_files)}")
    print(f"📊 每项目实验数: 340")
    print(f"📊 总实验数: {len(data_files) * 340}")
    
    # 运行所有项目
    total_stats = {'success': 0, 'failed': 0, 'errors': []}
    
    for i, (project_id, data_file) in enumerate(data_files, 1):
        print(f"\n🔄 项目进度: {i}/{len(data_files)}")
        
        project_stats = run_project_experiments(project_id, data_file, all_config_files, drive_save_dir)
        
        # 累计统计
        total_stats['success'] += project_stats['success']
        total_stats['failed'] += project_stats['failed']
        total_stats['errors'].extend(project_stats['errors'])
    
    # 最终统计
    print(f"\n🎉 所有实验完成!")
    print(f"✅ 总成功: {total_stats['success']}")
    print(f"❌ 总失败: {total_stats['failed']}")
    print(f"📁 结果保存在: {drive_save_dir}")
    
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
