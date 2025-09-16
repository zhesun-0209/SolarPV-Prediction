#!/usr/bin/env python3
"""
SolarPV项目 - 批量实验脚本 (逆序版本)
在Colab上运行100个项目的完整实验，保存结果到Google Drive
从最大项目ID开始，逆序训练
"""

import os
import sys
import time
import subprocess
import yaml
import pandas as pd
from pathlib import Path
import re

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
    
    csv_files.sort(key=lambda x: int(x[0]), reverse=True)  # 逆序排序
    return csv_files

def get_config_files():
    """获取所有配置文件"""
    config_dir = "config/projects"
    all_config_files = []
    
    if os.path.exists(config_dir):
        for project_dir in os.listdir(config_dir):
            project_path = os.path.join(config_dir, project_dir)
            if os.path.isdir(project_path):
                for config_file in os.listdir(project_path):
                    if config_file.endswith('.yaml') and config_file != 'config_index.yaml':
                        all_config_files.append(os.path.join(project_path, config_file))
    
    return all_config_files

def create_project_csv(project_id, drive_path):
    """为项目创建CSV文件"""
    csv_file = os.path.join(drive_path, f"{project_id}_results.csv")
    
    if not os.path.exists(csv_file):
        # 创建CSV文件头
        columns = [
            'model', 'use_pv', 'use_hist_weather', 'use_forecast', 'weather_category',
            'use_time_encoding', 'past_days', 'model_complexity', 'epochs', 'batch_size',
            'learning_rate', 'use_ideal_nwp', 'train_time_sec', 'inference_time_sec', 'param_count',
            'samples_count', 'best_epoch', 'final_lr', 'mse', 'rmse', 'mae', 'nrmse',
            'r_square', 'smape', 'gpu_memory_used'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)
        print(f"📄 创建项目CSV文件: {csv_file}")
        return True
    else:
        print(f"📄 项目CSV文件已存在: {csv_file}")
        return True

def get_completed_experiments_count(project_id, drive_path):
    """获取已完成的实验数量"""
    csv_file = os.path.join(drive_path, f"{project_id}_results.csv")
    completed_count = 0
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            completed_count = len(df)
            print(f"📊 发现 {completed_count} 个已完成实验")
        except Exception as e:
            print(f"⚠️ 无法读取现有结果文件: {e}")
            completed_count = 0
    
    return completed_count

def get_completed_experiment_configs(project_id, drive_path):
    """获取已完成的实验配置名称列表"""
    csv_file = os.path.join(drive_path, f"{project_id}_results.csv")
    completed_configs = set()
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            # 从CSV中提取配置信息，重建配置名称
            for _, row in df.iterrows():
                # 根据CSV中的参数重建配置名称
                model = row['model']
                complexity = row['model_complexity']
                use_pv = row['use_pv']
                use_hist_weather = row['use_hist_weather']
                use_forecast = row['use_forecast']
                use_ideal_nwp = row.get('use_ideal_nwp', False)
                past_days = row['past_days']
                use_time_encoding = row['use_time_encoding']
                
                # 确定输入类别
                if use_pv and not use_hist_weather and not use_forecast:
                    input_cat = 'PV'
                elif use_pv and not use_hist_weather and use_forecast and not use_ideal_nwp:
                    input_cat = 'PV_plus_NWP'
                elif use_pv and not use_hist_weather and use_forecast and use_ideal_nwp:
                    input_cat = 'PV_plus_NWP_plus'
                elif use_pv and use_hist_weather and not use_forecast:
                    input_cat = 'PV_plus_HW'
                elif not use_pv and not use_hist_weather and use_forecast and not use_ideal_nwp:
                    input_cat = 'NWP'
                elif not use_pv and not use_hist_weather and use_forecast and use_ideal_nwp:
                    input_cat = 'NWP_plus'
                else:
                    continue  # 跳过无法识别的组合
                
                # 确定回看小时数
                lookback_hours = past_days * 24
                
                # 确定时间编码后缀
                te_suffix = 'TE' if use_time_encoding else 'noTE'
                
                # 重建配置名称
                config_name = f"{model}_{complexity}_{input_cat}_{lookback_hours}h_{te_suffix}"
                completed_configs.add(config_name)
                
        except Exception as e:
            print(f"⚠️ 无法读取现有结果文件: {e}")
    
    return completed_configs

def run_experiment(config_file, data_file, project_id):
    """运行单个实验"""
    try:
        # 加载配置文件
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 修改配置文件中的data_path
        config['data_path'] = data_file
        config['plant_id'] = project_id
        
        # 创建临时配置文件
        temp_config = f"temp_config_{project_id}_{int(time.time())}.yaml"
        with open(temp_config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 运行实验并记录时间
        cmd = ['python', 'main.py', '--config', temp_config]
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        duration = time.time() - start_time
        
        # 清理临时文件
        if os.path.exists(temp_config):
            os.remove(temp_config)
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr, duration, config
        else:
            return False, result.stdout, result.stderr, duration, config
            
    except Exception as e:
        return False, "", str(e), 0.0, {}

def parse_experiment_output(output, config_file, duration, config):
    """解析实验输出，提取结果"""
    try:
        # 提取基本指标
        mse_match = re.search(r'mse=([0-9.]+)', output)
        rmse_match = re.search(r'rmse=([0-9.]+)', output)
        mae_match = re.search(r'mae=([0-9.]+)', output)
        r_square_match = re.search(r'r_square=([0-9.]+)', output)
        
        # 初始化额外字段
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
        for line in output.split('\n'):
            if "[METRICS]" in line:
                print(f"   找到METRICS行: {line}")
        
        # 使用METRICS标签提取额外信息（与colab_batch_experiments.py保持一致）
        for line in output.split('\n'):
            if "[METRICS]" in line:
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
        
        # 从配置文件名解析参数
        config_filename = os.path.basename(config_file)
        parts = config_filename.replace('.yaml', '').split('_')
        
        model_name = parts[0]
        complexity = parts[1]
        
        # 解析输入类别和时间编码
        if len(parts) > 2:
            # 处理包含下划线的输入类别名称
            if parts[2] == 'PV' and len(parts) > 3:
                if parts[3] == 'plus' and len(parts) > 4:
                    if parts[4] == 'NWP' and len(parts) > 5 and parts[5] == 'plus':
                        input_category = 'PV_plus_NWP_plus'
                        lookback_hours = parts[6].replace('h', '') if len(parts) > 6 else '24'
                        time_encoding = parts[7] == 'TE' if len(parts) > 7 else False
                    elif parts[4] == 'NWP':
                        input_category = 'PV_plus_NWP'
                        lookback_hours = parts[5].replace('h', '') if len(parts) > 5 else '24'
                        time_encoding = parts[6] == 'TE' if len(parts) > 6 else False
                    elif parts[4] == 'HW':
                        input_category = 'PV_plus_HW'
                        lookback_hours = parts[5].replace('h', '') if len(parts) > 5 else '24'
                        time_encoding = parts[6] == 'TE' if len(parts) > 6 else False
                    else:
                        input_category = 'PV'
                        lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                        time_encoding = parts[4] == 'TE' if len(parts) > 4 else False
                else:
                    input_category = 'PV'
                    lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                    time_encoding = parts[4] == 'TE' if len(parts) > 4 else False
            elif parts[2] == 'NWP' and len(parts) > 3 and parts[3] == 'plus':
                input_category = 'NWP_plus'
                lookback_hours = parts[4].replace('h', '') if len(parts) > 4 else '24'
                time_encoding = parts[5] == 'TE' if len(parts) > 5 else False
            elif parts[2] == 'NWP':
                input_category = 'NWP'
                lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                time_encoding = parts[4] == 'TE' if len(parts) > 4 else False
            else:
                input_category = parts[2]
                lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                time_encoding = parts[4] == 'TE' if len(parts) > 4 else False
        else:
            input_category = 'unknown'
            lookback_hours = '24'
            time_encoding = False
        
        # 根据input_category确定其他参数
        use_pv = input_category in ['PV', 'PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW']
        use_hist_weather = input_category in ['PV_plus_HW']
        use_forecast = input_category in ['PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW', 'NWP', 'NWP_plus']
        
        # 计算past_days
        past_days = int(int(lookback_hours) / 24) if lookback_hours.isdigit() else 1
        
        # 判断模型类型
        is_dl_model = model_name in ['Transformer', 'LSTM', 'GRU', 'TCN']
        has_learning_rate = model_name in ['XGB', 'LGBM']
        
        # 创建结果行
        result_row = {
            'model': model_name,
            'use_pv': use_pv,
            'use_hist_weather': use_hist_weather,
            'use_forecast': use_forecast,
            'weather_category': 'all_weather',
            'use_time_encoding': time_encoding,
            'past_days': past_days,
            'model_complexity': complexity,
            'epochs': 50 if complexity == 'high' else 15 if is_dl_model else 0,
            'batch_size': 32 if is_dl_model else 0,
            'learning_rate': 0.001 if has_learning_rate else 0.0,
            'use_ideal_nwp': config.get('use_ideal_nwp', False),
            'train_time_sec': round(duration, 4),  # 使用传入的duration参数
            'inference_time_sec': inference_time,
            'param_count': param_count,
            'samples_count': samples_count,
            'best_epoch': best_epoch if is_dl_model else 0,
            'final_lr': final_lr if is_dl_model else 0.0,
            'mse': float(mse_match.group(1)) if mse_match else 0.0,
            'rmse': float(rmse_match.group(1)) if rmse_match else 0.0,
            'mae': float(mae_match.group(1)) if mae_match else 0.0,
            'nrmse': nrmse,
            'r_square': float(r_square_match.group(1)) if r_square_match else 0.0,
            'smape': smape,
            'gpu_memory_used': gpu_memory_used
        }
        
        return result_row
        
    except Exception as e:
        print(f"❌ 解析实验输出失败: {e}")
        return None

def main():
    """主函数"""
    print("🌟 SolarPV项目 - 批量实验脚本 (逆序版本)")
    print("=" * 50)
    
    # 检查Google Drive
    if not check_drive_mount():
        return
    
    # 设置路径
    drive_path = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(drive_path, exist_ok=True)
    
    # 获取数据文件
    print("📁 扫描数据文件...")
    data_files = get_data_files()
    print(f"📊 找到 {len(data_files)} 个项目: {[pid for pid, _ in data_files[:10]]}...")
    
    # 获取配置文件
    print("📁 检查配置文件...")
    config_files = get_config_files()
    print(f"📊 找到 {len(config_files)} 个配置文件")
    
    # 检查是否需要生成配置文件
    if len(config_files) < len(data_files) * 100:
        print("🔧 配置文件不足，正在生成...")
        try:
            result = subprocess.run([
                'python', 'scripts/generate_dynamic_project_configs.py'
            ], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ 配置文件生成完成")
                config_files = get_config_files()
                print(f"📊 现在有 {len(config_files)} 个配置文件")
            else:
                print(f"❌ 配置文件生成失败: {result.stderr}")
                return
        except Exception as e:
            print(f"❌ 配置文件生成异常: {e}")
            return
    
    if not data_files or not config_files:
        print("❌ 没有找到数据文件或配置文件")
        return
    
    print("🚀 开始批量实验!")
    print(f"📊 总项目数: {len(data_files)}")
    print(f"📊 每项目实验数: {len(config_files) // len(data_files)}")
    print(f"📊 总实验数: {len(data_files) * (len(config_files) // len(data_files))}")
    
    total_experiments = 0
    successful_experiments = 0
    failed_experiments = 0
    
    # 遍历每个项目
    for project_idx, (project_id, data_file) in enumerate(data_files, 1):
        print(f"\n{'='*80}")
        print(f"🚀 开始项目 {project_id} 的实验 (逆序: {project_idx}/{len(data_files)})")
        print(f"{'='*80}")
        
        # 创建项目CSV文件
        if not create_project_csv(project_id, drive_path):
            print(f"❌ 无法为项目 {project_id} 创建CSV文件")
            continue
        
        # 获取该项目的配置文件
        project_configs = [cf for cf in config_files if f"/{project_id}/" in cf]
        
        # 检查已完成的实验
        completed_count = get_completed_experiments_count(project_id, drive_path)
        completed_configs = get_completed_experiment_configs(project_id, drive_path)
        
        print(f"📊 项目 {project_id}: 将运行 {len(project_configs)} 个实验")
        print(f"📁 结果保存到: {drive_path}")
        print(f"📊 已完成实验: {len(completed_configs)} 个")
        
        # 运行实验
        for exp_idx, config_file in enumerate(project_configs, 1):
            config_name = os.path.basename(config_file)
            # 移除.yaml后缀获取配置名称
            config_name_without_ext = config_name.replace('.yaml', '')
            
            # 跳过已完成的实验（基于配置名称判断）
            if config_name_without_ext in completed_configs:
                print(f"⏭️ 跳过已完成实验: {config_name}")
                continue
                
            print(f"\n🔄 进度: {exp_idx}/{len(project_configs)} - {config_name}")
            
            # 运行实验
            success, stdout, stderr, duration, config = run_experiment(config_file, data_file, project_id)
            total_experiments += 1
            
            if success:
                print(f"✅ 实验成功! 用时: {duration:.1f}秒")
                successful_experiments += 1
                
                # 解析结果
                result_row = parse_experiment_output(stdout, config_file, duration, config)
                if result_row:
                    # 保存结果到CSV
                    csv_file = os.path.join(drive_path, f"{project_id}_results.csv")
                    
                    # 读取现有CSV
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                    else:
                        df = pd.DataFrame()
                    
                    # 添加新行
                    new_row_df = pd.DataFrame([result_row])
                    df = pd.concat([df, new_row_df], ignore_index=True)
                    
                    # 保存CSV
                    df.to_csv(csv_file, index=False)
                    print(f"💾 结果已保存到: {csv_file}")
                    print(f"📊 CSV文件当前行数: {len(df)}")
                    print(f"📊 最新实验: {result_row['model']} - {result_row['mse']:.4f}")
                    print(f"🔍 解析的配置信息:")
                    print(f"   模型: {result_row['model']}, 复杂度: {result_row['model_complexity']}")
                    print(f"   时间编码: {result_row['use_time_encoding']}")
                    print(f"   PV: {result_row['use_pv']}, 历史天气: {result_row['use_hist_weather']}, 预测天气: {result_row['use_forecast']}")
                    print(f"🔍 提取的额外字段:")
                    print(f"   推理时间: {result_row['inference_time_sec']}s, 参数数量: {result_row['param_count']}, 样本数量: {result_row['samples_count']}")
                    print(f"   最佳轮次: {result_row['best_epoch']}, 最终学习率: {result_row['final_lr']}")
                    print(f"   NRMSE: {result_row['nrmse']}, SMAPE: {result_row['smape']}, GPU内存: {result_row['gpu_memory_used']}MB")
                else:
                    print("⚠️ 无法解析实验结果")
            else:
                print(f"❌ 实验失败!")
                print(f"   错误: {stderr}")
                failed_experiments += 1
        
        print(f"✅ 项目 {project_id} 完成!")
    
    # 总结
    print(f"\n{'='*80}")
    print("🎉 批量实验完成!")
    print(f"📊 总实验数: {total_experiments}")
    print(f"✅ 成功: {successful_experiments}")
    print(f"❌ 失败: {failed_experiments}")
    print(f"📁 结果保存在: {drive_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
