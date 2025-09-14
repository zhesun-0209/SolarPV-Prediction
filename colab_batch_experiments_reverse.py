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
                    if config_file.endswith('.yaml'):
                        all_config_files.append(os.path.join(project_path, config_file))
    
    return all_config_files

def create_project_csv(project_id, drive_path):
    """为项目创建CSV文件"""
    csv_file = os.path.join(drive_path, f"{project_id}.csv")
    
    if not os.path.exists(csv_file):
        # 创建CSV文件头
        columns = [
            'model', 'use_pv', 'use_hist_weather', 'use_forecast', 'weather_category',
            'use_time_encoding', 'past_days', 'model_complexity', 'epochs', 'batch_size',
            'learning_rate', 'train_time_sec', 'inference_time_sec', 'param_count',
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

def get_completed_experiments(project_id, drive_path):
    """获取已完成的实验"""
    csv_file = os.path.join(drive_path, f"{project_id}.csv")
    completed_experiments = set()
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if 'config_name' in df.columns:
                completed_experiments = set(df['config_name'].tolist())
            else:
                # 如果没有config_name列，使用行数判断
                completed_experiments = {f"experiment_{i}" for i in range(len(df))}
            print(f"📊 发现 {len(completed_experiments)} 个已完成实验")
        except Exception as e:
            print(f"⚠️ 无法读取现有结果文件: {e}")
    
    return completed_experiments

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
        
        # 运行实验
        cmd = ['python', 'main.py', '--config', temp_config]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        # 清理临时文件
        if os.path.exists(temp_config):
            os.remove(temp_config)
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        else:
            return False, result.stdout, result.stderr
            
    except Exception as e:
        return False, "", str(e)

def parse_experiment_output(output, config_file):
    """解析实验输出，提取结果"""
    try:
        # 提取基本指标
        mse_match = re.search(r'mse=([0-9.]+)', output)
        rmse_match = re.search(r'rmse=([0-9.]+)', output)
        mae_match = re.search(r'mae=([0-9.]+)', output)
        r_square_match = re.search(r'r_square=([0-9.]+)', output)
        
        # 提取训练时间
        train_time_match = re.search(r'训练用时: ([0-9.]+)秒', output)
        train_time = float(train_time_match.group(1)) if train_time_match else 0.0
        
        # 提取推理时间
        inference_time_match = re.search(r'推理用时: ([0-9.]+)秒', output)
        inference_time = float(inference_time_match.group(1)) if inference_time_match else 0.0
        
        # 提取参数数量
        param_count_match = re.search(r'参数数量: (\d+)', output)
        param_count = int(param_count_match.group(1)) if param_count_match else 0
        
        # 提取样本数量
        samples_count_match = re.search(r'样本数量: (\d+)', output)
        samples_count = int(samples_count_match.group(1)) if samples_count_match else 0
        
        # 提取最佳轮次
        best_epoch_match = re.search(r'最佳轮次: (\d+)', output)
        best_epoch = int(best_epoch_match.group(1)) if best_epoch_match else 0
        
        # 提取最终学习率
        final_lr_match = re.search(r'最终学习率: ([0-9.e-]+)', output)
        final_lr = float(final_lr_match.group(1)) if final_lr_match else 0.0
        
        # 提取NRMSE
        nrmse_match = re.search(r'NRMSE: ([0-9.]+)', output)
        nrmse = float(nrmse_match.group(1)) if nrmse_match else 0.0
        
        # 提取SMAPE
        smape_match = re.search(r'SMAPE: ([0-9.]+)', output)
        smape = float(smape_match.group(1)) if smape_match else 0.0
        
        # 提取GPU内存使用
        gpu_memory_match = re.search(r'GPU内存: ([0-9.]+)MB', output)
        gpu_memory_used = float(gpu_memory_match.group(1)) if gpu_memory_match else 0.0
        
        # 从配置文件名解析参数
        config_filename = os.path.basename(config_file)
        parts = config_filename.replace('.yaml', '').split('_')
        
        model_name = parts[0]
        complexity = parts[1]
        
        # 解析输入类别和时间编码
        if len(parts) > 2:
            if parts[2] == 'NWP' and len(parts) > 3 and parts[3] == 'plus':
                input_category = 'NWP_plus'
                lookback_hours = parts[4].replace('h', '') if len(parts) > 4 else '24'
                time_encoding = parts[5] == 'TE' if len(parts) > 5 else False
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
            'train_time_sec': train_time,
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
        completed_experiments = get_completed_experiments(project_id, drive_path)
        
        print(f"📊 项目 {project_id}: 将运行 {len(project_configs)} 个实验")
        print(f"📁 结果保存到: {drive_path}")
        
        # 运行实验
        for exp_idx, config_file in enumerate(project_configs, 1):
            config_name = os.path.basename(config_file)
            
            # 跳过已完成的实验
            if config_name in completed_experiments:
                print(f"⏭️ 跳过已完成实验: {config_name}")
                continue
                
            print(f"\n🔄 进度: {exp_idx}/{len(project_configs)} - {config_name}")
            
            # 运行实验
            success, stdout, stderr = run_experiment(config_file, data_file, project_id)
            total_experiments += 1
            
            if success:
                print(f"✅ 实验成功!")
                successful_experiments += 1
                
                # 解析结果
                result_row = parse_experiment_output(stdout, config_file)
                if result_row:
                    # 保存结果到CSV
                    csv_file = os.path.join(drive_path, f"{project_id}.csv")
                    
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
