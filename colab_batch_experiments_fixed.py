#!/usr/bin/env python3
"""
SolarPV项目 - 批量实验脚本 (修复版本)
在Colab上运行100个项目的完整实验，保存结果到Google Drive
修复了临时文件冲突、超时设置等问题
"""

import os
import sys
import time
import subprocess
import yaml
import pandas as pd
from pathlib import Path
import re
import uuid

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
    
    csv_files.sort(key=lambda x: int(x[0]))  # 正序排序
    return csv_files

def get_config_files():
    """获取所有配置文件"""
    config_dir = "config/projects"
    all_config_files = []
    
    if os.path.exists(config_dir):
        for project_dir in sorted(os.listdir(config_dir)):
            project_path = os.path.join(config_dir, project_dir)
            if os.path.isdir(project_path):
                for config_file in sorted(os.listdir(project_path)):
                    if config_file.endswith('.yaml'):
                        all_config_files.append(os.path.join(project_path, config_file))
    
    return all_config_files

def get_completed_experiments(drive_path):
    """获取已完成的实验列表"""
    completed_configs = set()
    
    try:
        results_file = os.path.join(drive_path, "SolarPV_Results", "all_results.csv")
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            if 'config_file' in df.columns:
                completed_configs = set(df['config_file'].tolist())
                print(f"📊 发现 {len(completed_configs)} 个已完成的实验")
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
        
        # 创建临时配置文件（使用UUID避免冲突）
        temp_config = f"temp_config_{project_id}_{uuid.uuid4().hex[:8]}.yaml"
        with open(temp_config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 运行实验并记录时间
        cmd = ['python', 'main.py', '--config', temp_config]
        start_time = time.time()
        
        # 增加超时时间到60分钟
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        duration = time.time() - start_time
        
        # 清理临时文件
        if os.path.exists(temp_config):
            os.remove(temp_config)
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr, duration, config
        else:
            return False, result.stdout, result.stderr, duration, config
            
    except subprocess.TimeoutExpired:
        # 处理超时
        if os.path.exists(temp_config):
            os.remove(temp_config)
        return False, "", "实验超时（60分钟）", 3600.0, {}
    except Exception as e:
        return False, "", str(e), 0.0, {}

def parse_experiment_output(output, config_file, duration, config):
    """解析实验输出，提取结果"""
    try:
        # 提取模型信息
        model_name = config.get('model', 'Unknown')
        complexity = config.get('model_complexity', 'unknown')
        
        # 提取训练参数
        use_pv = config.get('use_pv', False)
        use_hist_weather = config.get('use_hist_weather', False)
        use_forecast = config.get('use_forecast', False)
        weather_category = config.get('weather_category', 'none')
        time_encoding = config.get('use_time_encoding', False)
        past_days = config.get('past_days', 1)
        use_ideal_nwp = config.get('use_ideal_nwp', False)
        
        # 确定输入类别
        if use_pv and use_hist_weather:
            input_category = 'PV_plus_HW'
        elif use_pv and use_forecast and use_ideal_nwp:
            input_category = 'PV_plus_NWP_plus'
        elif use_pv and use_forecast:
            input_category = 'PV_plus_NWP'
        elif use_pv:
            input_category = 'PV'
        elif use_forecast and use_ideal_nwp:
            input_category = 'NWP_plus'
        elif use_forecast:
            input_category = 'NWP'
        else:
            input_category = 'Unknown'
        
        # 判断是否为深度学习模型
        is_dl_model = model_name in ['LSTM', 'GRU', 'Transformer', 'TCN']
        has_learning_rate = is_dl_model or model_name in ['XGB', 'LGBM']
        
        # 提取指标（使用正确的格式：mse=0.1234）
        mse_match = re.search(r'mse=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', output)
        rmse_match = re.search(r'rmse=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', output)
        mae_match = re.search(r'mae=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', output)
        r_square_match = re.search(r'r_square=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', output)
        
        # 调试：显示匹配结果
        print(f"🔍 调试: MSE匹配: {mse_match.group(1) if mse_match else 'None'}")
        print(f"🔍 调试: RMSE匹配: {rmse_match.group(1) if rmse_match else 'None'}")
        print(f"🔍 调试: MAE匹配: {mae_match.group(1) if mae_match else 'None'}")
        print(f"🔍 调试: R²匹配: {r_square_match.group(1) if r_square_match else 'None'}")
        
        # 计算其他指标
        mse = float(mse_match.group(1)) if mse_match else 0.0
        rmse = float(rmse_match.group(1)) if rmse_match else 0.0
        mae = float(mae_match.group(1)) if mae_match else 0.0
        r_square = float(r_square_match.group(1)) if r_square_match else 0.0
        
        # 计算NRMSE和SMAPE
        nrmse = (rmse / (mae + 1e-8)) * 100 if mae > 0 else 0.0
        smape = (2 * mae / (mae + 1e-8)) * 100 if mae > 0 else 0.0
        
        # 提取训练信息
        best_epoch = 0
        final_lr = 0.0
        if is_dl_model:
            epoch_match = re.search(r'Best epoch: (\d+)', output)
            lr_match = re.search(r'Final LR: ([\d.]+)', output)
            best_epoch = int(epoch_match.group(1)) if epoch_match else 0
            final_lr = float(lr_match.group(1)) if lr_match else 0.0
        
        # 提取参数数量
        param_match = re.search(r'Total parameters: ([\d,]+)', output)
        param_count = int(param_match.group(1).replace(',', '')) if param_match else 0
        
        # 提取样本数量
        samples_match = re.search(r'Training samples: (\d+)', output)
        samples_count = int(samples_match.group(1)) if samples_match else 0
        
        # 提取推理时间
        inference_match = re.search(r'Inference time: ([\d.]+)s', output)
        inference_time = float(inference_match.group(1)) if inference_match else 0.0
        
        # 提取GPU内存使用
        gpu_memory_match = re.search(r'GPU memory used: ([\d.]+)MB', output)
        gpu_memory_used = float(gpu_memory_match.group(1)) if gpu_memory_match else 0.0
        
        # 创建结果行
        result_row = {
            'model': model_name,
            'use_pv': use_pv,
            'use_hist_weather': use_hist_weather,
            'use_forecast': use_forecast,
            'weather_category': weather_category,
            'use_time_encoding': time_encoding,
            'past_days': past_days,
            'model_complexity': complexity,
            'epochs': config.get('epochs', 80 if complexity == 'high' else 50) if is_dl_model else 0,
            'batch_size': config.get('train_params', {}).get('batch_size', 64) if is_dl_model else 0,
            'learning_rate': config.get('train_params', {}).get('learning_rate', 0.001) if has_learning_rate else 0.0,
            'use_ideal_nwp': use_ideal_nwp,
            'input_category': input_category,
            'train_time_sec': round(duration, 4),
            'inference_time_sec': inference_time,
            'param_count': param_count,
            'samples_count': samples_count,
            'best_epoch': best_epoch if is_dl_model else 0,
            'final_lr': final_lr if is_dl_model else 0.0,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'nrmse': nrmse,
            'r_square': r_square,
            'smape': smape,
            'gpu_memory_used': gpu_memory_used,
            'config_file': os.path.basename(config_file)
        }
        
        return result_row
        
    except Exception as e:
        print(f"⚠️ 解析实验结果失败: {e}")
        return None

def save_results_to_drive(results, drive_path):
    """保存结果到Google Drive"""
    try:
        print(f"🔍 调试: 准备保存 {len(results)} 个结果到 {drive_path}")
        
        results_dir = os.path.join(drive_path, "SolarPV_Results")
        print(f"🔍 调试: 结果目录: {results_dir}")
        
        # 确保目录存在
        os.makedirs(results_dir, exist_ok=True)
        print(f"🔍 调试: 目录创建成功: {os.path.exists(results_dir)}")
        
        # 保存到CSV
        results_file = os.path.join(results_dir, "all_results.csv")
        print(f"🔍 调试: CSV文件路径: {results_file}")
        
        if os.path.exists(results_file):
            print(f"🔍 调试: 读取现有CSV文件")
            # 读取现有结果
            existing_df = pd.read_csv(results_file)
            print(f"🔍 调试: 现有结果数量: {len(existing_df)}")
            new_df = pd.DataFrame(results)
            print(f"🔍 调试: 新结果数量: {len(new_df)}")
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"🔍 调试: 合并后结果数量: {len(combined_df)}")
        else:
            print(f"🔍 调试: 创建新的CSV文件")
            combined_df = pd.DataFrame(results)
            print(f"🔍 调试: 新结果数量: {len(combined_df)}")
        
        # 保存CSV
        combined_df.to_csv(results_file, index=False)
        print(f"✅ 结果已保存到: {results_file}")
        print(f"🔍 调试: 文件大小: {os.path.getsize(results_file)} 字节")
        
        # 保存到Excel
        excel_file = os.path.join(results_dir, "all_results.xlsx")
        combined_df.to_excel(excel_file, index=False)
        print(f"✅ Excel结果已保存到: {excel_file}")
        print(f"🔍 调试: Excel文件大小: {os.path.getsize(excel_file)} 字节")
        
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
        import traceback
        print(f"🔍 调试: 详细错误信息: {traceback.format_exc()}")

def main():
    """主函数"""
    print("🌟 SolarPV项目 - 批量实验脚本 (修复版本)")
    print("=" * 60)
    
    # 检查Google Drive
    if not check_drive_mount():
        return
    
    drive_path = "/content/drive/MyDrive"
    
    # 获取数据文件
    print("📁 扫描数据文件...")
    data_files = get_data_files()
    if not data_files:
        print("❌ 没有找到数据文件")
        return
    
    print(f"📊 找到 {len(data_files)} 个数据文件")
    
    # 获取配置文件
    print("📁 扫描配置文件...")
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
    
    # 获取已完成的实验
    completed_configs = get_completed_experiments(drive_path)
    
    # 开始实验
    all_results = []
    total_experiments = 0
    completed_experiments = 0
    failed_experiments = 0
    
    for project_idx, (project_id, data_file) in enumerate(data_files, 1):
        print(f"\n🚀 开始项目 {project_id} 的实验 ({project_idx}/{len(data_files)})")
        print(f"📁 数据文件: {data_file}")
        
        # 获取该项目的配置文件
        project_configs = sorted([cf for cf in config_files if f"/{project_id}/" in cf])
        print(f"📊 找到 {len(project_configs)} 个配置文件")
        
        if not project_configs:
            print(f"⚠️ 项目 {project_id} 没有配置文件，跳过")
            continue
        
        # 显示一些已完成的实验示例
        if completed_configs:
            sample_completed = list(completed_configs)[:5]
            print(f"🔍 已完成实验示例: {sample_completed}")
        
        project_results = []
        skipped_count = 0
        
        for config_file in project_configs:
            config_name = os.path.basename(config_file)
            total_experiments += 1
            
            # 检查是否已完成
            if config_name in completed_configs:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"⏭️ 跳过已完成实验: {config_name}")
                continue
            
            print(f"\n🔄 运行实验: {config_name}")
            
            # 运行实验
            success, stdout, stderr, duration, config = run_experiment(config_file, data_file, project_id)
            
            if success:
                # 解析结果
                print(f"🔍 调试: 开始解析实验结果: {config_name}")
                result_row = parse_experiment_output(stdout, config_file, duration, config)
                if result_row:
                    project_results.append(result_row)
                    completed_experiments += 1
                    print(f"✅ 实验完成: {config_name} ({duration:.1f}s) - MSE: {result_row['mse']:.4f}")
                    print(f"🔍 调试: 解析成功，结果字段: {list(result_row.keys())}")
                else:
                    failed_experiments += 1
                    print(f"⚠️ 无法解析实验结果: {config_name}")
                    print(f"🔍 调试: 实验输出前500字符: {stdout[:500]}")
            else:
                failed_experiments += 1
                print(f"❌ 实验失败: {config_name}")
                print(f"   错误: {stderr}")
                print(f"🔍 调试: 标准输出: {stdout[:200]}")
        
        if skipped_count > 5:
            print(f"⏭️ ... 还有 {skipped_count - 5} 个已完成的实验被跳过")
        
        # 保存项目结果
        if project_results:
            save_results_to_drive(project_results, drive_path)
            all_results.extend(project_results)
            print(f"💾 项目 {project_id} 完成，保存了 {len(project_results)} 个结果")
        
        print(f"📊 项目 {project_id} 统计:")
        print(f"   总实验: {len(project_configs)}")
        print(f"   跳过: {skipped_count}")
        print(f"   完成: {len(project_results)}")
        print(f"   失败: {len(project_configs) - skipped_count - len(project_results)}")
    
    # 最终统计
    print("\n" + "=" * 60)
    print("🎉 所有实验完成！")
    print(f"📊 总实验数: {total_experiments}")
    print(f"✅ 完成: {completed_experiments}")
    print(f"❌ 失败: {failed_experiments}")
    print(f"⏭️ 跳过: {total_experiments - completed_experiments - failed_experiments}")
    
    if all_results:
        print(f"💾 总共保存了 {len(all_results)} 个结果")
        print(f"📁 结果保存在: {os.path.join(drive_path, 'SolarPV_Results')}")

if __name__ == "__main__":
    main()