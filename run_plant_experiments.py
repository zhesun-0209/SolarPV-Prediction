#!/usr/bin/env python3
"""
运行单个厂的所有252个实验组合
每个厂生成多个summary.csv文件，不创建Excel文件
"""

import os
import sys
import subprocess
import time
import pandas as pd
import numpy as np
import glob

def check_existing_experiments(plant_id, save_dir):
    """
    检查已有的实验，从Excel文件中读取已完成的实验ID
    
    Args:
        plant_id: 厂ID
        save_dir: 保存目录
    
    Returns:
        set: 已完成的实验ID集合
    """
    existing_experiments = set()
    
    # 只检查Drive结果
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    
    result_dirs = []
    if os.path.exists(drive_dir):
        result_dirs.append(drive_dir)
    
    # 查找现有Excel结果
    print(f"🔍 调试: 检查 {len(result_dirs)} 个目录")
    for i, result_dir in enumerate(result_dirs):
        plant_dir = os.path.join(result_dir, plant_id)
        excel_file = os.path.join(plant_dir, f"{plant_id}_results.xlsx")
        
        print(f"🔍 调试 {i+1}: 检查路径 {excel_file}")
        print(f"🔍 调试 {i+1}: 目录存在 {os.path.exists(result_dir)}")
        print(f"🔍 调试 {i+1}: 厂目录存在 {os.path.exists(plant_dir)}")
        print(f"🔍 调试 {i+1}: Excel文件存在 {os.path.exists(excel_file)}")
        
        if os.path.exists(excel_file):
            try:
                df = pd.read_excel(excel_file)
                print(f"🔍 调试 {i+1}: Excel行数 {len(df)}")
                print(f"🔍 调试 {i+1}: Excel列 {list(df.columns)}")
                if not df.empty:
                    # 从Excel文件生成实验ID（不需要exp_id列）
                    existing_experiments = set()
                    for _, row in df.iterrows():
                        exp_id = f"{row['model']}_feat{str(row['use_hist_weather']).lower()}_fcst{str(row['use_forecast']).lower()}_days{row['past_days']}_comp{row['model_complexity']}"
                        existing_experiments.add(exp_id)
                    print(f"🔍 调试 {i+1}: 找到实验ID {len(existing_experiments)} 个")
                    break  # 找到就停止
                else:
                    print(f"🔍 调试 {i+1}: Excel为空")
            except Exception as e:
                print(f"⚠️  读取Excel文件失败 {excel_file}: {e}")
        else:
            print(f"🔍 调试 {i+1}: Excel文件不存在")
    
    return existing_experiments

# 移除 summary.csv 相关功能，只保留 Excel 文件保存

def run_plant_experiments(plant_id, data_file):
    """运行单个厂的所有252个实验"""
    
    print(f"🏭 开始运行厂 {plant_id} 的所有实验")
    print(f"   数据文件: {data_file}")
    print(f"   结果保存到: /content/drive/MyDrive/Solar PV electricity/results")
    print("=" * 80)
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    # 设置保存路径 - 每个厂一个目录
    base_save_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    save_dir = os.path.join(base_save_dir, plant_id)  # 每个厂一个目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查已有结果
    existing_experiments = check_existing_experiments(plant_id, save_dir)
    print(f"🔍 检查Drive路径: /content/drive/MyDrive/Solar PV electricity/results/{plant_id}/{plant_id}_results.xlsx")
    print(f"🔍 找到已有实验: {len(existing_experiments)} 个")
    if existing_experiments:
        print(f"📊 已有 {len(existing_experiments)} 个实验结果，将跳过已完成的实验")
        print(f"🔍 已有实验示例: {list(existing_experiments)[:5]}")
    
    # 定义所有实验组合
    models = ['Transformer', 'LSTM', 'GRU', 'TCN', 'RF', 'XGB', 'LGBM']
    feature_configs = [
        (False, False),  # 无特征
        (True, False),   # 历史天气
        (False, True),   # 预测天气
        (True, True),    # 历史+预测天气
        (False, True, True)  # 仅预测天气（无历史发电量）
    ]
    correlation_levels = ['high', 'medium', 'all']  # 相关度档位
    complexities = ['low', 'medium', 'high']
    past_days_options = [1, 3, 7]
    
    # 根据复杂度设置epoch数
    epoch_map = {'low': 15, 'medium': 30, 'high': 50}
    
    # 计算总实验数
    normal_configs = 4  # 前4种配置使用past_days_options
    forecast_only_configs = 1  # 最后1种配置不使用past_days
    total_experiments = (len(models) * normal_configs * len(correlation_levels) * len(complexities) * len(past_days_options) + 
                        len(models) * forecast_only_configs * len(correlation_levels) * len(complexities) * 1)
    
    print(f"📊 总实验数: {total_experiments}")
    print(f"📊 正常模式: {len(models)} × 4 × {len(correlation_levels)} × {len(complexities)} × {len(past_days_options)} = {len(models) * normal_configs * len(correlation_levels) * len(complexities) * len(past_days_options)}")
    print(f"📊 仅预测模式: {len(models)} × 1 × {len(correlation_levels)} × {len(complexities)} × 1 = {len(models) * forecast_only_configs * len(correlation_levels) * len(complexities)}")
    print(f"📊 相关度档位: {correlation_levels} (高/中/全相关度)")
    
    completed = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    # 收集所有实验结果
    all_results = []
    
    for model in models:
        for feature_config in feature_configs:
            # 处理不同的特征配置格式
            if len(feature_config) == 2:
                hist_weather, forecast = feature_config
                no_hist_power = False
            else:  # len == 3
                hist_weather, forecast, no_hist_power = feature_config
            
            for correlation_level in correlation_levels:
                for complexity in complexities:
                    if no_hist_power:
                        # 仅预测天气模式：不使用past_days，只运行一次
                        past_days_list = [0]  # 0表示不使用历史数据
                    else:
                        # 正常模式：使用所有past_days选项
                        past_days_list = past_days_options
                    
                    for past_days in past_days_list:
                        # 生成实验ID
                        if no_hist_power:
                            feat_str = f"feat{str(hist_weather).lower()}_fcst{str(forecast).lower()}_nohist_{correlation_level}_comp{complexity}"
                        else:
                            feat_str = f"feat{str(hist_weather).lower()}_fcst{str(forecast).lower()}_days{past_days}_{correlation_level}_comp{complexity}"
                        exp_id = f"{model}_{feat_str}"
                    
                    # 检查是否已存在
                    if exp_id in existing_experiments:
                        print(f"⏭️  跳过已完成实验: {exp_id}")
                        skipped += 1
                        continue
                    
                    print(f"🚀 运行实验: {exp_id} (不在已有实验中)")
                    
                    # 构建命令
                    epochs = epoch_map[complexity]
                    cmd = [
                        sys.executable, 'main.py',
                        '--config', 'config/default.yaml',
                        '--model', model,
                        '--use_hist_weather', str(hist_weather).lower(),
                        '--use_forecast', str(forecast).lower(),
                        '--correlation_level', correlation_level,
                        '--model_complexity', complexity,
                        '--epochs', str(epochs),
                        '--data_path', data_file,
                        '--plant_id', plant_id,
                        '--save_dir', save_dir,
                    ]
                    
                    # 添加past_days参数（仅对非仅预测天气模式）
                    if not no_hist_power:
                        cmd.extend(['--past_days', str(past_days)])
                    
                    # 添加无历史发电量标志
                    if no_hist_power:
                        cmd.extend(['--no_hist_power', 'true'])
                    
                    # 运行实验
                    exp_start = time.time()
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
                        exp_end = time.time()
                        exp_duration = exp_end - exp_start
                        
                        if result.returncode == 0:
                            print(f"✅ 实验完成 (耗时: {exp_duration:.1f}秒)")
                            completed += 1
                            
                        else:
                            print(f"❌ 实验失败")
                            print("错误输出:")
                            print(result.stderr)
                            failed += 1
                            
                    except subprocess.TimeoutExpired:
                        print(f"❌ 实验超时 (30分钟)")
                        failed += 1
                    except Exception as e:
                        print(f"❌ 实验异常: {e}")
                        failed += 1
                    
                    # 显示进度
                    current_total = completed + failed + skipped
                    remaining = total_experiments - current_total
                    print(f"📈 进度: {current_total}/{total_experiments} ({current_total/total_experiments*100:.1f}%) - 剩余: {remaining}")
                    
                    # 每20个实验显示一次统计
                    if current_total % 20 == 0:
                        print(f"   ✅ 成功: {completed} | ❌ 失败: {failed} | ⏭️ 跳过: {skipped}")
    
    # 最终统计
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 厂 {plant_id} 所有实验完成!")
    print("=" * 80)
    print(f"总实验数: {total_experiments}")
    print(f"成功: {completed}")
    print(f"跳过: {skipped}")
    print(f"失败: {failed}")
    print(f"总耗时: {total_duration/3600:.1f}小时")
    if completed > 0:
        print(f"平均每实验: {total_duration/completed/60:.1f}分钟")
    
    # 检查summary.csv文件
    # 检查Excel结果文件
    excel_file = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    if os.path.exists(excel_file):
        try:
            df = pd.read_excel(excel_file)
            print(f"📊 总共生成了 {len(df)} 个实验结果")
            print(f"📁 结果文件: {excel_file}")
        except Exception as e:
            print(f"⚠️  读取Excel文件失败: {e}")
    else:
        print(f"❌ Excel文件未生成: {excel_file}")
    
    return completed > 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行单个厂的所有252个实验')
    parser.add_argument('plant_id', help='厂ID')
    parser.add_argument('data_file', help='数据文件路径')
    
    args = parser.parse_args()
    
    success = run_plant_experiments(args.plant_id, args.data_file)
    sys.exit(0 if success else 1)
