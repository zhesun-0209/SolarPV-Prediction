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
import re
import glob

def check_existing_experiments(plant_id, save_dir):
    """
    检查已有的实验，返回已完成的实验ID集合
    
    Args:
        plant_id: 厂ID
        save_dir: 保存目录
    
    Returns:
        set: 已完成的实验ID集合
    """
    existing_experiments = set()
    
    # 查找所有summary.csv文件
    summary_files = glob.glob(os.path.join(save_dir, "**", "summary.csv"), recursive=True)
    
    for summary_file in summary_files:
        try:
            # 从文件路径提取实验ID
            # 假设路径格式为: save_dir/exp_id/summary.csv
            path_parts = summary_file.split(os.sep)
            if len(path_parts) >= 2:
                exp_id = path_parts[-2]  # 目录名就是实验ID
                existing_experiments.add(exp_id)
        except Exception as e:
            print(f"⚠️  解析文件路径失败 {summary_file}: {e}")
    
    return existing_experiments

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
    if existing_experiments:
        print(f"📊 已有 {len(existing_experiments)} 个实验结果")
    
    # 定义所有实验组合
    models = ['Transformer', 'LSTM', 'GRU', 'TCN', 'RF', 'XGB', 'LGBM']
    feature_configs = [
        (False, False),  # 无特征
        (True, False),   # 历史天气
        (False, True),   # 预测天气
        (True, True)     # 历史+预测天气
    ]
    complexities = ['low', 'medium', 'high']
    past_days_options = [1, 3, 7]
    
    # 根据复杂度设置epoch数
    epoch_map = {'low': 15, 'medium': 30, 'high': 50}
    
    total_experiments = len(models) * len(feature_configs) * len(complexities) * len(past_days_options)
    print(f"📊 总实验数: {total_experiments}")
    
    completed = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    # 收集所有实验结果
    all_results = []
    
    for model in models:
        for hist_weather, forecast in feature_configs:
            for complexity in complexities:
                for past_days in past_days_options:
                    # 生成实验ID
                    feat_str = f"feat{str(hist_weather).lower()}_fcst{str(forecast).lower()}_days{past_days}_comp{complexity}"
                    exp_id = f"{model}_{feat_str}"
                    
                    # 检查是否已存在
                    if exp_id in existing_experiments:
                        print(f"⏭️  跳过已完成实验: {exp_id}")
                        skipped += 1
                        continue
                    
                    print(f"\n🚀 运行实验: {exp_id}")
                    
                    # 为每个实验创建子目录
                    exp_save_dir = os.path.join(save_dir, exp_id)
                    os.makedirs(exp_save_dir, exist_ok=True)
                    
                    # 构建命令
                    epochs = epoch_map[complexity]
                    
                    cmd = [
                        sys.executable, 'main.py',
                        '--config', 'config/default.yaml',
                        '--model', model,
                        '--use_hist_weather', str(hist_weather).lower(),
                        '--use_forecast', str(forecast).lower(),
                        '--model_complexity', complexity,
                        '--past_days', str(past_days),
                        '--epochs', str(epochs),
                        '--data_path', data_file,
                        '--plant_id', plant_id,
                        '--save_dir', exp_save_dir,  # 每个实验一个子目录
                        '--save_summary', 'true'  # 确保保存summary.csv
                    ]
                    
                    # 运行实验
                    exp_start = time.time()
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
                        exp_end = time.time()
                        exp_duration = exp_end - exp_start
                        
                        if result.returncode == 0:
                            print(f"✅ 实验完成 (耗时: {exp_duration:.1f}秒)")
                            completed += 1
                            
                            # 检查summary.csv是否生成
                            summary_file = os.path.join(exp_save_dir, "summary.csv")
                            if os.path.exists(summary_file):
                                print(f"✅ summary.csv已生成: {summary_file}")
                            else:
                                print(f"⚠️  summary.csv未生成: {summary_file}")
                            
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
                    print(f"📈 进度: {current_total}/{total_experiments} ({current_total/total_experiments*100:.1f}%)")
    
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
    
    # 检查summary.csv文件数量
    summary_files = glob.glob(os.path.join(save_dir, "**", "summary.csv"), recursive=True)
    print(f"📊 总共生成了 {len(summary_files)} 个summary.csv文件")
    
    return completed > 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行单个厂的所有252个实验')
    parser.add_argument('plant_id', help='厂ID')
    parser.add_argument('data_file', help='数据文件路径')
    
    args = parser.parse_args()
    
    success = run_plant_experiments(args.plant_id, args.data_file)
    sys.exit(0 if success else 1)
