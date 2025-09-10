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
    检查已有的实验，从summary.csv文件中读取已完成的实验ID
    
    Args:
        plant_id: 厂ID
        save_dir: 保存目录
    
    Returns:
        set: 已完成的实验ID集合
    """
    existing_experiments = set()
    
    # 检查厂级别的summary.csv文件
    summary_file = os.path.join(save_dir, "summary.csv")
    
    if os.path.exists(summary_file):
        try:
            df = pd.read_csv(summary_file)
            if not df.empty and 'exp_id' in df.columns:
                existing_experiments = set(df['exp_id'].tolist())
        except Exception as e:
            print(f"⚠️  读取summary.csv失败: {e}")
    
    return existing_experiments

def append_experiment_to_summary(plant_id, save_dir, exp_id, model, hist_weather, forecast, 
                                past_days, complexity, epochs, exp_duration, result_stdout):
    """
    将实验结果追加到summary.csv文件
    
    Args:
        plant_id: 厂ID
        save_dir: 保存目录
        exp_id: 实验ID
        model: 模型名称
        hist_weather: 是否使用历史天气
        forecast: 是否使用预测天气
        past_days: 过去天数
        complexity: 模型复杂度
        epochs: 训练轮数
        exp_duration: 实验耗时
        result_stdout: main.py的输出
    """
    summary_file = os.path.join(save_dir, "summary.csv")
    
    # 解析test_loss
    test_loss = 0
    try:
        test_loss_match = re.search(r'test_loss=([\d.]+)', result_stdout)
        if test_loss_match:
            test_loss = float(test_loss_match.group(1))
    except:
        pass
    
    # 构建实验数据行
    exp_data = {
        'exp_id': exp_id,
        'plant_id': plant_id,
        'model': model,
        'use_hist_weather': hist_weather,
        'use_forecast': forecast,
        'past_days': past_days,
        'model_complexity': complexity,
        'epochs': epochs,
        'train_time_sec': round(exp_duration, 4),
        'test_loss': test_loss,
        'rmse': 0,  # 暂时设为0，后续可以从summary.csv读取
        'mae': 0,
        'nrmse': 0,
        'r_square': 0,
        'mape': 0,
        'smape': 0,
        'param_count': 0,
        'samples_count': 0,
        'best_epoch': np.nan,
        'final_lr': np.nan,
        'gpu_memory_used': 0
    }
    
    # 追加到summary.csv
    try:
        if os.path.exists(summary_file):
            # 读取现有数据
            df = pd.read_csv(summary_file)
            # 检查是否已存在该实验
            if exp_id not in df['exp_id'].values:
                # 追加新行
                new_row = pd.DataFrame([exp_data])
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                # 更新现有行
                df.loc[df['exp_id'] == exp_id, list(exp_data.keys())] = list(exp_data.values())
        else:
            # 创建新文件
            df = pd.DataFrame([exp_data])
        
        # 保存文件
        df.to_csv(summary_file, index=False)
        print(f"✅ 实验结果已保存到: {summary_file}")
        
    except Exception as e:
        print(f"⚠️  保存实验结果失败: {e}")

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
                        '--save_dir', save_dir,  # 直接使用厂级目录
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
                            
                            # 将实验结果追加到summary.csv
                            append_experiment_to_summary(
                                plant_id, save_dir, exp_id, model, hist_weather, forecast,
                                past_days, complexity, epochs, exp_duration, result.stdout
                            )
                            
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
    
    # 检查summary.csv文件
    summary_file = os.path.join(save_dir, "summary.csv")
    if os.path.exists(summary_file):
        try:
            df = pd.read_csv(summary_file)
            print(f"📊 总共生成了 {len(df)} 个实验结果")
            print(f"📁 结果文件: {summary_file}")
        except Exception as e:
            print(f"⚠️  读取summary.csv失败: {e}")
    else:
        print(f"❌ summary.csv文件未生成: {summary_file}")
    
    return completed > 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行单个厂的所有252个实验')
    parser.add_argument('plant_id', help='厂ID')
    parser.add_argument('data_file', help='数据文件路径')
    
    args = parser.parse_args()
    
    success = run_plant_experiments(args.plant_id, args.data_file)
    sys.exit(0 if success else 1)
