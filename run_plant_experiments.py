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
    print(f"🔍 [DEBUG] 开始保存实验结果: {exp_id}")
    print(f"🔍 [DEBUG] 保存目录: {save_dir}")
    
    summary_file = os.path.join(save_dir, "summary.csv")
    print(f"🔍 [DEBUG] summary.csv路径: {summary_file}")
    
    # 解析test_loss和其他指标
    test_loss = 0
    rmse = 0
    mae = 0
    print(f"🔍 [DEBUG] 开始解析指标...")
    print(f"🔍 [DEBUG] main.py输出长度: {len(result_stdout)}")
    print(f"🔍 [DEBUG] main.py输出前500字符: {result_stdout[:500]}")
    
    try:
        # 解析test_loss
        test_loss_match = re.search(r'test_loss=([\d.]+)', result_stdout)
        if test_loss_match:
            test_loss = float(test_loss_match.group(1))
            print(f"🔍 [DEBUG] 成功解析test_loss: {test_loss}")
        else:
            print(f"🔍 [DEBUG] 未找到test_loss模式")
        
        # 解析rmse
        rmse_match = re.search(r'rmse=([\d.]+)', result_stdout)
        if rmse_match:
            rmse = float(rmse_match.group(1))
            print(f"🔍 [DEBUG] 成功解析rmse: {rmse}")
        
        # 解析mae
        mae_match = re.search(r'mae=([\d.]+)', result_stdout)
        if mae_match:
            mae = float(mae_match.group(1))
            print(f"🔍 [DEBUG] 成功解析mae: {mae}")
            
    except Exception as e:
        print(f"🔍 [DEBUG] 解析指标失败: {e}")
    
    # 构建实验数据行
    print(f"🔍 [DEBUG] 构建实验数据...")
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
        'rmse': rmse,  # 使用解析到的真实值
        'mae': mae,    # 使用解析到的真实值
        'nrmse': 0,    # 暂时设为0，后续可以计算
        'r_square': 0, # 暂时设为0，后续可以计算
        'mape': 0,     # 暂时设为0，后续可以计算
        'smape': 0,    # 暂时设为0，后续可以计算
        'param_count': 0,
        'samples_count': 0,
        'best_epoch': np.nan,
        'final_lr': np.nan,
        'gpu_memory_used': 0
    }
    print(f"🔍 [DEBUG] 实验数据: {exp_data}")
    
    # 追加到summary.csv
    print(f"🔍 [DEBUG] 开始保存到summary.csv...")
    try:
        if os.path.exists(summary_file):
            print(f"🔍 [DEBUG] summary.csv已存在，读取现有数据...")
            # 读取现有数据
            df = pd.read_csv(summary_file)
            print(f"🔍 [DEBUG] 现有数据行数: {len(df)}")
            print(f"🔍 [DEBUG] 现有列: {list(df.columns)}")
            
            # 检查是否已存在该实验
            if 'exp_id' in df.columns and exp_id in df['exp_id'].values:
                print(f"🔍 [DEBUG] 实验 {exp_id} 已存在，更新数据...")
                # 更新现有行
                df.loc[df['exp_id'] == exp_id, list(exp_data.keys())] = list(exp_data.values())
            else:
                print(f"🔍 [DEBUG] 实验 {exp_id} 不存在，追加新行...")
                # 追加新行
                new_row = pd.DataFrame([exp_data])
                df = pd.concat([df, new_row], ignore_index=True)
        else:
            print(f"🔍 [DEBUG] summary.csv不存在，创建新文件...")
            # 创建新文件
            df = pd.DataFrame([exp_data])
        
        print(f"🔍 [DEBUG] 最终数据行数: {len(df)}")
        print(f"🔍 [DEBUG] 最终列: {list(df.columns)}")
        
        # 保存文件
        df.to_csv(summary_file, index=False)
        print(f"✅ 实验结果已保存到: {summary_file}")
        print(f"🔍 [DEBUG] 文件大小: {os.path.getsize(summary_file)} bytes")
        
    except Exception as e:
        print(f"⚠️  保存实验结果失败: {e}")
        import traceback
        traceback.print_exc()

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
    print(f"🔍 [DEBUG] 检查已有实验结果...")
    existing_experiments = check_existing_experiments(plant_id, save_dir)
    print(f"🔍 [DEBUG] 找到已有实验: {existing_experiments}")
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
                    print(f"🔍 [DEBUG] 实验参数: model={model}, hist_weather={hist_weather}, forecast={forecast}, past_days={past_days}, complexity={complexity}")
                    
                    # 构建命令
                    epochs = epoch_map[complexity]
                    print(f"🔍 [DEBUG] 使用epochs: {epochs}")
                    
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
                    print(f"🔍 [DEBUG] 运行命令: {' '.join(cmd)}")
                    
                    # 运行实验
                    exp_start = time.time()
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
                        exp_end = time.time()
                        exp_duration = exp_end - exp_start
                        
                        if result.returncode == 0:
                            print(f"✅ 实验完成 (耗时: {exp_duration:.1f}秒)")
                            print(f"🔍 [DEBUG] main.py返回码: {result.returncode}")
                            print(f"🔍 [DEBUG] stdout长度: {len(result.stdout)}")
                            print(f"🔍 [DEBUG] stderr长度: {len(result.stderr)}")
                            print(f"🔍 [DEBUG] stdout前200字符: {result.stdout[:200]}")
                            if result.stderr:
                                print(f"🔍 [DEBUG] stderr: {result.stderr}")
                            
                            completed += 1
                            
                            # 将实验结果追加到summary.csv
                            print(f"🔍 [DEBUG] 开始调用append_experiment_to_summary...")
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
