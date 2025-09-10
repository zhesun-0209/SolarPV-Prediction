#!/usr/bin/env python3
"""
运行单个厂的所有252个实验组合
每个厂只生成一个Excel文件，不创建子文件夹
"""

import os
import sys
import subprocess
import time
import pandas as pd
import numpy as np
import re
from eval.excel_utils import save_plant_excel_results, load_plant_excel_results

def parse_experiment_output(output_text, save_dir, exp_id):
    """
    解析实验输出，从summary.csv文件读取指标
    
    Args:
        output_text: main.py的标准输出文本
        save_dir: 保存目录
        exp_id: 实验ID
    
    Returns:
        dict: 解析出的指标字典
    """
    metrics = {
        'test_loss': 0,
        'rmse': 0,
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
    
    try:
        # 首先尝试从stdout解析test_loss
        test_loss_match = re.search(r'test_loss=([\d.]+)', output_text)
        if test_loss_match:
            metrics['test_loss'] = float(test_loss_match.group(1))
        
        # 然后尝试从summary.csv文件读取完整指标
        summary_file = os.path.join(save_dir, "summary.csv")
        if os.path.exists(summary_file):
            try:
                df = pd.read_csv(summary_file)
                if len(df) > 0:
                    row = df.iloc[0]
                    metrics.update({
                        'test_loss': row.get('test_loss', metrics['test_loss']),
                        'rmse': row.get('rmse', 0),
                        'mae': row.get('mae', 0),
                        'nrmse': row.get('nrmse', 0),
                        'r_square': row.get('r_square', 0),
                        'mape': row.get('mape', 0),
                        'smape': row.get('smape', 0),
                        'param_count': row.get('param_count', 0),
                        'samples_count': row.get('samples_count', 0),
                        'best_epoch': row.get('best_epoch', np.nan),
                        'final_lr': row.get('final_lr', np.nan),
                        'gpu_memory_used': row.get('gpu_memory_used', 0)
                    })
                    print(f"✅ 从summary.csv读取到指标: test_loss={metrics['test_loss']:.4f}, rmse={metrics['rmse']:.4f}")
            except Exception as e:
                print(f"⚠️  读取summary.csv失败: {e}")
        
    except Exception as e:
        print(f"⚠️  解析输出失败: {e}")
    
    return metrics

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
    existing_results = load_plant_excel_results(plant_id, save_dir)
    existing_experiments = set()
    if not existing_results.empty:
        for _, row in existing_results.iterrows():
            feat_str = f"feat{str(row['use_hist_weather']).lower()}_fcst{str(row['use_forecast']).lower()}_days{row['past_days']}_comp{row['model_complexity']}"
            exp_id = f"{row['model']}_{feat_str}"
            existing_experiments.add(exp_id)
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
                        '--plant_id', plant_id
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
                            
                            # 解析main.py的输出
                            parsed_metrics = parse_experiment_output(result.stdout, save_dir, exp_id)
                            
                            # 立即保存到Excel文件
                            try:
                                # 构建实验结果数据
                                result_data = {
                                    'config': {
                                        'model': model,
                                        'use_hist_weather': hist_weather,
                                        'use_forecast': forecast,
                                        'past_days': past_days,
                                        'model_complexity': complexity,
                                        'epochs': epochs,
                                        'batch_size': 32,  # 默认值
                                        'learning_rate': 0.001  # 默认值
                                    },
                                    'metrics': {
                                        'train_time_sec': exp_duration,
                                        'inference_time_sec': parsed_metrics.get('inference_time_sec', 0),
                                        'param_count': parsed_metrics.get('param_count', 0),
                                        'samples_count': parsed_metrics.get('samples_count', 0),
                                        'test_loss': parsed_metrics.get('test_loss', 0),
                                        'rmse': parsed_metrics.get('rmse', 0),
                                        'mae': parsed_metrics.get('mae', 0),
                                        'nrmse': parsed_metrics.get('nrmse', 0),
                                        'r_square': parsed_metrics.get('r_square', 0),
                                        'mape': parsed_metrics.get('mape', 0),
                                        'smape': parsed_metrics.get('smape', 0),
                                        'best_epoch': parsed_metrics.get('best_epoch', np.nan),
                                        'final_lr': parsed_metrics.get('final_lr', np.nan),
                                        'gpu_memory_used': parsed_metrics.get('gpu_memory_used', 0)
                                    }
                                }
                                
                                # 保存到Excel
                                from eval.excel_utils import append_plant_excel_results
                                append_plant_excel_results(plant_id, [result_data], save_dir)
                                
                            except Exception as e:
                                print(f"⚠️  保存Excel结果失败: {e}")
                            
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
    
    # 检查Excel文件是否生成
    excel_file = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    if os.path.exists(excel_file):
        print(f"✅ Excel结果文件已生成: {excel_file}")
    else:
        print(f"❌ Excel结果文件未生成: {excel_file}")
    
    return completed > 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行单个厂的所有252个实验')
    parser.add_argument('plant_id', help='厂ID')
    parser.add_argument('data_file', help='数据文件路径')
    
    args = parser.parse_args()
    
    success = run_plant_experiments(args.plant_id, args.data_file)
    sys.exit(0 if success else 1)
