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
    
    # 检查厂级别的Excel文件
    excel_file = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    
    if os.path.exists(excel_file):
        try:
            df = pd.read_excel(excel_file)
            if not df.empty and 'exp_id' in df.columns:
                existing_experiments = set(df['exp_id'].tolist())
        except Exception as e:
            print(f"⚠️  读取Excel文件失败: {e}")
    
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
                        # --save_summary 已移除，不再保存summary.csv
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
                            
                            # 实验结果已通过 main.py 保存到 Excel 文件
                            print(f"✅ 实验结果已保存到 Excel 文件")
                            
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
                    
                    # 显示详细进度
                    current_total = completed + failed + skipped
                    remaining = total_experiments - current_total
                    print(f"📈 进度: {current_total}/{total_experiments} ({current_total/total_experiments*100:.1f}%) - 剩余: {remaining}")
                    
                    # 每10个实验显示一次详细统计
                    if current_total % 10 == 0:
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
