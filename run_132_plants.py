#!/usr/bin/env python3
"""
运行132个厂的完整实验
支持断点续传和进度跟踪
"""

import os
import sys
import time
import subprocess
import pandas as pd
import glob
from datetime import datetime

def find_plant_data_files():
    """查找所有厂的数据文件"""
    
    data_dir = 'data'
    plant_files = []
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    for file in csv_files:
        filename = os.path.basename(file)
        # 跳过非厂数据文件
        if filename in ['Project1033.csv']:
            continue
        
        # 假设厂数据文件格式为 Plant_XXXX.csv 或类似
        if filename.endswith('.csv'):
            plant_id = filename.replace('.csv', '')
            plant_files.append((plant_id, file))
    
    return sorted(plant_files)

def check_existing_results(plant_id):
    """检查厂是否已有结果"""
    
    # 检查Drive和本地结果
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    local_dir = 'result'
    
    result_dirs = []
    if os.path.exists(drive_dir):
        result_dirs.append(drive_dir)
    if os.path.exists(local_dir):
        result_dirs.append(local_dir)
    
    for result_dir in result_dirs:
        plant_result_dir = os.path.join(result_dir, plant_id)
        if os.path.exists(plant_result_dir):
            # 检查是否有完整的结果
            summary_files = glob.glob(os.path.join(plant_result_dir, '**/summary.csv'), recursive=True)
            if len(summary_files) >= 24:  # 至少24个实验（3个模型 × 4个特征组合 × 2个复杂度）
                return True, plant_result_dir
    
    return False, None

def run_plant_experiments(plant_id, data_file):
    """运行单个厂的所有实验"""
    
    print(f"\n🏭 开始处理厂: {plant_id}")
    print(f"   数据文件: {data_file}")
    print("=" * 80)
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    # 检查是否已有结果
    has_results, result_dir = check_existing_results(plant_id)
    if has_results:
        print(f"✅ 厂 {plant_id} 已有结果，跳过")
        return True
    
    # 运行实验
    cmd = [
        sys.executable, 'colab_gpu_experiments.py',
        '--plant_id', plant_id,
        '--data_file', data_file
    ]
    
    print(f"🚀 运行命令: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 厂 {plant_id} 实验完成 (耗时: {duration:.1f}秒)")
            return True
        else:
            print(f"❌ 厂 {plant_id} 实验失败")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ 厂 {plant_id} 实验超时 (1小时)")
        return False
    except Exception as e:
        print(f"❌ 厂 {plant_id} 实验异常: {e}")
        return False

def run_all_plants():
    """运行所有厂的实验"""
    
    print("🚀 开始运行所有132个厂的完整实验")
    print("=" * 80)
    
    # 查找所有厂数据文件
    plant_files = find_plant_data_files()
    
    if not plant_files:
        print("❌ 未找到任何厂数据文件")
        print("请确保数据文件在 data/ 目录下")
        return
    
    print(f"✅ 找到 {len(plant_files)} 个厂数据文件")
    
    # 统计信息
    total_plants = len(plant_files)
    completed_plants = 0
    failed_plants = 0
    skipped_plants = 0
    
    start_time = time.time()
    
    for i, (plant_id, data_file) in enumerate(plant_files, 1):
        print(f"\n📊 进度: {i}/{total_plants} ({i/total_plants*100:.1f}%)")
        
        # 检查是否已有结果
        has_results, _ = check_existing_results(plant_id)
        if has_results:
            print(f"⏭️  厂 {plant_id} 已有结果，跳过")
            skipped_plants += 1
            continue
        
        # 运行实验
        success = run_plant_experiments(plant_id, data_file)
        
        if success:
            completed_plants += 1
        else:
            failed_plants += 1
        
        # 显示当前统计
        print(f"\n📈 当前统计:")
        print(f"   已完成: {completed_plants}")
        print(f"   已跳过: {skipped_plants}")
        print(f"   失败: {failed_plants}")
        print(f"   剩余: {total_plants - completed_plants - skipped_plants - failed_plants}")
    
    # 最终统计
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 所有厂实验完成!")
    print("=" * 80)
    print(f"总厂数: {total_plants}")
    print(f"已完成: {completed_plants}")
    print(f"已跳过: {skipped_plants}")
    print(f"失败: {failed_plants}")
    print(f"总耗时: {total_duration/3600:.1f}小时")
    print(f"平均每厂: {total_duration/total_plants/60:.1f}分钟")

def analyze_results():
    """分析所有厂的结果"""
    
    print("\n📊 分析所有厂的结果...")
    print("=" * 80)
    
    # 查找所有结果文件
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    local_dir = 'result'
    
    result_dirs = []
    if os.path.exists(drive_dir):
        result_dirs.append(drive_dir)
    if os.path.exists(local_dir):
        result_dirs.append(local_dir)
    
    all_results = []
    
    for result_dir in result_dirs:
        # 查找所有summary.csv文件
        summary_files = glob.glob(os.path.join(result_dir, '**/summary.csv'), recursive=True)
        
        for file in summary_files:
            try:
                df = pd.read_csv(file)
                df['result_file'] = file
                all_results.append(df)
            except Exception as e:
                print(f"❌ 读取结果文件失败 {file}: {e}")
    
    if not all_results:
        print("❌ 未找到任何结果文件")
        return
    
    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print(f"✅ 找到 {len(combined_df)} 个实验结果")
    
    # 按厂统计
    plant_stats = combined_df.groupby('plant_id').size().sort_values(ascending=False)
    print(f"\n📈 各厂实验数量:")
    print(plant_stats.head(10))
    
    # 按模型统计
    model_stats = combined_df.groupby('model').size()
    print(f"\n🤖 各模型实验数量:")
    print(model_stats)
    
    # 保存分析结果
    analysis_file = 'all_plants_analysis.csv'
    combined_df.to_csv(analysis_file, index=False)
    print(f"\n💾 分析结果保存到: {analysis_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行所有132个厂的完整实验')
    parser.add_argument('--analyze', action='store_true', help='只分析结果，不运行实验')
    parser.add_argument('--plant_id', type=str, help='只运行指定厂')
    parser.add_argument('--data_file', type=str, help='指定数据文件')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_results()
    elif args.plant_id and args.data_file:
        run_plant_experiments(args.plant_id, args.data_file)
    else:
        run_all_plants()
