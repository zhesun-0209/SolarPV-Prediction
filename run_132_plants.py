#!/usr/bin/env python3
"""
运行132个厂的完整实验
支持断点续传、进度跟踪和GPU加速
"""

import os
import sys
import time
import subprocess
import pandas as pd
import glob
import argparse
from datetime import datetime

def find_plant_data_files():
    """查找所有厂的数据文件"""
    
    data_dir = 'data'
    plant_files = []
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    for file in csv_files:
        filename = os.path.basename(file)
        # 包含所有CSV文件作为厂数据文件
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
            if len(summary_files) >= 36:  # 至少36个实验（3个模型 × 4个特征组合 × 3个复杂度）
                return True, plant_result_dir
    
    return False, None

def check_partial_results(plant_id):
    """检查厂是否有部分结果，返回缺失的实验"""
    
    # 检查Drive和本地结果
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    local_dir = 'result'
    
    result_dirs = []
    if os.path.exists(drive_dir):
        result_dirs.append(drive_dir)
    if os.path.exists(local_dir):
        result_dirs.append(local_dir)
    
    # 定义所有应该存在的实验
    models = ['Transformer', 'LSTM', 'GRU', 'TCN', 'RF', 'XGB', 'LGBM']
    feature_configs = [
        (False, False),  # 无特征
        (True, False),   # 历史天气
        (False, True),   # 预测天气
        (True, True)     # 历史+预测天气
    ]
    complexities = ['low', 'medium', 'high']
    past_days_options = [1, 3, 7]
    
    expected_experiments = []
    for model in models:
        for hist_weather, forecast in feature_configs:
            for complexity in complexities:
                for past_days in past_days_options:
                    # 生成实验ID
                    feat_str = f"feat{str(hist_weather).lower()}_fcst{str(forecast).lower()}_days{past_days}_comp{complexity}"
                    expected_experiments.append(f"{model}_{feat_str}")
    
    # 查找现有结果
    existing_experiments = set()
    for result_dir in result_dirs:
        plant_result_dir = os.path.join(result_dir, plant_id)
        if os.path.exists(plant_result_dir):
            # 查找所有summary.csv文件
            summary_files = glob.glob(os.path.join(plant_result_dir, '**/summary.csv'), recursive=True)
            for file in summary_files:
                # 从文件路径提取实验ID
                path_parts = file.split(os.sep)
                if len(path_parts) >= 2:
                    exp_id = path_parts[-2]  # 假设实验ID是目录名
                    existing_experiments.add(exp_id)
    
    # 找出缺失的实验
    missing_experiments = set(expected_experiments) - existing_experiments
    
    return len(missing_experiments) == 0, missing_experiments, len(existing_experiments)

def run_plant_experiments(plant_id, data_file, force_rerun=False):
    """运行单个厂的所有实验"""
    
    print(f"\n🏭 开始处理厂: {plant_id}")
    print(f"   数据文件: {data_file}")
    print("=" * 80)
    
    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    # 检查是否已有完整结果
    has_complete_results, result_dir = check_existing_results(plant_id)
    if has_complete_results and not force_rerun:
        print(f"✅ 厂 {plant_id} 已有完整结果，跳过")
        return True
    
    # 检查部分结果
    is_complete, missing_experiments, existing_count = check_partial_results(plant_id)
    
    if is_complete and not force_rerun:
        print(f"✅ 厂 {plant_id} 所有实验已完成，跳过")
        return True
    elif existing_count > 0:
        print(f"📊 厂 {plant_id} 已有 {existing_count} 个实验，缺失 {len(missing_experiments)} 个")
        print(f"   缺失实验: {list(missing_experiments)[:5]}{'...' if len(missing_experiments) > 5 else ''}")
    
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

def run_all_plants(force_rerun=False):
    """运行所有厂的实验"""
    
    print("🚀 开始运行所有132个厂的完整实验")
    if force_rerun:
        print("⚠️  强制重新运行模式")
    print("=" * 80)
    
    # 查找所有厂数据文件
    plant_files = find_plant_data_files()
    
    if not plant_files:
        print("❌ 未找到任何厂数据文件")
        print("请确保数据文件在 data/ 目录下")
        return
    
    print(f"✅ 找到 {len(plant_files)} 个厂数据文件")
    
    # 计算总实验数
    models = ['Transformer', 'LSTM', 'GRU', 'TCN', 'RF', 'XGB', 'LGBM']
    feature_configs = 4  # 无特征, 历史天气, 预测天气, 历史+预测天气
    complexities = 3     # low, medium, high
    past_days_options = 3  # 1, 3, 7天
    experiments_per_plant = len(models) * feature_configs * complexities * past_days_options
    total_experiments = len(plant_files) * experiments_per_plant
    
    print(f"📊 实验规模:")
    print(f"   每厂实验数: {experiments_per_plant}")
    print(f"   总实验数: {total_experiments:,}")
    print(f"   预计时间: {total_experiments * 2 / 60:.1f} 小时 (假设每实验2分钟)")
    
    # 统计信息
    total_plants = len(plant_files)
    completed_plants = 0
    failed_plants = 0
    skipped_plants = 0
    partial_plants = 0
    
    start_time = time.time()
    
    for i, (plant_id, data_file) in enumerate(plant_files, 1):
        print(f"\n📊 进度: {i}/{total_plants} ({i/total_plants*100:.1f}%)")
        
        # 检查结果状态
        has_complete_results, _ = check_existing_results(plant_id)
        is_complete, missing_experiments, existing_count = check_partial_results(plant_id)
        
        if has_complete_results and not force_rerun:
            print(f"⏭️  厂 {plant_id} 已有完整结果，跳过")
            skipped_plants += 1
            continue
        elif is_complete and not force_rerun:
            print(f"⏭️  厂 {plant_id} 所有实验已完成，跳过")
            skipped_plants += 1
            continue
        elif existing_count > 0:
            print(f"🔄 厂 {plant_id} 部分完成 ({existing_count} 个实验)，继续运行")
            partial_plants += 1
        
        # 运行实验
        success = run_plant_experiments(plant_id, data_file, force_rerun)
        
        if success:
            completed_plants += 1
        else:
            failed_plants += 1
        
        # 显示当前统计
        print(f"\n📈 当前统计:")
        print(f"   已完成: {completed_plants}")
        print(f"   部分完成: {partial_plants}")
        print(f"   已跳过: {skipped_plants}")
        print(f"   失败: {failed_plants}")
        print(f"   剩余: {total_plants - completed_plants - partial_plants - skipped_plants - failed_plants}")
    
    # 最终统计
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 所有厂实验完成!")
    print("=" * 80)
    print(f"总厂数: {total_plants}")
    print(f"已完成: {completed_plants}")
    print(f"部分完成: {partial_plants}")
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
    parser.add_argument('--force_rerun', action='store_true', help='强制重新运行所有实验')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_results()
    elif args.plant_id and args.data_file:
        run_plant_experiments(args.plant_id, args.data_file, args.force_rerun)
    else:
        run_all_plants(args.force_rerun)
