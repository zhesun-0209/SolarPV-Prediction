#!/usr/bin/env python3
"""
检查132个厂的实验进度
支持断点续传分析
"""

import os
import glob
import pandas as pd
from collections import defaultdict

def check_plant_progress(plant_id):
    """检查单个厂的进度"""
    
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
        # 首先尝试标准路径: result_dir/plant_id/
        plant_result_dir = os.path.join(result_dir, plant_id)
        if os.path.exists(plant_result_dir):
            print(f"   找到标准路径: {plant_result_dir}")
            # 查找所有summary.csv文件
            summary_files = glob.glob(os.path.join(plant_result_dir, '**/summary.csv'), recursive=True)
            for file in summary_files:
                # 从文件路径提取实验ID
                path_parts = file.split(os.sep)
                if len(path_parts) >= 2:
                    exp_id = path_parts[-2]  # 假设实验ID是目录名
                    existing_experiments.add(exp_id)
        else:
            # 如果标准路径不存在，尝试在整个result_dir中查找包含plant_id的目录
            print(f"   标准路径不存在: {plant_result_dir}")
            print(f"   在 {result_dir} 中搜索包含 {plant_id} 的目录...")
            
            # 递归查找包含plant_id的目录
            for root, dirs, files in os.walk(result_dir):
                for dir_name in dirs:
                    if plant_id in dir_name:
                        found_dir = os.path.join(root, dir_name)
                        print(f"   找到相关目录: {found_dir}")
                        
                        # 查找summary.csv文件
                        summary_files = glob.glob(os.path.join(found_dir, '**/summary.csv'), recursive=True)
                        for file in summary_files:
                            path_parts = file.split(os.sep)
                            if len(path_parts) >= 2:
                                exp_id = path_parts[-2]
                                existing_experiments.add(exp_id)
                                print(f"   提取实验ID: {exp_id}")
            
            # 如果还是没找到，尝试查找所有summary.csv文件，看是否有匹配的
            if not existing_experiments:
                print(f"   在 {result_dir} 中查找所有summary.csv文件...")
                all_summary_files = glob.glob(os.path.join(result_dir, '**/summary.csv'), recursive=True)
                print(f"   找到 {len(all_summary_files)} 个summary.csv文件")
                
                for file in all_summary_files[:10]:  # 只显示前10个
                    print(f"     {file}")
                    path_parts = file.split(os.sep)
                    if len(path_parts) >= 2:
                        exp_id = path_parts[-2]
                        print(f"       实验ID: {exp_id}")
    
    # 计算进度
    total_expected = len(expected_experiments)
    total_existing = len(existing_experiments)
    missing_experiments = set(expected_experiments) - existing_experiments
    
    return {
        'plant_id': plant_id,
        'total_expected': total_expected,
        'total_existing': total_existing,
        'completion_rate': total_existing / total_expected * 100,
        'missing_count': len(missing_experiments),
        'missing_experiments': missing_experiments,
        'is_complete': len(missing_experiments) == 0
    }

def check_all_plants_progress():
    """检查所有厂的进度"""
    
    print("🔍 检查所有132个厂的实验进度")
    print("=" * 80)
    
    # 查找所有厂数据文件
    data_dir = 'data'
    plant_files = []
    
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    for file in csv_files:
        filename = os.path.basename(file)
        if filename.endswith('.csv'):
            plant_id = filename.replace('.csv', '')
            plant_files.append(plant_id)
    
    if not plant_files:
        print("❌ 未找到任何厂数据文件")
        return
    
    print(f"✅ 找到 {len(plant_files)} 个厂数据文件")
    
    # 检查每个厂的进度
    progress_data = []
    complete_plants = 0
    partial_plants = 0
    empty_plants = 0
    
    for plant_id in sorted(plant_files):
        progress = check_plant_progress(plant_id)
        progress_data.append(progress)
        
        if progress['is_complete']:
            complete_plants += 1
        elif progress['total_existing'] > 0:
            partial_plants += 1
        else:
            empty_plants += 1
        
        # 显示进度
        status = "✅" if progress['is_complete'] else "🔄" if progress['total_existing'] > 0 else "❌"
        print(f"{status} {plant_id}: {progress['total_existing']}/{progress['total_expected']} ({progress['completion_rate']:.1f}%)")
    
    # 统计汇总
    print(f"\n📊 进度汇总:")
    print(f"   已完成: {complete_plants} 个厂")
    print(f"   部分完成: {partial_plants} 个厂")
    print(f"   未开始: {empty_plants} 个厂")
    print(f"   总厂数: {len(plant_files)} 个厂")
    
    # 按模型统计缺失实验
    missing_by_model = defaultdict(int)
    for progress in progress_data:
        for exp_id in progress['missing_experiments']:
            model = exp_id.split('_')[0]
            missing_by_model[model] += 1
    
    if missing_by_model:
        print(f"\n🤖 各模型缺失实验数:")
        for model, count in sorted(missing_by_model.items()):
            print(f"   {model}: {count} 个")
    
    # 保存进度报告
    progress_df = pd.DataFrame(progress_data)
    progress_file = 'plant_progress_report.csv'
    progress_df.to_csv(progress_file, index=False)
    print(f"\n💾 进度报告保存到: {progress_file}")
    
    return progress_data

def find_incomplete_plants():
    """找出未完成的厂"""
    
    progress_data = check_all_plants_progress()
    
    incomplete_plants = []
    for progress in progress_data:
        if not progress['is_complete']:
            incomplete_plants.append(progress['plant_id'])
    
    print(f"\n🔄 未完成的厂 ({len(incomplete_plants)} 个):")
    for plant_id in incomplete_plants:
        print(f"   {plant_id}")
    
    return incomplete_plants

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='检查132个厂的实验进度')
    parser.add_argument('--plant_id', type=str, help='检查指定厂')
    parser.add_argument('--incomplete', action='store_true', help='只显示未完成的厂')
    
    args = parser.parse_args()
    
    if args.plant_id:
        progress = check_plant_progress(args.plant_id)
        print(f"\n🏭 厂 {args.plant_id} 进度:")
        print(f"   总实验数: {progress['total_expected']}")
        print(f"   已完成: {progress['total_existing']}")
        print(f"   完成率: {progress['completion_rate']:.1f}%")
        print(f"   缺失: {progress['missing_count']} 个")
        if progress['missing_experiments']:
            print(f"   缺失实验: {list(progress['missing_experiments'])[:10]}{'...' if len(progress['missing_experiments']) > 10 else ''}")
    elif args.incomplete:
        find_incomplete_plants()
    else:
        check_all_plants_progress()
