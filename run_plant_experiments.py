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
    models = ['Transformer', 'LSTM', 'GRU', 'TCN', 'RF', 'XGB', 'LGBM', 'Linear']
    
    # 20种特征配置：(use_pv, use_hist_weather, use_forecast, weather_category, use_time_encoding, past_days)
    feature_configs = [
        # 1-4: 仅历史PV
        (True, False, False, 'none', True, 1),   # 历史PV + 时间编码 + 24h
        (True, False, False, 'none', True, 3),   # 历史PV + 时间编码 + 72h
        (True, False, False, 'none', False, 1),  # 历史PV + 无时间编码 + 24h
        (True, False, False, 'none', False, 3),  # 历史PV + 无时间编码 + 72h
        
        # 5-8: 历史PV + 历史天气
        (True, True, False, 'irradiance', True, 1),   # 历史PV + 历史太阳辐射 + 时间编码 + 24h
        (True, True, False, 'irradiance', True, 3),   # 历史PV + 历史太阳辐射 + 时间编码 + 72h
        (True, True, False, 'all_weather', True, 1),  # 历史PV + 历史全部天气 + 时间编码 + 24h
        (True, True, False, 'all_weather', True, 3),  # 历史PV + 历史全部天气 + 时间编码 + 72h
        
        # 9-12: 历史PV + 预测天气
        (True, False, True, 'irradiance', True, 1),   # 历史PV + 预测太阳辐射 + 时间编码 + 24h
        (True, False, True, 'irradiance', True, 3),   # 历史PV + 预测太阳辐射 + 时间编码 + 72h
        (True, False, True, 'all_weather', True, 1),  # 历史PV + 预测全部天气 + 时间编码 + 24h
        (True, False, True, 'all_weather', True, 3),  # 历史PV + 预测全部天气 + 时间编码 + 72h
        
        # 13-16: 历史PV + 历史+预测天气
        (True, True, True, 'irradiance', True, 1),    # 历史PV + 历史+预测太阳辐射 + 时间编码 + 24h
        (True, True, True, 'irradiance', True, 3),    # 历史PV + 历史+预测太阳辐射 + 时间编码 + 72h
        (True, True, True, 'all_weather', True, 1),   # 历史PV + 历史+预测全部天气 + 时间编码 + 24h
        (True, True, True, 'all_weather', True, 3),   # 历史PV + 历史+预测全部天气 + 时间编码 + 72h
        
        # 17-20: 仅预测天气
        (False, False, True, 'irradiance', True, 0),  # 仅预测太阳辐射 + 时间编码
        (False, False, True, 'irradiance', False, 0), # 仅预测太阳辐射 + 无时间编码
        (False, False, True, 'all_weather', True, 0), # 仅预测全部天气 + 时间编码
        (False, False, True, 'all_weather', False, 0) # 仅预测全部天气 + 无时间编码
    ]
    
    # 模型复杂度 (只保留Low和High)
    complexities = ['low', 'high']
    
    # 根据复杂度设置epoch数
    epoch_map = {'low': 15, 'high': 50}
    
    # 计算总实验数
    other_models = [m for m in models if m != 'Linear']
    linear_models = [m for m in models if m == 'Linear']
    
    # 其他模型：使用所有复杂度
    other_experiments = len(other_models) * len(feature_configs) * len(complexities)
    
    # Linear模型：只有1个复杂度
    linear_experiments = len(linear_models) * len(feature_configs) * 1
    
    total_experiments = other_experiments + linear_experiments
    
    print(f"📊 总实验数: {total_experiments}")
    print(f"📊 其他模型: {len(other_models)} × {len(feature_configs)} × {len(complexities)} = {other_experiments}")
    print(f"📊 Linear模型: {len(linear_models)} × {len(feature_configs)} × 1 = {linear_experiments}")
    print(f"📊 模型类型: {len(models)} 种 (Linear无复杂度区分)")
    print(f"📊 特征组合: {len(feature_configs)} 种")
    print(f"   - 仅历史PV: 4种 (1-4)")
    print(f"   - 历史PV+历史天气: 4种 (5-8)")
    print(f"   - 历史PV+预测天气: 4种 (9-12)")
    print(f"   - 历史PV+历史+预测天气: 4种 (13-16)")
    print(f"   - 仅预测天气: 4种 (17-20)")
    print(f"📊 天气特征: 2种 (Irradiance/All Weather)")
    print(f"📊 时间编码: 2种 (开启/关闭)")
    print(f"📊 回望窗口: 2种 (1天/3天) + 无回望窗口")
    print(f"📊 模型复杂度: 2种 (Low/High)")
    
    completed = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    # 收集所有实验结果
    all_results = []
    
    for model in models:
        for feature_config in feature_configs:
            # 解析特征配置：(use_pv, use_hist_weather, use_forecast, weather_category, use_time_encoding, past_days)
            use_pv, use_hist_weather, use_forecast, weather_category, use_time_encoding, past_days = feature_config
            
            # Linear Regression不需要复杂度区分
            if model == 'Linear':
                complexity_list = ['default']  # 只有一个默认复杂度
            else:
                complexity_list = complexities  # 其他模型使用所有复杂度
            
            for complexity in complexity_list:
                # 生成实验ID
                time_str = "time" if use_time_encoding else "notime"
                weather_str = weather_category if weather_category != 'none' else 'none'
                
                if past_days == 0:
                    # 仅预测天气模式
                    if model == 'Linear':
                        feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_nohist"
                    else:
                        feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_nohist_comp{complexity}"
                else:
                    # 正常模式
                    if model == 'Linear':
                        feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_days{past_days}"
                    else:
                        feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_days{past_days}_comp{complexity}"
                
                exp_id = f"{model}_{feat_str}"
                
                # 检查是否已存在
                if exp_id in existing_experiments:
                    print(f"⏭️  跳过已完成实验: {exp_id}")
                    skipped += 1
                    continue
                
                print(f"🚀 运行实验: {exp_id} (不在已有实验中)")
                
                # 构建命令
                if model == 'Linear':
                    # Linear Regression不需要epochs和model_complexity参数
                    cmd = [
                        sys.executable, 'main.py',
                        '--config', 'config/default.yaml',
                        '--model', model,
                        '--use_pv', str(use_pv).lower(),
                        '--use_hist_weather', str(use_hist_weather).lower(),
                        '--use_forecast', str(use_forecast).lower(),
                        '--weather_category', weather_category,
                        '--use_time_encoding', str(use_time_encoding).lower(),
                        '--data_path', data_file,
                        '--plant_id', plant_id,
                        '--save_dir', save_dir,
                    ]
                else:
                    # 其他模型需要epochs和model_complexity参数
                    epochs = epoch_map[complexity]
                    cmd = [
                        sys.executable, 'main.py',
                        '--config', 'config/default.yaml',
                        '--model', model,
                        '--use_pv', str(use_pv).lower(),
                        '--use_hist_weather', str(use_hist_weather).lower(),
                        '--use_forecast', str(use_forecast).lower(),
                        '--weather_category', weather_category,
                        '--use_time_encoding', str(use_time_encoding).lower(),
                        '--model_complexity', complexity,
                        '--epochs', str(epochs),
                        '--data_path', data_file,
                        '--plant_id', plant_id,
                        '--save_dir', save_dir,
                    ]
                
                # 添加past_days参数（仅对非仅预测天气模式）
                if past_days > 0:
                    cmd.extend(['--past_days', str(past_days)])
                
                # 添加无历史发电量标志
                if not use_pv:
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
