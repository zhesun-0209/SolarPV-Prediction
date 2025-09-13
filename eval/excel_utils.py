#!/usr/bin/env python3
"""
Excel结果保存工具
为每个厂保存一个包含所有实验结果的Excel文件
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def save_plant_excel_results(
    plant_id: str,
    results: List[Dict[str, Any]],
    save_dir: str
):
    """
    保存单个厂的Excel结果文件
    
    Args:
        plant_id: 厂ID
        results: 实验结果列表，每个元素包含一个实验的结果
        save_dir: 保存目录
    """
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    excel_data = []
    
    for result in results:
        # 提取配置信息
        config = result.get('config', {})
        metrics = result.get('metrics', {})
        
        # 构建行数据 (27列)
        row_data = {
            # 实验配置列 (14列)
            'model': config.get('model', ''),
            'use_pv': config.get('use_pv', True),
            'use_hist_weather': config.get('use_hist_weather', False),
            'use_forecast': config.get('use_forecast', False),
            'weather_category': config.get('weather_category', 'irradiance'),
            'use_time_encoding': config.get('use_time_encoding', True),
            'past_days': config.get('past_days', 1),
            'model_complexity': config.get('model_complexity', 'low'),
            'epochs': config.get('epochs', 15),
            'batch_size': config.get('batch_size', 32),
            'learning_rate': config.get('learning_rate', 0.001),
            
            # 性能指标列 (6列)
            'train_time_sec': round(metrics.get('train_time_sec', 0), 4),
            'inference_time_sec': round(metrics.get('inference_time_sec', 0), 4),
            'param_count': metrics.get('param_count', 0),
            'samples_count': metrics.get('samples_count', 0),
            'best_epoch': metrics.get('best_epoch', np.nan),
            'final_lr': metrics.get('final_lr', np.nan),
            
            # 评估指标列 (7列)
            'mse': round(metrics.get('mse', 0), 4),
            'rmse': round(metrics.get('rmse', 0), 4),
            'mae': round(metrics.get('mae', 0), 4),
            'nrmse': round(metrics.get('nrmse', 0), 4),
            'r_square': round(metrics.get('r_square', 0), 4),
            'r2': round(metrics.get('r2', 0), 4),  # 添加r2列
            'smape': round(metrics.get('smape', 0), 4),
            'gpu_memory_used': metrics.get('gpu_memory_used', 0)
        }
        
        excel_data.append(row_data)
    
    # 创建DataFrame
    df = pd.DataFrame(excel_data)
    
    # 保存到Excel文件
    excel_path = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    df.to_excel(excel_path, index=False)
    
    print(f"✅ Excel结果已保存: {excel_path}")
    print(f"   总实验数: {len(excel_data)}")
    print(f"   列数: {len(df.columns)}")

def load_plant_excel_results(plant_id: str, save_dir: str) -> pd.DataFrame:
    """
    加载单个厂的Excel结果文件
    
    Args:
        plant_id: 厂ID
        save_dir: 保存目录
        
    Returns:
        DataFrame: 实验结果数据
    """
    excel_path = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    
    if not os.path.exists(excel_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(excel_path)
        return df
    except Exception as e:
        print(f"❌ 加载Excel文件失败: {e}")
        return pd.DataFrame()

def append_plant_excel_results(
    plant_id: str,
    result: Dict[str, Any],
    save_dir: str
):
    """
    向单个厂的Excel结果文件追加新的实验结果
    
    Args:
        plant_id: 厂ID
        result: 单个实验结果
        save_dir: 保存目录
    """
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取配置信息
    config = result.get('config', {})
    metrics = result.get('metrics', {})
    
    # 构建行数据 (27列)
    row_data = {
        # 实验配置列 (14列)
        'model': config.get('model', ''),
        'use_pv': config.get('use_pv', True),
        'use_hist_weather': config.get('use_hist_weather', False),
        'use_forecast': config.get('use_forecast', False),
        'weather_category': config.get('weather_category', 'irradiance'),
        'use_time_encoding': config.get('use_time_encoding', True),
        'past_days': config.get('past_days', 1),
        'model_complexity': config.get('model_complexity', 'low'),
        'epochs': config.get('epochs', 15),
        'batch_size': config.get('batch_size', 32),
        'learning_rate': config.get('learning_rate', 0.001),
        
        # 性能指标列 (6列)
        'train_time_sec': round(metrics.get('train_time_sec', 0), 4),
        'inference_time_sec': round(metrics.get('inference_time_sec', 0), 4),
        'param_count': metrics.get('param_count', 0),
        'samples_count': metrics.get('samples_count', 0),
        'best_epoch': metrics.get('best_epoch', np.nan),
        'final_lr': metrics.get('final_lr', np.nan),
        
        # 评估指标列 (7列)
        'mse': round(metrics.get('mse', 0), 4),
        'rmse': round(metrics.get('rmse', 0), 4),
        'mae': round(metrics.get('mae', 0), 4),
        'nrmse': round(metrics.get('nrmse', 0), 4),
        'r_square': round(metrics.get('r_square', 0), 4),
        'r2': round(metrics.get('r2', 0), 4),  # 添加r2列
        'smape': round(metrics.get('smape', 0), 4),
        'gpu_memory_used': metrics.get('gpu_memory_used', 0)
    }
    
    # 检查文件是否存在
    excel_path = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    
    if os.path.exists(excel_path):
        # 读取现有数据
        try:
            existing_df = pd.read_excel(excel_path)
            
            # 检查是否已存在相同的实验（基于关键配置列）
            key_columns = ['model', 'use_pv', 'use_hist_weather', 'use_forecast', 
                          'weather_category', 'use_time_encoding', 'past_days', 'model_complexity']
            
            # 创建新行DataFrame
            new_row_df = pd.DataFrame([row_data])
            
            # 检查重复
            is_duplicate = False
            for _, existing_row in existing_df.iterrows():
                if all(existing_row[col] == row_data[col] for col in key_columns):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                print(f"⚠️  实验已存在，跳过: {plant_id}")
                return
            
            # 合并数据
            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            
        except Exception as e:
            print(f"❌ 读取现有Excel文件失败: {e}")
            # 如果读取失败，创建新的DataFrame
            combined_df = pd.DataFrame([row_data])
    else:
        # 文件不存在，创建新的DataFrame
        combined_df = pd.DataFrame([row_data])
    
    # 保存到Excel文件
    combined_df.to_excel(excel_path, index=False)
    
    print(f"✅ Excel结果已更新: {excel_path}")
    print(f"   总实验数: {len(combined_df)}")
    print(f"   新增实验数: 1")

def get_existing_experiments(plant_id: str, save_dir: str) -> set:
    """
    获取已存在的实验ID集合
    
    Args:
        plant_id: 厂ID
        save_dir: 保存目录
        
    Returns:
        set: 已存在的实验ID集合
    """
    df = load_plant_excel_results(plant_id, save_dir)
    
    if df.empty:
        return set()
    
    # 从配置列生成实验ID
    existing_experiments = set()
    
    for _, row in df.iterrows():
        # 生成实验ID（与run_plant_experiments.py中的逻辑一致）
        model = row['model']
        use_pv = row['use_pv']
        use_hist_weather = row['use_hist_weather']
        use_forecast = row['use_forecast']
        weather_category = row['weather_category']
        use_time_encoding = row['use_time_encoding']
        past_days = row['past_days']
        model_complexity = row['model_complexity']
        
        time_str = "time" if use_time_encoding else "notime"
        weather_str = weather_category if weather_category != 'none' else 'none'
        
        if past_days == 0:
            # 仅预测天气模式
            if model == 'Linear':
                feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_nohist"
            else:
                feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_nohist_comp{model_complexity}"
        else:
            # 正常模式
            if model == 'Linear':
                feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_days{past_days}"
            else:
                feat_str = f"pv{str(use_pv).lower()}_hist{str(use_hist_weather).lower()}_fcst{str(use_forecast).lower()}_{weather_str}_{time_str}_days{past_days}_comp{model_complexity}"
        
        exp_id = f"{model}_{feat_str}"
        existing_experiments.add(exp_id)
    
    return existing_experiments