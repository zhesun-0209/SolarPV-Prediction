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
        
        # 构建行数据 (25列)
        row_data = {
            # 实验配置列 (13列)
            'model': config.get('model', ''),
            'use_hist_weather': config.get('use_hist_weather', False),
            'use_forecast': config.get('use_forecast', False),
            'past_days': config.get('past_days', 1),
            'model_complexity': config.get('model_complexity', 'medium'),
            'correlation_level': config.get('correlation_level', 'high'),
            'use_time_encoding': config.get('use_time_encoding', True),
            'no_hist_power': config.get('no_hist_power', False),
            'epochs': config.get('epochs', 50),
            'batch_size': config.get('batch_size', 32),
            'learning_rate': config.get('learning_rate', 0.001),
            
            # 性能指标列 (6列)
            'train_time_sec': round(metrics.get('train_time_sec', 0), 4),
            'inference_time_sec': round(metrics.get('inference_time_sec', 0), 4),
            'param_count': metrics.get('param_count', 0),
            'samples_count': metrics.get('samples_count', 0),
            'best_epoch': metrics.get('best_epoch', np.nan),
            'final_lr': metrics.get('final_lr', np.nan),
            
            # 评估指标列 (6列)
            'test_loss': round(metrics.get('test_loss', 0), 4),
            'rmse': round(metrics.get('rmse', 0), 4),
            'mae': round(metrics.get('mae', 0), 4),
            'nrmse': round(metrics.get('nrmse', 0), 4),
            'r_square': round(metrics.get('r_square', 0), 4),
            'mape': round(metrics.get('mape', 0), 4),
            'smape': round(metrics.get('smape', 0), 4),
            
            # GPU内存使用列 (1列)
            'gpu_memory_used': round(metrics.get('gpu_memory_used', 0), 4)
        }
        
        excel_data.append(row_data)
    
    # 创建DataFrame
    df = pd.DataFrame(excel_data)
    
    # 保存Excel文件
    excel_file = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    df.to_excel(excel_file, index=False, engine='openpyxl')
    
    print(f"✅ Excel结果已保存: {excel_file}")
    print(f"   包含 {len(df)} 个实验结果")
    
    return excel_file

def collect_plant_results(plant_id: str, result_dirs: List[str]) -> List[Dict[str, Any]]:
    """
    收集单个厂的所有实验结果
    
    Args:
        plant_id: 厂ID
        result_dirs: 结果目录列表
    
    Returns:
        List[Dict]: 实验结果列表
    """
    import glob
    import json
    
    results = []
    
    for result_dir in result_dirs:
        # 查找该厂的所有summary.csv文件
        summary_pattern = os.path.join(result_dir, '**', plant_id, '**', 'summary.csv')
        summary_files = glob.glob(summary_pattern, recursive=True)
        
        # 也尝试查找包含plant_id的目录
        if not summary_files:
            for root, dirs, files in os.walk(result_dir):
                for dir_name in dirs:
                    if (plant_id in dir_name or 
                        plant_id.replace('_', '') in dir_name or
                        plant_id.replace('Project_', '') in dir_name):
                        plant_dir = os.path.join(root, dir_name)
                        summary_files.extend(glob.glob(os.path.join(plant_dir, '**', 'summary.csv'), recursive=True))
        
        # 读取每个summary.csv文件
        for summary_file in summary_files:
            try:
                df = pd.read_csv(summary_file)
                if len(df) > 0:
                    # 提取配置信息
                    config = {
                        'model': df.iloc[0].get('model', ''),
                        'use_hist_weather': df.iloc[0].get('use_hist_weather', False),
                        'use_forecast': df.iloc[0].get('use_forecast', False),
                        'past_days': df.iloc[0].get('past_days', 1),
                        'model_complexity': df.iloc[0].get('model_complexity', 'medium'),
                        'epochs': df.iloc[0].get('epochs', 50),
                        'batch_size': df.iloc[0].get('batch_size', 32),
                        'learning_rate': df.iloc[0].get('learning_rate', 0.001)
                    }
                    
                    # 提取指标信息
                    metrics = {
                        'train_time_sec': df.iloc[0].get('train_time_sec', 0),
                        'inference_time_sec': df.iloc[0].get('inference_time_sec', 0),
                        'param_count': df.iloc[0].get('param_count', 0),
                        'samples_count': df.iloc[0].get('samples_count', 0),
                        'test_loss': df.iloc[0].get('test_loss', 0),
                        'rmse': df.iloc[0].get('rmse', 0),
                        'mae': df.iloc[0].get('mae', 0),
                        'nrmse': df.iloc[0].get('nrmse', 0),
                        'r_square': df.iloc[0].get('r_square', 0),
                        'mape': df.iloc[0].get('mape', 0),
                        'smape': df.iloc[0].get('smape', 0),
                        'best_epoch': df.iloc[0].get('best_epoch', np.nan),
                        'final_lr': df.iloc[0].get('final_lr', np.nan),
                        'gpu_memory_used': df.iloc[0].get('gpu_memory_used', 0)
                    }
                    
                    results.append({
                        'config': config,
                        'metrics': metrics,
                        'summary_file': summary_file
                    })
                    
            except Exception as e:
                print(f"❌ 读取summary文件失败 {summary_file}: {e}")
                continue
    
    return results

def load_plant_excel_results(plant_id: str, save_dir: str) -> pd.DataFrame:
    """
    加载单个厂的Excel结果文件
    
    Args:
        plant_id: 厂ID
        save_dir: 保存目录
    
    Returns:
        pd.DataFrame: 实验结果DataFrame，如果文件不存在则返回空DataFrame
    """
    
    excel_file = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    
    if os.path.exists(excel_file):
        try:
            df = pd.read_excel(excel_file)
            return df
        except Exception as e:
            print(f"❌ 读取Excel文件失败 {excel_file}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def append_plant_excel_results(
    plant_id: str,
    results: List[Dict[str, Any]],
    save_dir: str
):
    """
    追加实验结果到Excel文件（支持断点续传）
    
    Args:
        plant_id: 厂ID
        results: 新的实验结果列表
        save_dir: 保存目录
    """
    
    # 加载现有结果
    existing_df = load_plant_excel_results(plant_id, save_dir)
    
    # 准备新数据
    excel_data = []
    
    for result in results:
        # 提取配置信息
        config = result.get('config', {})
        metrics = result.get('metrics', {})
        
        # 构建行数据 (25列)
        row_data = {
            # 实验配置列 (13列)
            'model': config.get('model', ''),
            'use_hist_weather': config.get('use_hist_weather', False),
            'use_forecast': config.get('use_forecast', False),
            'past_days': config.get('past_days', 1),
            'model_complexity': config.get('model_complexity', 'medium'),
            'correlation_level': config.get('correlation_level', 'high'),
            'use_time_encoding': config.get('use_time_encoding', True),
            'no_hist_power': config.get('no_hist_power', False),
            'epochs': config.get('epochs', 50),
            'batch_size': config.get('batch_size', 32),
            'learning_rate': config.get('learning_rate', 0.001),
            
            # 性能指标列 (6列)
            'train_time_sec': round(metrics.get('train_time_sec', 0), 4),
            'inference_time_sec': round(metrics.get('inference_time_sec', 0), 4),
            'param_count': metrics.get('param_count', 0),
            'samples_count': metrics.get('samples_count', 0),
            'best_epoch': metrics.get('best_epoch', np.nan),
            'final_lr': metrics.get('final_lr', np.nan),
            
            # 评估指标列 (6列)
            'test_loss': round(metrics.get('test_loss', 0), 4),
            'rmse': round(metrics.get('rmse', 0), 4),
            'mae': round(metrics.get('mae', 0), 4),
            'nrmse': round(metrics.get('nrmse', 0), 4),
            'r_square': round(metrics.get('r_square', 0), 4),
            'mape': round(metrics.get('mape', 0), 4),
            'smape': round(metrics.get('smape', 0), 4),
            
            # GPU内存使用列 (1列)
            'gpu_memory_used': round(metrics.get('gpu_memory_used', 0), 4)
        }
        
        excel_data.append(row_data)
    
    # 创建新DataFrame
    new_df = pd.DataFrame(excel_data)
    
    # 合并现有数据和新数据
    if not existing_df.empty:
        # 去重：基于实验配置去重
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=['model', 'use_hist_weather', 'use_forecast', 'past_days', 'model_complexity', 'correlation_level', 'use_time_encoding'],
            keep='last'
        )
    else:
        combined_df = new_df
    
    # 保存合并后的Excel文件
    excel_file = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    combined_df.to_excel(excel_file, index=False, engine='openpyxl')
    
    print(f"✅ Excel结果已更新: {excel_file}")
    print(f"   总实验数: {len(combined_df)}")
    print(f"   新增实验数: {len(new_df)}")
    
    return excel_file
