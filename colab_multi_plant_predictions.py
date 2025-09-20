#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab环境下的多Plant预测结果保存脚本
为Plant 171、172、186生成配置并保存预测结果到Google Drive

使用方法：
1. 在Colab中先运行以下命令：
   !pip install -q pyyaml pandas numpy scikit-learn xgboost lightgbm
   from google.colab import drive
   drive.mount('/content/drive')
   !git clone https://github.com/zhesun-0209/SolarPV-Prediction.git /content/SolarPV-Prediction
   !python /content/SolarPV-Prediction/generate_multi_plant_configs.py

2. 然后运行此脚本：
   !python /content/SolarPV-Prediction/colab_multi_plant_predictions.py
"""

import os
import sys

# 导入必要的库
import sys
sys.path.append('/content/SolarPV-Prediction')

import yaml
import pandas as pd
import numpy as np
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# 导入训练模块
from train.train_dl import train_lstm, train_gru, train_tcn, train_transformer
from train.train_ml import train_rf, train_xgb, train_lgbm
from data.data_utils import load_and_preprocess_data

def get_scenario_name(config):
    """根据配置获取场景名称"""
    use_pv = config.get('use_pv', False)
    use_forecast = config.get('use_forecast', False)
    use_hist_weather = config.get('use_hist_weather', False)
    use_ideal_nwp = config.get('use_ideal_nwp', False)
    
    if use_pv and use_forecast and use_ideal_nwp:
        return 'PV+NWP+'
    elif use_pv and use_forecast:
        return 'PV+NWP'
    elif use_pv and use_hist_weather:
        return 'PV+HW'
    elif use_pv:
        return 'PV'
    elif use_forecast and use_ideal_nwp:
        return 'NWP+'
    elif use_forecast:
        return 'NWP'
    else:
        return 'Unknown'

def train_single_model(config_path, plant_id):
    """训练单个模型"""
    try:
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_name = config['model']
        print(f"🔄 训练 {model_name} 模型...")
        
        # 加载和预处理数据
        data_path = config['data_path']
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            return None
        
        X_tr, X_te, y_tr, y_te, dates_te = load_and_preprocess_data(
            data_path=data_path,
            past_hours=config['past_hours'],
            future_hours=config['future_hours'],
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            use_pv=config['use_pv'],
            use_forecast=config['use_forecast'],
            use_hist_weather=config['use_hist_weather'],
            use_ideal_nwp=config['use_ideal_nwp'],
            use_time_encoding=config['use_time_encoding'],
            weather_category=config['weather_category']
        )
        
        # 训练模型
        if model_name in ['LSTM', 'GRU', 'TCN', 'Transformer']:
            # 深度学习模型
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                complexity = config.get('model_complexity', 'low')
                if complexity == 'high' and 'ml_high' in config['model_params']:
                    dl_params = config['model_params']['ml_high']
                elif complexity == 'low' and 'ml_low' in config['model_params']:
                    dl_params = config['model_params']['ml_low']
                else:
                    dl_params = config['model_params']
                
                if model_name == 'LSTM':
                    model = train_lstm(X_tr, y_tr, dl_params)
                elif model_name == 'GRU':
                    model = train_gru(X_tr, y_tr, dl_params)
                elif model_name == 'TCN':
                    model = train_tcn(X_tr, y_tr, dl_params)
                elif model_name == 'Transformer':
                    model = train_transformer(X_tr, y_tr, dl_params)
        else:
            # 机器学习模型
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                complexity = config.get('model_complexity', 'low')
                if complexity == 'high' and 'ml_high' in config['model_params']:
                    ml_params = config['model_params']['ml_high']
                elif complexity == 'low' and 'ml_low' in config['model_params']:
                    ml_params = config['model_params']['ml_low']
                else:
                    ml_params = config['model_params']
                
                if model_name == 'RF':
                    model = train_rf(X_tr, y_tr, ml_params)
                elif model_name == 'XGB':
                    model = train_xgb(X_tr, y_tr, ml_params)
                elif model_name == 'LGBM':
                    model = train_lgbm(X_tr, y_tr, ml_params)
                elif model_name in ['Linear', 'LSR']:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(X_tr, y_tr)
        
        # 预测
        y_pred = model.predict(X_te)
        
        # 获取配置信息
        scenario = get_scenario_name(config)
        lookback = config['past_hours']
        te = config.get('use_time_encoding', False)
        complexity = config.get('model_complexity', 'low')
        
        # 模型名称映射：LSR -> Linear
        model_name_mapped = 'Linear' if model_name == 'LSR' else model_name
        
        print(f"✅ {model_name} 模型训练完成 - {scenario}")
        
        return {
            'y_true': y_te,
            'y_pred': y_pred,
            'model_name': model_name_mapped,
            'scenario': scenario,
            'lookback': lookback,
            'te': te,
            'complexity': complexity,
            'config_path': config_path,
            'dates': dates_te,
            'plant_id': plant_id
        }
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return None

def save_predictions_to_drive(results, plant_id, drive_path):
    """保存预测结果到Google Drive"""
    print(f"💾 保存Plant {plant_id}的预测结果到Google Drive...")
    
    # 创建Plant专用目录
    plant_dir = os.path.join(drive_path, f'Plant_{plant_id}')
    os.makedirs(plant_dir, exist_ok=True)
    
    # 1. 保存汇总CSV
    summary_data = []
    
    for i, result in enumerate(results):
        if result is None:
            continue
            
        # 取前168小时的数据（7天）
        y_true = result['y_true'][:168].flatten()
        y_pred = result['y_pred'][:168].flatten()
        
        # 确保长度一致
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # 创建时间序列数据
        for j in range(min_len):
            summary_data.append({
                'experiment_id': i + 1,
                'model': result['model_name'],
                'scenario': result['scenario'],
                'lookback': result['lookback'],
                'te': result['te'],
                'complexity': result['complexity'],
                'timestep': j,
                'ground_truth': y_true[j],
                'prediction': y_pred[j],
                'config_file': os.path.basename(result['config_path'])
            })
    
    # 保存汇总CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(plant_dir, f'Plant_{plant_id}_all_predictions_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ 汇总CSV已保存: {summary_path}")
    
    # 2. 按场景保存CSV文件
    scenarios = ['PV', 'PV+NWP', 'PV+NWP+', 'PV+HW', 'NWP', 'NWP+']
    
    # 创建by_scenario子目录
    by_scenario_dir = os.path.join(plant_dir, 'by_scenario')
    os.makedirs(by_scenario_dir, exist_ok=True)
    
    for scenario in scenarios:
        scenario_data = []
        
        for i, result in enumerate(results):
            if result is None or result['scenario'] != scenario:
                continue
                
            # 取前168小时的数据
            y_true = result['y_true'][:168].flatten()
            y_pred = result['y_pred'][:168].flatten()
            
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # 创建时间序列数据
            for j in range(min_len):
                scenario_data.append({
                    'experiment_id': i + 1,
                    'model': result['model_name'],
                    'lookback': result['lookback'],
                    'te': result['te'],
                    'complexity': result['complexity'],
                    'timestep': j,
                    'ground_truth': y_true[j],
                    'prediction': y_pred[j],
                    'config_file': os.path.basename(result['config_path'])
                })
        
        if scenario_data:
            scenario_df = pd.DataFrame(scenario_data)
            scenario_path = os.path.join(by_scenario_dir, f'{scenario}_predictions.csv')
            scenario_df.to_csv(scenario_path, index=False)
            print(f"✅ {scenario} CSV已保存: {scenario_path}")
    
    # 3. 按模型保存CSV文件
    models = ['LSTM', 'GRU', 'TCN', 'Transformer', 'RF', 'XGB', 'LGBM', 'Linear']
    
    # 创建by_model子目录
    by_model_dir = os.path.join(plant_dir, 'by_model')
    os.makedirs(by_model_dir, exist_ok=True)
    
    for model in models:
        model_data = []
        
        for i, result in enumerate(results):
            if result is None or result['model_name'] != model:
                continue
                
            # 取前168小时的数据
            y_true = result['y_true'][:168].flatten()
            y_pred = result['y_pred'][:168].flatten()
            
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # 创建时间序列数据
            for j in range(min_len):
                model_data.append({
                    'experiment_id': i + 1,
                    'scenario': result['scenario'],
                    'lookback': result['lookback'],
                    'te': result['te'],
                    'complexity': result['complexity'],
                    'timestep': j,
                    'ground_truth': y_true[j],
                    'prediction': y_pred[j],
                    'config_file': os.path.basename(result['config_path'])
                })
        
        if model_data:
            model_df = pd.DataFrame(model_data)
            model_path = os.path.join(by_model_dir, f'{model}_predictions.csv')
            model_df.to_csv(model_path, index=False)
            print(f"✅ {model} CSV已保存: {model_path}")
    
    # 4. 保存实验配置汇总
    config_summary = []
    
    for i, result in enumerate(results):
        if result is None:
            continue
        
        config_summary.append({
            'experiment_id': i + 1,
            'model': result['model_name'],
            'scenario': result['scenario'],
            'lookback': result['lookback'],
            'te': result['te'],
            'complexity': result['complexity'],
            'config_file': os.path.basename(result['config_path']),
            'plant_id': result['plant_id']
        })
    
    config_df = pd.DataFrame(config_summary)
    config_path = os.path.join(plant_dir, f'Plant_{plant_id}_experiment_configs.csv')
    config_df.to_csv(config_path, index=False)
    print(f"✅ 实验配置CSV已保存: {config_path}")
    
    print(f"🎉 Plant {plant_id} 所有结果已保存到: {plant_dir}")

def process_plant(plant_id, drive_path):
    """处理单个Plant的所有实验"""
    print(f"\n🚀 开始处理Plant {plant_id}...")
    
    # 获取配置文件列表
    config_dir = f"config/projects/{plant_id}"
    if not os.path.exists(config_dir):
        print(f"❌ 配置目录不存在: {config_dir}")
        return
    
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    print(f"📁 找到 {len(config_files)} 个配置文件")
    
    # 训练所有模型
    results = []
    for i, config_file in enumerate(config_files):
        config_path = os.path.join(config_dir, config_file)
        print(f"\n📊 进度: {i+1}/{len(config_files)} - {config_file}")
        
        result = train_single_model(config_path, plant_id)
        results.append(result)
    
    # 保存结果到Google Drive
    save_predictions_to_drive(results, plant_id, drive_path)
    
    print(f"✅ Plant {plant_id} 处理完成！")

# 主执行部分
print("🚀 多Plant预测结果保存到Google Drive...")

# Google Drive路径
drive_path = "/content/drive/MyDrive/Solar PV electricity/plot"

# 确保Drive路径存在
os.makedirs(drive_path, exist_ok=True)

# 要处理的Plant列表
plant_ids = [171, 172, 186]

# 首先生成配置文件
print("📝 生成配置文件...")
!python generate_multi_plant_configs.py

for plant_id in plant_ids:
    process_plant(plant_id, drive_path)

print(f"\n🎉 所有Plant处理完成！")
print(f"📁 结果保存在: {drive_path}")
