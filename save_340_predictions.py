#!/usr/bin/env python3
"""
保存340个组合的预测结果
将ground truth和prediction保存到CSV文件中
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('.')

from data.data_utils import (
    load_raw_data,
    preprocess_features,
    create_sliding_windows,
    split_data
)
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model
from models.rnn_models import LSTM, GRU
from models.tcn import TCNModel
from models.transformer import Transformer
from models.ml_models import train_rf, train_xgb, train_lgbm

def load_and_prepare_data(project_id):
    """加载和准备数据"""
    print(f"📊 加载项目 {project_id} 数据...")
    
    data_path = f"data/Project{project_id}.csv"
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    df = load_raw_data(data_path)
    print(f"✅ 数据加载完成: {len(df)} 行")
    
    return df

def train_and_predict_single_model(df, config_path):
    """训练单个模型并预测"""
    print(f"🚀 训练模型: {os.path.basename(config_path)}...")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 预处理特征
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = preprocess_features(df, config)
        
        # 创建滑动窗口
        X_hist, X_fcst, y, hours, dates = create_sliding_windows(
            df_clean, 
            config['past_hours'], 
            config['future_hours'], 
            hist_feats, 
            fcst_feats, 
            no_hist_power=not config.get('use_pv', True)
        )
        
        # 分割数据
        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
        Xh_va, Xf_va, y_va, hrs_va, dates_va, \
        Xh_te, Xf_te, y_te, hrs_te, dates_te = split_data(
            X_hist, X_fcst, y, hours, dates,
            train_ratio=config["train_ratio"],
            val_ratio=config["val_ratio"]
        )
        
        model_name = config['model']
        
        # 训练模型
        if model_name in ['LSTM', 'GRU', 'TCN', 'Transformer']:
            # 深度学习模型
            train_data = (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr)
            val_data = (Xh_va, Xf_va, y_va, hrs_va, dates_va)
            test_data = (Xh_te, Xf_te, y_te, hrs_te, dates_te)
            scalers = (scaler_hist, scaler_fcst, scaler_target)
            
            # 抑制训练过程中的输出
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
            
            # 预测
            model.eval()
            with torch.no_grad():
                device = next(model.parameters()).device
                Xh_tensor = torch.FloatTensor(Xh_te).to(device)
                Xf_tensor = torch.FloatTensor(Xf_te).to(device) if Xf_te is not None else None
                
                if Xf_tensor is not None:
                    y_pred = model(Xh_tensor, Xf_tensor)
                else:
                    y_pred = model(Xh_tensor)
                
                y_pred = y_pred.cpu().numpy()
        else:
            # 机器学习模型
            if Xf_tr is not None:
                X_tr = np.concatenate([Xh_tr.reshape(Xh_tr.shape[0], -1), Xf_tr.reshape(Xf_tr.shape[0], -1)], axis=1)
            else:
                X_tr = Xh_tr.reshape(Xh_tr.shape[0], -1)
            
            if Xf_te is not None:
                X_te = np.concatenate([Xh_te.reshape(Xh_te.shape[0], -1), Xf_te.reshape(Xh_te.shape[0], -1)], axis=1)
            else:
                X_te = Xh_te.reshape(Xh_te.shape[0], -1)
            
            # 训练模型
            import io
            from contextlib import redirect_stdout, redirect_stderr
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
                elif model_name == 'Linear':
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
        
        print(f"✅ {model_name} 模型训练完成 - {scenario}")
        
        return {
            'y_true': y_te,
            'y_pred': y_pred,
            'model_name': model_name,
            'scenario': scenario,
            'lookback': lookback,
            'te': te,
            'complexity': complexity,
            'config_path': config_path,
            'dates': dates_te
        }
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return None

def get_scenario_name(config):
    """根据配置获取场景名称"""
    use_pv = config.get('use_pv', False)
    use_forecast = config.get('use_forecast', False)
    use_hist_weather = config.get('use_hist_weather', False)
    use_ideal_nwp = config.get('use_ideal_nwp', False)
    
    if use_pv and use_hist_weather:
        return 'PV+HW'
    elif use_pv and use_forecast and use_ideal_nwp:
        return 'PV+NWP+'
    elif use_pv and use_forecast:
        return 'PV+NWP'
    elif use_pv:
        return 'PV'
    elif use_forecast and use_ideal_nwp:
        return 'NWP+'
    elif use_forecast:
        return 'NWP'
    else:
        return 'Unknown'

def save_predictions_to_csv(results, output_dir):
    """保存预测结果到CSV文件"""
    print("💾 保存预测结果到CSV文件...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存汇总CSV文件
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
    summary_path = os.path.join(output_dir, 'all_predictions_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ 汇总CSV已保存: {summary_path}")
    
    # 2. 按场景保存CSV文件
    scenarios = ['PV', 'PV+NWP', 'PV+NWP+', 'PV+HW', 'NWP', 'NWP+']
    
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
            scenario_path = os.path.join(output_dir, f'{scenario}_predictions.csv')
            scenario_df.to_csv(scenario_path, index=False)
            print(f"✅ {scenario} CSV已保存: {scenario_path}")
    
    # 3. 按模型保存CSV文件
    models = ['LSTM', 'GRU', 'TCN', 'Transformer', 'RF', 'XGB', 'LGBM', 'Linear', 'LSR']
    
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
            model_path = os.path.join(output_dir, f'{model}_predictions.csv')
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
            'data_points': len(result['y_true'][:168].flatten())
        })
    
    config_df = pd.DataFrame(config_summary)
    config_path = os.path.join(output_dir, 'experiment_configs.csv')
    config_df.to_csv(config_path, index=False)
    print(f"✅ 实验配置汇总已保存: {config_path}")

def main():
    """主函数"""
    print("🚀 保存340个组合的预测结果...")
    
    # 参数设置
    project_id = 1140  # 使用1140项目的数据
    config_dir = "config/ablation"
    
    # 加载数据
    df = load_and_prepare_data(project_id)
    if df is None:
        print("❌ 无法加载数据，退出")
        return
    
    # 获取所有配置文件
    config_files = []
    for file in os.listdir(config_dir):
        if file.endswith('.yaml'):
            config_files.append(os.path.join(config_dir, file))
    
    print(f"📊 找到 {len(config_files)} 个配置文件")
    
    # 存储结果
    results = []
    
    # 训练所有模型
    for i, config_path in enumerate(config_files):
        print(f"\n--- 处理配置文件 {i+1}/{len(config_files)}: {os.path.basename(config_path)} ---")
        
        result = train_and_predict_single_model(df, config_path)
        results.append(result)
    
    # 保存结果
    output_dir = '340_predictions_results'
    save_predictions_to_csv(results, output_dir)
    
    print(f"\n✅ 所有预测结果保存完成！")
    print(f"📁 结果保存在: {os.path.abspath(output_dir)}")
    
    # 列出生成的文件
    files = os.listdir(output_dir)
    files.sort()
    print(f"\n📋 生成的文件列表:")
    for i, file in enumerate(files, 1):
        print(f"{i:2d}. {file}")

if __name__ == "__main__":
    main()
