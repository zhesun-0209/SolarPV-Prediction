#!/usr/bin/env python3
"""
绘制340个组合的模型对比图
按照6种情况 × 8个模型 × 多种参数组合的方式展示
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 抑制LGBM和TCN的调试信息
import logging
logging.getLogger('lightgbm').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

# 抑制其他可能的调试输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制TensorFlow信息
os.environ['LIGHTGBM_VERBOSE'] = '0'     # 抑制LightGBM信息

# 设置matplotlib参数
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 16

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
    
    # 加载数据
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
            'config_path': config_path
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

def plot_scenario_comparison(results_by_scenario, output_dir):
    """绘制每个场景的模型对比图"""
    print("🎨 绘制场景对比图...")
    
    # 模型名称映射
    model_names = {
        'LSTM': 'LSTM',
        'GRU': 'GRU', 
        'TCN': 'TCN',
        'Transformer': 'Transformer',
        'RF': 'Random Forest',
        'XGB': 'XGBoost',
        'LGBM': 'LightGBM',
        'Linear': 'Linear',
        'LSR': 'LSR'
    }
    
    scenarios = ['PV', 'PV+NWP', 'PV+NWP+', 'PV+HW', 'NWP', 'NWP+']
    
    for scenario in scenarios:
        if scenario not in results_by_scenario:
            continue
            
        print(f"📊 绘制场景: {scenario}")
        
        # 创建子图：2行4列
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # 获取该场景的所有模型结果
        scenario_results = results_by_scenario[scenario]
        
        # 按模型分组
        models = list(scenario_results.keys())
        
        for i, model_name in enumerate(models):
            if i >= 8:  # 最多8个模型
                break
                
            ax = axes[i]
            model_results = scenario_results[model_name]
            
            # 获取ground truth（所有结果都相同）
            y_true = None
            for result in model_results:
                if result['y_true'] is not None:
                    y_true = result['y_true']
                    break
            
            if y_true is None:
                ax.text(0.5, 0.5, f'{model_names.get(model_name, model_name)}\n无数据', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_names.get(model_name, model_name)}', fontweight='bold')
                continue
            
            # 取前168小时的数据
            n_samples = min(168, len(y_true))
            y_true_plot = y_true[:n_samples].flatten()
            timesteps = range(len(y_true_plot))
            
            # 绘制Ground Truth
            ax.plot(timesteps, y_true_plot, 'gray', linewidth=2, label='Ground Truth', alpha=0.8)
            
            # 定义颜色
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
            
            # 绘制不同参数组合的预测结果
            for j, result in enumerate(model_results):
                if result['y_pred'] is not None:
                    y_pred_plot = result['y_pred'][:n_samples].flatten()
                    min_len = min(len(y_true_plot), len(y_pred_plot))
                    
                    # 生成标签
                    te_str = 'TE' if result['te'] else 'noTE'
                    label = f"{result['lookback']}h-{te_str}-{result['complexity']}"
                    
                    ax.plot(timesteps[:min_len], y_pred_plot[:min_len], 
                           color=colors[j % len(colors)], 
                           linewidth=1.5, 
                           label=label, 
                           alpha=0.7)
            
            ax.set_title(f'{model_names.get(model_name, model_name)}', fontweight='bold')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Capacity Factor (%)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        # 隐藏多余的子图
        for i in range(len(models), len(axes)):
            axes[i].set_visible(False)
        
        # 设置总标题
        plt.suptitle(f'Scenario: {scenario} - Model Comparison', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f'scenario_{scenario}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 图片已保存: {output_path}")
        
        plt.close()

def plot_model_comparison(results_by_model, output_dir):
    """绘制每个模型的场景对比图"""
    print("🎨 绘制模型对比图...")
    
    # 场景名称映射
    scenario_names = {
        'PV': 'PV',
        'PV+NWP': 'PV+NWP',
        'PV+NWP+': 'PV+NWP+',
        'PV+HW': 'PV+HW',
        'NWP': 'NWP',
        'NWP+': 'NWP+'
    }
    
    models = ['LSTM', 'GRU', 'TCN', 'Transformer', 'RF', 'XGB', 'LGBM', 'Linear', 'LSR']
    
    for model_name in models:
        if model_name not in results_by_model:
            continue
            
        print(f"📊 绘制模型: {model_name}")
        
        # 创建子图：2行3列（6个场景）
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 获取该模型的所有场景结果
        model_results = results_by_model[model_name]
        
        scenarios = ['PV', 'PV+NWP', 'PV+NWP+', 'PV+HW', 'NWP', 'NWP+']
        
        for i, scenario in enumerate(scenarios):
            if i >= 6:  # 最多6个场景
                break
                
            ax = axes[i]
            
            if scenario not in model_results:
                ax.text(0.5, 0.5, f'{scenario}\n无数据', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{scenario}', fontweight='bold')
                continue
            
            scenario_data = model_results[scenario]
            
            # 获取ground truth
            y_true = None
            for result in scenario_data:
                if result['y_true'] is not None:
                    y_true = result['y_true']
                    break
            
            if y_true is None:
                ax.text(0.5, 0.5, f'{scenario}\n无数据', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{scenario}', fontweight='bold')
                continue
            
            # 取前168小时的数据
            n_samples = min(168, len(y_true))
            y_true_plot = y_true[:n_samples].flatten()
            timesteps = range(len(y_true_plot))
            
            # 绘制Ground Truth
            ax.plot(timesteps, y_true_plot, 'gray', linewidth=2, label='Ground Truth', alpha=0.8)
            
            # 定义颜色
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
            
            # 绘制不同参数组合的预测结果
            for j, result in enumerate(scenario_data):
                if result['y_pred'] is not None:
                    y_pred_plot = result['y_pred'][:n_samples].flatten()
                    min_len = min(len(y_true_plot), len(y_pred_plot))
                    
                    # 生成标签
                    te_str = 'TE' if result['te'] else 'noTE'
                    label = f"{result['lookback']}h-{te_str}-{result['complexity']}"
                    
                    ax.plot(timesteps[:min_len], y_pred_plot[:min_len], 
                           color=colors[j % len(colors)], 
                           linewidth=1.5, 
                           label=label, 
                           alpha=0.7)
            
            ax.set_title(f'{scenario}', fontweight='bold')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Capacity Factor (%)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        # 隐藏多余的子图
        for i in range(len(scenarios), len(axes)):
            axes[i].set_visible(False)
        
        # 设置总标题
        plt.suptitle(f'Model: {model_name} - Scenario Comparison', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f'model_{model_name}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 图片已保存: {output_path}")
        
        plt.close()

def main():
    """主函数"""
    print("🚀 绘制340个组合的模型对比图...")
    
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
    results_by_scenario = {}
    results_by_model = {}
    
    # 训练所有模型
    for i, config_path in enumerate(config_files):
        print(f"\n--- 处理配置文件 {i+1}/{len(config_files)}: {os.path.basename(config_path)} ---")
        
        result = train_and_predict_single_model(df, config_path)
        if result is None:
            continue
        
        # 按场景分组
        scenario = result['scenario']
        if scenario not in results_by_scenario:
            results_by_scenario[scenario] = {}
        
        model_name = result['model_name']
        if model_name not in results_by_scenario[scenario]:
            results_by_scenario[scenario][model_name] = []
        
        results_by_scenario[scenario][model_name].append(result)
        
        # 按模型分组
        if model_name not in results_by_model:
            results_by_model[model_name] = {}
        
        if scenario not in results_by_model[model_name]:
            results_by_model[model_name][scenario] = []
        
        results_by_model[model_name][scenario].append(result)
    
    # 创建输出目录
    output_dir = '340_combinations_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制场景对比图
    plot_scenario_comparison(results_by_scenario, output_dir)
    
    # 绘制模型对比图
    plot_model_comparison(results_by_model, output_dir)
    
    print(f"\n✅ 所有图片生成完成！")
    print(f"📁 图片保存在: {os.path.abspath(output_dir)}")
    
    # 列出生成的文件
    files = os.listdir(output_dir)
    files.sort()
    print(f"\n📋 生成的文件列表:")
    for i, file in enumerate(files, 1):
        print(f"{i:2d}. {file}")

if __name__ == "__main__":
    main()
