#!/usr/bin/env python3
"""
绘制171、172、186项目在72小时、no time encoding、low complexity、PV+NWP+配置下
7个模型（除了LSR）的ground truth和预测值对比图
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

def load_config(project_id, model_name):
    """加载指定项目的配置文件"""
    config_path = f"config/projects/{project_id}/{model_name}_low_PV_plus_NWP_plus_72h_noTE.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_test_data(project_id):
    """加载测试数据"""
    data_path = f"data/Plant_{project_id}.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    # 加载原始数据
    df = load_raw_data(data_path)
    print(f"📊 项目 {project_id} 数据加载完成: {len(df)} 行")
    
    return df

def prepare_data(df, config):
    """准备训练数据"""
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
    
    return (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr,
            Xh_va, Xf_va, y_va, hrs_va, dates_va,
            Xh_te, Xf_te, y_te, hrs_te, dates_te), (scaler_hist, scaler_fcst, scaler_target)

def train_model(config, train_data, val_data, test_data, scalers, model_name):
    """训练指定模型"""
    print(f"🚀 开始训练 {model_name} 模型...")
    
    try:
        if model_name in ['LSTM', 'GRU', 'TCN', 'Transformer']:
            # 深度学习模型
            model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
        else:
            # 机器学习模型
            Xh_tr, Xf_tr, y_tr, _, _ = train_data
            Xh_te, Xf_te, y_te, _, _ = test_data
            scaler_hist, scaler_fcst, scaler_target = scalers
            
            # 准备ML模型输入
            if Xf_tr is not None:
                X_tr = np.concatenate([Xh_tr.reshape(Xh_tr.shape[0], -1), Xf_tr.reshape(Xf_tr.shape[0], -1)], axis=1)
                X_te = np.concatenate([Xh_te.reshape(Xh_te.shape[0], -1), Xf_te.reshape(Xf_te.shape[0], -1)], axis=1)
            else:
                X_tr = Xh_tr.reshape(Xh_tr.shape[0], -1)
                X_te = Xh_te.reshape(Xh_te.shape[0], -1)
            
            # 训练模型
            if model_name == 'RF':
                model = train_rf(X_tr, y_tr, config['model_params']['low'])
            elif model_name == 'XGB':
                model = train_xgb(X_tr, y_tr, config['model_params']['low'])
            elif model_name == 'LGBM':
                model = train_lgbm(X_tr, y_tr, config['model_params']['low'])
            else:
                raise ValueError(f"不支持的模型: {model_name}")
            
            # 预测
            y_pred = model.predict(X_te)
            metrics = {'mse': np.mean((y_te - y_pred) ** 2)}
        
        print(f"✅ {model_name} 模型训练完成")
        return model, metrics
        
    except Exception as e:
        print(f"❌ {model_name} 模型训练失败: {e}")
        return None, None

def predict_test_data(model, test_data, scalers, model_name, config):
    """预测测试数据"""
    Xh_te, Xf_te, y_te, hrs_te, dates_te = test_data
    scaler_hist, scaler_fcst, scaler_target = scalers
    
    if model_name in ['LSTM', 'GRU', 'TCN', 'Transformer']:
        # 深度学习模型预测
        model.eval()
        with torch.no_grad():
            # 转换为tensor
            Xh_tensor = torch.FloatTensor(Xh_te)
            Xf_tensor = torch.FloatTensor(Xf_te) if Xf_te is not None else None
            
            # 预测
            if Xf_tensor is not None:
                y_pred = model(Xh_tensor, Xf_tensor)
            else:
                y_pred = model(Xh_tensor)
            
            y_pred = y_pred.cpu().numpy()
    else:
        # 机器学习模型预测
        if Xf_te is not None:
            X_te = np.concatenate([Xh_te.reshape(Xh_te.shape[0], -1), Xf_te.reshape(Xf_te.shape[0], -1)], axis=1)
        else:
            X_te = Xh_te.reshape(Xh_te.shape[0], -1)
        
        y_pred = model.predict(X_te)
    
    # 反标准化
    if scaler_target is not None:
        y_te_orig = scaler_target.inverse_transform(y_te)
        y_pred_orig = scaler_target.inverse_transform(y_pred)
    else:
        y_te_orig = y_te
        y_pred_orig = y_pred
    
    return y_te_orig, y_pred_orig, dates_te

def plot_model_comparison(project_id, models_to_plot):
    """绘制模型对比图"""
    print(f"🎨 开始绘制项目 {project_id} 的模型对比图...")
    
    # 加载数据
    df = load_test_data(project_id)
    if df is None:
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # 模型名称映射
    model_names = {
        'LSTM': 'LSTM',
        'GRU': 'GRU', 
        'TCN': 'TCN',
        'Transformer': 'Transformer',
        'RF': 'Random Forest',
        'XGB': 'XGBoost',
        'LGBM': 'LightGBM'
    }
    
    for i, model_name in enumerate(models_to_plot):
        print(f"📊 处理模型: {model_name}")
        
        # 加载配置
        config = load_config(project_id, model_name)
        if config is None:
            continue
        
        # 准备数据
        data_splits, scalers = prepare_data(df, config)
        train_data, val_data, test_data = data_splits[0:3], data_splits[3:6], data_splits[6:9]
        
        # 训练模型
        model, metrics = train_model(config, train_data, val_data, test_data, scalers, model_name)
        if model is None:
            continue
        
        # 预测测试数据
        y_te_orig, y_pred_orig, dates_te = predict_test_data(model, test_data, scalers, model_name, config)
        
        # 取前3天的数据（72小时）
        n_samples = min(72, len(y_te_orig))
        y_true_plot = y_te_orig[:n_samples].flatten()
        y_pred_plot = y_pred_orig[:n_samples].flatten()
        
        # 绘制
        ax = axes[i]
        timesteps = range(len(y_true_plot))
        
        ax.plot(timesteps, y_true_plot, 'gray', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(timesteps, y_pred_plot, 'red', linewidth=2, label=f'{model_names[model_name]}', alpha=0.8)
        
        ax.set_title(f'{model_names[model_name]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Capacity Factor', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # 计算并显示RMSE
        rmse = np.sqrt(np.mean((y_true_plot - y_pred_plot) ** 2))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 隐藏多余的子图
    for i in range(len(models_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Project {project_id}: Day-ahead Forecasting Results (72h, noTE, low, PV+NWP+)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图片
    output_path = f'project_{project_id}_model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图片已保存: {output_path}")
    
    plt.show()

def main():
    """主函数"""
    print("🚀 开始绘制模型对比图...")
    
    # 要绘制的项目
    projects = [171, 172, 186]
    
    # 要绘制的模型（除了LSR）
    models_to_plot = ['LSTM', 'GRU', 'TCN', 'Transformer', 'RF', 'XGB', 'LGBM']
    
    for project_id in projects:
        print(f"\n{'='*50}")
        print(f"📊 处理项目 {project_id}")
        print(f"{'='*50}")
        
        try:
            plot_model_comparison(project_id, models_to_plot)
        except Exception as e:
            print(f"❌ 项目 {project_id} 处理失败: {e}")
            continue
    
    print("\n✅ 所有项目处理完成！")

if __name__ == "__main__":
    main()
