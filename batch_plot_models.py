#!/usr/bin/env python3
"""
批量绘制模型对比图
专门用于绘制171、172、186项目在指定配置下的7个模型对比
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

# 设置matplotlib参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

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
    data_path = f"data/Plant_{project_id}.csv"
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None, None
    
    df = load_raw_data(data_path)
    print(f"✅ 数据加载完成: {len(df)} 行")
    
    return df

def train_and_predict_model(df, project_id, model_name):
    """训练模型并预测"""
    print(f"🚀 训练 {model_name} 模型...")
    
    try:
        # 加载配置
        config_path = f"config/projects/{project_id}/{model_name}_low_PV_plus_NWP_plus_72h_noTE.yaml"
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return None, None, None
        
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
        
        # 训练模型
        if model_name in ['LSTM', 'GRU', 'TCN', 'Transformer']:
            # 深度学习模型
            train_data = (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr)
            val_data = (Xh_va, Xf_va, y_va, hrs_va, dates_va)
            test_data = (Xh_te, Xf_te, y_te, hrs_te, dates_te)
            scalers = (scaler_hist, scaler_fcst, scaler_target)
            
            model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
            
            # 预测
            model.eval()
            with torch.no_grad():
                Xh_tensor = torch.FloatTensor(Xh_te)
                Xf_tensor = torch.FloatTensor(Xf_te) if Xf_te is not None else None
                
                if Xf_tensor is not None:
                    y_pred = model(Xh_tensor, Xf_tensor)
                else:
                    y_pred = model(Xh_tensor)
                
                y_pred = y_pred.cpu().numpy()
        else:
            # 机器学习模型
            if Xf_te is not None:
                X_te = np.concatenate([Xh_te.reshape(Xh_te.shape[0], -1), Xf_te.reshape(Xf_te.shape[0], -1)], axis=1)
            else:
                X_te = Xh_te.reshape(Xh_te.shape[0], -1)
            
            if model_name == 'RF':
                model = train_rf(X_te, y_te, config['model_params']['low'])
            elif model_name == 'XGB':
                model = train_xgb(X_te, y_te, config['model_params']['low'])
            elif model_name == 'LGBM':
                model = train_lgbm(X_te, y_te, config['model_params']['low'])
            
            y_pred = model.predict(X_te)
        
        # 反标准化
        if scaler_target is not None:
            y_te_orig = scaler_target.inverse_transform(y_te)
            y_pred_orig = scaler_target.inverse_transform(y_pred)
        else:
            y_te_orig = y_te
            y_pred_orig = y_pred
        
        print(f"✅ {model_name} 模型训练完成")
        return y_te_orig, y_pred_orig, model_name
        
    except Exception as e:
        print(f"❌ {model_name} 模型训练失败: {e}")
        return None, None, None

def plot_project_models(project_id, df):
    """绘制单个项目的所有模型对比"""
    print(f"🎨 绘制项目 {project_id} 的模型对比图...")
    
    # 要绘制的模型
    models = ['LSTM', 'GRU', 'TCN', 'Transformer', 'RF', 'XGB', 'LGBM']
    
    # 存储结果
    results = {}
    
    # 训练所有模型
    for model_name in models:
        y_true, y_pred, name = train_and_predict_model(df, project_id, model_name)
        if y_true is not None:
            results[model_name] = (y_true, y_pred)
    
    if not results:
        print(f"❌ 项目 {project_id} 没有成功训练的模型")
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
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
    
    # 绘制每个模型
    for i, (model_name, (y_true, y_pred)) in enumerate(results.items()):
        ax = axes[i]
        
        # 取前72小时的数据
        n_samples = min(72, len(y_true))
        y_true_plot = y_true[:n_samples].flatten() * 100  # 转换为百分数
        y_pred_plot = y_pred[:n_samples].flatten() * 100  # 转换为百分数
        
        # 绘制
        timesteps = range(len(y_true_plot))
        
        ax.plot(timesteps, y_true_plot, 'gray', linewidth=3, label='Ground Truth', alpha=0.8)
        ax.plot(timesteps, y_pred_plot, 'red', linewidth=3, label=f'{model_names[model_name]}', alpha=0.8)
        
        ax.set_title(f'{model_names[model_name]}', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Capacity Factor (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    # 隐藏多余的子图
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Project {project_id}: Day-ahead Forecasting Results (72h, noTE, low, PV+NWP+)', 
                 fontweight='bold', y=0.98)
    
    # 创建输出目录
    output_dir = 'model_comparison_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    output_path = os.path.join(output_dir, f'project_{project_id}_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图片已保存: {output_path}")
    
    plt.show()

def main():
    """主函数"""
    print("🚀 批量绘制模型对比图...")
    
    # 要绘制的项目
    projects = [171, 172, 186]
    
    for project_id in projects:
        print(f"\n{'='*60}")
        print(f"📊 处理项目 {project_id}")
        print(f"{'='*60}")
        
        # 加载数据
        df = load_and_prepare_data(project_id)
        if df is None:
            continue
        
        # 绘制模型对比
        try:
            plot_project_models(project_id, df)
        except Exception as e:
            print(f"❌ 项目 {project_id} 绘制失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n✅ 所有项目处理完成！")

if __name__ == "__main__":
    main()
