#!/usr/bin/env python3
"""
快速测试脚本 - 绘制单个项目的模型对比图
用于调试和验证绘图功能
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

def quick_plot_single_model(project_id=171, model_name='LSTM'):
    """快速绘制单个模型的预测结果"""
    print(f"🎨 绘制项目 {project_id} 的 {model_name} 模型...")
    
    # 加载数据
    data_path = f"data/Plant_{project_id}.csv"
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    df = load_raw_data(data_path)
    print(f"📊 数据加载完成: {len(df)} 行")
    
    # 加载配置
    config_path = f"config/projects/{project_id}/{model_name}_low_PV_plus_NWP_plus_72h_noTE.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 配置加载完成: {config['model']}")
    
    # 预处理特征
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = preprocess_features(df, config)
    print(f"🔧 特征预处理完成: 历史特征{len(hist_feats)}个, 预测特征{len(fcst_feats)}个")
    
    # 创建滑动窗口
    X_hist, X_fcst, y, hours, dates = create_sliding_windows(
        df_clean, 
        config['past_hours'], 
        config['future_hours'], 
        hist_feats, 
        fcst_feats, 
        no_hist_power=not config.get('use_pv', True)
    )
    print(f"🪟 滑动窗口创建完成: {X_hist.shape[0]} 个样本")
    
    # 分割数据
    Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
    Xh_va, Xf_va, y_va, hrs_va, dates_va, \
    Xh_te, Xf_te, y_te, hrs_te, dates_te = split_data(
        X_hist, X_fcst, y, hours, dates,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"]
    )
    print(f"✂️ 数据分割完成: 训练集{len(y_tr)}, 验证集{len(y_va)}, 测试集{len(y_te)}")
    
    # 训练模型
    print(f"🚀 开始训练 {model_name} 模型...")
    try:
        if model_name in ['LSTM', 'GRU', 'TCN', 'Transformer']:
            # 深度学习模型
            train_data = (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr)
            val_data = (Xh_va, Xf_va, y_va, hrs_va, dates_va)
            test_data = (Xh_te, Xf_te, y_te, hrs_te, dates_te)
            scalers = (scaler_hist, scaler_fcst, scaler_target)
            
            model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
            
            # 预测测试数据
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
        
        print(f"✅ 模型训练完成")
        
        # 反标准化
        if scaler_target is not None:
            y_te_orig = scaler_target.inverse_transform(y_te)
            y_pred_orig = scaler_target.inverse_transform(y_pred)
        else:
            y_te_orig = y_te
            y_pred_orig = y_pred
        
        # 取前72小时的数据
        n_samples = min(72, len(y_te_orig))
        y_true_plot = y_te_orig[:n_samples].flatten()
        y_pred_plot = y_pred_orig[:n_samples].flatten()
        
        # 绘制
        plt.figure(figsize=(12, 6))
        timesteps = range(len(y_true_plot))
        
        plt.plot(timesteps, y_true_plot, 'gray', linewidth=2, label='Ground Truth', alpha=0.8)
        plt.plot(timesteps, y_pred_plot, 'red', linewidth=2, label=f'{model_name}', alpha=0.8)
        
        plt.title(f'Project {project_id}: {model_name} Forecasting Results (72h, noTE, low, PV+NWP+)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Capacity Factor', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        
        # 计算并显示RMSE
        rmse = np.sqrt(np.mean((y_true_plot - y_pred_plot) ** 2))
        plt.text(0.02, 0.98, f'RMSE: {rmse:.4f}', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        output_path = f'project_{project_id}_{model_name}_quick_test.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 图片已保存: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 快速测试脚本启动...")
    
    # 测试单个模型
    project_id = 171
    model_name = 'LSTM'  # 可以改为其他模型: 'GRU', 'TCN', 'Transformer', 'RF', 'XGB', 'LGBM'
    
    quick_plot_single_model(project_id, model_name)

if __name__ == "__main__":
    main()
