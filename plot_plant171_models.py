#!/usr/bin/env python3
"""
绘制厂171的8个模型在6种情况下的预测图
2*4子图布局，每个模型一个子图，包含ground truth+6个情况预测图
支持不同参数组合：lookback, TE, complexity
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
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 18

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

def train_and_predict_single_model(df, project_id, model_name, scenario, lookback, te, complexity):
    """训练单个模型并预测"""
    print(f"🚀 训练 {model_name} 模型 - {scenario} - {lookback}h - {'TE' if te else 'noTE'} - {complexity}...")
    
    try:
        # 构建配置文件路径
        te_str = 'TE' if te else 'noTE'
        config_path = f"config/projects/{project_id}/{model_name}_{complexity}_{scenario}_{lookback}h_{te_str}.yaml"
        
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
            
            # 抑制训练过程中的输出
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            # 重定向stdout和stderr来抑制训练输出
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
            
            # 预测
            model.eval()
            with torch.no_grad():
                # 确保数据在正确的设备上
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
            # 准备训练数据
            if Xf_tr is not None:
                X_tr = np.concatenate([Xh_tr.reshape(Xh_tr.shape[0], -1), Xf_tr.reshape(Xf_tr.shape[0], -1)], axis=1)
            else:
                X_tr = Xh_tr.reshape(Xh_tr.shape[0], -1)
            
            # 准备测试数据
            if Xf_te is not None:
                X_te = np.concatenate([Xh_te.reshape(Xh_te.shape[0], -1), Xf_te.reshape(Xh_te.shape[0], -1)], axis=1)
            else:
                X_te = Xh_te.reshape(Xh_te.shape[0], -1)
            
            # 训练模型（抑制输出）
            import io
            from contextlib import redirect_stdout, redirect_stderr
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                # 根据复杂度选择参数
                complexity = config.get('model_complexity', 'low')
                if complexity == 'high' and 'ml_high' in config['model_params']:
                    ml_params = config['model_params']['ml_high']
                elif complexity == 'low' and 'ml_low' in config['model_params']:
                    ml_params = config['model_params']['ml_low']
                else:
                    ml_params = config['model_params']  # 回退
                
                if model_name == 'RF':
                    model = train_rf(X_tr, y_tr, ml_params)
                elif model_name == 'XGB':
                    model = train_xgb(X_tr, y_tr, ml_params)
                elif model_name == 'LGBM':
                    model = train_lgbm(X_tr, y_tr, ml_params)
            
            # 预测
            y_pred = model.predict(X_te)
        
        # Capacity Factor不需要反标准化（已经是0-100范围）
        y_te_orig = y_te
        y_pred_orig = y_pred
        
        print(f"✅ {model_name} 模型训练完成")
        
        return y_te_orig, y_pred_orig, model_name
        
    except Exception as e:
        print(f"❌ {model_name} 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def plot_plant171_models(project_id, results, lookback, te, complexity, scenarios):
    """绘制厂171的所有模型对比（2*4子图形式）"""
    print(f"🎨 绘制项目 {project_id} 的所有模型对比...")
    
    # 模型名称映射
    model_names = {
        'LSTM': 'LSTM',
        'GRU': 'GRU', 
        'TCN': 'TCN',
        'Transformer': 'Transformer',
        'RF': 'Random Forest',
        'XGB': 'XGBoost',
        'LGBM': 'LightGBM',
        'Linear': 'Linear'
    }
    
    # 场景名称映射
    scenario_names = {
        'PV': 'PV',
        'PV_plus_NWP': 'PV+NWP',
        'PV_plus_NWP_plus': 'PV+NWP+',
        'PV_plus_HW': 'PV+HW',
        'NWP': 'NWP',
        'NWP_plus': 'NWP+'
    }
    
    # 创建子图：2行4列（8个模型）
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    # 定义颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    # 绘制每个模型的子图
    for i, (model_name, model_results) in enumerate(results.items()):
        ax = axes[i]
        
        # 获取ground truth（所有场景的ground truth都相同）
        y_true = None
        for scenario, (y_te, y_pred, _) in model_results.items():
            if y_te is not None:
                y_true = y_te
                break
        
        if y_true is None:
            ax.text(0.5, 0.5, f'{model_names[model_name]}\n无数据', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model_names[model_name]}', fontweight='bold')
            continue
        
        # 取前168小时的数据（7天）
        n_samples = min(168, len(y_true))
        y_true_plot = y_true[:n_samples].flatten()
        
        # 确保只取前168个时间步
        if len(y_true_plot) > 168:
            y_true_plot = y_true_plot[:168]
        
        # 重新计算timesteps以确保长度一致
        timesteps = range(len(y_true_plot))
        
        # 绘制Ground Truth
        ax.plot(timesteps, y_true_plot, 'gray', linewidth=2, label='Ground Truth', alpha=0.8)
        
        # 绘制每个场景的预测结果
        color_idx = 0
        for scenario, (y_te, y_pred, _) in model_results.items():
            if y_pred is not None:
                y_pred_plot = y_pred[:n_samples].flatten()
                
                # 确保只取前168个时间步
                if len(y_pred_plot) > 168:
                    y_pred_plot = y_pred_plot[:168]
                
                # 确保两个数组长度一致
                min_len = min(len(y_true_plot), len(y_pred_plot))
                y_pred_plot = y_pred_plot[:min_len]
                
                # 绘制预测结果
                ax.plot(timesteps[:min_len], y_pred_plot, 
                       color=colors[color_idx % len(colors)], 
                       linewidth=1.5, 
                       label=scenario_names[scenario], 
                       alpha=0.7)
                color_idx += 1
        
        ax.set_title(f'{model_names[model_name]}', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Capacity Factor (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    # 隐藏多余的子图
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    # 设置总标题
    te_str = 'TE' if te else 'noTE'
    plt.suptitle(f'Plant {project_id}: 8 Models × 6 Scenarios Prediction Results\n'
                 f'Lookback: {lookback}h, Time Encoding: {te_str}, Complexity: {complexity}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 创建输出目录
    output_dir = 'plant171_comparison_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    te_str = 'TE' if te else 'noTE'
    output_path = os.path.join(output_dir, f'plant_{project_id}_models_{lookback}h_{te_str}_{complexity}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图片已保存: {output_path}")
    
    plt.close()  # 关闭图形以释放内存

def main():
    """主函数"""
    print("🚀 绘制厂171的8个模型在6种情况下的预测图...")
    
    # 参数设置
    project_id = 171
    models = ['LSTM', 'GRU', 'TCN', 'Transformer', 'RF', 'XGB', 'LGBM', 'Linear']
    scenarios = ['PV', 'PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW', 'NWP', 'NWP_plus']
    lookbacks = [24, 72]  # 24小时和72小时
    te_options = [True, False]  # 时间编码
    complexities = ['low', 'high']  # 复杂度
    
    # 加载数据
    df = load_and_prepare_data(project_id)
    if df is None:
        print("❌ 无法加载数据，退出")
        return
    
    total_plots = 0
    
    # 遍历所有参数组合
    for lookback in lookbacks:
        for te in te_options:
            for complexity in complexities:
                print(f"\n{'='*80}")
                print(f"📊 处理参数组合: Lookback={lookback}h, TE={te}, Complexity={complexity}")
                print(f"{'='*80}")
                
                # 存储该参数组合的所有模型结果
                param_results = {}
                
                # 训练所有模型
                for model_name in models:
                    print(f"\n--- 处理 {model_name} 模型 ---")
                    
                    # 存储该模型在所有场景下的结果
                    model_results = {}
                    
                    for scenario in scenarios:
                        print(f"  🎯 场景: {scenario}")
                        
                        # 训练模型并预测
                        y_true, y_pred, name = train_and_predict_single_model(
                            df, project_id, model_name, scenario, lookback, te, complexity
                        )
                        
                        if y_true is not None:
                            model_results[scenario] = (y_true, y_pred, name)
                        else:
                            print(f"    ❌ 跳过 {scenario} 场景")
                    
                    # 存储该模型的结果
                    if model_results:
                        param_results[model_name] = model_results
                    else:
                        print(f"❌ 跳过 {model_name} 模型")
                
                # 绘制该参数组合的所有模型对比图
                if param_results:
                    plot_plant171_models(project_id, param_results, lookback, te, complexity, scenarios)
                    total_plots += 1
                else:
                    print(f"❌ 参数组合 {lookback}h-{te}-{complexity} 没有成功训练的模型")
    
    print(f"\n✅ 所有图片生成完成！")
    print(f"📊 总共生成了 {total_plots} 张图片")
    print(f"📁 图片保存在: {os.path.abspath('plant171_comparison_plots')}")
    
    # 列出生成的文件
    output_dir = 'plant171_comparison_plots'
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        files.sort()
        print(f"\n📋 生成的文件列表:")
        for i, file in enumerate(files, 1):
            print(f"{i:2d}. {file}")

if __name__ == "__main__":
    main()
