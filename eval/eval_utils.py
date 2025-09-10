"""
eval/eval_utils.py

Utilities to save summary, predictions, training logs, and call plotting routines.
"""

import os
import pandas as pd
import numpy as np
from eval.plot_utils import plot_forecast, plot_training_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===== Define Deep Learning model names =====
DL_MODELS = {"Transformer", "LSTM", "GRU", "TCN"}

def save_results(
    model,
    metrics: dict,
    dates: list,
    y_true: np.ndarray,
    Xh_test: np.ndarray,
    Xf_test: np.ndarray,
    config: dict
):
    """
    Save summary.csv, predictions.csv, training_log.csv, and generate plots
    under config['save_dir'].

    Args:
        model:   Trained DL or sklearn model
        metrics: Dictionary containing:
                 'test_loss', 'train_time_sec', 'param_count', 'rmse', 'mae',
                 'predictions' (n,h), 'y_true' (n,h),
                 'dates' (n), 'epoch_logs' (list of dicts)
        dates:   List of datetime strings
        y_true, Xh_test, Xf_test: Used for legacy or optional plots
        config:  Dictionary with keys like 'save_dir', 'model', 'plot_days', 'scaler_target'
    """
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Extract predictions and ground truth
    preds = metrics['predictions']
    yts   = metrics['y_true']

    # ===== Capacity Factor不需要逆标准化（已经是0-100范围） =====
    # 数据已经是原始尺度，直接使用

    # ===== 计算损失指标 =====
    # 所有计算方式在数学上等价，直接计算一次即可
    test_mse = np.mean((preds - yts) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(preds - yts))
    
    # 只计算原始尺度指标

    # 获取保存选项
    save_options = config.get('save_options', {})
    
    # ===== 1. Save summary.csv =====
    if save_options.get('save_summary', True):
        # 使用原始尺度（kWh）作为主要评估指标
        summary = {
            'model':           config['model'],
            'use_hist_weather': config.get('use_hist_weather', False),
            'use_forecast':    config.get('use_forecast', False),
            'past_hours':      config['past_hours'],
            'future_hours':    config['future_hours'],
            
            # 主要指标
            'test_loss':       test_mse,   # 整个测试集MSE (Capacity Factor²)
            'rmse':            test_rmse,  # 整个测试集RMSE (Capacity Factor)
            'mae':             test_mae,   # 整个测试集MAE (Capacity Factor)
            
            # 性能指标
            'train_time_sec':  metrics.get('train_time_sec'),
            'inference_time_sec': metrics.get('inference_time_sec', np.nan),
            'param_count':     metrics.get('param_count'),
            'samples_count':   len(preds),  # 测试样本数量
        }
        pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    # ===== 2. Save predictions.csv =====
    if save_options.get('save_predictions', True):
        hrs = metrics.get('hours')
        dates_list = metrics.get('dates', dates)
        records = []
        n_samples, horizon = preds.shape
        
        # Handle case where hours information is not available
        if hrs is None:
            # Generate default hour sequence if not provided
            hrs = np.tile(np.arange(horizon), (n_samples, 1))
        
        for i in range(n_samples):
            start = pd.to_datetime(dates_list[i]) - pd.Timedelta(hours=horizon - 1)
            for h in range(horizon):
                dt = start + pd.Timedelta(hours=h)
                records.append({
                    'window_index':      i,
                    'forecast_datetime': dt,
                    'hour':              int(hrs[i, h]) if hrs is not None else dt.hour,
                    'y_true':            float(yts[i, h]),
                    'y_pred':            float(preds[i, h])
                })
        pd.DataFrame(records).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    # ===== 3. Save training log (only if DL) =====
    is_dl = config['model'] in DL_MODELS
    if is_dl and 'epoch_logs' in metrics and save_options.get('save_training_log', True):
        pd.DataFrame(metrics['epoch_logs']).to_csv(
            os.path.join(save_dir, "training_log.csv"), index=False
        )

    # ===== 4. Save plots =====
    days = config.get('plot_days', None)
    
    # 保存预测对比图
    if save_options.get('save_forecast_plot', False):
        plot_forecast(dates_list, yts, preds, save_dir, model_name=config['model'], days=days)

    # 保存训练曲线图
    if is_dl and 'epoch_logs' in metrics and save_options.get('save_training_curve', True):
        plot_training_curve(metrics['epoch_logs'], save_dir, model_name=config['model'])
    


    print(f"[INFO] Results saved in {save_dir}")
