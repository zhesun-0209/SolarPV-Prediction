"""
eval/eval_utils.py

Utilities to save summary, predictions, training logs, and call plotting routines.
"""

import os
import pandas as pd
import numpy as np
from eval.plot_utils import plot_forecast, plot_training_curve, plot_val_loss_over_time
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

    # ===== [NEW] Optional inverse_transform (only if not already done) =====
    scaler = config.get("scaler_target", None)
    already_inverse = metrics.get("inverse_transformed", False)
    if scaler is not None and not already_inverse:
        preds = scaler.inverse_transform(pd.DataFrame(preds.reshape(-1, 1), columns=['Electricity Generated'])).reshape(preds.shape)
        yts   = scaler.inverse_transform(pd.DataFrame(yts.reshape(-1, 1), columns=['Electricity Generated'])).reshape(yts.shape)

    # ===== 计算多种损失指标 =====
    def calculate_losses(preds, yts, method='overall'):
        """计算不同方式的损失函数"""
        if method == 'overall':
            # 整个测试集平均（当前方式）
            mse = np.mean((preds - yts) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(preds - yts))
            
        elif method == 'hourly':
            # 每小时平均：先按小时计算，再求平均
            hourly_errors = []
            hourly_mae_errors = []
            for h in range(preds.shape[1]):  # future_hours
                hour_mse = np.mean((preds[:, h] - yts[:, h]) ** 2)
                hour_mae = np.mean(np.abs(preds[:, h] - yts[:, h]))
                hourly_errors.append(hour_mse)
                hourly_mae_errors.append(hour_mae)
            mse = np.mean(hourly_errors)
            rmse = np.sqrt(mse)
            mae = np.mean(hourly_mae_errors)
            
        elif method == 'daily':
            # 每天平均：每个样本（天）的误差
            daily_errors = []
            daily_mae_errors = []
            for i in range(preds.shape[0]):
                day_mse = np.mean((preds[i] - yts[i]) ** 2)
                day_mae = np.mean(np.abs(preds[i] - yts[i]))
                daily_errors.append(day_mse)
                daily_mae_errors.append(day_mae)
            mse = np.mean(daily_errors)
            rmse = np.sqrt(mse)
            mae = np.mean(daily_mae_errors)
            
        elif method == 'sample':
            # 每个样本平均：先计算每个样本的误差，再求平均
            sample_errors = []
            sample_mae_errors = []
            for i in range(preds.shape[0]):
                sample_mse = np.mean((preds[i] - yts[i]) ** 2)
                sample_mae = np.mean(np.abs(preds[i] - yts[i]))
                sample_errors.append(sample_mse)
                sample_mae_errors.append(sample_mae)
            mse = np.mean(sample_errors)
            rmse = np.sqrt(mse)
            mae = np.mean(sample_mae_errors)
        
        return mse, rmse, mae

    # 计算主要指标（overall方式）
    test_mse, test_rmse, test_mae = calculate_losses(preds, yts, 'overall')
    
    # 计算辅助指标
    hourly_mse, hourly_rmse, hourly_mae = calculate_losses(preds, yts, 'hourly')
    daily_mse, daily_rmse, daily_mae = calculate_losses(preds, yts, 'daily')
    sample_mse, sample_rmse, sample_mae = calculate_losses(preds, yts, 'sample')
    
    # 计算标准化误差（用于辅助分析）
    if scaler is not None and not already_inverse:
        # 使用训练集的scaler进行标准化
        preds_norm = scaler.transform(pd.DataFrame(preds.reshape(-1, 1), columns=['Electricity Generated'])).flatten()
        yts_norm   = scaler.transform(pd.DataFrame(yts.reshape(-1, 1), columns=['Electricity Generated'])).flatten()
        
        norm_mse  = mean_squared_error(yts_norm, preds_norm)
        norm_rmse = np.sqrt(norm_mse)
        norm_mae  = mean_absolute_error(yts_norm, preds_norm)
    else:
        # 如果已经反标准化或没有scaler，计算相对误差
        if np.mean(yts) > 0:
            norm_mse  = np.mean(((preds - yts) / yts) ** 2)
            norm_rmse = np.sqrt(norm_mse)
            norm_mae  = np.mean(np.abs((preds - yts) / yts))
        else:
            norm_mse = norm_rmse = norm_mae = np.nan

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
            
            # 主要指标（overall方式）
            'test_loss':       test_mse,   # 整个测试集MSE (kWh²)
            'rmse':            test_rmse,  # 整个测试集RMSE (kWh)
            'mae':             test_mae,   # 整个测试集MAE (kWh)
            
            # 辅助指标
            'hourly_rmse':     hourly_rmse,  # 每小时平均RMSE
            'hourly_mae':      hourly_mae,   # 每小时平均MAE
            'daily_rmse':      daily_rmse,   # 每天平均RMSE
            'daily_mae':       daily_mae,    # 每天平均MAE
            'sample_rmse':     sample_rmse,  # 每样本平均RMSE
            'sample_mae':      sample_mae,   # 每样本平均MAE
            
            # 性能指标
            'train_time_sec':  metrics.get('train_time_sec'),
            'inference_time_sec': metrics.get('inference_time_sec', np.nan),
            'param_count':     metrics.get('param_count'),
            'samples_count':   len(preds),  # 测试样本数量
            
            # 标准化指标
            'norm_test_loss':  norm_mse,   # 标准化/相对MSE
            'norm_rmse':       norm_rmse,  # 标准化/相对RMSE
            'norm_mae':        norm_mae,   # 标准化/相对MAE
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
    
    # 保存验证损失图
    if is_dl and 'epoch_logs' in metrics and save_options.get('save_val_loss_plot', False):
        plot_val_loss_over_time(metrics['epoch_logs'], save_dir, model_name=config['model'])


    print(f"[INFO] Results saved in {save_dir}")
