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
        preds = scaler.inverse_transform(pd.DataFrame(preds.reshape(-1, 1), columns=['Electricity Generated'])).values.reshape(preds.shape)
        yts   = scaler.inverse_transform(pd.DataFrame(yts.reshape(-1, 1), columns=['Electricity Generated'])).values.reshape(yts.shape)

    # ===== [NEW] Compute normalized errors using train-set scaler (if needed) =====
    if scaler is not None:
        # Fit scaler on flattened y_true (real values), get normalization transform
        norm_scaler = scaler  # We assume same scaler used for train/val/test
        preds_norm = norm_scaler.transform(pd.DataFrame(preds.reshape(-1, 1), columns=['Electricity Generated'])).flatten()
        yts_norm   = norm_scaler.transform(pd.DataFrame(yts.reshape(-1, 1), columns=['Electricity Generated'])).flatten()
  
        norm_mse  = mean_squared_error(yts_norm, preds_norm)
        norm_rmse = np.sqrt(norm_mse)
        norm_mae  = mean_absolute_error(yts_norm, preds_norm)
    else:
        norm_mse = norm_rmse = norm_mae = np.nan

    # ===== 1. Save summary.csv =====
    summary = {
        'model':           config['model'],
        'use_feature':     config.get('use_feature', True),
        'past_hours':      config['past_hours'],
        'future_hours':    config['future_hours'],
        'test_loss':       metrics.get('test_loss'),
        'train_time_sec':  metrics.get('train_time_sec'),
        'param_count':     metrics.get('param_count'),
        'rmse':            metrics.get('rmse', np.nan),
        'mae':             metrics.get('mae', np.nan),
        'norm_test_loss':  norm_mse,
        'norm_rmse':       norm_rmse,
        'norm_mae':        norm_mae,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    # ===== 2. Save predictions.csv =====
    hrs = metrics.get('hours')
    dates_list = metrics.get('dates', dates)
    records = []
    n_samples, horizon = preds.shape
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
    if is_dl and 'epoch_logs' in metrics:
        pd.DataFrame(metrics['epoch_logs']).to_csv(
            os.path.join(save_dir, "training_log.csv"), index=False
        )

    # ===== 4. Save plots =====
    days = config.get('plot_days', None)
    plot_forecast(dates_list, yts, preds, save_dir, model_name=config['model'], days=days)

    if is_dl and 'epoch_logs' in metrics:
        plot_training_curve(metrics['epoch_logs'], save_dir, model_name=config['model'])
        plot_val_loss_over_time(metrics['epoch_logs'], save_dir, model_name=config['model'])

    print(f"[INFO] Results saved in {save_dir}")
