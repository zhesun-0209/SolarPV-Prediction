"""
eval/eval_utils.py

Evaluation and result saving utilities.
"""
import os
import pandas as pd
import numpy as np
from eval.plot_utils import plot_forecast, plot_training_curve, plot_val_loss_over_time


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
    Save summary.csv, predictions.csv, training_log.csv, and generate plots.

    Args:
        model: trained model or sklearn estimator
        metrics: dict with keys:
            - test_loss, train_time_sec, param_count, rmse, mae
            - predictions (n_samples, horizon)
            - y_true       (n_samples, horizon)
            - dates        list of end-of-window datetimes
            - epoch_logs   list of dicts with epoch, train_loss, val_loss, epoch_time, cum_time
        dates: list of datetime for test windows (fallback)
        y_true: ground truth array
        Xh_test/Xf_test: unused
        config: dict with save_dir, model, plot_days, use_feature, past_hours, future_hours
    """
    save_dir   = config['save_dir']
    model_name = config['model']
    model_dir  = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 1) summary.csv
    summary = {
        'model':           model_name,
        'use_feature':     config.get('use_feature', True),
        'past_hours':      config['past_hours'],
        'future_hours':    config['future_hours'],
        'test_loss':       metrics.get('test_loss'),
        'train_time_sec':  metrics.get('train_time_sec'),
        'param_count':     metrics.get('param_count'),
        'rmse':            metrics.get('rmse', np.nan),
        'mae':             metrics.get('mae', np.nan)
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(model_dir, 'summary.csv'), index=False
    )

    # 2) predictions.csv
    preds      = metrics['predictions']
    yts        = metrics['y_true']
    hrs        = metrics.get('hours')
    dates_list = metrics.get('dates', dates)
    records = []
    n_samples, horizon = preds.shape
    for i in range(n_samples):
        start = pd.to_datetime(dates_list[i]) - pd.Timedelta(hours=horizon-1)
        for h in range(horizon):
            dt = start + pd.Timedelta(hours=h)
            records.append({
                'window_index':      i,
                'forecast_datetime': dt,
                'hour':              int(hrs[i, h]) if hrs is not None else dt.hour,
                'y_true':            float(yts[i, h]),
                'y_pred':            float(preds[i, h])
            })
    pd.DataFrame(records).to_csv(
        os.path.join(model_dir, 'predictions.csv'), index=False
    )

    # 3) training_log.csv
    if 'epoch_logs' in metrics:
        df_log = pd.DataFrame(metrics['epoch_logs'])
        df_log.to_csv(
            os.path.join(model_dir, 'training_log.csv'), index=False
        )

    # 4) Generate plots
    # Forecast plot for specified days
    days = config.get('plot_days', config['future_hours'] // 24)
    plot_forecast(
        metrics['dates'],
        metrics['y_true'],
        metrics['predictions'],
        model_dir,
        model_name=model_name,
        days=days
    )
    # Loss curves
    if 'epoch_logs' in metrics:
        plot_training_curve(
            metrics['epoch_logs'], model_dir, model_name=model_name
        )
        plot_val_loss_over_time(
            metrics['epoch_logs'], model_dir, model_name=model_name
        )

    print(f"[INFO] Results saved in {model_dir}")
