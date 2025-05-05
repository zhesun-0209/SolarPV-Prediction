"""
eval/eval_utils.py

Evaluation and result saving utilities.
"""
import os
import pandas as pd
import numpy as np
import torch
from eval.plot_utils import plot_forecast, plot_training_curve


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

    metrics must include:
      - 'test_loss', 'train_time_sec'
      - 'rmse', 'mae' (for ML)
      - 'param_count'
      - 'predictions': ndarray (n_samples, horizon)
      - 'y_true':      ndarray (n_samples, horizon)
      - 'dates':       list of end-of-window datetimes
      - 'hours':       ndarray (n_samples, horizon), optional
      - 'epoch_logs':  list of dicts {'epoch','train_loss','val_loss'}

    Args:
        model: trained model
        metrics: dict of metrics and arrays
        dates: list of datetime64 for each window end
        y_true: numpy array (n_samples, horizon)
        Xh_test, Xf_test: unused here
        config: contains 'save_dir', 'model', 'use_feature', 'past_hours', 'future_hours'
    """
    save_dir = config['save_dir']
    model_name = config['model']
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 1) summary.csv
    summary = {
        'model':         model_name,
        'use_feature':   config.get('use_feature', True),
        'past_hours':    config['past_hours'],
        'future_hours':  config['future_hours'],
        'test_loss':     metrics.get('test_loss'),
        'train_time_sec':metrics.get('train_time_sec'),
        'param_count':   metrics.get('param_count'),
        'rmse':          metrics.get('rmse', np.nan),
        'mae':           metrics.get('mae', np.nan)
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(model_dir, 'summary.csv'), index=False
    )

    # 2) predictions.csv
    preds = metrics['predictions']
    yts   = metrics['y_true']
    hrs   = metrics.get('hours')
    dates = metrics.get('dates', dates)
    records = []
    n_samples, horizon = preds.shape
    for i in range(n_samples):
        # compute full datetime index for this window
        start_dt = pd.to_datetime(dates[i]) - pd.Timedelta(hours=horizon-1)
        times = [start_dt + pd.Timedelta(hours=h) for h in range(horizon)]
        for h, dt in enumerate(times):
            records.append({
                'window_index':     i,
                'forecast_datetime': dt,
                'hour':             int(hrs[i,h]) if hrs is not None else dt.hour,
                'true':             float(yts[i,h]),
                'predicted':        float(preds[i,h])
            })
    pd.DataFrame(records).to_csv(
        os.path.join(model_dir, 'predictions.csv'), index=False
    )

    # 3) training_log.csv
    if 'epoch_logs' in metrics:
        pd.DataFrame(metrics['epoch_logs']).to_csv(
            os.path.join(model_dir, 'training_log.csv'), index=False
        )

    # 4) plot figures
    plot_forecast(
        dates, metrics['y_true'], metrics['predictions'], model_dir,
        model_name=model_name, days=config.get('plot_days', 7)
    )
    if 'epoch_logs' in metrics:
        plot_training_curve(metrics['epoch_logs'], model_dir, model_name=model_name)

    print(f"[INFO] Results saved in {model_dir}")
