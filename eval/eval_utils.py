"""
eval/eval_utils.py

Utilities to save summary, predictions, training logs, and call plotting routines.
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
    Save summary.csv, predictions.csv, training_log.csv, and generate plots
    under config['save_dir'].

    Args:
      - model: trained DL or sklearn model
      - metrics: must contain keys:
          'test_loss','train_time_sec','param_count','rmse','mae',
          'predictions'(n,h), 'y_true'(n,h),
          'dates'(n), 'epoch_logs' (list of dicts with 'epoch','train_loss','val_loss','epoch_time','cum_time')
      - dates: fallback list of datetimes
      - y_true / Xh_test / Xf_test: unused here
      - config: includes 'save_dir','model','plot_days','scaler_target'
    """
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 1) summary.csv
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
    }
    pd.DataFrame([summary]).to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    # 2) predictions.csv
    preds = metrics['predictions']
    yts   = metrics['y_true']

    scaler = config.get('scaler_target', None)
    if scaler is not None:
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        yts   = scaler.inverse_transform(yts.reshape(-1, 1)).reshape(yts.shape)

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
                'hour':              int(hrs[i,h]) if hrs is not None else dt.hour,
                'y_true':            float(yts[i,h]),
                'y_pred':            float(preds[i,h])
            })
    pd.DataFrame(records).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    # 3) training_log.csv
    if 'epoch_logs' in metrics:
        pd.DataFrame(metrics['epoch_logs']).to_csv(
            os.path.join(save_dir, "training_log.csv"), index=False
        )

    # 4) plots
    days = config.get('plot_days', None)
    plot_forecast(
        dates_list, yts, preds,
        save_dir, model_name=config['model'], days=days
    )
    if 'epoch_logs' in metrics:
        plot_training_curve(metrics['epoch_logs'], save_dir, model_name=config['model'])
        plot_val_loss_over_time(metrics['epoch_logs'], save_dir, model_name=config['model'])

    print(f"[INFO] Results saved in {save_dir}")
