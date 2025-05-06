# eval/plot_utils.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_forecast(
    dates: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    model_name: str = None,
    days: int = None
):
    """
    Plot continuous forecast vs true values for the test set and save.

    Args:
        dates:      list of datetime objects (end of each window)
        y_true:     array shape (n_windows, horizon) of true values
        y_pred:     array shape (n_windows, horizon) of predictions
        save_dir:   directory to save the figure
        model_name: optional label for the title
        days:       number of days to display (None shows all)
    """
    os.makedirs(save_dir, exist_ok=True)
    horizon = y_true.shape[1]

    # Build full hourly timeline and flatten
    times, true_vals, pred_vals = [], [], []
    for i, dt_end in enumerate(dates):
        start = pd.to_datetime(dt_end) - pd.Timedelta(hours=horizon - 1)
        idx = pd.date_range(start, periods=horizon, freq='h')
        times.extend(idx)
        true_vals.extend(y_true[i])
        pred_vals.extend(y_pred[i])

    df = pd.DataFrame({'datetime': times, 'true': true_vals, 'pred': pred_vals})
    df = df.set_index('datetime')

    # Optionally limit to first `days` days
    if days is not None:
        max_points = days * 24
        df = df.iloc[:max_points]

    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df['true'], label='True')
    plt.plot(df.index, df['pred'], '--', label='Predicted')
    plt.xlabel('Datetime')
    plt.ylabel('Electricity Generated')
    title = 'Forecast vs True'
    if model_name:
        title = f'{model_name} ' + title
    if days is not None:
        title += f' (first {days} days)'
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'forecast_comparison.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved forecast plot to {fig_path}")


def plot_training_curve(
    epoch_logs: list,
    save_dir: str,
    model_name: str = None
):
    """
    Plot training and validation loss curves over epochs and save.

    Args:
        epoch_logs: list of dicts with keys 'epoch', 'train_loss', 'val_loss'
        save_dir:   directory to save the figure
        model_name: optional label for the title
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs       = [log['epoch'] for log in epoch_logs]
    train_losses = [log['train_loss'] for log in epoch_logs]
    val_losses   = [log['val_loss']   for log in epoch_logs]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = 'Training Curve'
    if model_name:
        title = f'{model_name} ' + title
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved training curve to {fig_path}")


def plot_val_loss_over_time(
    epoch_logs: list,
    save_dir: str,
    model_name: str = None
):
    """
    Plot validation loss vs cumulative training time and save.

    Args:
        epoch_logs: list of dicts with keys 'cum_time' and 'val_loss'
        save_dir:   directory to save the figure
        model_name: optional label for the title
    """
    os.makedirs(save_dir, exist_ok=True)
    times      = [log['cum_time'] for log in epoch_logs]
    val_losses = [log['val_loss'] for log in epoch_logs]

    plt.figure(figsize=(10, 5))
    plt.plot(times, val_losses, marker='o')
    plt.xlabel('Cumulative Training Time (s)')
    plt.ylabel('Validation Loss')
    title = 'Validation Loss over Time'
    if model_name:
        title = f'{model_name} ' + title
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'val_loss_over_time.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved validation-loss-over-time plot to {fig_path}")
