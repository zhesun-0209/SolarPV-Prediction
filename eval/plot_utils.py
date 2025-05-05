"""
eval/plot_utils.py

Plotting utilities for solar power forecasting results and training curves.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    dates: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    model_name: str = None,
    days: int = 7
):
    """
    Plot continuous forecast vs true values for the first `days` days of the test set and save.

    Args:
        dates:     list of datetime objects for each window end
        y_true:    array of shape (n_windows, horizon)
        y_pred:    array of shape (n_windows, horizon)
        save_dir:  directory to save the figure
        model_name: optional label for title
        days:      number of days to plot (default 7)
    """
    os.makedirs(save_dir, exist_ok=True)
    horizon = y_true.shape[1]

    # Build full hourly timeline and flatten
    times, true_vals, pred_vals = [], [], []
    for i, dt_end in enumerate(dates):
        start = pd.to_datetime(dt_end) - pd.Timedelta(hours=horizon - 1)
        idx = pd.date_range(start, periods=horizon, freq='H')
        times.extend(idx)
        true_vals.extend(y_true[i])
        pred_vals.extend(y_pred[i])

    df = pd.DataFrame({
        'datetime': times,
        'true':      true_vals,
        'pred':      pred_vals
    }).set_index('datetime')

    # Determine how many hours to plot
    max_hours = days * 24
    df_plot = df.iloc[:max_hours]

    plt.figure(figsize=(15, 5))
    plt.plot(df_plot.index, df_plot['true'], label='True')
    plt.plot(df_plot.index, df_plot['pred'], '--', label='Predicted')
    plt.xlabel('Datetime')
    plt.ylabel('Electricity Generated')
    title = f"Forecast vs True (first {days} days)"
    if model_name:
        title = f"{model_name} {title}"
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig_name = f"forecast_{days}d.png"
    fig_path = os.path.join(save_dir, fig_name)
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
        epoch_logs: list of dicts with keys 'epoch','train_loss','val_loss'
        save_dir:   directory to save the figure
        model_name: optional label for title
    """
    os.makedirs(save_dir, exist_ok=True)
    df_logs = pd.DataFrame(epoch_logs)

    plt.figure(figsize=(10, 6))
    plt.plot(df_logs['epoch'], df_logs['train_loss'], label='Train Loss')
    plt.plot(df_logs['epoch'], df_logs['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = 'Training Curve'
    if model_name:
        title = f"{model_name} {title}"
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"training_curve.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved training curve to {fig_path}")
