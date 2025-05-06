# ===== File: train/loss_utils.py =====
import torch
import os
import matplotlib.pyplot as plt


def weighted_mse_loss_with_hour(y_pred, y_true, hour_tensor,
                                 alpha=3.0, peak_start=10, peak_end=14):
    """
    Compute MSE loss with extra weight during peak hours.

    Args:
      y_pred:      Tensor of shape [B, T], model predictions
      y_true:      Tensor of shape [B, T], ground truth
      hour_tensor: Tensor of shape [B, T], hour index (0–23)
      alpha:       float, additional weight factor for peak hours
      peak_start:  int, start of peak hour window
      peak_end:    int, end of peak hour window
    Returns:
      scalar Tensor: weighted MSE
    """
    peak_mask = ((hour_tensor >= peak_start) & (hour_tensor <= peak_end)).float()
    weights = 1.0 + alpha * peak_mask
    return torch.mean(weights * (y_pred - y_true) ** 2)


def compute_dynamic_hour_weights(hour_errors, alpha=3.0, threshold=0.005):
    """
    Compute dynamic hour-wise weights based on validation errors.

    Args:
      hour_errors: dict mapping hour (0–23) to list of absolute errors
      alpha:       float, scaling factor for weights
      threshold:   float, errors below threshold are set to zero before scaling
    Returns:
      Tensor of shape [24]: normalized hour weights
    """
    hour_avg = torch.tensor([
        torch.tensor(hour_errors[h]).float().mean() if len(hour_errors[h]) > 0 else 0.0
        for h in range(24)
    ])
    hour_avg = torch.where(hour_avg < threshold, torch.zeros_like(hour_avg), hour_avg)
    nonzero = hour_avg[hour_avg > 0]
    if nonzero.numel() > 0:
        hour_avg = hour_avg / nonzero.mean()
    else:
        hour_avg = torch.ones(24)
    return hour_avg * alpha


def plot_hour_weights(hour_weights, output_path):
    """
    Plot and save dynamic hour-wise training weights.

    Args:
      hour_weights: Tensor of shape [24]
      output_path:  str, file path to save the plot PNG
    """
    hours = list(range(24))
    plt.figure(figsize=(10, 4))
    plt.bar(hours, hour_weights.cpu().numpy())
    plt.xlabel("Hour of Day")
    plt.ylabel("Weight")
    plt.title("Dynamic Hour-wise Loss Weights")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
