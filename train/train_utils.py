"""
train/train_utils.py

Helper utilities for training and evaluation:
  - Optimizer and scheduler creation
  - Early stopping
  - Model parameter counting
  - Dynamic hour-wise weight computation for meta-weighted loss
  - Plotting dynamic hour-wise weights per epoch
"""
import torch
from typing import Dict, List, Optional


def get_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float
) -> torch.optim.Optimizer:
    """
    Create an Adam optimizer for the given model.

    Args:
        model: PyTorch model
        lr: learning rate
        weight_decay: L2 regularization weight
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_params: dict
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Create a ReduceLROnPlateau scheduler.
    Patience is half of the early stopping patience by default.

    Args:
        optimizer: optimizer to wrap
        train_params: dict containing 'early_stop_patience'
    """
    patience = max(1, train_params.get('early_stop_patience', 10) // 2)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience
    )


class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss no longer improves.
    Saves the best model state dict.

    Args:
        patience: number of epochs to wait without improvement
        delta: minimum change in validation loss to qualify as improvement
    """
    def __init__(
        self,
        patience: int = 10,
        delta: float = 1e-4
    ):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state: Dict[str, torch.Tensor] = None

    def step(
        self,
        val_loss: float,
        model: torch.nn.Module
    ) -> bool:
        """
        Call after each validation epoch.
        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Returns:
        Total trainable parameter count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_dynamic_hour_weights(
    hour_errors: Dict[int, List[float]],
    alpha: float = 3.0,
    threshold: float = 0.005,
    save_dir: Optional[str] = None,
    epoch: Optional[int] = None
) -> torch.Tensor:
    """
    Compute dynamic hour-wise weights based on validation errors.

    Steps:
      1. Compute mean absolute error per hour (0–23).
      2. Zero out entries below `threshold`.
      3. Normalize non-zero values to mean=1.
      4. Scale by `alpha`.
      5. Optionally plot weights for the epoch.

    Args:
        hour_errors: mapping from hour to list of error values
        alpha: scaling factor for weight magnitudes
        threshold: errors below this are ignored
        save_dir: root directory to save plots (if provided)
        epoch: epoch number for naming plot file

    Returns:
        A Tensor of shape (24,) containing the weight for each hour.
    """
    hour_avg = torch.tensor([
        torch.tensor(hour_errors.get(h, [])).float().mean()
        if hour_errors.get(h) else 0.0
        for h in range(24)
    ])
    hour_avg = torch.where(hour_avg < threshold, torch.zeros_like(hour_avg), hour_avg)
    nonzero = hour_avg[hour_avg > 0]
    if nonzero.numel() > 0:
        hour_avg = hour_avg / nonzero.mean()
    else:
        hour_avg = torch.ones(24)
    weights = hour_avg * alpha
    # Optional plotting
    if save_dir and epoch is not None:
        plot_hour_weights(weights, save_dir, epoch)
    return weights


def plot_hour_weights(
    weights: torch.Tensor,
    save_dir: str,
    epoch: int
) -> None:
    """
    Plot and save dynamic hour-wise weight bar chart for a given epoch.

    Args:
        weights: Tensor of shape (24,) with hour weights.
        save_dir: root directory where 'hour_weights/' subfolder will be created
        epoch: epoch number for filename
    """
    import matplotlib.pyplot as plt
    import os

    out_dir = os.path.join(save_dir, 'hour_weights')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'epoch_{epoch:03d}.png')

    hours = list(range(24))
    plt.figure(figsize=(8, 3))
    plt.bar(hours, weights.cpu().numpy())
    plt.xticks(hours)
    plt.xlabel("Hour of Day")
    plt.ylabel("Weight")
    plt.title(f"Dynamic Hour Weights - Epoch {epoch}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
