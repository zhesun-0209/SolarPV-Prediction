"""
train/train_utils.py

Helper utilities for training and evaluation:
  - Optimizer and scheduler creation
  - Early stopping
  - Model parameter counting
"""
import torch
from typing import Dict, List, Optional


def get_optimizer(
    model: torch.nn.Module,
    lr: float
) -> torch.optim.Optimizer:
    """
    Create an Adam optimizer for the given model.

    Args:
        model: PyTorch model
        lr: learning rate
    """
    return torch.optim.Adam(model.parameters(), lr=lr)


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
        patience: int = 20,
        delta: float = 1e-5
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



def get_loss_function(loss_type: str):
    """
    Return a loss function according to the selected type.

    Args:
        loss_type: 'mse'

    Returns:
        A callable loss function
    """
    mse_fn = torch.nn.MSELoss()

    if loss_type == "mse":
        return lambda preds, target: mse_fn(preds, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
