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
    Create an AdamW optimizer for the given model.
    使用AdamW解决周期性问题，有更好的泛化能力

    Args:
        model: PyTorch model
        lr: learning rate
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_params: dict
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Create a ReduceLROnPlateau scheduler for learning rate decay.
    更好的学习率调度策略，解决周期性问题

    Args:
        optimizer: optimizer to wrap
        train_params: dict containing training parameters
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,     # 学习率减半
        patience=8,     # 8个epoch没有改善就降低学习率
        min_lr=1e-6     # 最小学习率
    )




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
