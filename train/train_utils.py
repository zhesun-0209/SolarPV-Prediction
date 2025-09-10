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
) -> torch.optim.lr_scheduler.StepLR:
    """
    Create a StepLR scheduler for learning rate decay.

    Args:
        optimizer: optimizer to wrap
        train_params: dict containing training parameters
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,  # 每20个epoch衰减一次
        gamma=0.5      # 学习率减半
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
