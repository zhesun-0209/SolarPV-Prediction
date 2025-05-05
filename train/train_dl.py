#!/usr/bin/env python3
"""
train/train_dl.py

Deep learning training pipeline for solar power forecasting.
Supports dynamic meta-weighted loss and various architectures (Transformer, LSTM, GRU, TCN).
Records per-epoch timing and validation loss over time for plotting.
"""
import os
import time
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from train.train_utils import (
    get_optimizer, get_scheduler, EarlyStopping,
    count_parameters, compute_dynamic_hour_weights
)
from models.transformer import Transformer
from models.rnn_models import LSTM, GRU
from models.tcn import TCNModel


def train_dl_model(
    config: dict,
    train_data: tuple,
    val_data:   tuple,
    test_data:  tuple,
    scalers:    tuple
):
    """
    Train and evaluate a deep learning model with optional dynamic meta-weighted loss.

    Args:
        config: configuration dict
        train_data: (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr)
        val_data:   (Xh_va, Xf_va, y_va, hrs_va, dates_va)
        test_data:  (Xh_te, Xf_te, y_te, hrs_te, dates_te)
        scalers:    (scaler_hist, scaler_fcst, scaler_target)

    Returns:
        model: trained PyTorch model
        metrics: dict with test_loss, epoch_logs, param_count
    """
    # Unpack data
    Xh_tr, Xf_tr, y_tr, hrs_tr, _ = train_data
    Xh_va, Xf_va, y_va, hrs_va, _ = val_data
    Xh_te, Xf_te, y_te, hrs_te, dates_te = test_data
    _, _, scaler_target = scalers

    # Build DataLoader
    def make_loader(Xh, Xf, y, hrs, bs, shuffle=False):
        tensors = [torch.tensor(Xh, dtype=torch.float32),
                   torch.tensor(hrs, dtype=torch.long)]
        if Xf is not None:
            tensors.insert(1, torch.tensor(Xf, dtype=torch.float32))
        tensors.append(torch.tensor(y, dtype=torch.float32))
        return DataLoader(TensorDataset(*tensors), batch_size=bs, shuffle=shuffle)

    bs = config['train_params']['batch_size']
    train_loader = make_loader(Xh_tr, Xf_tr, y_tr, hrs_tr, bs, shuffle=True)
    val_loader   = make_loader(Xh_va, Xf_va, y_va, hrs_va, bs)
    test_loader  = make_loader(Xh_te, Xf_te, y_te, hrs_te, bs)

    # Instantiate model
    model_name = config['model']
    hist_dim = Xh_tr.shape[2]
    fcst_dim = Xf_tr.shape[2] if Xf_tr is not None else 0

    if model_name == 'Transformer':
        model = Transformer(hist_dim, fcst_dim, config['model_params'])
    elif model_name == 'LSTM':
        model = LSTM(hist_dim, fcst_dim, config['model_params'])
    elif model_name == 'GRU':
        model = GRU(hist_dim, fcst_dim, config['model_params'])
    elif model_name == 'TCN':
        model = TCNModel(hist_dim, fcst_dim, config['model_params'])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Setup optimizer, scheduler, early stopping
    opt     = get_optimizer(model,
                            config['train_params']['learning_rate'],
                            config['train_params']['weight_decay'])
    sched   = get_scheduler(opt, config['train_params'])
    stopper = EarlyStopping(config['train_params']['early_stop_patience'])

    # Loss functions
    mse_fn   = torch.nn.MSELoss()
    use_meta = config.get('use_meta', False)
    # initialize hour weights if meta
    hour_weights = torch.ones(config['future_hours'], device=device) if use_meta else None

    logs = []
    total_time = 0.0
    # training loop
    for ep in range(1, config['train_params']['epochs'] + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if Xf_tr is not None:
                xh, xf, hrs, yb = batch
                xh, xf, hrs, yb = xh.to(device), xf.to(device), hrs.to(device), yb.to(device)
                preds = model(xh, xf)
            else:
                xh, hrs, yb = batch
                xh, hrs, yb = xh.to(device), hrs.to(device), yb.to(device)
                preds = model(xh)

            if use_meta:
                weights = hour_weights[hrs]
                loss = torch.mean(weights * (preds - yb) ** 2)
            else:
                loss = mse_fn(preds, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        hour_errors = defaultdict(list) if use_meta else None
        with torch.no_grad():
            for batch in val_loader:
                if Xf_va is not None:
                    xh, xf, hrs, yb = batch
                    xh, xf, hrs, yb = xh.to(device), xf.to(device), hrs.to(device), yb.to(device)
                    preds = model(xh, xf)
                else:
                    xh, hrs, yb = batch
                    xh, hrs, yb = xh.to(device), hrs.to(device), yb.to(device)
                    preds = model(xh)

                val_loss += mse_fn(preds, yb).item()
                if use_meta:
                    err = torch.abs(preds - yb)
                    for i in range(err.size(0)):
                        for j in range(err.size(1)):
                            hour_errors[int(hrs[i, j])].append(err[i, j].item())
        val_loss /= len(val_loader)

        # epoch timing
        epoch_time = time.time() - epoch_start
        total_time += epoch_time

        # update scheduler and dynamic weights
        sched.step(val_loss)
        if use_meta:
            new_w = compute_dynamic_hour_weights(
                hour_errors,
                alpha=config['model_params'].get('meta_alpha', 3.0),
                threshold=config['model_params'].get('meta_threshold', 0.005)
            )
            hour_weights = new_w.to(device)

        logs.append({
            'epoch':      ep,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'epoch_time': epoch_time,
            'cum_time':   total_time
        })

        if stopper.step(val_loss, model):
            print(f"Early stopping at epoch {ep}")
            break

    # load best model
    model.load_state_dict(stopper.best_state)

    # test inference
    model.eval()
    test_loss = 0.0
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            if Xf_te is not None:
                xh, xf, hrs, yb = batch
                xh, xf = xh.to(device), xf.to(device)
                preds = model(xh, xf)
            else:
                xh, hrs, yb = batch
                xh = xh.to(device)
                preds = model(xh)

            if use_meta:
                weights = hour_weights[hrs]
                test_loss += torch.mean(weights * (preds - yb.to(device))**2).item()
            else:
                test_loss += mse_fn(preds, yb.to(device)).item()
            all_preds.append(preds.cpu().numpy())
    test_loss /= len(test_loader)
    preds_arr = np.vstack(all_preds)

    # inverse scale
    y_flat = y_te.reshape(-1, 1)
    p_flat = preds_arr.reshape(-1,1)
    y_inv  = scaler_target.inverse_transform(y_flat).flatten()
    p_inv  = scaler_target.inverse_transform(p_flat).flatten()
    raw_mse = np.mean((y_inv - p_inv)**2)

    metrics = {
        'test_loss':   raw_mse,
        'epoch_logs':  logs,
        'param_count': count_parameters(model)
    }
    return model, metrics
