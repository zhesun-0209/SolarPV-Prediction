#!/usr/bin/env python3
"""
train/train_dl.py

Deep learning training pipeline for solar power forecasting.
Supports various architectures (Transformer, LSTM, GRU, TCN).
Records per-epoch timing and validation loss over time for plotting.
"""

import time
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from train.train_utils import (
    get_optimizer, get_scheduler, EarlyStopping,
    count_parameters
)
from models.transformer import Transformer
from models.rnn_models import LSTM, GRU
from models.tcn import TCNModel

def train_dl_model(
    config: dict,
    train_data: tuple,
    val_data: tuple,
    test_data: tuple,
    scalers: tuple
):
    """
    Train and evaluate a deep learning model.

    Returns:
        model:   trained PyTorch model
        metrics: dict with inverse-transformed predictions, loss, etc.
    """
    # Unpack data
    Xh_tr, Xf_tr, y_tr, hrs_tr, _ = train_data
    Xh_va, Xf_va, y_va, hrs_va, _ = val_data
    Xh_te, Xf_te, y_te, hrs_te, dates_te = test_data
    _, _, scaler_target = scalers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    def make_loader(Xh, Xf, y, hrs, bs, shuffle=False):
        tensors = [torch.tensor(Xh, dtype=torch.float32),
                   torch.tensor(hrs, dtype=torch.long)]
        if Xf is not None:
            tensors.insert(1, torch.tensor(Xf, dtype=torch.float32))
        tensors.append(torch.tensor(y, dtype=torch.float32))
        return DataLoader(TensorDataset(*tensors), batch_size=bs, shuffle=shuffle)

    bs = int(config['train_params']['batch_size'])
    train_loader = make_loader(Xh_tr, Xf_tr, y_tr, hrs_tr, bs, shuffle=True)
    val_loader   = make_loader(Xh_va, Xf_va, y_va, hrs_va, bs)
    test_loader  = make_loader(Xh_te, Xf_te, y_te, hrs_te, bs)

    # Model setup
    mp = config['model_params'].copy()
    mp['use_forecast'] = config.get('use_forecast', False)
    mp['past_hours'] = config['past_hours']
    mp['future_hours'] = config['future_hours']

    hist_dim = Xh_tr.shape[2]
    fcst_dim = Xf_tr.shape[2] if Xf_tr is not None else 0

    model_name = config['model']
    if model_name == 'Transformer':
        model = Transformer(hist_dim, fcst_dim, mp)
    elif model_name == 'LSTM':
        model = LSTM(hist_dim, fcst_dim, mp)
    elif model_name == 'GRU':
        model = GRU(hist_dim, fcst_dim, mp)
    elif model_name == 'TCN':
        model = TCNModel(hist_dim, fcst_dim, mp)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(device)

    # Training utils
    train_params = config['train_params']
    opt = get_optimizer(
        model,
        lr=float(train_params['learning_rate'])
    )
    sched = get_scheduler(opt, train_params)
    stopper = EarlyStopping(int(train_params['early_stop_patience']))

    mse_fn = torch.nn.MSELoss()
    logs = []
    total_time = 0.0
    total_train_time = 0.0
    total_inference_time = 0.0

    for ep in range(1, int(train_params['epochs']) + 1):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()

        for batch in train_loader:
            batch_start = time.time()
            if Xf_tr is not None:
                xh, xf, hrs, yb = batch
                xh, xf, hrs, yb = xh.to(device), xf.to(device), hrs.to(device), yb.to(device)
                preds = model(xh, xf)
            else:
                xh, hrs, yb = batch
                xh, hrs, yb = xh.to(device), hrs.to(device), yb.to(device)
                preds = model(xh)

            loss = mse_fn(preds, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            
            batch_time = time.time() - batch_start
            total_train_time += batch_time

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
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

        val_loss /= len(val_loader)
        sched.step(val_loss)
        
        # Fix time calculation: separate epoch time from cumulative time
        epoch_time = time.time() - epoch_start
        total_time += epoch_time


        logs.append({
            'epoch': ep,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'cum_time': total_time
        })

        if stopper.step(val_loss, model):
            print(f"Early stopping at epoch {ep}")
            break

    model.load_state_dict(stopper.best_state)

        # Test phase
    model.eval()
    all_preds = []
    inference_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            if Xf_te is not None:
                xh, xf, hrs, _ = batch
                xh, xf = xh.to(device), xf.to(device)
                preds = model(xh, xf)
            else:
                xh, hrs, _ = batch
                xh = xh.to(device)
                preds = model(xh)
            all_preds.append(preds.cpu().numpy())
    
    total_inference_time = time.time() - inference_start

    preds_arr = np.vstack(all_preds)
    y_true_arr = y_te  # already numpy

    # Inverse normalization (with column name for compatibility)
    p_flat = preds_arr.reshape(-1, 1)
    y_flat = y_true_arr.reshape(-1, 1)

    p_df = pd.DataFrame(p_flat, columns=['Electricity Generated'])
    y_df = pd.DataFrame(y_flat, columns=['Electricity Generated'])

    p_inv = scaler_target.inverse_transform(p_df).flatten()
    y_inv = scaler_target.inverse_transform(y_df).flatten()

    # === Raw test metrics (kWh scale) ===
    raw_mse = np.mean((y_inv - p_inv) ** 2)
    raw_rmse = np.sqrt(raw_mse)
    raw_mae = np.mean(np.abs(y_inv - p_inv))

    # === Normalized test metrics (0–1 scale)
    p_norm = scaler_target.transform(p_df).flatten()
    y_norm = scaler_target.transform(y_df).flatten()
    norm_mse = np.mean((y_norm - p_norm) ** 2)
    norm_rmse = np.sqrt(norm_mse)
    norm_mae = np.mean(np.abs(y_norm - p_norm))


    # 根据配置决定是否保存模型
    save_options = config.get('save_options', {})
    if save_options.get('save_model', False):
        save_dir = config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

    metrics = {
        'test_loss': raw_mse,
        'rmse': raw_rmse,
        'mae': raw_mae,
        'norm_test_loss': norm_mse,
        'norm_rmse': norm_rmse,
        'norm_mae': norm_mae,
        'epoch_logs': logs,
        'param_count': count_parameters(model),
        'train_time_sec': total_train_time,
        'inference_time_sec': total_inference_time,
        'predictions': p_inv.reshape(y_te.shape),
        'y_true': y_inv.reshape(y_te.shape),
        'dates': dates_te,
        'inverse_transformed': True
    }

    return model, metrics
