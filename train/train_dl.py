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
    get_optimizer, get_scheduler,
    count_parameters
)
from eval.metrics_utils import calculate_metrics, calculate_mse
from utils.gpu_utils import get_gpu_memory_used
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
    
    # 根据模型复杂度获取epoch数
    complexity = config.get('model_complexity', 'medium')
    epoch_params = config.get('epoch_params', {'low': 20, 'medium': 50, 'high': 100})
    epochs = epoch_params.get(complexity, 50)

    mse_fn = torch.nn.MSELoss()
    logs = []
    total_time = 0.0
    total_train_time = 0.0
    total_inference_time = 0.0

    for ep in range(1, epochs + 1):
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

        # 不再使用早停，训练完整的epoch数

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

    # Capacity Factor不需要逆标准化（已经是0-100范围）
    p_inv = preds_arr.flatten()
    y_inv = y_true_arr.flatten()

    # === 计算所有评估指标 ===
    # 计算MSE
    raw_mse = calculate_mse(y_true_arr, preds_arr)
    
    # 计算所有指标
    all_metrics = calculate_metrics(y_true_arr, preds_arr)
    
    # 提取基本指标
    raw_rmse = all_metrics['rmse']
    raw_mae = all_metrics['mae']

    # 根据配置决定是否保存模型
    save_options = config.get('save_options', {})
    if save_options.get('save_model', False):
        save_dir = config['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

    # 获取最佳epoch和最终学习率
    best_epoch = max(logs, key=lambda x: x['val_loss'])['epoch'] if logs else 1
    final_lr = opt.param_groups[0]['lr']
    
    # 获取GPU内存使用量
    gpu_memory_used = get_gpu_memory_used()
    
    metrics = {
        'test_loss': raw_mse,
        'rmse': raw_rmse,
        'mae': raw_mae,
        'nrmse': all_metrics['nrmse'],
        'r_square': all_metrics['r_square'],
        'mape': all_metrics['mape'],
        'smape': all_metrics['smape'],
        'best_epoch': best_epoch,
        'final_lr': final_lr,
        'gpu_memory_used': gpu_memory_used,
        'epoch_logs': logs,
        'param_count': count_parameters(model),
        'train_time_sec': total_train_time,
        'inference_time_sec': total_inference_time,
        'samples_count': len(y_te),
        'predictions': p_inv.reshape(y_te.shape),
        'y_true': y_inv.reshape(y_te.shape),
        'dates': dates_te,
        'inverse_transformed': True
    }

    return model, metrics
