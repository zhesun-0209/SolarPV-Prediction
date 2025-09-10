#!/usr/bin/env python3
"""
评估指标计算工具
根据论文定义的指标计算MAE, RMSE, NRMSE, R², MAPE, sMAPE
"""

import numpy as np
from sklearn.metrics import r2_score

def calculate_metrics(y_true, y_pred):
    """
    计算所有评估指标
    
    Args:
        y_true: 真实值 (n_samples, n_hours)
        y_pred: 预测值 (n_samples, n_hours)
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 展平为一维数组
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'nrmse': np.nan,
            'r_square': np.nan,
            'mape': np.nan,
            'smape': np.nan
        }
    
    T = len(y_true_clean)
    
    # MAE
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    
    # NRMSE = RMSE / ȳ (其中ȳ是真实值的均值)
    y_mean = np.mean(y_true_clean)
    if y_mean != 0:
        nrmse = rmse / y_mean
    else:
        nrmse = np.nan
    
    # R²
    r_square = r2_score(y_true_clean, y_pred_clean)
    
    # MAPE (当y_t > 0时)
    mape_mask = y_true_clean > 0
    if np.any(mape_mask):
        mape = np.mean(np.abs(y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])
    else:
        mape = np.nan
    
    # sMAPE (对称平均绝对百分比误差)
    # 公式: sMAPE = (1/n) * Σ[2 * |y_t - ŷ_t| / (|y_t| + |ŷ_t|)]
    # 特殊情况: 当y_t = 0且ŷ_t = 0时，该项定义为0
    
    # 计算分母 (|y_t| + |ŷ_t|)
    denominator = np.abs(y_true_clean) + np.abs(y_pred_clean)
    
    # 避免除零错误：当分母为0时（即y_t = 0且ŷ_t = 0），该项为0
    smape_mask = denominator > 0
    
    if np.any(smape_mask):
        smape = np.mean(2 * np.abs(y_true_clean[smape_mask] - y_pred_clean[smape_mask]) / 
                       denominator[smape_mask])
    else:
        smape = np.nan
    
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'nrmse': round(nrmse, 4),
        'r_square': round(r_square, 4),
        'mape': round(mape, 4),
        'smape': round(smape, 4)
    }

def calculate_mse(y_true, y_pred):
    """
    计算MSE (用于test_loss)
    
    Args:
        y_true: 真实值 (n_samples, n_hours)
        y_pred: 预测值 (n_samples, n_hours)
    
    Returns:
        float: MSE值
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return np.nan
    
    mse = np.mean((y_true_clean - y_pred_clean) ** 2)
    return round(mse, 4)
