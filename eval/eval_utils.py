"""
eval/eval_utils.py

Utilities to save summary, predictions, training logs, and call plotting routines.
"""

import os
import pandas as pd
import numpy as np
# 绘图功能已移除，默认不保存图片
from eval.excel_utils import save_plant_excel_results
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===== Define Deep Learning model names =====
DL_MODELS = {"Transformer", "LSTM", "GRU", "TCN"}

def save_results(
    model,
    metrics: dict,
    dates: list,
    y_true: np.ndarray,
    Xh_test: np.ndarray,
    Xf_test: np.ndarray,
    config: dict
):
    """
    Save summary.csv, predictions.csv, training_log.csv, and generate plots
    under config['save_dir'].

    Args:
        model:   Trained DL or sklearn model
        metrics: Dictionary containing:
                 'test_loss', 'train_time_sec', 'param_count', 'rmse', 'mae',
                 'predictions' (n,h), 'y_true' (n,h),
                 'dates' (n), 'epoch_logs' (list of dicts)
        dates:   List of datetime strings
        y_true, Xh_test, Xf_test: Used for legacy or optional plots
        config:  Dictionary with keys like 'save_dir', 'model', 'plot_days', 'scaler_target'
    """
    # 硬编码Drive路径，删除本地保存
    save_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(save_dir, exist_ok=True)

    # Extract predictions and ground truth
    preds = metrics['predictions']
    yts   = metrics['y_true']

    # ===== Capacity Factor不需要逆标准化（已经是0-100范围） =====
    # 数据已经是原始尺度，直接使用

    # ===== 计算损失指标 =====
    # 所有计算方式在数学上等价，直接计算一次即可
    test_mse = np.mean((preds - yts) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(preds - yts))
    
    # 计算R² (决定系数)
    y_mean = np.mean(yts)
    ss_tot = np.sum((yts - y_mean) ** 2)  # 总平方和
    ss_res = np.sum((yts - preds) ** 2)   # 残差平方和
    r_square = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 只计算原始尺度指标

    # 获取保存选项
    save_options = config.get('save_options', {})
    print(f"🔍 调试: save_options = {save_options}")
    print(f"🔍 调试: save_excel_results = {save_options.get('save_excel_results', True)}")
    
    # ===== 1. summary.csv 已完全禁用 =====
    # 不再保存summary.csv文件，只保存Excel文件
    # 定义summary变量供Excel保存使用
    summary = {
        'model':           config['model'],
        'use_hist_weather': config.get('use_hist_weather', False),
        'use_forecast':    config.get('use_forecast', False),
        'past_days':       config.get('past_days', 1),
        'model_complexity': config.get('model_complexity', 'low'),
        'correlation_level': config.get('correlation_level', 'high'),
        'use_time_encoding': config.get('use_time_encoding', True),
        'past_hours':      config['past_hours'],
        'future_hours':    config['future_hours'],
        
        # 主要指标
        'mse':             test_mse,   # 整个测试集MSE (Capacity Factor²)
        'rmse':            test_rmse,  # 整个测试集RMSE (Capacity Factor)
        'mae':             test_mae,   # 整个测试集MAE (Capacity Factor)
        'r_square':        r_square,   # R²决定系数
        
        # 性能指标
        'train_time_sec':  metrics.get('train_time_sec'),
        'inference_time_sec': metrics.get('inference_time_sec', np.nan),
        'param_count':     metrics.get('param_count'),
        'samples_count':   len(preds),  # 测试样本数量
    }
    # 不保存summary.csv，只保存Excel文件

    # ===== 2. Save predictions.csv =====
    if save_options.get('save_predictions', True):
        hrs = metrics.get('hours')
        dates_list = metrics.get('dates', dates)
        records = []
        n_samples, horizon = preds.shape
        
        # Handle case where hours information is not available
        if hrs is None:
            # Generate default hour sequence if not provided
            hrs = np.tile(np.arange(horizon), (n_samples, 1))
        
        for i in range(n_samples):
            start = pd.to_datetime(dates_list[i]) - pd.Timedelta(hours=horizon - 1)
            for h in range(horizon):
                dt = start + pd.Timedelta(hours=h)
                records.append({
                    'window_index':      i,
                    'forecast_datetime': dt,
                    'hour':              int(hrs[i, h]) if hrs is not None else dt.hour,
                    'y_true':            float(yts[i, h]),
                    'y_pred':            float(preds[i, h])
                })
        pd.DataFrame(records).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    # ===== 3. Save training log (only if DL) =====
    is_dl = config['model'] in DL_MODELS
    if is_dl and 'epoch_logs' in metrics and save_options.get('save_training_log', True):
        pd.DataFrame(metrics['epoch_logs']).to_csv(
            os.path.join(save_dir, "training_log.csv"), index=False
        )

    # ===== 4. Save plots =====
    days = config.get('plot_days', None)
    
    # 绘图功能已移除，默认不保存图片
    # 如需保存图片，请设置相应的save_options为True
    
    # 保存Excel结果文件（如果启用）
    print(f"🔍 调试: 准备保存Excel结果，条件判断: {save_options.get('save_excel_results', True)}")
    if save_options.get('save_excel_results', True):
        print(f"🔍 调试: 进入Excel保存逻辑")
        # 构建实验结果数据
        result_data = {
            'config': {
                'model': config['model'],
                'use_pv': config.get('use_pv', True),
                'use_hist_weather': config.get('use_hist_weather', False),
                'use_forecast': config.get('use_forecast', False),
                'weather_category': config.get('weather_category', 'irradiance'),
                'use_time_encoding': config.get('use_time_encoding', True),
                'past_days': config.get('past_days', 1),
                'model_complexity': config.get('model_complexity', 'low'),
                'epochs': config.get('epochs', 15),
                'batch_size': config.get('batch_size', 32),
                'learning_rate': config.get('learning_rate', 0.001)
            },
            'metrics': {
                'train_time_sec': summary['train_time_sec'],
                'inference_time_sec': summary['inference_time_sec'],
                'param_count': summary['param_count'],
                'samples_count': summary['samples_count'],
                'mse': summary['mse'],
                'rmse': summary['rmse'],
                'mae': summary['mae'],
                'nrmse': metrics.get('nrmse', np.nan),
                'r_square': summary['r_square'],
                'smape': metrics.get('smape', np.nan),
                'best_epoch': metrics.get('best_epoch', np.nan),
                'final_lr': metrics.get('final_lr', np.nan),
                'gpu_memory_used': metrics.get('gpu_memory_used', 0)
            }
        }
        
        # 保存到CSV文件（追加模式）
        from eval.excel_utils import append_plant_excel_results
        print(f"🔍 调试: plant_id={config.get('plant_id', 'unknown')}, save_dir={save_dir}")
        csv_file = append_plant_excel_results(
            plant_id=config.get('plant_id', 'unknown'),
            result=result_data,
            save_dir=save_dir
        )
        print(f"🔍 调试: CSV文件已保存到 {csv_file}")
    else:
        print(f"🔍 调试: 跳过Excel保存，save_excel_results = False")

    print(f"[INFO] Results saved in {save_dir}")

def save_season_hour_results(
    model,
    metrics: dict,
    dates: list,
    y_true: np.ndarray,
    Xh_test: np.ndarray,
    Xf_test: np.ndarray,
    config: dict
):
    """
    保存season and hour analysis结果
    为每个厂保存prediction.csv和summary.csv到指定的Drive路径
    
    Args:
        model: 训练好的模型
        metrics: 包含预测结果和指标的字典
        dates: 日期列表
        y_true: 真实值
        Xh_test, Xf_test: 测试数据
        config: 配置字典
    """
    # 设置Drive路径
    drive_path = "/content/drive/MyDrive/Solar PV electricity/hour and season analysis"
    os.makedirs(drive_path, exist_ok=True)
    
    # 提取预测结果和真实值
    preds = metrics['predictions']
    yts = metrics['y_true']
    
    # 计算指标
    test_mse = np.mean((preds - yts) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(preds - yts))
    
    # 计算R²
    y_mean = np.mean(yts)
    ss_tot = np.sum((yts - y_mean) ** 2)
    ss_res = np.sum((yts - preds) ** 2)
    r_square = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 计算NRMSE和SMAPE
    nrmse = (test_rmse / (test_mae + 1e-8)) * 100 if test_mae > 0 else 0
    smape = (2 * test_mae / (test_mae + 1e-8)) * 100 if test_mae > 0 else 0
    
    # 获取项目ID
    project_id = config.get('plant_id', 'unknown')
    model_name = config.get('model', 'unknown')
    
    # 1. 保存prediction.csv
    prediction_file = os.path.join(drive_path, f"{project_id}_prediction.csv")
    
    # 准备预测结果数据
    hrs = metrics.get('hours')
    dates_list = metrics.get('dates', dates)
    records = []
    n_samples, horizon = preds.shape
    
    # 处理小时信息
    if hrs is None:
        hrs = np.tile(np.arange(horizon), (n_samples, 1))
    
    for i in range(n_samples):
        start = pd.to_datetime(dates_list[i]) - pd.Timedelta(hours=horizon - 1)
        for h in range(horizon):
            dt = start + pd.Timedelta(hours=h)
            records.append({
                'date': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'ground_truth': float(yts[i, h]),
                'prediction': float(preds[i, h]),
                'model': model_name,
                'project_id': project_id,
                'window_index': i,
                'hour': int(hrs[i, h]) if hrs is not None else dt.hour
            })
    
    # 保存预测结果
    pred_df = pd.DataFrame(records)
    if os.path.exists(prediction_file):
        # 如果文件已存在，追加数据
        existing_df = pd.read_csv(prediction_file)
        pred_df = pd.concat([existing_df, pred_df], ignore_index=True)
    pred_df.to_csv(prediction_file, index=False)
    print(f"💾 预测结果已保存: {prediction_file}")
    
    # 2. 保存summary.csv
    summary_file = os.path.join(drive_path, f"{project_id}_summary.csv")
    
    # 准备summary数据
    summary_data = {
        'model': model_name,
        'weather_level': config.get('weather_category', 'unknown'),
        'lookback_hours': config.get('past_hours', 24),
        'complexity_level': config.get('model_complexity', 'unknown').replace('level', ''),
        'dataset_scale': '80%',
        'use_pv': config.get('use_pv', False),
        'use_hist_weather': config.get('use_hist_weather', False),
        'use_forecast': config.get('use_forecast', False),
        'use_time_encoding': config.get('use_time_encoding', False),
        'past_days': config.get('past_days', 1),
        'use_ideal_nwp': config.get('use_ideal_nwp', False),
        'selected_weather_features': str(config.get('selected_weather_features', [])),
        'epochs': config.get('epochs', 0),
        'batch_size': config.get('train_params', {}).get('batch_size', 0),
        'learning_rate': config.get('train_params', {}).get('learning_rate', 0.0),
        'train_time_sec': round(metrics.get('train_time_sec', 0), 4),
        'inference_time_sec': round(metrics.get('inference_time_sec', 0), 4),
        'param_count': metrics.get('param_count', 0),
        'samples_count': len(preds),
        'best_epoch': metrics.get('best_epoch', 0),
        'final_lr': metrics.get('final_lr', 0.0),
        'mse': round(test_mse, 4),
        'rmse': round(test_rmse, 4),
        'mae': round(test_mae, 4),
        'nrmse': round(nrmse, 4),
        'r_square': round(r_square, 4),
        'smape': round(smape, 4),
        'gpu_memory_used': metrics.get('gpu_memory_used', 0),
        'config_file': f"season_hour_{model_name.lower()}.yaml"
    }
    
    # 保存summary结果
    summary_df = pd.DataFrame([summary_data])
    if os.path.exists(summary_file):
        # 如果文件已存在，追加数据
        existing_df = pd.read_csv(summary_file)
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 汇总结果已保存: {summary_file}")
    
    print(f"✅ Season and Hour Analysis结果已保存到: {drive_path}")
    
    return summary_data
