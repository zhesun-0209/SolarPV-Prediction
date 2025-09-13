import os
import time
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from eval.metrics_utils import calculate_metrics, calculate_mse
from utils.gpu_utils import get_gpu_memory_used
from models.ml_models import train_rf, train_xgb, train_lgbm

def train_ml_model(
    config: dict,
    Xh_train: np.ndarray,
    Xf_train: np.ndarray,
    y_train: np.ndarray,
    Xh_test: np.ndarray,
    Xf_test: np.ndarray,
    y_test: np.ndarray,
    dates_test: list,
    scaler_target=None
):
    """
    Train a traditional ML model and evaluate on the test set.
    """

    def flatten(Xh, Xf):
        """
        简单的特征展平，保持DL和ML模型特征一致性
        """
        h = Xh.reshape(Xh.shape[0], -1)
        if Xf is not None:
            f = Xf.reshape(Xf.shape[0], -1)
            return np.concatenate([h, f], axis=1)
        return h

    X_train_flat = flatten(Xh_train, Xf_train)
    X_test_flat  = flatten(Xh_test, Xf_test)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_test_flat  = y_test.reshape(y_test.shape[0], -1)

    name = config['model']

    ml_param_keys = {
        'RF':     ['n_estimators', 'max_depth', 'random_state'],
        'XGB':    ['n_estimators', 'max_depth', 'learning_rate', 'verbosity'],
        'LGBM':   ['n_estimators', 'max_depth', 'learning_rate', 'random_state'],
        'Linear': []  # Linear Regression has no hyperparameters
    }
    all_params = config.get('model_params', {})
    allowed_keys = ml_param_keys.get(name, [])
    params = {k: all_params[k] for k in allowed_keys if k in all_params}

    if 'learning_rate' in params:
        params['learning_rate'] = float(params['learning_rate'])
    if 'n_estimators' in params:
        params['n_estimators'] = int(params['n_estimators'])
    if 'max_depth' in params and params['max_depth'] is not None:
        params['max_depth'] = int(params['max_depth'])
    if 'random_state' in params:
        params['random_state'] = int(params['random_state'])
    if 'verbosity' in params:
        params['verbosity'] = int(params['verbosity'])

    if name == 'RF':
        trainer = train_rf
    elif name == 'XGB':
        trainer = train_xgb
    elif name == 'LGBM':
        trainer = train_lgbm
    elif name == 'Linear':
        # Linear Regression不需要特殊的trainer函数
        trainer = None
    else:
        raise ValueError(f"Unsupported ML model: {name}")

    start_time = time.time()
    if name == 'Linear':
        # 直接使用LinearRegression
        model = LinearRegression()
        model.fit(X_train_flat, y_train_flat)
    else:
        model = trainer(X_train_flat, y_train_flat, params)
    train_time = time.time() - start_time

    # Measure inference time
    inference_start = time.time()
    preds_flat = model.predict(X_test_flat)
    inference_time = time.time() - inference_start
    train_preds_flat = model.predict(X_train_flat)

    # Capacity Factor不需要逆标准化（已经是0-100范围）
    # 数据已经是原始尺度

    fh = int(config['future_hours'])
    y_matrix = y_test_flat.reshape(-1, fh)
    p_matrix = preds_flat.reshape(-1, fh)

    # === 计算所有评估指标 ===
    # 计算MSE
    mse = calculate_mse(y_matrix, p_matrix)
    
    # 计算所有指标
    all_metrics = calculate_metrics(y_matrix, p_matrix)
    
    # 提取基本指标
    rmse = all_metrics['rmse']
    mae = all_metrics['mae']
    
    train_mse = mean_squared_error(y_train_flat, train_preds_flat)

    # 获取GPU内存使用量
    gpu_memory_used = get_gpu_memory_used()

    save_dir  = config['save_dir']
    
    # 根据配置决定是否保存模型
    save_options = config.get('save_options', {})
    if save_options.get('save_model', False):
        model_dir = os.path.join(save_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))

    metrics = {
        'mse':            mse,
        'rmse':           rmse,
        'mae':            mae,
        'nrmse':          all_metrics['nrmse'],
        'r_square':       all_metrics['r_square'],
        'r2':             all_metrics['r2'],  # 添加r2别名
        'smape':          all_metrics['smape'],
        'mape':           all_metrics['mape'],  # 添加mape指标
        'best_epoch':     np.nan,  # ML模型没有epoch概念
        'final_lr':       np.nan,  # ML模型没有学习率概念
        'gpu_memory_used': gpu_memory_used,
        'train_time_sec': round(train_time, 2),
        'inference_time_sec': round(inference_time, 2),
        'param_count':    X_train_flat.shape[1],
        'samples_count':  len(y_matrix),
        'predictions':    p_matrix,
        'y_true':         y_matrix,
        'dates':          dates_test,
        'epoch_logs':     [{'epoch': 1, 'train_loss': train_mse, 'val_loss': mse}],
        'inverse_transformed': False  # Capacity Factor不需要逆标准化
    }

    return model, metrics
