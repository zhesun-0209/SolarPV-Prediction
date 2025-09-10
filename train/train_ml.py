import os
import time
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
        'RF':   ['n_estimators', 'max_depth', 'random_state'],
        'XGB':  ['n_estimators', 'max_depth', 'learning_rate', 'verbosity'],
        'LGBM': ['n_estimators', 'max_depth', 'learning_rate', 'random_state']
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
    else:
        raise ValueError(f"Unsupported ML model: {name}")

    start_time = time.time()
    model = trainer(X_train_flat, y_train_flat, params)
    train_time = time.time() - start_time

    # Measure inference time
    inference_start = time.time()
    preds_flat = model.predict(X_test_flat)
    inference_time = time.time() - inference_start
    train_preds_flat = model.predict(X_train_flat)

    # Capacity Factor不需要逆标准化（已经是0-100范围）
    # 数据已经是原始尺度

    mse  = mean_squared_error(y_test_flat, preds_flat)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test_flat, preds_flat)
    train_mse = mean_squared_error(y_train_flat, train_preds_flat)

    fh = int(config['future_hours'])
    y_matrix = y_test_flat.reshape(-1, fh)
    p_matrix = preds_flat.reshape(-1, fh)

    # 相对误差（用于辅助分析）
    if np.mean(y_test_flat) > 0:
        norm_mse = np.mean(((y_test_flat - preds_flat) / y_test_flat) ** 2)
        norm_rmse = np.sqrt(norm_mse)
        norm_mae = np.mean(np.abs((y_test_flat - preds_flat) / y_test_flat))
    else:
        norm_mse = norm_rmse = norm_mae = np.nan

    save_dir  = config['save_dir']
    
    # 根据配置决定是否保存模型
    save_options = config.get('save_options', {})
    if save_options.get('save_model', False):
        model_dir = os.path.join(save_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))

    metrics = {
        'test_loss':      mse,
        'rmse':           rmse,
        'mae':            mae,
        'norm_test_loss': norm_mse,
        'norm_rmse':      norm_rmse,
        'norm_mae':       norm_mae,
        'train_time_sec': round(train_time, 2),
        'inference_time_sec': round(inference_time, 2),
        'param_count':    X_train_flat.shape[1],
        'predictions':    p_matrix,
        'y_true':         y_matrix,
        'dates':          dates_test,
        'epoch_logs':     [{'epoch': 1, 'train_loss': train_mse, 'val_loss': mse}],
        'inverse_transformed': False  # Capacity Factor不需要逆标准化
    }

    return model, metrics
