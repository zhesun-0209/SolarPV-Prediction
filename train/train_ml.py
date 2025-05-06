import os
import time
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.ml_models import train_rf, train_gbr, train_xgb, train_lgbm

def train_ml_model(
    config: dict,
    Xh_train: np.ndarray,
    Xf_train: np.ndarray,
    y_train: np.ndarray,
    Xh_test: np.ndarray,
    Xf_test: np.ndarray,
    y_test: np.ndarray,
    scaler_target=None  # ← NEW: allow inverse-transforming if provided
):
    """
    Train a traditional ML model and evaluate on the test set.

    Returns:
        model: trained sklearn model
        metrics: dict containing predictions, y_true, test_loss, rmse, mae, epoch_logs, etc.
    """

    def flatten(Xh, Xf):
        h = Xh.reshape(Xh.shape[0], -1)
        if Xf is not None:
            f = Xf.reshape(Xf.shape[0], -1)
            return np.concatenate([h, f], axis=1)
        return h

    # Flatten data
    X_train_flat = flatten(Xh_train, Xf_train)
    X_test_flat  = flatten(Xh_test, Xf_test)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_test_flat  = y_test.reshape(y_test.shape[0], -1)

    # Parse model and parameters
    name   = config['model']
    params = config['model_params']
    if 'learning_rate' in params:
        params['learning_rate'] = float(params['learning_rate'])
    if 'n_estimators' in params:
        params['n_estimators'] = int(params['n_estimators'])
    if 'max_depth' in params and params['max_depth'] is not None:
        params['max_depth'] = int(params['max_depth'])

    if name == 'RF':
        trainer = train_rf
    elif name == 'GBR':
        trainer = train_gbr
    elif name == 'XGB':
        trainer = train_xgb
    elif name == 'LGBM':
        trainer = train_lgbm
    else:
        raise ValueError(f"Unsupported ML model: {name}")

    # Train model
    start_time = time.time()
    model = trainer(X_train_flat, y_train_flat, params)
    train_time = time.time() - start_time

    # Predict
    preds_flat = model.predict(X_test_flat)
    train_preds_flat = model.predict(X_train_flat)

    # Inverse transform if scaler is provided
    if scaler_target is not None:
        preds_flat = scaler_target.inverse_transform(preds_flat.reshape(-1, 1)).flatten()
        y_test_flat = scaler_target.inverse_transform(y_test_flat.reshape(-1, 1)).flatten()
        train_preds_flat = scaler_target.inverse_transform(train_preds_flat.reshape(-1, 1)).flatten()
        y_train_flat = scaler_target.inverse_transform(y_train_flat.reshape(-1, 1)).flatten()

    # Compute metrics
    mse  = mean_squared_error(y_test_flat, preds_flat)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test_flat, preds_flat)
    train_mse = mean_squared_error(y_train_flat, train_preds_flat)

    # Reshape to (num_windows, future_hours)
    fh = int(config['train_params']['future_hours'])
    y_matrix = y_test_flat.reshape(-1, fh)
    p_matrix = preds_flat.reshape(-1, fh)

    # Save model
    save_dir  = config['save_dir']
    model_dir = os.path.join(save_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))

    # Return
    metrics = {
        'test_loss':      mse,
        'rmse':           rmse,
        'mae':            mae,
        'train_time_sec': round(train_time, 2),
        'param_count':    X_train_flat.shape[1],
        'predictions':    p_matrix,
        'y_true':         y_matrix,
        'epoch_logs':     [{'epoch': 1, 'train_loss': train_mse, 'val_loss': mse}],
        'inverse_transformed': scaler_target is not None  # ← KEY
    }

    return model, metrics
