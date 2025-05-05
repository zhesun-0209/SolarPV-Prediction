"""
train/train_ml.py

Machine learning training pipeline for solar power forecasting.
Supports RF, GBR, XGB, and LGBM. Saves model, predictions, metrics, and a single-entry training log.
"""
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
    y_test:  np.ndarray
):
    """
    Train a traditional ML model and evaluate on the test set.

    Args:
        config: dict with 'model','model_params','train_params','save_dir'
        Xh_train: historical inputs (n_train, past_hours, n_hist_feats)
        Xf_train: forecast inputs  (n_train, future_hours, n_fcst_feats) or None
        y_train:  targets         (n_train, future_hours)
        Xh_test:  historical inputs for test (n_test, ...)
        Xf_test:  forecast inputs for test   (n_test, ...) or None
        y_test:   targets for test (n_test, future_hours)

    Returns:
        model: trained sklearn model
        metrics: dict containing
            - test_loss (MSE)
            - rmse
            - mae
            - train_time_sec
            - param_count (number of input features)
            - predictions: np.ndarray (n_test, future_hours)
            - y_true:      np.ndarray (n_test, future_hours)
            - epoch_logs: [{'epoch', 'train_loss', 'val_loss'}]
    """
    # Helper to flatten features
    def flatten(Xh, Xf):
        h_flat = Xh.reshape(Xh.shape[0], -1)
        if Xf is not None:
            f_flat = Xf.reshape(Xf.shape[0], -1)
            return np.concatenate([h_flat, f_flat], axis=1)
        return h_flat

    # Flatten datasets
    X_train_flat = flatten(Xh_train, Xf_train)
    X_test_flat  = flatten(Xh_test,  Xf_test)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_test_flat  = y_test.reshape(y_test.shape[0],  -1)

    # Select and train model
    name   = config['model']
    params = config['model_params']
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

    start_time = time.time()
    model = trainer(X_train_flat, y_train_flat, params)
    train_time = time.time() - start_time

    # Predict on test set
    preds_flat = model.predict(X_test_flat)

    # Compute test metrics
    y_true_flat = y_test_flat.flatten()
    y_pred_flat = preds_flat.flatten()
    mse  = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true_flat, y_pred_flat)

    # Compute training loss for logging
    train_preds_flat = model.predict(X_train_flat)
    train_mse = mean_squared_error(y_train_flat.flatten(), train_preds_flat.flatten())

    # Reshape to (n_samples, future_hours)
    future_hours = config['train_params']['future_hours']
    y_matrix = y_test_flat.reshape(-1, future_hours)
    p_matrix = preds_flat.reshape(-1, future_hours)

    # Save model weights
    save_dir  = config['save_dir']
    model_dir = os.path.join(save_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.pkl')
    joblib.dump(model, model_path)

    # Package metrics
    metrics = {
        'test_loss':      mse,
        'rmse':           rmse,
        'mae':            mae,
        'train_time_sec': round(train_time, 2),
        'param_count':    X_train_flat.shape[1],
        'predictions':    p_matrix,
        'y_true':         y_matrix,
        # single epoch log for ML
        'epoch_logs':     [{'epoch': 1, 'train_loss': train_mse, 'val_loss': mse}]
    }
    return model, metrics
