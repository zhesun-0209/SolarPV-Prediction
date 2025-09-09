"""
Machine learning regressors with config-driven parameters.
Uses GPU-accelerated versions for Random Forest and Gradient Boosting.
"""
try:
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from cuml.ensemble import GradientBoostingRegressor as cuGradientBoostingRegressor
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor as cuGradientBoostingRegressor
    GPU_AVAILABLE = False
    print("Warning: cuML not available, falling back to CPU versions")

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def train_rf(X_train, y_train, params: dict):
    """Train GPU-accelerated Random Forest regressor with multi-output support."""
    if GPU_AVAILABLE:
        # Use GPU version directly (cuML supports multi-output natively)
        model = cuRandomForestRegressor(**params)
        model.fit(X_train, y_train)
        return model
    else:
        # Fallback to CPU version with MultiOutputRegressor
        base = cuRandomForestRegressor(**params)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model

def train_gbr(X_train, y_train, params: dict):
    """Train GPU-accelerated Gradient Boosting regressor with multi-output support."""
    if GPU_AVAILABLE:
        # Use GPU version directly (cuML supports multi-output natively)
        model = cuGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        return model
    else:
        # Fallback to CPU version with MultiOutputRegressor
        base = cuGradientBoostingRegressor(**params)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model

def train_xgb(X_train, y_train, params: dict):
    """Train XGBoost regressor with multi-output support."""
    base = XGBRegressor(**params)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model

def train_lgbm(X_train, y_train, params: dict):
    """Train LightGBM regressor with multi-output support."""
    base = LGBMRegressor(**params)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model
