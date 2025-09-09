"""
Machine learning regressors with config-driven parameters.
Uses GPU-accelerated versions for Random Forest and Gradient Boosting.
"""
try:
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    # 尝试导入GradientBoostingRegressor，如果失败则使用其他模型
    try:
        from cuml.ensemble import GradientBoostingRegressor as cuGradientBoostingRegressor
    except ImportError:
        # cuML 25.06+ 版本可能没有GradientBoostingRegressor，使用其他模型
        try:
            from cuml.linear_model import LinearRegression as cuGradientBoostingRegressor
            print("Warning: Using LinearRegression instead of GradientBoostingRegressor for GPU")
        except ImportError:
            # 如果都失败，回退到CPU版本
            raise ImportError("No suitable GPU model found")
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
