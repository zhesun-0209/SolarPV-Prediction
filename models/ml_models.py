"""
Machine learning regressors with config-driven parameters.
"""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def train_rf(
    X_train, y_train, params: dict
):
    """Train Random Forest regressor."""
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_gbr(X_train, y_train, params: dict):
    """Train Gradient Boosting regressor with multi-output support."""
    base = GradientBoostingRegressor(**params)
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
