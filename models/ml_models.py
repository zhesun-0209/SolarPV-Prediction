"""
Machine learning regressors with config-driven parameters.
"""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def train_rf(
    X_train, y_train, params: dict
):
    """Train Random Forest regressor."""
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_gbr(
    X_train, y_train, params: dict
):
    """Train Gradient Boosting regressor."""
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_xgb(
    X_train, y_train, params: dict
):
    """Train XGBoost regressor."""
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_lgbm(
    X_train, y_train, params: dict
):
    """Train LightGBM regressor."""
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model
