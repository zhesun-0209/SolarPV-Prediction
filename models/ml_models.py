"""
Machine learning regressors with config-driven parameters.
Uses GPU-accelerated versions for Random Forest and Gradient Boosting.
"""
try:
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    GPU_AVAILABLE = True
    print("✅ cuML RandomForestRegressor 可用")
except ImportError:
    from sklearn.ensemble import RandomForestRegressor as cuRandomForestRegressor
    GPU_AVAILABLE = False
    print("Warning: cuML not available, falling back to CPU versions")

# 检查XGBoost GPU支持
XGB_GPU_AVAILABLE = False
try:
    import xgboost as xgb
    # 测试XGBoost GPU支持
    try:
        test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
        XGB_GPU_AVAILABLE = True
        print("✅ XGBoost GPU 可用")
    except:
        print("⚠️ XGBoost GPU 不可用，使用CPU版本")
except ImportError:
    print("❌ XGBoost 不可用")

# 检查LightGBM GPU支持
LGB_GPU_AVAILABLE = False
try:
    import lightgbm as lgb
    # 测试LightGBM GPU支持
    try:
        test_model = lgb.LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0, n_estimators=1)
        LGB_GPU_AVAILABLE = True
        print("✅ LightGBM GPU 可用")
    except:
        print("⚠️ LightGBM GPU 不可用，使用CPU版本")
except ImportError:
    print("❌ LightGBM 不可用")

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def train_rf(X_train, y_train, params: dict):
    """Train GPU-accelerated Random Forest regressor with multi-output support."""
    if GPU_AVAILABLE:
        # cuML Random Forest不支持多输出，需要使用MultiOutputRegressor
        base = cuRandomForestRegressor(**params)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model
    else:
        # Fallback to CPU version with MultiOutputRegressor
        base = cuRandomForestRegressor(**params)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model

# GBR已移除，使用XGBoost和LightGBM替代

def train_xgb(X_train, y_train, params: dict):
    """Train XGBoost regressor with multi-output support."""
    if XGB_GPU_AVAILABLE:
        # 使用XGBoost GPU版本
        gpu_params = params.copy()
        gpu_params.update({
            'tree_method': 'hist',
            'device': 'cuda'
        })
        base = XGBRegressor(**gpu_params)
    else:
        # 使用XGBoost CPU版本
        base = XGBRegressor(**params)
    
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model

def train_lgbm(X_train, y_train, params: dict):
    """Train LightGBM regressor with multi-output support."""
    if LGB_GPU_AVAILABLE:
        # 使用LightGBM GPU版本
        gpu_params = params.copy()
        gpu_params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        })
        base = LGBMRegressor(**gpu_params)
    else:
        # 使用LightGBM CPU版本
        base = LGBMRegressor(**params)
    
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    return model
