"""
Machine learning regressors with config-driven parameters.
Uses GPU-accelerated versions for Random Forest and Gradient Boosting.
"""
import numpy as np
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
    try:
        if GPU_AVAILABLE:
            # cuML Random Forest不支持多输出，需要使用MultiOutputRegressor
            # 需要确保数据是NumPy数组格式
            import cupy as cp
            if hasattr(X_train, 'get'):  # 如果是CuPy数组
                X_train = X_train.get()
            if hasattr(y_train, 'get'):  # 如果是CuPy数组
                y_train = y_train.get()
            
            # 检查数据有效性
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                print("⚠️ 检测到NaN或Inf值，进行清理")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
            if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
                print("⚠️ 检测到NaN或Inf值，进行清理")
                y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # cuML Random Forest参数 - 只保留RF相关的参数
            rf_params = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 10),
                'random_state': 42
            }
            base = cuRandomForestRegressor(**rf_params)
            model = MultiOutputRegressor(base)
            model.fit(X_train, y_train)
            return model
        else:
            # Fallback to CPU version with MultiOutputRegressor - 过滤RF不支持的参数
            rf_params = {k: v for k, v in params.items() if k in ['n_estimators', 'max_depth', 'random_state']}
            base = cuRandomForestRegressor(**rf_params)
            model = MultiOutputRegressor(base)
            model.fit(X_train, y_train)
            return model
    except Exception as e:
        print(f"❌ Random Forest训练失败: {e}")
        # 回退到CPU版本 - 过滤RF不支持的参数
        from sklearn.ensemble import RandomForestRegressor
        rf_params = {k: v for k, v in params.items() if k in ['n_estimators', 'max_depth', 'random_state']}
        base = RandomForestRegressor(**rf_params)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model

# GBR已移除，使用XGBoost和LightGBM替代

def train_xgb(X_train, y_train, params: dict):
    """Train XGBoost regressor with multi-output support."""
    try:
        # 检查数据有效性
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("⚠️ 检测到NaN或Inf值，进行清理")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("⚠️ 检测到NaN或Inf值，进行清理")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
    except Exception as e:
        print(f"❌ XGBoost训练失败: {e}")
        # 回退到CPU版本
        base = XGBRegressor(**params)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model

def train_lgbm(X_train, y_train, params: dict):
    """Train LightGBM regressor with multi-output support."""
    try:
        # 检查数据有效性
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("⚠️ 检测到NaN或Inf值，进行清理")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("⚠️ 检测到NaN或Inf值，进行清理")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
    except Exception as e:
        print(f"❌ LightGBM训练失败: {e}")
        # 回退到CPU版本
        base = LGBMRegressor(**params)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return model
