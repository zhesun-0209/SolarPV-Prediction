#!/usr/bin/env python3
"""
检查GPU版本的ML模型是否启用
"""

def check_gpu_ml():
    """检查GPU版本的ML模型"""
    print("🔍 检查GPU版本的ML模型")
    print("=" * 50)
    
    # 检查cuML
    try:
        import cuml
        print(f"✅ cuML已安装，版本: {cuml.__version__}")
        
        # 检查GPU可用性
        if cuml.is_gpu_available():
            print("✅ cuML GPU可用")
        else:
            print("❌ cuML GPU不可用")
            
    except ImportError:
        print("❌ cuML未安装")
        return False
    
    # 检查模型导入
    try:
        from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
        from cuml.ensemble import GradientBoostingRegressor as cuGradientBoostingRegressor
        print("✅ cuML模型导入成功")
        
        # 测试创建模型
        print("\n🧪 测试创建GPU模型...")
        
        # 创建测试数据
        import numpy as np
        X_test = np.random.rand(1000, 10).astype(np.float32)
        y_test = np.random.rand(1000, 24).astype(np.float32)
        
        # 测试Random Forest
        print("测试Random Forest...")
        rf_model = cuRandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(X_test, y_test)
        print("✅ Random Forest GPU模型创建成功")
        
        # 测试Gradient Boosting
        print("测试Gradient Boosting...")
        gbr_model = cuGradientBoostingRegressor(n_estimators=10, random_state=42)
        gbr_model.fit(X_test, y_test)
        print("✅ Gradient Boosting GPU模型创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ cuML模型测试失败: {e}")
        return False

def check_ml_models_import():
    """检查ml_models.py中的导入"""
    print("\n🔍 检查ml_models.py导入...")
    
    try:
        from models.ml_models import train_rf, train_gbr
        print("✅ ml_models导入成功")
        
        # 检查GPU_AVAILABLE变量
        import models.ml_models as ml_models
        print(f"GPU_AVAILABLE: {ml_models.GPU_AVAILABLE}")
        
        if ml_models.GPU_AVAILABLE:
            print("✅ ml_models将使用GPU版本")
        else:
            print("❌ ml_models将使用CPU版本")
            
        return ml_models.GPU_AVAILABLE
        
    except Exception as e:
        print(f"❌ ml_models导入失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 GPU版本ML模型检查工具")
    print("=" * 50)
    
    # 检查cuML
    cuml_ok = check_gpu_ml()
    
    # 检查ml_models导入
    ml_models_ok = check_ml_models_import()
    
    print("\n📊 总结:")
    if cuml_ok and ml_models_ok:
        print("✅ GPU版本ML模型已正确启用")
        print("💡 如果运行时间仍然很长，可能是数据量太大或模型复杂度太高")
    else:
        print("❌ GPU版本ML模型未正确启用")
        print("💡 请检查cuML安装和导入")

if __name__ == "__main__":
    main()
