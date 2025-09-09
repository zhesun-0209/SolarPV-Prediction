#!/usr/bin/env python3
"""
检查cuML 25.06版本中可用的模型
"""

def check_cuml_models():
    """检查cuML中可用的模型"""
    print("🔍 检查cuML 25.06版本中可用的模型")
    print("=" * 50)
    
    try:
        import cuml
        print(f"✅ cuML版本: {cuml.__version__}")
        
        # 检查ensemble模块
        print("\n📦 检查ensemble模块...")
        try:
            from cuml.ensemble import RandomForestRegressor
            print("✅ RandomForestRegressor 可用")
        except ImportError as e:
            print(f"❌ RandomForestRegressor 不可用: {e}")
        
        try:
            from cuml.ensemble import GradientBoostingRegressor
            print("✅ GradientBoostingRegressor 可用")
        except ImportError as e:
            print(f"❌ GradientBoostingRegressor 不可用: {e}")
        
        # 检查linear_model模块
        print("\n📦 检查linear_model模块...")
        try:
            from cuml.linear_model import LinearRegression
            print("✅ LinearRegression 可用")
        except ImportError as e:
            print(f"❌ LinearRegression 不可用: {e}")
        
        try:
            from cuml.linear_model import Ridge
            print("✅ Ridge 可用")
        except ImportError as e:
            print(f"❌ Ridge 不可用: {e}")
        
        try:
            from cuml.linear_model import Lasso
            print("✅ Lasso 可用")
        except ImportError as e:
            print(f"❌ Lasso 不可用: {e}")
        
        # 检查其他可能的模块
        print("\n📦 检查其他模块...")
        try:
            from cuml.svm import SVR
            print("✅ SVR 可用")
        except ImportError as e:
            print(f"❌ SVR 不可用: {e}")
        
        try:
            from cuml.neighbors import KNeighborsRegressor
            print("✅ KNeighborsRegressor 可用")
        except ImportError as e:
            print(f"❌ KNeighborsRegressor 不可用: {e}")
        
        # 检查多输出支持
        print("\n🔍 检查多输出支持...")
        try:
            from cuml.ensemble import RandomForestRegressor
            import numpy as np
            
            # 测试多输出
            X = np.random.rand(100, 5).astype(np.float32)
            y = np.random.rand(100, 3).astype(np.float32)  # 3个输出
            
            model = RandomForestRegressor(n_estimators=5, random_state=42)
            model.fit(X, y)
            print("✅ RandomForestRegressor 支持多输出")
            
        except Exception as e:
            print(f"❌ RandomForestRegressor 不支持多输出: {e}")
            
            # 测试单输出
            try:
                y_single = np.random.rand(100, 1).astype(np.float32)  # 1个输出
                model = RandomForestRegressor(n_estimators=5, random_state=42)
                model.fit(X, y_single)
                print("✅ RandomForestRegressor 支持单输出")
            except Exception as e2:
                print(f"❌ RandomForestRegressor 单输出也失败: {e2}")
        
    except ImportError as e:
        print(f"❌ cuML导入失败: {e}")

if __name__ == "__main__":
    check_cuml_models()
