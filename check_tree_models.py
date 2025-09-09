#!/usr/bin/env python3
"""
检查cuML中可用的树模型，寻找GBR的替代
"""

def check_tree_models():
    """检查cuML中可用的树模型"""
    print("🌳 检查cuML中可用的树模型")
    print("=" * 50)
    
    try:
        import cuml
        print(f"✅ cuML版本: {cuml.__version__}")
        
        # 检查所有可能的树模型
        tree_models = []
        
        # 检查ensemble模块
        print("\n📦 检查ensemble模块...")
        try:
            from cuml.ensemble import RandomForestRegressor
            tree_models.append(("RandomForestRegressor", RandomForestRegressor))
            print("✅ RandomForestRegressor 可用")
        except ImportError as e:
            print(f"❌ RandomForestRegressor 不可用: {e}")
        
        try:
            from cuml.ensemble import GradientBoostingRegressor
            tree_models.append(("GradientBoostingRegressor", GradientBoostingRegressor))
            print("✅ GradientBoostingRegressor 可用")
        except ImportError as e:
            print(f"❌ GradientBoostingRegressor 不可用: {e}")
        
        # 检查其他可能的树模型
        print("\n📦 检查其他可能的树模型...")
        
        # 检查XGBoost GPU版本
        try:
            import xgboost as xgb
            if hasattr(xgb, 'XGBRegressor'):
                # 检查是否有GPU支持
                try:
                    model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
                    tree_models.append(("XGBoost-GPU", xgb.XGBRegressor))
                    print("✅ XGBoost GPU 可用")
                except:
                    print("⚠️ XGBoost GPU 不可用，但CPU版本可用")
            else:
                print("❌ XGBoost 不可用")
        except ImportError as e:
            print(f"❌ XGBoost 不可用: {e}")
        
        # 检查LightGBM GPU版本
        try:
            import lightgbm as lgb
            if hasattr(lgb, 'LGBMRegressor'):
                # 检查是否有GPU支持
                try:
                    model = lgb.LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0)
                    tree_models.append(("LightGBM-GPU", lgb.LGBMRegressor))
                    print("✅ LightGBM GPU 可用")
                except:
                    print("⚠️ LightGBM GPU 不可用，但CPU版本可用")
            else:
                print("❌ LightGBM 不可用")
        except ImportError as e:
            print(f"❌ LightGBM 不可用: {e}")
        
        # 测试多输出支持
        print("\n🔍 测试多输出支持...")
        import numpy as np
        from sklearn.multioutput import MultiOutputRegressor
        
        X = np.random.rand(100, 5).astype(np.float32)
        y = np.random.rand(100, 3).astype(np.float32)  # 3个输出
        
        for name, model_class in tree_models:
            print(f"\n测试 {name}...")
            try:
                # 直接测试多输出
                model = model_class(n_estimators=5, random_state=42)
                model.fit(X, y)
                print(f"✅ {name} 直接支持多输出")
            except Exception as e:
                print(f"❌ {name} 直接多输出失败: {e}")
                
                # 测试MultiOutputRegressor包装
                try:
                    base_model = model_class(n_estimators=5, random_state=42)
                    wrapped_model = MultiOutputRegressor(base_model)
                    wrapped_model.fit(X, y)
                    print(f"✅ {name} 通过MultiOutputRegressor支持多输出")
                except Exception as e2:
                    print(f"❌ {name} MultiOutputRegressor也失败: {e2}")
        
        # 推荐替代方案
        print("\n💡 推荐替代方案:")
        print("1. RandomForestRegressor + MultiOutputRegressor (cuML GPU)")
        print("2. XGBoost + MultiOutputRegressor (如果有GPU支持)")
        print("3. LightGBM + MultiOutputRegressor (如果有GPU支持)")
        print("4. 回退到CPU版本的GradientBoostingRegressor")
        
    except ImportError as e:
        print(f"❌ cuML导入失败: {e}")

if __name__ == "__main__":
    check_tree_models()
