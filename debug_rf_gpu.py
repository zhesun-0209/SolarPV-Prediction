#!/usr/bin/env python3
"""
调试RF模型GPU运行和参数配置问题
"""

import os
import pandas as pd
import numpy as np
import traceback
import time
import subprocess
import sys

def check_gpu_availability():
    """检查GPU可用性"""
    
    print("🔍 检查GPU可用性...")
    print("=" * 60)
    
    # 检查cuML
    try:
        import cuml
        import cupy as cp
        
        if cp.cuda.is_available():
            print("✅ cuML GPU可用")
            print(f"   GPU数量: {cp.cuda.runtime.getDeviceCount()}")
            print(f"   当前GPU: {cp.cuda.runtime.getDevice()}")
            
            # 检查GPU内存
            mempool = cp.get_default_memory_pool()
            print(f"   GPU内存: {mempool.total_bytes() / (1024**3):.1f} GB")
            
            return True
        else:
            print("❌ cuML GPU不可用")
            return False
            
    except ImportError:
        print("❌ cuML未安装")
        return False
    except Exception as e:
        print(f"❌ GPU检查失败: {e}")
        return False

def test_rf_gpu_parameters():
    """测试RF模型GPU参数配置"""
    
    print("\n🔍 测试RF模型GPU参数配置...")
    print("=" * 60)
    
    # 检查GPU
    if not check_gpu_availability():
        print("❌ GPU不可用，无法测试")
        return
    
    try:
        from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        import cupy as cp
        
        # 创建测试数据（GPU上）
        print("📊 创建测试数据...")
        np.random.seed(42)
        X_test = cp.random.randn(1000, 10)  # 使用CuPy
        y_test = cp.random.randn(1000, 24)  # 24小时预测
        
        print(f"   数据形状: X={X_test.shape}, y={y_test.shape}")
        
        # 测试不同复杂度参数
        complexities = {
            'low': {
                'n_estimators': 50,
                'max_depth': 5,
                'random_state': 42
            },
            'medium': {
                'n_estimators': 100,
                'max_depth': 15,  # 修复：设置具体值
                'random_state': 42
            },
            'high': {
                'n_estimators': 200,
                'max_depth': 25,  # 修复：设置具体值
                'random_state': 42
            }
        }
        
        for complexity, params in complexities.items():
            print(f"\n🧪 测试 {complexity} 复杂度参数:")
            print(f"   参数: {params}")
            
            try:
                start_time = time.time()
                
                # 测试单输出RF
                print("  - 测试单输出cuRandomForestRegressor...")
                rf_single = cuRandomForestRegressor(**params)
                rf_single.fit(X_test, y_test[:, 0])  # 只使用第一列
                print("    ✅ 单输出RF成功")
                
                # 测试多输出RF（需要转换数据格式）
                print("  - 测试多输出MultiOutputRegressor...")
                # 将CuPy数组转换为NumPy数组
                X_test_np = X_test.get()
                y_test_np = y_test.get()
                rf_multi = MultiOutputRegressor(cuRandomForestRegressor(**params))
                rf_multi.fit(X_test_np, y_test_np)
                print("    ✅ 多输出RF成功")
                
                # 测试预测
                pred = rf_multi.predict(X_test_np[:5])
                print(f"    ✅ 预测成功，形状: {pred.shape}")
                
                end_time = time.time()
                print(f"    ⏱️  训练时间: {end_time - start_time:.2f}秒")
                
            except Exception as e:
                print(f"    ❌ 错误: {e}")
                print(f"    📋 详细错误:")
                traceback.print_exc()
                
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保安装了cuML: pip install cuml-cu12")

def test_rf_cpu_fallback():
    """测试RF模型CPU回退"""
    
    print("\n🔍 测试RF模型CPU回退...")
    print("=" * 60)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        
        # 创建测试数据
        np.random.seed(42)
        X_test = np.random.randn(1000, 10)
        y_test = np.random.randn(1000, 24)
        
        print(f"   数据形状: X={X_test.shape}, y={y_test.shape}")
        
        # 测试参数
        params = {
            'n_estimators': 200,
            'max_depth': 25,
            'random_state': 42
        }
        
        print(f"   参数: {params}")
        
        start_time = time.time()
        
        # 测试多输出RF
        rf_multi = MultiOutputRegressor(RandomForestRegressor(**params))
        rf_multi.fit(X_test, y_test)
        
        # 测试预测
        pred = rf_multi.predict(X_test[:5])
        print(f"    ✅ CPU RF成功，预测形状: {pred.shape}")
        
        end_time = time.time()
        print(f"    ⏱️  训练时间: {end_time - start_time:.2f}秒")
        
    except Exception as e:
        print(f"    ❌ CPU RF错误: {e}")
        traceback.print_exc()

def check_rf_config_file():
    """检查RF模型配置文件"""
    
    print("\n🔍 检查RF模型配置文件...")
    print("=" * 60)
    
    config_path = 'config/default.yaml'
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("当前RF配置:")
        for complexity in ['ml_low', 'ml_medium', 'ml_high']:
            if complexity in config:
                print(f"\n{complexity}:")
                for key, value in config[complexity].items():
                    print(f"  {key}: {value} ({type(value).__name__})")
                    
                    # 检查问题
                    if key == 'max_depth' and value is None:
                        print(f"    ⚠️  {key}=None 可能导致问题")
                    if key == 'n_estimators' and value > 150:
                        print(f"    ⚠️  {key}={value} 较大，可能内存不足")
            else:
                print(f"\n❌ {complexity} 配置缺失")
                
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")

def fix_rf_config():
    """修复RF配置文件"""
    
    print("\n🔧 修复RF配置文件...")
    print("=" * 60)
    
    config_path = 'config/default.yaml'
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 修复RF参数
        if 'ml_medium' in config:
            config['ml_medium']['max_depth'] = 15
            print("✅ 修复 ml_medium max_depth")
        
        if 'ml_high' in config:
            config['ml_high']['max_depth'] = 25
            print("✅ 修复 ml_high max_depth")
        
        # 保存修复后的配置
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("✅ 配置文件已修复")
        
    except Exception as e:
        print(f"❌ 修复配置文件失败: {e}")

def test_rf_training_pipeline():
    """测试RF训练管道"""
    
    print("\n🔍 测试RF训练管道...")
    print("=" * 60)
    
    try:
        # 导入训练模块
        from models.ml_models import train_rf
        from data.data_utils import load_raw_data, preprocess_features, create_sliding_windows, split_data
        
        # 加载数据
        print("📊 加载数据...")
        df = load_raw_data('data/Project1033.csv')
        print(f"   数据形状: {df.shape}")
        
        # 预处理
        print("🔧 预处理数据...")
        config = {
            'use_hist_weather': True,
            'use_forecast': False,
            'past_days': 1
        }
        
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = preprocess_features(df, config)
        print(f"   清理后数据形状: {df_clean.shape}")
        print(f"   历史特征: {len(hist_feats)}")
        print(f"   预测特征: {len(fcst_feats)}")
        
        # 创建滑动窗口
        print("🪟 创建滑动窗口...")
        windows = create_sliding_windows(df_clean, 24, 24, hist_feats, fcst_feats)
        print(f"   窗口数量: {len(windows)}")
        
        # 分割数据
        print("✂️ 分割数据...")
        # 创建hours和dates数据
        hours = np.tile(np.arange(24), (len(windows), 1))
        dates = [f"2024-01-{i+1:02d}" for i in range(len(windows))]
        train_data, val_data, test_data = split_data(windows, 0.8, 0.1, hours, dates)
        print(f"   训练集: {len(train_data)}")
        print(f"   验证集: {len(val_data)}")
        print(f"   测试集: {len(test_data)}")
        
        # 测试RF训练
        print("🤖 测试RF训练...")
        rf_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42
        }
        
        start_time = time.time()
        model, metrics = train_rf(train_data, val_data, test_data, rf_params, config)
        end_time = time.time()
        
        print(f"✅ RF训练成功")
        print(f"   训练时间: {end_time - start_time:.2f}秒")
        print(f"   测试损失: {metrics['test_loss']:.4f}")
        
    except Exception as e:
        print(f"❌ RF训练管道测试失败: {e}")
        traceback.print_exc()

def generate_rf_rerun_commands():
    """生成RF模型重新运行命令"""
    
    print("\n🚀 生成RF模型重新运行命令...")
    print("=" * 60)
    
    # 缺失的实验
    missing_experiments = [
        'rf/featFalse_fcstFalse_days1_compmedium',
        'rf/featFalse_fcstFalse_days3_compmedium',
        'rf/featFalse_fcstFalse_days7_compmedium',
        'rf/featFalse_fcstFalse_days1_comphigh',
        'rf/featFalse_fcstFalse_days3_comphigh',
        'rf/featFalse_fcstFalse_days7_comphigh',
        'rf/featTrue_fcstFalse_days1_compmedium',
        'rf/featTrue_fcstFalse_days3_compmedium',
        'rf/featTrue_fcstFalse_days7_compmedium',
        'rf/featTrue_fcstFalse_days1_comphigh',
        'rf/featTrue_fcstFalse_days3_comphigh',
        'rf/featTrue_fcstFalse_days7_comphigh',
        'rf/featFalse_fcstTrue_days1_compmedium',
        'rf/featFalse_fcstTrue_days3_compmedium',
        'rf/featFalse_fcstTrue_days7_compmedium',
        'rf/featFalse_fcstTrue_days1_comphigh',
        'rf/featFalse_fcstTrue_days3_comphigh',
        'rf/featFalse_fcstTrue_days7_comphigh',
        'rf/featTrue_fcstTrue_days1_compmedium',
        'rf/featTrue_fcstTrue_days3_compmedium',
        'rf/featTrue_fcstTrue_days7_compmedium',
        'rf/featTrue_fcstTrue_days1_comphigh',
        'rf/featTrue_fcstTrue_days3_comphigh',
        'rf/featTrue_fcstTrue_days7_comphigh'
    ]
    
    print("建议的重新运行命令:")
    print("=" * 60)
    
    for exp in missing_experiments:
        # 解析实验参数
        parts = exp.split('/')[1].split('_')
        feat = parts[0].replace('feat', '') == 'True'
        fcst = parts[1].replace('fcst', '') == 'True'
        days = int(parts[2].replace('days', ''))
        comp = parts[3].replace('comp', '')
        
        cmd = f"!python main.py --config config/default.yaml --model RF --use_hist_weather {str(feat).lower()} --use_forecast {str(fcst).lower()} --model_complexity {comp} --past_days {days}"
        print(cmd)

def run_single_rf_test():
    """运行单个RF测试"""
    
    print("\n🧪 运行单个RF测试...")
    print("=" * 60)
    
    # 测试medium复杂度
    cmd = [
        'python', 'main.py',
        '--config', 'config/default.yaml',
        '--model', 'RF',
        '--use_hist_weather', 'false',
        '--use_forecast', 'false',
        '--model_complexity', 'medium',
        '--past_days', '1'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ RF测试成功")
            print("输出:")
            print(result.stdout)
        else:
            print("❌ RF测试失败")
            print("错误:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ RF测试超时（5分钟）")
    except Exception as e:
        print(f"❌ RF测试异常: {e}")

def test_rf_direct():
    """直接测试RF模型（不通过main.py）"""
    
    print("\n🧪 直接测试RF模型...")
    print("=" * 60)
    
    try:
        from models.ml_models import train_rf
        import numpy as np
        
        # 创建测试数据
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randn(100, 24)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randn(20, 24)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randn(20, 24)
        
        # 创建数据元组
        train_data = (X_train, None, y_train, None, None)
        val_data = (X_val, None, y_val, None, None)
        test_data = (X_test, None, y_test, None, None)
        
        # RF参数
        rf_params = {
            'n_estimators': 50,
            'max_depth': 15,
            'random_state': 42
        }
        
        config = {'model': 'RF'}
        
        print("开始训练RF模型...")
        start_time = time.time()
        
        model, metrics = train_rf(train_data, val_data, test_data, rf_params, config)
        
        end_time = time.time()
        
        print(f"✅ RF直接测试成功")
        print(f"   训练时间: {end_time - start_time:.2f}秒")
        print(f"   测试损失: {metrics['test_loss']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        
    except Exception as e:
        print(f"❌ RF直接测试失败: {e}")
        traceback.print_exc()

# 运行调试
if __name__ == "__main__":
    print("🚀 RF模型GPU调试开始...")
    print("=" * 60)
    
    # 1. 检查GPU
    check_gpu_availability()
    
    # 2. 检查配置文件
    check_rf_config_file()
    
    # 3. 修复配置文件
    fix_rf_config()
    
    # 4. 测试RF GPU参数
    test_rf_gpu_parameters()
    
    # 5. 测试RF CPU回退
    test_rf_cpu_fallback()
    
    # 6. 测试RF训练管道
    test_rf_training_pipeline()
    
    # 7. 直接测试RF模型
    test_rf_direct()
    
    # 8. 运行单个RF测试
    run_single_rf_test()
    
    # 9. 生成重新运行命令
    generate_rf_rerun_commands()
    
    print("\n✅ 调试完成！")
