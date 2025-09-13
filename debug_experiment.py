#!/usr/bin/env python3
"""
实验调试代码
用于诊断实验运行中的问题
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import subprocess
from datetime import datetime

def test_single_experiment_run():
    """测试单个实验的完整运行流程"""
    print("🔍 调试单个实验运行")
    print("=" * 60)
    
    # 选择一个简单的配置进行测试
    config_name = "LSR_low_PV_24h_noTE"
    config_path = f"config/projects/1140/{config_name}.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    print(f"✅ 使用配置文件: {config_path}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ 配置加载成功")
    print(f"   数据路径: {config.get('data_path')}")
    print(f"   保存路径: {config.get('save_dir')}")
    print(f"   模型: {config.get('model')}")
    print(f"   复杂度: {config.get('model_complexity')}")
    
    # 检查数据文件
    data_path = config.get('data_path')
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return False
    
    print(f"✅ 数据文件存在: {data_path}")
    
    # 运行单个实验
    print(f"\n🚀 开始运行实验: {config_name}")
    
    try:
        # 构建main.py命令
        cmd = [
            'python', 'main.py',
            '--config', config_path
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 运行实验
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        print(f"返回码: {result.returncode}")
        
        if result.stdout:
            print(f"\n📤 标准输出:")
            print(result.stdout[-2000:])  # 只显示最后2000字符
        
        if result.stderr:
            print(f"\n❌ 错误输出:")
            print(result.stderr[-2000:])  # 只显示最后2000字符
        
        if result.returncode == 0:
            print(f"✅ 实验运行成功")
            
            # 检查结果文件
            save_dir = Path(config.get('save_dir'))
            if save_dir.exists():
                print(f"✅ 结果目录存在: {save_dir}")
                
                # 查找结果文件
                result_files = list(save_dir.glob("*"))
                print(f"   结果文件: {[f.name for f in result_files]}")
                
                # 检查Excel文件
                excel_files = list(save_dir.glob("*.xlsx"))
                if excel_files:
                    excel_file = excel_files[0]
                    print(f"✅ 找到Excel文件: {excel_file}")
                    
                    try:
                        df = pd.read_excel(excel_file)
                        print(f"   Excel内容形状: {df.shape}")
                        print(f"   Excel列名: {list(df.columns)}")
                        
                        # 检查关键指标
                        key_metrics = ['mae', 'rmse', 'r2', 'mape', 'train_time_sec', 'inference_time_sec']
                        for metric in key_metrics:
                            if metric in df.columns:
                                value = df[metric].iloc[0] if len(df) > 0 else None
                                print(f"   {metric}: {value}")
                            else:
                                print(f"   ❌ 缺少列: {metric}")
                        
                    except Exception as e:
                        print(f"❌ 读取Excel文件失败: {e}")
                
                # 检查CSV文件
                csv_files = list(save_dir.glob("*.csv"))
                if csv_files:
                    csv_file = csv_files[0]
                    print(f"✅ 找到CSV文件: {csv_file}")
                    
                    try:
                        df = pd.read_csv(csv_file)
                        print(f"   CSV内容形状: {df.shape}")
                        print(f"   CSV列名: {list(df.columns)}")
                        
                        # 检查关键指标
                        key_metrics = ['mae', 'rmse', 'r2', 'mape', 'train_time_sec', 'inference_time_sec']
                        for metric in key_metrics:
                            if metric in df.columns:
                                value = df[metric].iloc[0] if len(df) > 0 else None
                                print(f"   {metric}: {value}")
                            else:
                                print(f"   ❌ 缺少列: {metric}")
                        
                    except Exception as e:
                        print(f"❌ 读取CSV文件失败: {e}")
                
            else:
                print(f"❌ 结果目录不存在: {save_dir}")
            
        else:
            print(f"❌ 实验运行失败，返回码: {result.returncode}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ 实验运行超时")
        return False
    except Exception as e:
        print(f"❌ 实验运行异常: {e}")
        traceback.print_exc()
        return False

def test_data_preprocessing():
    """测试数据预处理流程"""
    print("\n🔍 调试数据预处理")
    print("=" * 60)
    
    try:
        # 导入数据工具
        sys.path.append('.')
        from data.data_utils import load_raw_data, preprocess_features
        
        # 加载数据
        data_path = "data/Project1140.csv"
        print(f"📊 加载数据: {data_path}")
        
        df = load_raw_data(data_path)
        print(f"✅ 数据加载成功，形状: {df.shape}")
        print(f"   列名: {list(df.columns)}")
        
        # 测试配置
        config = {
            'use_pv': True,
            'use_hist_weather': False,
            'use_forecast': False,
            'weather_category': 'none',
            'use_time_encoding': False,
            'past_hours': 24
        }
        
        print(f"📝 使用配置: {config}")
        
        # 预处理数据
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = preprocess_features(df, config)
        
        print(f"✅ 数据预处理成功")
        print(f"   清理后数据形状: {df_clean.shape}")
        print(f"   历史特征: {hist_feats}")
        print(f"   预测特征: {fcst_feats}")
        print(f"   目标列: Capacity Factor")
        
        # 检查目标列
        if 'Capacity Factor' in df_clean.columns:
            target_values = df_clean['Capacity Factor'].dropna()
            print(f"✅ 目标列存在，有效值数量: {len(target_values)}")
            print(f"   目标值范围: {target_values.min():.3f} - {target_values.max():.3f}")
        else:
            print(f"❌ 目标列不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        traceback.print_exc()
        return False

def test_model_training():
    """测试模型训练流程"""
    print("\n🔍 调试模型训练")
    print("=" * 60)
    
    try:
        # 检查依赖包
        missing_packages = []
        
        try:
            import xgboost
            print("✅ XGBoost 可用")
        except ImportError:
            missing_packages.append('xgboost')
            print("❌ XGBoost 不可用")
        
        try:
            import lightgbm
            print("✅ LightGBM 可用")
        except ImportError:
            missing_packages.append('lightgbm')
            print("❌ LightGBM 不可用")
        
        try:
            import sklearn
            print("✅ Scikit-learn 可用")
        except ImportError:
            missing_packages.append('scikit-learn')
            print("❌ Scikit-learn 不可用")
        
        if missing_packages:
            print(f"⚠️  缺少依赖包: {missing_packages}")
            print("   在Colab环境中这些包应该是可用的")
            return False
        
        # 如果所有包都可用，测试训练流程
        from train.train_ml import train_ml_model
        
        # 创建测试数据
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100)
        X_val = np.random.rand(20, 5)
        y_val = np.random.rand(20)
        
        # 测试配置
        config = {
            'model': 'Linear',
            'model_params': {
                'learning_rate': 0.001
            }
        }
        
        print(f"📝 测试LSR模型训练")
        print(f"   训练数据形状: X={X_train.shape}, y={y_train.shape}")
        print(f"   验证数据形状: X={X_val.shape}, y={y_val.shape}")
        
        # 创建测试数据（添加缺失的参数）
        Xf_train = np.random.rand(100, 3)
        Xh_test = np.random.rand(20, 5)
        Xf_test = np.random.rand(20, 3)
        y_test = np.random.rand(20)
        dates_test = [f"2024-01-01 {i:02d}:00:00" for i in range(20)]
        
        print(f"   完整训练数据形状: Xh={X_train.shape}, Xf={Xf_train.shape}, y={y_train.shape}")
        print(f"   完整测试数据形状: Xh={Xh_test.shape}, Xf={Xf_test.shape}, y={y_test.shape}")
        
        # 训练模型
        model, metrics = train_ml_model(
            config=config,
            Xh_train=X_train,
            Xf_train=Xf_train,
            y_train=y_train,
            Xh_test=Xh_test,
            Xf_test=Xf_test,
            y_test=y_test,
            dates_test=dates_test
        )
        
        print(f"✅ 模型训练成功")
        print(f"   模型类型: {type(model)}")
        print(f"   性能指标: {metrics}")
        
        # 检查关键指标
        key_metrics = ['mae', 'rmse', 'r2', 'mape']
        for metric in key_metrics:
            if metric in metrics:
                print(f"   {metric}: {metrics[metric]}")
            else:
                print(f"   ❌ 缺少指标: {metric}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        traceback.print_exc()
        return False

def test_result_saving():
    """测试结果保存流程"""
    print("\n🔍 调试结果保存")
    print("=" * 60)
    
    try:
        # 导入评估工具
        sys.path.append('.')
        from eval.eval_utils import save_results
        
        # 创建测试结果（使用正确的格式）
        test_metrics = {
            'mae': 0.123,
            'rmse': 0.234,
            'r2': 0.567,
            'mape': 12.34,
            'train_time_sec': 5.67,
            'inference_time_sec': 0.12,
            'param_count': 100,
            'samples_count': 1000,
            'predictions': np.random.rand(20, 1),
            'y_true': np.random.rand(20, 1)
        }
        
        # 创建测试数据
        test_dates = [f"2024-01-01 {i:02d}:00:00" for i in range(20)]
        test_y_true = np.random.rand(20)
        test_Xh = np.random.rand(20, 5)
        test_Xf = np.random.rand(20, 3)
        
        # 测试保存目录
        test_save_dir = Path("debug_test_results")
        test_save_dir.mkdir(exist_ok=True)
        
        test_config = {
            'save_dir': str(test_save_dir),
            'model': 'Linear',
            'plot_days': 7,
            'past_hours': 24,
            'future_hours': 24
        }
        
        print(f"📝 测试结果保存到: {test_save_dir}")
        print(f"   测试指标: {list(test_metrics.keys())}")
        
        # 创建模拟模型
        class MockModel:
            def predict(self, X):
                return np.random.rand(X.shape[0], 1)
        
        mock_model = MockModel()
        
        # 保存结果（使用正确的参数格式）
        save_results(
            model=mock_model,
            metrics=test_metrics,
            dates=test_dates,
            y_true=test_y_true,
            Xh_test=test_Xh,
            Xf_test=test_Xf,
            config=test_config
        )
        
        print(f"✅ 结果保存成功")
        
        # 检查保存的文件
        result_files = list(test_save_dir.glob("*"))
        print(f"   保存的文件: {[f.name for f in result_files]}")
        
        # 检查Excel文件
        excel_files = list(test_save_dir.glob("*.xlsx"))
        if excel_files:
            excel_file = excel_files[0]
            df = pd.read_excel(excel_file)
            print(f"   Excel文件内容形状: {df.shape}")
            print(f"   Excel列名: {list(df.columns)}")
            
            # 验证关键指标
            key_metrics = ['mae', 'rmse', 'r2', 'mape', 'train_time_sec']
            for metric in key_metrics:
                if metric in df.columns:
                    saved_value = df[metric].iloc[0]
                    original_value = test_metrics[metric]
                    print(f"   ✅ {metric}: 期望{original_value}, 实际{saved_value}")
                else:
                    print(f"   ❌ 缺少列: {metric}")
        
        # 清理测试文件
        import shutil
        shutil.rmtree(test_save_dir)
        print(f"✅ 测试文件已清理")
        
        return True
        
    except Exception as e:
        print(f"❌ 结果保存失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主调试函数"""
    print("🚀 开始实验调试")
    print("=" * 80)
    
    debug_results = {}
    
    # 测试1: 数据预处理
    debug_results['data_preprocessing'] = test_data_preprocessing()
    
    # 测试2: 模型训练
    debug_results['model_training'] = test_model_training()
    
    # 测试3: 结果保存
    debug_results['result_saving'] = test_result_saving()
    
    # 测试4: 完整实验运行
    debug_results['full_experiment'] = test_single_experiment_run()
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 调试总结")
    print("=" * 80)
    
    for test_name, result in debug_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    total_passed = sum(debug_results.values())
    total_tests = len(debug_results)
    
    print(f"\n📊 总体结果: {total_passed}/{total_tests} 测试通过")
    
    if total_passed == total_tests:
        print("🎉 所有调试测试通过！")
    else:
        print("⚠️  部分调试测试失败，需要进一步检查")
    
    return debug_results

if __name__ == "__main__":
    main()
