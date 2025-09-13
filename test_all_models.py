#!/usr/bin/env python3
"""
全面测试所有DL和ML模型
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

def test_all_model_types():
    """测试所有模型类型"""
    print("🔍 测试所有模型类型")
    print("=" * 60)
    
    # 模型列表
    models = {
        'ML': ['Linear', 'RF', 'XGB', 'LGBM'],
        'DL': ['LSTM', 'GRU', 'TCN', 'Transformer']
    }
    
    results = {}
    
    for model_type, model_list in models.items():
        print(f"\n📋 测试{model_type}模型:")
        results[model_type] = {}
        
        for model in model_list:
            print(f"\n  🔧 测试 {model} 模型:")
            
            # 创建测试配置
            config = create_test_config(model, model_type)
            
            # 测试模型训练
            success = test_single_model(model, config)
            results[model_type][model] = success
            
            if success:
                print(f"    ✅ {model} 测试通过")
            else:
                print(f"    ❌ {model} 测试失败")
    
    return results

def create_test_config(model, model_type):
    """创建测试配置"""
    config = {
        'model': model,
        'data_path': 'data/Project1140.csv',
        'save_dir': f'temp_test_results/{model}',
        'past_hours': 24,
        'future_hours': 24,
        'use_pv': True,
        'use_hist_weather': False,
        'use_forecast': False,
        'weather_category': 'none',
        'use_time_encoding': False,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'epochs': 2,  # 减少epochs用于快速测试
        'batch_size': 32,
        'learning_rate': 0.001,
        'loss_type': 'mse'
    }
    
    # 添加模型特定参数（使用正确的结构）
    if model_type == 'ML':
        if model == 'Linear':
            config['model_params'] = {
                'ml_low': {},
                'ml_high': {}
            }
        elif model == 'RF':
            config['model_params'] = {
                'ml_low': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                },
                'ml_high': {
                    'n_estimators': 20,
                    'max_depth': 5,
                    'random_state': 42
                }
            }
        elif model == 'XGB':
            config['model_params'] = {
                'ml_low': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'verbosity': 0
                },
                'ml_high': {
                    'n_estimators': 20,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'verbosity': 0
                }
            }
        elif model == 'LGBM':
            config['model_params'] = {
                'ml_low': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'ml_high': {
                    'n_estimators': 20,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'random_state': 42
                }
            }
    else:  # DL models
        config['model_params'] = {
            'low': {
                'd_model': 32,
                'num_heads': 2,
                'num_layers': 2,
                'hidden_dim': 16,
                'dropout': 0.1,
                'tcn_channels': [16, 32],
                'kernel_size': 3
            },
            'high': {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 4,
                'hidden_dim': 32,
                'dropout': 0.2,
                'tcn_channels': [32, 64],
                'kernel_size': 5
            }
        }
        # 添加train_params
        config['train_params'] = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'loss_type': 'mse'
        }
    
    return config

def test_single_model(model, config):
    """测试单个模型"""
    try:
        # 创建测试数据（确保数据量足够）
        future_hours = config.get('future_hours', 24)
        past_hours = config.get('past_hours', 24)
        
        # 根据模型类型创建不同维度的数据
        if config['model'] in ['Linear', 'RF', 'XGB', 'LGBM']:
            # ML模型使用2D数据
            X_train = np.random.rand(100, 5)
            Xf_train = np.random.rand(100, 3)
            y_train = np.random.rand(100, future_hours)
            Xh_test = np.random.rand(30, 5)
            Xf_test = np.random.rand(30, 3)
            y_test = np.random.rand(30, future_hours)
        else:
            # DL模型使用3D数据 (samples, sequence_length, features)
            X_train = np.random.rand(100, past_hours, 5)
            Xf_train = np.random.rand(100, past_hours, 3)
            y_train = np.random.rand(100, future_hours)  # 输出保持2D
            Xh_test = np.random.rand(30, past_hours, 5)
            Xf_test = np.random.rand(30, past_hours, 3)
            y_test = np.random.rand(30, future_hours)  # 输出保持2D
            
        dates_test = [f"2024-01-01 {i:02d}:00:00" for i in range(30)]
        
        # 根据模型类型选择训练函数
        if config['model'] in ['Linear', 'RF', 'XGB', 'LGBM']:
            from train.train_ml import train_ml_model
            model_obj, metrics = train_ml_model(
                config=config,
                Xh_train=X_train,
                Xf_train=Xf_train,
                y_train=y_train,
                Xh_test=Xh_test,
                Xf_test=Xf_test,
                y_test=y_test,
                dates_test=dates_test
            )
        else:  # DL models - 使用正确的参数结构
            from train.train_dl import train_dl_model
            
            # 创建DL模型所需的数据结构
            train_data = (X_train, Xf_train, y_train, np.zeros(100), None)
            val_data = (X_train[:20], Xf_train[:20], y_train[:20], np.zeros(20), None)
            test_data = (Xh_test, Xf_test, y_test, np.zeros(30), dates_test)
            scalers = (None, None, None)
            
            model_obj, metrics = train_dl_model(
                config=config,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                scalers=scalers
            )
        
        # 检查关键指标
        key_metrics = ['mae', 'rmse', 'r2', 'mape']
        missing_metrics = [m for m in key_metrics if m not in metrics]
        
        if missing_metrics:
            print(f"    ⚠️  缺少指标: {missing_metrics}")
            return False
        
        print(f"    📊 性能指标: {[(k, f'{v:.4f}') for k, v in metrics.items() if k in key_metrics]}")
        return True
        
    except Exception as e:
        print(f"    ❌ 错误: {e}")
        traceback.print_exc()
        return False

def test_result_saving_all_models():
    """测试所有模型的结果保存"""
    print("\n🔍 测试结果保存功能")
    print("=" * 60)
    
    try:
        from eval.eval_utils import save_results
        
        # 创建测试数据
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
        
        test_dates = [f"2024-01-01 {i:02d}:00:00" for i in range(20)]
        test_y_true = np.random.rand(20)
        test_Xh = np.random.rand(20, 5)
        test_Xf = np.random.rand(20, 3)
        
        # 测试保存目录
        test_save_dir = Path("debug_test_results_all")
        test_save_dir.mkdir(exist_ok=True)
        
        test_config = {
            'save_dir': str(test_save_dir),
            'model': 'Linear',
            'plot_days': 7,
            'past_hours': 24,
            'future_hours': 24
        }
        
        # 创建模拟模型
        class MockModel:
            def predict(self, X):
                return np.random.rand(X.shape[0], 1)
        
        mock_model = MockModel()
        
        # 保存结果
        save_results(
            model=mock_model,
            metrics=test_metrics,
            dates=test_dates,
            y_true=test_y_true,
            Xh_test=test_Xh,
            Xf_test=test_Xf,
            config=test_config
        )
        
        print("✅ 结果保存成功")
        
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
        print("✅ 测试文件已清理")
        
        return True
        
    except Exception as e:
        print(f"❌ 结果保存失败: {e}")
        traceback.print_exc()
        return False

def test_full_experiment_pipeline():
    """测试完整实验流程"""
    print("\n🔍 测试完整实验流程")
    print("=" * 60)
    
    # 测试配置
    test_configs = [
        {'model': 'Linear', 'complexity': 'low'},
        {'model': 'RF', 'complexity': 'low'},
        {'model': 'LSTM', 'complexity': 'low'}
    ]
    
    results = {}
    
    for config_info in test_configs:
        model = config_info['model']
        complexity = config_info['complexity']
        
        print(f"\n  🔧 测试 {model}_{complexity} 完整流程:")
        
        # 创建配置文件
        config = create_test_config(model, 'ML' if model in ['Linear', 'RF', 'XGB', 'LGBM'] else 'DL')
        config['model_complexity'] = complexity
        
        # 保存配置文件
        config_dir = Path("temp_test_configs")
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / f"{model}_{complexity}_test.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # 运行实验
            cmd = ['python', 'main.py', '--config', str(config_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"    ✅ {model}_{complexity} 实验运行成功")
                
                # 检查结果文件
                save_dir = Path(config['save_dir'])
                if save_dir.exists():
                    result_files = list(save_dir.glob("*"))
                    print(f"    📁 结果文件: {[f.name for f in result_files]}")
                    
                    # 检查Excel文件
                    excel_files = list(save_dir.glob("*.xlsx"))
                    if excel_files:
                        df = pd.read_excel(excel_files[0])
                        print(f"    📊 Excel内容: {df.shape}, 列: {list(df.columns)}")
                        
                        # 检查关键指标
                        key_metrics = ['mae', 'rmse', 'r2', 'mape']
                        for metric in key_metrics:
                            if metric in df.columns:
                                value = df[metric].iloc[0]
                                print(f"    ✅ {metric}: {value}")
                            else:
                                print(f"    ❌ 缺少指标: {metric}")
                    
                    results[f"{model}_{complexity}"] = True
                else:
                    print(f"    ❌ 结果目录不存在: {save_dir}")
                    results[f"{model}_{complexity}"] = False
            else:
                print(f"    ❌ {model}_{complexity} 实验运行失败")
                print(f"    错误输出: {result.stderr[-500:]}")
                results[f"{model}_{complexity}"] = False
                
        except subprocess.TimeoutExpired:
            print(f"    ⏰ {model}_{complexity} 实验超时")
            results[f"{model}_{complexity}"] = False
        except Exception as e:
            print(f"    ❌ {model}_{complexity} 实验异常: {e}")
            results[f"{model}_{complexity}"] = False
        
        # 清理配置文件
        config_file.unlink()
    
    # 清理配置目录
    config_dir.rmdir()
    
    return results

def main():
    """主测试函数"""
    print("🚀 开始全面模型测试")
    print("=" * 80)
    
    test_results = {}
    
    # 测试1: 所有模型类型
    test_results['model_types'] = test_all_model_types()
    
    # 测试2: 结果保存功能
    test_results['result_saving'] = test_result_saving_all_models()
    
    # 测试3: 完整实验流程
    test_results['full_pipeline'] = test_full_experiment_pipeline()
    
    # 总结
    print("\n" + "=" * 80)
    print("🎯 全面测试总结")
    print("=" * 80)
    
    # 模型类型测试结果
    print("\n📋 模型类型测试结果:")
    for model_type, models in test_results['model_types'].items():
        print(f"  {model_type}模型:")
        for model, success in models.items():
            status = "✅ 通过" if success else "❌ 失败"
            print(f"    {model}: {status}")
    
    # 结果保存测试结果
    print(f"\n📊 结果保存测试: {'✅ 通过' if test_results['result_saving'] else '❌ 失败'}")
    
    # 完整流程测试结果
    print(f"\n🔄 完整流程测试结果:")
    for test_name, success in test_results['full_pipeline'].items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    # 总体统计
    total_tests = sum(len(models) for models in test_results['model_types'].values())
    passed_tests = sum(sum(models.values()) for models in test_results['model_types'].values())
    
    if test_results['result_saving']:
        passed_tests += 1
    total_tests += 1
    
    passed_tests += sum(test_results['full_pipeline'].values())
    total_tests += len(test_results['full_pipeline'])
    
    print(f"\n📊 总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！系统可以正常运行实验")
    else:
        print("⚠️  部分测试失败，需要进一步检查")
    
    return test_results

if __name__ == "__main__":
    main()
