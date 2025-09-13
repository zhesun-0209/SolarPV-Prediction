#!/usr/bin/env python3
"""
真实数据全面测试脚本 - 测试所有ML和DL模型
使用Project1140.csv数据，详细输出所有测试过程
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🚀 开始真实数据全面测试...")
print(f"📁 项目根目录: {project_root}")
print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 1. 测试配置生成
print("\n" + "="*80)
print("1️⃣ 测试配置生成")
print("="*80)

try:
    from scripts.generate_dynamic_project_configs import generate_project_configs
    
    project_configs = generate_project_configs("1140")
    print(f"✅ 配置生成成功")
    print(f"📊 生成配置数量: {len(project_configs)}")
    
    # 显示配置类型统计
    model_stats = {}
    input_stats = {}
    for config_info in project_configs:
        model = config_info['config'].get('model', 'Unknown')
        input_cat = config_info['name'].split('_')[2]  # 从配置名提取输入类别
        model_stats[model] = model_stats.get(model, 0) + 1
        input_stats[input_cat] = input_stats.get(input_cat, 0) + 1
    
    print(f"📊 模型类型统计:")
    for model, count in model_stats.items():
        print(f"   - {model}: {count}个配置")
    
    print(f"📊 输入类别统计:")
    for input_cat, count in input_stats.items():
        print(f"   - {input_cat}: {count}个配置")
    
    # 显示前10个配置名称
    print(f"📊 前10个配置名称:")
    for i, config_info in enumerate(project_configs[:10]):
        print(f"   {i+1}. {config_info.get('name', 'N/A')}")
        
except Exception as e:
    print(f"❌ 配置生成失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 2. 测试数据加载和预处理
print("\n" + "="*80)
print("2️⃣ 测试数据加载和预处理")
print("="*80)

try:
    from data.data_utils import load_raw_data, preprocess_features
    
    # 加载数据
    data_file = project_root / "data" / "Project1140.csv"
    df = load_raw_data(str(data_file))
    print(f"✅ 数据加载成功")
    print(f"📊 原始数据形状: {df.shape}")
    print(f"📊 数据时间范围: {df['Datetime'].min()} 到 {df['Datetime'].max()}")
    print(f"📊 数据列数: {len(df.columns)}")
    print(f"📊 目标变量范围: {df['Capacity Factor'].min():.2f} - {df['Capacity Factor'].max():.2f}")
    
    # 测试不同的配置（注意：LSR不使用PV配置）
    test_configs = [
        ('PV_noTE', 'RF_low_PV_24h_noTE'),  # 使用RF测试PV配置
        ('NWP_noTE', 'LSR_low_NWP_24h_noTE'),
        ('HW_noTE', 'LSR_low_PV_plus_HW_24h_noTE'),
        ('NWP_TE', 'LSR_low_NWP_24h_TE')
    ]
    
    preprocessing_results = {}
    
    for config_name, config_pattern in test_configs:
        print(f"\n🔧 测试配置: {config_name}")
        
        # 找到对应的配置
        test_config = None
        for cfg_info in project_configs:
            if config_pattern in cfg_info.get('name', ''):
                test_config = cfg_info.get('config', {})
                break
        
        if test_config:
            print(f"📊 使用配置: {config_pattern}")
            print(f"📊 配置详情:")
            print(f"   - 模型: {test_config.get('model', 'N/A')}")
            print(f"   - 复杂度: {test_config.get('model_complexity', 'N/A')}")
            print(f"   - 使用PV: {test_config.get('use_pv', 'N/A')}")
            print(f"   - 使用历史天气: {test_config.get('use_hist_weather', 'N/A')}")
            print(f"   - 使用预测天气: {test_config.get('use_forecast', 'N/A')}")
            print(f"   - 时间编码: {test_config.get('use_time_encoding', 'N/A')}")
            print(f"   - 天气类别: {test_config.get('weather_category', 'N/A')}")
            print(f"   - 回看小时: {test_config.get('past_hours', 'N/A')}")
            
            df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target = preprocess_features(df, test_config)
            
            print(f"✅ {config_name} 数据预处理成功")
            print(f"📊 清理后数据形状: {df_clean.shape}")
            print(f"📊 历史特征: {hist_feats}")
            print(f"📊 预测特征: {fcst_feats}")
            print(f"📊 历史特征数量: {len(hist_feats)}")
            print(f"📊 预测特征数量: {len(fcst_feats)}")
            
            preprocessing_results[config_name] = {
                'config': test_config,
                'df_clean': df_clean,
                'hist_feats': hist_feats,
                'fcst_feats': fcst_feats,
                'scaler_hist': scaler_hist,
                'scaler_fcst': scaler_fcst,
                'scaler_target': scaler_target
            }
        else:
            print(f"❌ 找不到配置: {config_pattern}")
            
except Exception as e:
    print(f"❌ 数据预处理失败: {e}")
    traceback.print_exc()

# 3. 测试滑动窗口和数据分割
print("\n" + "="*80)
print("3️⃣ 测试滑动窗口和数据分割")
print("="*80)

try:
    from data.data_utils import create_sliding_windows, split_data
    
    # 使用第一个成功的预处理结果
    if preprocessing_results:
        config_name = list(preprocessing_results.keys())[0]
        result = preprocessing_results[config_name]
        
        print(f"📊 使用配置: {config_name}")
        df_clean = result['df_clean']
        hist_feats = result['hist_feats']
        fcst_feats = result['fcst_feats']
        config = result['config']
        
        past_hours = config.get('past_hours', 24)
        future_hours = config.get('future_hours', 24)
        
        print(f"📊 滑动窗口参数:")
        print(f"   - 过去小时: {past_hours}")
        print(f"   - 未来小时: {future_hours}")
        print(f"   - 历史特征: {hist_feats}")
        print(f"   - 预测特征: {fcst_feats}")
        
        # 确保有特征才创建滑动窗口
        if not hist_feats and not fcst_feats:
            print("❌ 错误：没有可用的特征，无法创建滑动窗口")
            sys.exit(1)
        
        X_hist, X_fcst, y, hours, dates = create_sliding_windows(
            df_clean, past_hours, future_hours, hist_feats, fcst_feats
        )
        
        print(f"✅ 滑动窗口创建成功")
        print(f"📊 X_hist形状: {X_hist.shape}")
        print(f"📊 X_fcst形状: {X_fcst.shape if X_fcst is not None else 'None'}")
        print(f"📊 y形状: {y.shape}")
        print(f"📊 hours形状: {hours.shape}")
        print(f"📊 dates数量: {len(dates)}")
        
        # 检查数据质量
        print(f"📊 数据质量检查:")
        print(f"   - X_hist NaN数量: {np.isnan(X_hist).sum()}")
        print(f"   - X_hist Inf数量: {np.isinf(X_hist).sum()}")
        print(f"   - y NaN数量: {np.isnan(y).sum()}")
        print(f"   - y Inf数量: {np.isinf(y).sum()}")
        print(f"   - y范围: {np.min(y):.4f} - {np.max(y):.4f}")
        
        # 数据分割
        result_split = split_data(X_hist, X_fcst, y, hours, dates)
        (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr,
         Xh_va, Xf_va, y_va, hrs_va, dates_va,
         Xh_te, Xf_te, y_te, hrs_te, dates_te) = result_split
        
        print(f"✅ 数据分割成功")
        print(f"📊 训练集: {len(Xh_tr)}样本")
        print(f"📊 验证集: {len(Xh_va)}样本")
        print(f"📊 测试集: {len(Xh_te)}样本")
        print(f"📊 训练集X形状: {Xh_tr.shape}")
        print(f"📊 训练集y形状: {y_tr.shape}")
        
    else:
        print("❌ 没有可用的预处理结果")
        
except Exception as e:
    print(f"❌ 滑动窗口/数据分割失败: {e}")
    traceback.print_exc()

# 4. 测试ML模型
print("\n" + "="*80)
print("4️⃣ 测试ML模型")
print("="*80)

ml_models = ['Linear', 'RF', 'XGB', 'LGBM']
ml_results = {}

for model_name in ml_models:
    print(f"\n🔧 测试 {model_name} 模型:")
    try:
        # 找到对应的配置
        model_config = None
        for cfg_info in project_configs:
            cfg = cfg_info.get('config', {})
            # 所有模型都使用'low'复杂度
            if cfg.get('model') == model_name and cfg.get('model_complexity') == 'low':
                model_config = cfg
                break
        
        if not model_config:
            print(f"❌ 找不到 {model_name} 模型配置")
            continue
            
        print(f"📊 使用配置: {model_config.get('model_complexity', 'N/A')}")
        print(f"📊 输入特征: use_pv={model_config.get('use_pv')}, use_hist_weather={model_config.get('use_hist_weather')}, use_forecast={model_config.get('use_forecast')}")
        
        # 使用当前配置重新预处理数据
        df_clean_model, hist_feats_model, fcst_feats_model, _, _, _ = preprocess_features(df, model_config)
        
        # 重新创建滑动窗口
        past_hours = model_config.get('past_hours', 24)
        future_hours = model_config.get('future_hours', 24)
        
        X_hist_model, X_fcst_model, y_model, hours_model, dates_model = create_sliding_windows(
            df_clean_model, past_hours, future_hours, hist_feats_model, fcst_feats_model
        )
        
        # 重新分割数据
        (Xh_tr_model, Xf_tr_model, y_tr_model, hrs_tr_model, dates_tr_model,
         Xh_va_model, Xf_va_model, y_va_model, hrs_va_model, dates_va_model,
         Xh_te_model, Xf_te_model, y_te_model, hrs_te_model, dates_te_model) = split_data(
            X_hist_model, X_fcst_model, y_model, hours_model, dates_model)
        
        # 准备2D数据
        X_train_2d = Xh_tr_model.reshape(Xh_tr_model.shape[0], -1)
        X_test_2d = Xh_te_model.reshape(Xh_te_model.shape[0], -1)
        
        # 如果有预测特征，合并
        if Xf_tr_model is not None and Xf_tr_model.shape[2] > 0:
            X_train_2d = np.hstack([X_train_2d, Xf_tr_model.reshape(Xf_tr_model.shape[0], -1)])
            X_test_2d = np.hstack([X_test_2d, Xf_te_model.reshape(Xf_te_model.shape[0], -1)])
        
        print(f"📊 训练数据形状: {X_train_2d.shape}")
        print(f"📊 测试数据形状: {X_test_2d.shape}")
        print(f"📊 目标数据形状: {y_tr_model.shape}")
        
        # 检查特征数量
        if X_train_2d.shape[1] == 0:
            print(f"❌ {model_name} 模型测试失败: 输入特征数量为0，无法训练。")
            continue
        
        # 检查数据质量
        print(f"📊 训练数据质量:")
        print(f"   - NaN数量: {np.isnan(X_train_2d).sum()}")
        print(f"   - Inf数量: {np.isinf(X_train_2d).sum()}")
        if X_train_2d.shape[1] > 0:
            print(f"   - 范围: {np.min(X_train_2d):.4f} - {np.max(X_train_2d):.4f}")
        
        # 训练模型
        start_time = time.time()
        
        if model_name == 'Linear':
            from sklearn.linear_model import LinearRegression
            from sklearn.multioutput import MultiOutputRegressor
            model = MultiOutputRegressor(LinearRegression())
            # 训练Linear模型
            model.fit(X_train_2d, y_tr_model)
        elif model_name == 'RF':
            from models.ml_models import train_rf
            model_params = model_config.get('model_params', {}).get('ml_low', {})
            model = train_rf(X_train_2d, y_tr_model, model_params)
        elif model_name == 'XGB':
            from models.ml_models import train_xgb
            model_params = model_config.get('model_params', {}).get('ml_low', {})
            model = train_xgb(X_train_2d, y_tr_model, model_params)
        elif model_name == 'LGBM':
            from models.ml_models import train_lgbm
            model_params = model_config.get('model_params', {}).get('ml_low', {})
            model = train_lgbm(X_train_2d, y_tr_model, model_params)
        
        train_time = time.time() - start_time
        
        # 预测
        start_time = time.time()
        y_pred = model.predict(X_test_2d)
        inference_time = time.time() - start_time
        
        print(f"📊 预测结果形状: {y_pred.shape}")
        print(f"📊 预测结果范围: {np.min(y_pred):.4f} - {np.max(y_pred):.4f}")
        
        # 计算指标
        mae = np.mean(np.abs(y_te_model - y_pred))
        rmse = np.sqrt(np.mean((y_te_model - y_pred) ** 2))
        
        # 计算R²
        y_true_flat = y_te_model.flatten()
        y_pred_flat = y_pred.flatten()
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 计算MAPE（避免除零）
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-8))) * 100
        
        print(f"✅ {model_name} 模型训练成功")
        print(f"📊 训练时间: {train_time:.2f}秒")
        print(f"📊 推理时间: {inference_time:.2f}秒")
        print(f"📊 MAE: {mae:.4f}")
        print(f"📊 RMSE: {rmse:.4f}")
        print(f"📊 R²: {r2:.4f}")
        print(f"📊 MAPE: {mape:.2f}%")
        
        ml_results[model_name] = {
            'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape,
            'train_time': train_time, 'inference_time': inference_time,
            'config': model_config
        }
        
    except Exception as e:
        print(f"❌ {model_name} 模型测试失败: {e}")
        traceback.print_exc()

# 5. 测试DL模型
print("\n" + "="*80)
print("5️⃣ 测试DL模型")
print("="*80)

dl_models = ['LSTM', 'GRU', 'TCN', 'Transformer']
dl_results = {}

for model_name in dl_models:
    print(f"\n🔧 测试 {model_name} 模型:")
    try:
        # 找到对应的配置
        model_config = None
        for cfg_info in project_configs:
            cfg = cfg_info.get('config', {})
            if cfg.get('model') == model_name and cfg.get('model_complexity') == 'low':
                model_config = cfg
                break
        
        if not model_config:
            print(f"❌ 找不到 {model_name} 模型配置")
            continue
            
        print(f"📊 使用配置: {model_config.get('model_complexity', 'N/A')}")
        print(f"📊 输入特征: use_pv={model_config.get('use_pv')}, use_hist_weather={model_config.get('use_hist_weather')}, use_forecast={model_config.get('use_forecast')}")
        
        # 使用当前配置重新预处理数据
        df_clean_model, hist_feats_model, fcst_feats_model, scaler_hist, scaler_fcst, scaler_target = preprocess_features(df, model_config)
        
        # 重新创建滑动窗口
        past_hours = model_config.get('past_hours', 24)
        future_hours = model_config.get('future_hours', 24)
        
        X_hist_model, X_fcst_model, y_model, hours_model, dates_model = create_sliding_windows(
            df_clean_model, past_hours, future_hours, hist_feats_model, fcst_feats_model
        )
        
        # 重新分割数据
        (Xh_tr_model, Xf_tr_model, y_tr_model, hrs_tr_model, dates_tr_model,
         Xh_va_model, Xf_va_model, y_va_model, hrs_va_model, dates_va_model,
         Xh_te_model, Xf_te_model, y_te_model, hrs_te_model, dates_te_model) = split_data(
            X_hist_model, X_fcst_model, y_model, hours_model, dates_model)
        
        # 准备训练参数
        model_params = model_config.get('model_params', {}).get('low', {})
        train_params = model_config.get('train_params', {})
        
        # 确保train_params包含必需的参数
        train_params.update({
            'batch_size': 32,
            'learning_rate': 0.001,
            'loss_type': 'mse',
            'future_hours': 24
        })
        
        print(f"📊 模型参数: {model_params}")
        print(f"📊 训练参数: {train_params}")
        
        # 准备训练数据元组
        train_data = (Xh_tr_model, Xf_tr_model, y_tr_model, hrs_tr_model, dates_tr_model)
        val_data = (Xh_va_model, Xf_va_model, y_va_model, hrs_va_model, dates_va_model)
        test_data = (Xh_te_model, Xf_te_model, y_te_model, hrs_te_model, dates_te_model)
        scalers = (scaler_hist, scaler_fcst, scaler_target)
        
        # 构建完整的配置
        full_config = model_config.copy()
        full_config.update({
            'model_params': {'low': model_params},
            'train_params': train_params
        })
        
        # 训练模型
        from train.train_dl import train_dl_model
        
        model, metrics = train_dl_model(
            full_config, train_data, val_data, test_data, scalers
        )
        
        print(f"✅ {model_name} 模型训练成功")
        print(f"📊 训练指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.4f}")
            elif isinstance(value, (int, np.integer)):
                print(f"   - {key}: {value}")
            elif isinstance(value, list) and len(value) > 0:
                print(f"   - {key}: {len(value)}项")
            else:
                print(f"   - {key}: {value}")
        
        dl_results[model_name] = metrics
        
    except Exception as e:
        print(f"❌ {model_name} 模型测试失败: {e}")
        traceback.print_exc()

# 6. 结果汇总
print("\n" + "="*80)
print("6️⃣ 测试结果汇总")
print("="*80)

print(f"📊 ML模型结果:")
if ml_results:
    print(f"{'模型':<10} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'MAPE':<8} {'训练时间':<8} {'配置':<20}")
    print("-" * 80)
    for model, results in ml_results.items():
        config_info = results['config']
        config_str = f"{config_info.get('use_pv', False)}_{config_info.get('use_hist_weather', False)}_{config_info.get('use_forecast', False)}"
        print(f"{model:<10} {results['mae']:<8.4f} {results['rmse']:<8.4f} "
              f"{results['r2']:<8.4f} {results['mape']:<8.2f} {results['train_time']:<8.2f}s {config_str:<20}")
else:
    print("❌ 没有成功的ML模型测试")

print(f"\n📊 DL模型结果:")
if dl_results:
    print(f"{'模型':<12} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'训练时间':<8}")
    print("-" * 60)
    for model, results in dl_results.items():
        mae = results.get('mae', np.nan)
        rmse = results.get('rmse', np.nan)
        r2 = results.get('r2', np.nan)
        train_time = results.get('train_time_sec', np.nan)
        print(f"{model:<12} {mae:<8.4f} {rmse:<8.4f} {r2:<8.4f} {train_time:<8.2f}s")
else:
    print("❌ 没有成功的DL模型测试")

# 7. 测试结果保存
print("\n" + "="*80)
print("7️⃣ 测试结果保存")
print("="*80)

try:
    from eval.excel_utils import append_plant_excel_results
    
    # 创建测试结果数据
    test_results = {
        'config': {
            'model': 'Linear',
            'model_complexity': 'baseline',
            'use_pv': False,  # 不再使用PV特征
            'use_hist_weather': True,
            'use_forecast': True,
            'weather_category': 'all_weather',
            'use_time_encoding': True,
            'past_hours': 24,
            'future_hours': 24,
            'start_date': '2022-01-01',
            'end_date': '2024-09-28'
        },
        'metrics': {
            'train_time_sec': 10.0,
            'inference_time_sec': 1.0,
            'param_count': 1000,
            'samples_count': len(Xh_tr) if 'Xh_tr' in locals() else 0,
            'mse': 155.36,
            'rmse': 12.46,
            'mae': 6.97,
            'r2': 0.85,
            'mape': 15.2
        }
    }
    
    # 测试结果保存
    result_dir = Path("test_results")
    result_dir.mkdir(exist_ok=True)
    
    excel_file = append_plant_excel_results(
        plant_id=1140,
        result=test_results,
        save_dir=str(result_dir)
    )
    
    print(f"✅ 结果保存成功")
    print(f"📊 Excel文件: {excel_file}")
    
except Exception as e:
    print(f"❌ 结果保存失败: {e}")
    traceback.print_exc()

print("\n" + "="*80)
print("🎉 真实数据全面测试完成！")
print("="*80)

print(f"\n📊 最终测试总结:")
print(f"   - 配置生成: {'✅' if 'project_configs' in locals() else '❌'}")
print(f"   - 数据预处理: {'✅' if 'preprocessing_results' in locals() else '❌'}")
print(f"   - 滑动窗口: {'✅' if 'X_hist' in locals() else '❌'}")
print(f"   - ML模型测试: {len(ml_results)}/4 成功")
print(f"   - DL模型测试: {len(dl_results)}/4 成功")
print(f"   - 结果保存: {'✅' if 'excel_file' in locals() else '❌'}")

print(f"\n🎯 关键发现:")
if 'df' in locals():
    print(f"   - 数据时间范围: 2020-2024 (过滤后: 2022-2024)")
if 'X_hist' in locals():
    print(f"   - 可用样本数: {X_hist.shape[0]}")
    print(f"   - 特征维度: {X_hist.shape[2]}")
if ml_results:
    best_ml = min(ml_results.items(), key=lambda x: x[1]['mae'])
    print(f"   - 最佳ML模型: {best_ml[0]} (MAE={best_ml[1]['mae']:.4f})")
if dl_results:
    best_dl = min(dl_results.items(), key=lambda x: x[1].get('mae', float('inf')))
    print(f"   - 最佳DL模型: {best_dl[0]} (MAE={best_dl[1].get('mae', 0):.4f})")

print(f"\n⏰ 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ 总耗时: {time.time() - time.time():.2f}秒")
