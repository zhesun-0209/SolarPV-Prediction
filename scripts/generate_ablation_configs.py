#!/usr/bin/env python3
"""
生成Project1140消融实验的所有配置文件
根据3.7 Controlled ablation study设计生成360个实验配置
"""

import os
import yaml
from itertools import product

def create_config_dir():
    """创建配置目录"""
    config_dir = "config/ablation"
    os.makedirs(config_dir, exist_ok=True)
    return config_dir

def get_model_complexity_params():
    """获取模型复杂度参数"""
    return {
        'low': {
            'epochs': 15,
            'tree_params': {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1
            },
            'dl_params': {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 6,
                'hidden_dim': 32,
                'dropout': 0.1,
                'tcn_channels': [64, 64, 32],
                'kernel_size': 3
            }
        },
        'high': {
            'epochs': 50,
            'tree_params': {
                'n_estimators': 200,
                'max_depth': 12,
                'learning_rate': 0.01
            },
            'dl_params': {
                'd_model': 256,
                'num_heads': 16,
                'num_layers': 18,
                'hidden_dim': 128,
                'dropout': 0.3,
                'tcn_channels': [128, 128, 64, 32],
                'kernel_size': 3
            }
        }
    }

def generate_base_config():
    """生成基础配置模板"""
    return {
        'data_path': './data/Project1140.csv',
        'save_dir': './results/ablation',
        'future_hours': 24,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'plot_days': 7,
        'start_date': '2022-01-01',
        'end_date': '2024-09-28',
        'train_params': {
            'batch_size': 32,
            'learning_rate': 5e-4,
            'loss_type': 'mse',
            'future_hours': 24
        },
        'save_options': {
            'save_model': False,
            'save_summary': True,
            'save_predictions': True,
            'save_training_log': False,
            'save_excel_results': True
        }
    }

def create_feature_config(input_category, lookback_hours, use_time_encoding):
    """创建特征配置"""
    # 11个天气特征 (与_pred后缀对应的历史天气特征)
    weather_features = [
        'global_tilted_irradiance', 'vapour_pressure_deficit', 'relative_humidity_2m',
        'temperature_2m', 'wind_gusts_10m', 'cloud_cover_low', 'wind_speed_100m',
        'snow_depth', 'dew_point_2m', 'precipitation', 'surface_pressure'
    ]
    
    config = {
        'past_hours': lookback_hours,
        'use_time_encoding': use_time_encoding,
        'weather_category': 'ablation_11_features'
    }
    
    # 根据输入类别设置特征
    if input_category == 'PV':
        config.update({
            'use_pv': True,
            'use_hist_weather': False,
            'use_forecast': False,
            'input_category': 'PV_only'
        })
    elif input_category == 'PV+NWP':
        config.update({
            'use_pv': True,
            'use_hist_weather': False,
            'use_forecast': True,
            'input_category': 'PV_plus_NWP'
        })
    elif input_category == 'PV+NWP+':
        config.update({
            'use_pv': True,
            'use_hist_weather': False,
            'use_forecast': True,
            'use_ideal_nwp': True,
            'input_category': 'PV_plus_ideal_NWP'
        })
    elif input_category == 'PV+HW':
        config.update({
            'use_pv': True,
            'use_hist_weather': True,
            'use_forecast': False,
            'input_category': 'PV_plus_HW'
        })
    elif input_category == 'NWP':
        config.update({
            'use_pv': False,
            'use_hist_weather': False,
            'use_forecast': True,
            'input_category': 'NWP_only'
        })
    elif input_category == 'NWP+':
        config.update({
            'use_pv': False,
            'use_hist_weather': False,
            'use_forecast': True,
            'use_ideal_nwp': True,
            'input_category': 'ideal_NWP_only'
        })
    
    return config

def create_model_config(model_name, complexity, complexity_params):
    """创建模型配置"""
    config = {
        'model': model_name,
        'model_complexity': complexity
    }
    
    if model_name == 'LSR':
        # LSR基线模型，不区分复杂度
        config.update({
            'epoch_params': {'low': 1, 'high': 1},
            'model_params': {
                'low': {'fit_intercept': True},
                'high': {'fit_intercept': True}
            }
        })
    elif model_name in ['RF', 'XGB', 'LGBM']:
        # 树模型
        config.update({
            'epoch_params': complexity_params,
            'model_params': {
                'ml_low': complexity_params['low']['tree_params'],
                'ml_high': complexity_params['high']['tree_params']
            }
        })
    else:
        # 深度学习模型
        config.update({
            'epoch_params': complexity_params,
            'model_params': {
                'low': complexity_params['low']['dl_params'],
                'high': complexity_params['high']['dl_params']
            }
        })
    
    return config

def generate_all_configs():
    """生成所有360个实验配置"""
    config_dir = create_config_dir()
    complexity_params = get_model_complexity_params()
    base_config = generate_base_config()
    
    # 实验参数
    input_categories = ['PV', 'PV+NWP', 'PV+NWP+', 'PV+HW', 'NWP', 'NWP+']
    lookback_hours = [24, 72]
    time_encoding = [False, True]
    complexities = ['low', 'high']
    models = ['RF', 'XGB', 'LGBM', 'LSTM', 'GRU', 'TCN', 'Transformer']
    baseline_model = 'LSR'
    
    config_count = 0
    
    # 生成主要模型配置 (7个模型 × 2复杂度 = 336个配置)
    for input_cat, lookback, te, complexity, model in product(
        input_categories, lookback_hours, time_encoding, complexities, models
    ):
        config = base_config.copy()
        
        # 添加特征配置
        feature_config = create_feature_config(input_cat, lookback, te)
        config.update(feature_config)
        
        # 添加模型配置
        model_config = create_model_config(model, complexity, complexity_params)
        config.update(model_config)
        
        # 生成配置文件名
        config_name = f"{model}_{complexity}_{input_cat.replace('+', '_plus_')}_{lookback}h_{'TE' if te else 'noTE'}.yaml"
        config_path = os.path.join(config_dir, config_name)
        
        # 保存配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        config_count += 1
        if config_count % 50 == 0:
            print(f"已生成 {config_count} 个配置文件...")
    
    # 生成LSR基线模型配置 (24个配置，不区分复杂度)
    for input_cat, lookback, te in product(input_categories, lookback_hours, time_encoding):
        config = base_config.copy()
        
        # 添加特征配置
        feature_config = create_feature_config(input_cat, lookback, te)
        config.update(feature_config)
        
        # 添加LSR模型配置
        model_config = create_model_config(baseline_model, 'low', complexity_params)
        config.update(model_config)
        
        # 生成配置文件名
        config_name = f"LSR_baseline_{input_cat.replace('+', '_plus_')}_{lookback}h_{'TE' if te else 'noTE'}.yaml"
        config_path = os.path.join(config_dir, config_name)
        
        # 保存配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        config_count += 1
    
    print(f"总共生成了 {config_count} 个配置文件")
    print(f"配置文件保存在: {config_dir}")
    
    # 生成配置索引文件
    generate_config_index(config_dir, config_count)
    
    return config_count

def generate_config_index(config_dir, total_configs):
    """生成配置索引文件"""
    index_file = os.path.join(config_dir, "config_index.yaml")
    
    index_data = {
        'total_configs': total_configs,
        'experiment_design': {
            'input_categories': 6,
            'lookback_hours': 2,
            'time_encoding': 2,
            'model_complexity': 2,
            'ml_models': 7,
            'baseline_model': 1
        },
        'calculation': {
            'main_models': '6 × 2 × 2 × 2 × 7 = 336',
            'baseline_model': '6 × 2 × 2 × 1 = 24',
            'total': '336 + 24 = 360'
        },
        'input_categories': [
            'PV - 仅历史PV功率',
            'PV+NWP - 历史PV + 目标日NWP',
            'PV+NWP+ - 历史PV + 理想NWP',
            'PV+HW - 历史PV + 历史HW',
            'NWP - 仅目标日NWP',
            'NWP+ - 仅理想NWP'
        ],
        'models': {
            'ml_models': ['RF', 'XGB', 'LGBM', 'LSTM', 'GRU', 'TCN', 'Transformer'],
            'baseline': ['LSR']
        },
        'weather_features': [
            'global_tilted_irradiance', 'vapour_pressure_deficit', 'relative_humidity_2m',
            'temperature_2m', 'wind_gusts_10m', 'cloud_cover_low', 'wind_speed_100m',
            'snow_depth', 'dew_point_2m', 'precipitation', 'surface_pressure'
        ]
    }
    
    with open(index_file, 'w', encoding='utf-8') as f:
        yaml.dump(index_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"配置索引文件已生成: {index_file}")

if __name__ == "__main__":
    print("开始生成Project1140消融实验配置文件...")
    print("实验设计: 6输入类别 × 2回看窗口 × 2时间编码 × 2复杂度 × 7模型 + 24基线 = 360配置")
    
    total_configs = generate_all_configs()
    
    print(f"\n✅ 配置生成完成!")
    print(f"📁 配置文件目录: config/ablation/")
    print(f"📊 总配置数: {total_configs}")
    print(f"📋 索引文件: config/ablation/config_index.yaml")
