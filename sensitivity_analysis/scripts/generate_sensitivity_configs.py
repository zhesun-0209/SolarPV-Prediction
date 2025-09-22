#!/usr/bin/env python3
"""
敏感性分析实验配置生成器
生成Weather feature adoption, Lookback window length, Model complexity, Training dataset scale四个维度的实验配置
"""

import os
import yaml
import itertools
from pathlib import Path
import re

def generate_base_config():
    """生成基础配置模板"""
    return {
        'data_path': '',  # 将在运行时动态设置
        'save_dir': '',   # 将在运行时动态设置
        'future_hours': 24,
        'plot_days': 7,
        'start_date': '2022-01-01',
        'end_date': '2024-09-28',
        'train_params': {
            'batch_size': 64,
            'learning_rate': 0.001,
            'loss_type': 'mse',
            'future_hours': 24
        },
        'epoch_params': {
            'level1': 30,   # 新增Level 1
            'level2': 50,   # 原low
            'level3': 65,   # 新增Level 3
            'level4': 80    # 原high
        },
        'save_options': {
            'save_model': False,
            'save_summary': False,
            'save_predictions': False,
            'save_training_log': False,
            'save_excel_results': False
        }
    }

def create_weather_feature_config(weather_level):
    """创建天气特征配置"""
    # 11个天气特征
    weather_features = [
        'global_tilted_irradiance', 'vapour_pressure_deficit', 'relative_humidity_2m',
        'temperature_2m', 'wind_gusts_10m', 'cloud_cover_low', 'wind_speed_100m',
        'snow_depth', 'dew_point_2m', 'precipitation', 'surface_pressure'
    ]
    
    # 预测天气特征（NWP）- 带_pred后缀
    forecast_features = [f + '_pred' for f in weather_features]
    
    config = {}
    
    if weather_level == 'SI':
        # 只用solar irradiance
        config['use_pv'] = True
        config['use_hist_weather'] = False
        config['use_forecast'] = True
        config['use_ideal_nwp'] = False
        config['weather_category'] = 'solar_irradiance_only'
        config['selected_weather_features'] = ['global_tilted_irradiance_pred']
    elif weather_level == 'H':
        # High: Solar irradiance, vapor pressure deficit, relative humidity
        config['use_pv'] = True
        config['use_hist_weather'] = False
        config['use_forecast'] = True
        config['use_ideal_nwp'] = False
        config['weather_category'] = 'high_weather'
        config['selected_weather_features'] = [
            'global_tilted_irradiance_pred', 'vapour_pressure_deficit_pred', 'relative_humidity_2m_pred'
        ]
    elif weather_level == 'M':
        # Medium: H + Temperature, wind gusts, cloud cover, wind speed
        config['use_pv'] = True
        config['use_hist_weather'] = False
        config['use_forecast'] = True
        config['use_ideal_nwp'] = False
        config['weather_category'] = 'medium_weather'
        config['selected_weather_features'] = [
            'global_tilted_irradiance_pred', 'vapour_pressure_deficit_pred', 'relative_humidity_2m_pred',
            'temperature_2m_pred', 'wind_gusts_10m_pred', 'cloud_cover_low_pred', 'wind_speed_100m_pred'
        ]
    elif weather_level == 'L':
        # Low: H + M + Snow depth, dewpoint, surface pressure, precipitation
        config['use_pv'] = True
        config['use_hist_weather'] = False
        config['use_forecast'] = True
        config['use_ideal_nwp'] = False
        config['weather_category'] = 'low_weather'
        config['selected_weather_features'] = [
            'global_tilted_irradiance_pred', 'vapour_pressure_deficit_pred', 'relative_humidity_2m_pred',
            'temperature_2m_pred', 'wind_gusts_10m_pred', 'cloud_cover_low_pred', 'wind_speed_100m_pred',
            'snow_depth_pred', 'dew_point_2m_pred', 'surface_pressure_pred', 'precipitation_pred'
        ]
    
    return config

def create_lookback_config(lookback_hours):
    """创建回看窗口配置"""
    config = {
        'past_hours': lookback_hours,
        'past_days': lookback_hours // 24
    }
    return config

def create_model_complexity_config(model, complexity_level):
    """创建模型复杂度配置"""
    config = {'model': model}
    
    # 初始化model_params和train_params
    config['model_params'] = {}
    config['train_params'] = {}
    
    if model == 'LSR':
        config['model'] = 'Linear'
        config['model_complexity'] = 'level2'  # LSR固定为level2
        config['epochs'] = 1
        config['model_params'] = {
            'ml_level1': {'learning_rate': 0.001},
            'ml_level2': {'learning_rate': 0.001},
            'ml_level3': {'learning_rate': 0.001},
            'ml_level4': {'learning_rate': 0.001}
        }
    elif model in ['RF', 'XGB', 'LGBM']:
        config['model_complexity'] = f'level{complexity_level}'
        config['epochs'] = 30 if complexity_level == 1 else (50 if complexity_level == 2 else (65 if complexity_level == 3 else 80))
        
        if complexity_level == 1:
            # Level 1: 最简单的设置
            config['model_params'] = {
                'ml_level1': {'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.2},
                'ml_level2': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1},
                'ml_level3': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05},
                'ml_level4': {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.01}
            }
            config['train_params'].update({
                'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.2
            })
        elif complexity_level == 2:
            # Level 2: 原low设置
            config['model_params'] = {
                'ml_level1': {'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.2},
                'ml_level2': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1},
                'ml_level3': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05},
                'ml_level4': {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.01}
            }
            config['train_params'].update({
                'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1
            })
        elif complexity_level == 3:
            # Level 3: 中等设置
            config['model_params'] = {
                'ml_level1': {'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.2},
                'ml_level2': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1},
                'ml_level3': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05},
                'ml_level4': {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.01}
            }
            config['train_params'].update({
                'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05
            })
        else:  # complexity_level == 4
            # Level 4: 原high设置
            config['model_params'] = {
                'ml_level1': {'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.2},
                'ml_level2': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1},
                'ml_level3': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05},
                'ml_level4': {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.01}
            }
            config['train_params'].update({
                'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.01
            })
    else:  # 深度学习模型
        config['model_complexity'] = f'level{complexity_level}'
        config['epochs'] = 30 if complexity_level == 1 else (50 if complexity_level == 2 else (65 if complexity_level == 3 else 80))
        
        if complexity_level == 1:
            # Level 1: 最简单的设置
            if model == 'TCN':
                config['model_params'] = {
                    'level1': {'tcn_channels': [16, 32], 'kernel_size': 2, 'dropout': 0.05},
                    'level2': {'tcn_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.1},
                    'level3': {'tcn_channels': [48, 96], 'kernel_size': 4, 'dropout': 0.2},
                    'level4': {'tcn_channels': [64, 128, 256], 'kernel_size': 5, 'dropout': 0.3}
                }
            else:
                config['model_params'] = {
                    'level1': {'d_model': 32, 'num_heads': 2, 'num_layers': 3, 'hidden_dim': 16, 'dropout': 0.05},
                    'level2': {'d_model': 64, 'num_heads': 4, 'num_layers': 6, 'hidden_dim': 32, 'dropout': 0.1},
                    'level3': {'d_model': 128, 'num_heads': 8, 'num_layers': 12, 'hidden_dim': 64, 'dropout': 0.2},
                    'level4': {'d_model': 256, 'num_heads': 16, 'num_layers': 18, 'hidden_dim': 128, 'dropout': 0.3}
                }
        elif complexity_level == 2:
            # Level 2: 原low设置
            if model == 'TCN':
                config['model_params'] = {
                    'level1': {'tcn_channels': [16, 32], 'kernel_size': 2, 'dropout': 0.05},
                    'level2': {'tcn_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.1},
                    'level3': {'tcn_channels': [48, 96], 'kernel_size': 4, 'dropout': 0.2},
                    'level4': {'tcn_channels': [64, 128, 256], 'kernel_size': 5, 'dropout': 0.3}
                }
            else:
                config['model_params'] = {
                    'level1': {'d_model': 32, 'num_heads': 2, 'num_layers': 3, 'hidden_dim': 16, 'dropout': 0.05},
                    'level2': {'d_model': 64, 'num_heads': 4, 'num_layers': 6, 'hidden_dim': 32, 'dropout': 0.1},
                    'level3': {'d_model': 128, 'num_heads': 8, 'num_layers': 12, 'hidden_dim': 64, 'dropout': 0.2},
                    'level4': {'d_model': 256, 'num_heads': 16, 'num_layers': 18, 'hidden_dim': 128, 'dropout': 0.3}
                }
        elif complexity_level == 3:
            # Level 3: 中等设置
            if model == 'TCN':
                config['model_params'] = {
                    'level1': {'tcn_channels': [16, 32], 'kernel_size': 2, 'dropout': 0.05},
                    'level2': {'tcn_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.1},
                    'level3': {'tcn_channels': [48, 96], 'kernel_size': 4, 'dropout': 0.2},
                    'level4': {'tcn_channels': [64, 128, 256], 'kernel_size': 5, 'dropout': 0.3}
                }
            else:
                config['model_params'] = {
                    'level1': {'d_model': 32, 'num_heads': 2, 'num_layers': 3, 'hidden_dim': 16, 'dropout': 0.05},
                    'level2': {'d_model': 64, 'num_heads': 4, 'num_layers': 6, 'hidden_dim': 32, 'dropout': 0.1},
                    'level3': {'d_model': 128, 'num_heads': 8, 'num_layers': 12, 'hidden_dim': 64, 'dropout': 0.2},
                    'level4': {'d_model': 256, 'num_heads': 16, 'num_layers': 18, 'hidden_dim': 128, 'dropout': 0.3}
                }
        else:  # complexity_level == 4
            # Level 4: 原high设置
            if model == 'TCN':
                config['model_params'] = {
                    'level1': {'tcn_channels': [16, 32], 'kernel_size': 2, 'dropout': 0.05},
                    'level2': {'tcn_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.1},
                    'level3': {'tcn_channels': [48, 96], 'kernel_size': 4, 'dropout': 0.2},
                    'level4': {'tcn_channels': [64, 128, 256], 'kernel_size': 5, 'dropout': 0.3}
                }
            else:
                config['model_params'] = {
                    'level1': {'d_model': 32, 'num_heads': 2, 'num_layers': 3, 'hidden_dim': 16, 'dropout': 0.05},
                    'level2': {'d_model': 64, 'num_heads': 4, 'num_layers': 6, 'hidden_dim': 32, 'dropout': 0.1},
                    'level3': {'d_model': 128, 'num_heads': 8, 'num_layers': 12, 'hidden_dim': 64, 'dropout': 0.2},
                    'level4': {'d_model': 256, 'num_heads': 16, 'num_layers': 18, 'hidden_dim': 128, 'dropout': 0.3}
                }
        
        if model == 'TCN':
            config['train_params'].update({
                'batch_size': 64,
                'learning_rate': 0.001
            })
        else:
            config['train_params'].update({
                'batch_size': 64,
                'learning_rate': 0.001
            })
    
    return config

def create_dataset_scale_config(dataset_scale):
    """创建数据集规模配置"""
    if dataset_scale == 'Low':
        config = {
            'train_ratio': 0.5,  # 20+10+10 = 40% train, 25% val, 25% test
            'val_ratio': 0.25,
            'test_ratio': 0.25
        }
    elif dataset_scale == 'Medium':
        config = {
            'train_ratio': 0.67,  # 40+10+10 = 67% train, 17% val, 17% test
            'val_ratio': 0.17,
            'test_ratio': 0.16
        }
    elif dataset_scale == 'High':
        config = {
            'train_ratio': 0.75,  # 60+10+10 = 75% train, 12.5% val, 12.5% test
            'val_ratio': 0.125,
            'test_ratio': 0.125
        }
    else:  # Full
        config = {
            'train_ratio': 0.8,  # 80+10+10 = 80% train, 10% val, 10% test
            'val_ratio': 0.1,
            'test_ratio': 0.1
        }
    
    return config

def generate_sensitivity_configs(project_id):
    """为单个Project生成所有敏感性分析配置"""
    configs = []
    
    # 默认配置
    default_lookback = 24
    default_complexity = 2  # Level 2 (原low)
    default_dataset = 'Full'  # 80%训练集
    default_weather = 'L'  # 所有天气特征
    
    # 实验参数
    weather_levels = ['SI', 'H', 'M', 'L']
    lookback_hours = [24, 72, 120, 168]
    complexity_levels = [1, 2, 3, 4]
    dataset_scales = ['Low', 'Medium', 'High', 'Full']
    models = ['LSR', 'RF', 'XGB', 'LGBM', 'LSTM', 'GRU', 'TCN', 'Transformer']
    
    config_count = 0
    
    # 1. Weather feature adoption实验
    print("生成Weather feature adoption实验配置...")
    for weather in weather_levels:
        for model in models:
            config_count += 1
            
            # 创建配置名称
            config_name = f"weather_{model}_{weather}"
            
            # 生成基础配置
            base_config = generate_base_config()
            base_config['data_path'] = f"data/Project{project_id}.csv"
            base_config['save_dir'] = f"sensitivity_analysis/results/{project_id}/{config_name}"
            
            # 添加特征配置
            weather_config = create_weather_feature_config(weather)
            base_config.update(weather_config)
            
            # 添加回看窗口配置（默认24h）
            lookback_config = create_lookback_config(default_lookback)
            base_config.update(lookback_config)
            
            # 添加模型配置（默认复杂度）
            model_config = create_model_complexity_config(model, default_complexity)
            base_config.update(model_config)
            
            # 添加数据集规模配置（默认80%）
            dataset_config = create_dataset_scale_config(default_dataset)
            base_config.update(dataset_config)
            
            # 时间编码配置（固定为False）
            base_config['use_time_encoding'] = False
            
            # 保存配置信息
            config_info = {
                'name': config_name,
                'config_id': config_count,
                'config': base_config,
                'experiment_type': 'weather_adoption',
                'weather_level': weather,
                'lookback_hours': default_lookback,
                'complexity_level': default_complexity,
                'dataset_scale': default_dataset,
                'model': model
            }
            configs.append(config_info)
    
    # 2. Lookback window length实验
    print("生成Lookback window length实验配置...")
    for lookback in lookback_hours:
        for model in models:
            if model == 'LSR':  # LSR不受回看窗口影响
                continue
                
            config_count += 1
            
            # 创建配置名称
            config_name = f"lookback_{model}_{lookback}h"
            
            # 生成基础配置
            base_config = generate_base_config()
            base_config['data_path'] = f"data/Project{project_id}.csv"
            base_config['save_dir'] = f"sensitivity_analysis/results/{project_id}/{config_name}"
            
            # 添加特征配置（默认所有天气特征）
            weather_config = create_weather_feature_config(default_weather)
            base_config.update(weather_config)
            
            # 添加回看窗口配置
            lookback_config = create_lookback_config(lookback)
            base_config.update(lookback_config)
            
            # 添加模型配置（默认复杂度）
            model_config = create_model_complexity_config(model, default_complexity)
            base_config.update(model_config)
            
            # 添加数据集规模配置（默认80%）
            # 注意：为了确保不同回看窗口使用相同的训练样本数量，
            # 我们需要调整数据集划分策略
            dataset_config = create_dataset_scale_config(default_dataset)
            base_config.update(dataset_config)
            
            # 添加样本数量控制配置
            # 使用固定的有效数据范围，确保所有回看窗口使用相同的样本数量
            base_config['fixed_sample_count'] = True
            base_config['max_lookback_hours'] = max(lookback_hours)  # 使用最大回看窗口作为基准
            
            # 时间编码配置（固定为False）
            base_config['use_time_encoding'] = False
            
            # 保存配置信息
            config_info = {
                'name': config_name,
                'config_id': config_count,
                'config': base_config,
                'experiment_type': 'lookback_length',
                'weather_level': default_weather,
                'lookback_hours': lookback,
                'complexity_level': default_complexity,
                'dataset_scale': default_dataset,
                'model': model,
                'note': f'使用固定样本数量策略，确保不同回看窗口的样本数量一致'
            }
            configs.append(config_info)
    
    # 3. Model complexity实验
    print("生成Model complexity实验配置...")
    for complexity in complexity_levels:
        for model in models:
            if model == 'LSR':  # LSR不受复杂度影响
                continue
                
            config_count += 1
            
            # 创建配置名称
            config_name = f"complexity_{model}_L{complexity}"
            
            # 生成基础配置
            base_config = generate_base_config()
            base_config['data_path'] = f"data/Project{project_id}.csv"
            base_config['save_dir'] = f"sensitivity_analysis/results/{project_id}/{config_name}"
            
            # 添加特征配置（默认所有天气特征）
            weather_config = create_weather_feature_config(default_weather)
            base_config.update(weather_config)
            
            # 添加回看窗口配置（默认24h）
            lookback_config = create_lookback_config(default_lookback)
            base_config.update(lookback_config)
            
            # 添加模型配置
            model_config = create_model_complexity_config(model, complexity)
            base_config.update(model_config)
            
            # 添加数据集规模配置（默认80%）
            dataset_config = create_dataset_scale_config(default_dataset)
            base_config.update(dataset_config)
            
            # 时间编码配置（固定为False）
            base_config['use_time_encoding'] = False
            
            # 保存配置信息
            config_info = {
                'name': config_name,
                'config_id': config_count,
                'config': base_config,
                'experiment_type': 'model_complexity',
                'weather_level': default_weather,
                'lookback_hours': default_lookback,
                'complexity_level': complexity,
                'dataset_scale': default_dataset,
                'model': model
            }
            configs.append(config_info)
    
    # 4. Training dataset scale实验
    print("生成Training dataset scale实验配置...")
    for dataset in dataset_scales:
        for model in models:
            config_count += 1
            
            # 创建配置名称
            config_name = f"dataset_{model}_{dataset}"
            
            # 生成基础配置
            base_config = generate_base_config()
            base_config['data_path'] = f"data/Project{project_id}.csv"
            base_config['save_dir'] = f"sensitivity_analysis/results/{project_id}/{config_name}"
            
            # 添加特征配置（默认所有天气特征）
            weather_config = create_weather_feature_config(default_weather)
            base_config.update(weather_config)
            
            # 添加回看窗口配置（默认24h）
            lookback_config = create_lookback_config(default_lookback)
            base_config.update(lookback_config)
            
            # 添加模型配置（默认复杂度）
            model_config = create_model_complexity_config(model, default_complexity)
            base_config.update(model_config)
            
            # 添加数据集规模配置
            dataset_config = create_dataset_scale_config(dataset)
            base_config.update(dataset_config)
            
            # 时间编码配置（固定为False）
            base_config['use_time_encoding'] = False
            
            # 保存配置信息
            config_info = {
                'name': config_name,
                'config_id': config_count,
                'config': base_config,
                'experiment_type': 'dataset_scale',
                'weather_level': default_weather,
                'lookback_hours': default_lookback,
                'complexity_level': default_complexity,
                'dataset_scale': dataset,
                'model': model
            }
            configs.append(config_info)
    
    return configs

def save_sensitivity_configs(project_id, configs):
    """保存敏感性分析配置到文件"""
    project_dir = Path("sensitivity_analysis/configs") / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个配置文件
    for config_info in configs:
        config_file = project_dir / f"{config_info['name']}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_info['config'], f, default_flow_style=False, allow_unicode=True)
    
    # 保存配置索引
    index_data = {
        'project_id': project_id,
        'experiment_type': 'sensitivity_analysis',
        'total_configs': len(configs),
        'configs': [
            {
                'name': c['name'],
                'config_id': c['config_id'],
                'file': f"{c['name']}.yaml",
                'weather_level': c['weather_level'],
                'lookback_hours': c['lookback_hours'],
                'complexity_level': c['complexity_level'],
                'dataset_scale': c['dataset_scale'],
                'model': c['model']
            }
            for c in configs
        ]
    }
    
    index_file = project_dir / "sensitivity_index.yaml"
    with open(index_file, 'w', encoding='utf-8') as f:
        yaml.dump(index_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ {project_id}: 生成 {len(configs)} 个敏感性分析配置文件")
    return len(configs)

def detect_project_files():
    """检测data目录中的Project文件"""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    # 查找所有Project*.csv文件
    csv_files = list(data_dir.glob("Project*.csv"))
    
    # 提取Project ID
    project_ids = []
    for csv_file in csv_files:
        # 使用正则表达式提取Project ID
        match = re.match(r'Project(\d+)\.csv', csv_file.name)
        if match:
            project_id = match.group(1)
            project_ids.append(project_id)
    
    return sorted(project_ids)

def main():
    """主函数"""
    print("🚀 开始生成敏感性分析实验配置")
    print("=" * 60)
    
    # 检测实际的Project文件
    project_ids = detect_project_files()
    
    if not project_ids:
        print("❌ 未找到任何Project CSV文件")
        print("请确保data/目录下有Project*.csv文件")
        return
    
    print(f"📊 检测到 {len(project_ids)} 个Project文件")
    
    # 只处理前20个厂
    target_projects = project_ids[:20]
    print(f"🎯 将处理前20个厂: {target_projects}")
    
    # 创建配置目录
    config_dir = Path("sensitivity_analysis/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    total_configs = 0
    successful_projects = 0
    
    # 为每个目标Project生成配置
    for project_id in target_projects:
        try:
            print(f"📝 生成 Project{project_id} 敏感性分析配置...")
            configs = generate_sensitivity_configs(project_id)
            count = save_sensitivity_configs(project_id, configs)
            total_configs += count
            successful_projects += 1
        except Exception as e:
            print(f"❌ Project{project_id} 配置生成失败: {e}")
    
    # 生成全局索引
    global_index = {
        'experiment_type': 'sensitivity_analysis',
        'total_projects': successful_projects,
        'total_configs': total_configs,
        'projects': []
    }
    
    for project_id in target_projects:
        project_dir = config_dir / project_id
        if project_dir.exists():
            global_index['projects'].append({
                'project_id': project_id,
                'config_dir': str(project_dir),
                'index_file': str(project_dir / "sensitivity_index.yaml")
            })
    
    # 保存全局索引
    global_index_file = config_dir / "sensitivity_global_index.yaml"
    with open(global_index_file, 'w', encoding='utf-8') as f:
        yaml.dump(global_index, f, default_flow_style=False, allow_unicode=True)
    
    print("=" * 60)
    print("🎉 敏感性分析配置生成完成!")
    print(f"✅ 成功生成 {successful_projects} 个Project的配置")
    print(f"📊 总配置数量: {total_configs}")
    print(f"📁 配置保存在: sensitivity_analysis/configs")
    print(f"📋 全局索引: sensitivity_analysis/configs/sensitivity_global_index.yaml")

if __name__ == "__main__":
    main()
