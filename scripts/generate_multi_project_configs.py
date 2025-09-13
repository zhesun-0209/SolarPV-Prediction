#!/usr/bin/env python3
"""
为100个Project生成消融实验配置
每个Project生成360个实验配置，总共36000个配置
"""

import os
import yaml
import itertools
from pathlib import Path

def generate_base_config():
    """生成基础配置模板"""
    return {
        'data_path': '',  # 将在运行时动态设置
        'save_dir': '',   # 将在运行时动态设置
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
            'save_summary': False,
            'save_predictions': False,
            'save_training_log': False,
            'save_excel_results': False  # 不使用Excel保存，改用CSV
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
    
    # 预测天气特征（_pred后缀）
    forecast_features = [f + '_pred' for f in weather_features]
    
    config = {}
    
    if input_category == 'PV':
        config.update({
            'use_pv': True,
            'use_hist_weather': False,
            'use_forecast': False,
            'weather_features': [],
            'forecast_features': []
        })
    elif input_category == 'PV_plus_NWP':
        config.update({
            'use_pv': True,
            'use_hist_weather': False,
            'use_forecast': True,
            'weather_features': [],
            'forecast_features': forecast_features
        })
    elif input_category == 'PV_plus_NWP_plus':
        config.update({
            'use_pv': True,
            'use_hist_weather': False,
            'use_forecast': True,
            'weather_features': [],
            'forecast_features': weather_features  # 使用历史天气作为理想预测
        })
    elif input_category == 'PV_plus_HW':
        config.update({
            'use_pv': True,
            'use_hist_weather': True,
            'use_forecast': False,
            'weather_features': weather_features,
            'forecast_features': []
        })
    elif input_category == 'NWP':
        config.update({
            'use_pv': False,
            'use_hist_weather': False,
            'use_forecast': True,
            'weather_features': [],
            'forecast_features': forecast_features
        })
    elif input_category == 'NWP_plus':
        config.update({
            'use_pv': False,
            'use_hist_weather': False,
            'use_forecast': True,
            'weather_features': [],
            'forecast_features': weather_features  # 使用历史天气作为理想预测
        })
    
    # 回看窗口配置
    config['past_hours'] = lookback_hours
    config['past_days'] = lookback_hours // 24
    
    # 时间编码配置
    config['use_time_encoding'] = use_time_encoding
    
    return config

def create_model_config(model, complexity):
    """创建模型配置"""
    config = {'model': model}
    
    if model == 'LSR':
        config['model_complexity'] = 'baseline'
        config['epochs'] = 1
    elif model in ['RF', 'XGB', 'LGBM']:
        config['model_complexity'] = complexity
        config['epochs'] = 15 if complexity == 'low' else 50
        if complexity == 'low':
            config['train_params'].update({
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1
            })
        else:  # high
            config['train_params'].update({
                'n_estimators': 200,
                'max_depth': 12,
                'learning_rate': 0.01
            })
    else:  # 深度学习模型
        config['model_complexity'] = complexity
        config['epochs'] = 15 if complexity == 'low' else 50
        if complexity == 'low':
            config['train_params'].update({
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 6,
                'hidden_dim': 32,
                'dropout': 0.1
            })
        else:  # high
            config['train_params'].update({
                'd_model': 256,
                'num_heads': 16,
                'num_layers': 18,
                'hidden_dim': 128,
                'dropout': 0.3
            })
    
    return config

def generate_project_configs(project_id):
    """为单个Project生成所有配置"""
    configs = []
    
    # 实验参数
    input_categories = ['PV', 'PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW', 'NWP', 'NWP_plus']
    lookback_hours = [24, 72]
    time_encodings = [False, True]  # noTE, TE
    complexities = ['low', 'high']
    models = ['LSR', 'RF', 'XGB', 'LGBM', 'LSTM', 'GRU', 'TCN', 'Transformer']
    
    config_count = 0
    
    for input_cat in input_categories:
        for lookback in lookback_hours:
            for te in time_encodings:
                for complexity in complexities:
                    for model in models:
                        # 跳过LSR的复杂度设置
                        if model == 'LSR' and complexity != 'low':
                            continue
                        
                        config_count += 1
                        
                        # 生成配置
                        base_config = generate_base_config()
                        feature_config = create_feature_config(input_cat, lookback, te)
                        model_config = create_model_config(model, complexity)
                        
                        # 合并配置
                        config = {**base_config, **feature_config, **model_config}
                        
                        # 设置Project特定路径
                        config['data_path'] = f'./data/Project{project_id}.csv'
                        config['save_dir'] = f'./temp_results/Project{project_id}'
                        
                        # 生成配置名称
                        te_str = 'TE' if te else 'noTE'
                        config_name = f"{model}_{complexity}_{input_cat}_{lookback}h_{te_str}"
                        
                        configs.append({
                            'config': config,
                            'name': config_name,
                            'project_id': project_id,
                            'config_id': config_count
                        })
    
    return configs

def save_project_configs(project_id, configs):
    """保存单个Project的所有配置"""
    project_dir = Path(f"config/projects/Project{project_id}")
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个配置
    for config_info in configs:
        config_file = project_dir / f"{config_info['name']}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_info['config'], f, default_flow_style=False, allow_unicode=True)
    
    # 保存配置索引
    index_data = {
        'project_id': project_id,
        'total_configs': len(configs),
        'configs': [
            {
                'name': c['name'],
                'config_id': c['config_id'],
                'file': f"{c['name']}.yaml"
            }
            for c in configs
        ]
    }
    
    index_file = project_dir / "config_index.yaml"
    with open(index_file, 'w', encoding='utf-8') as f:
        yaml.dump(index_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Project{project_id}: 生成 {len(configs)} 个配置文件")
    return len(configs)

def main():
    """主函数"""
    print("🚀 开始为100个Project生成消融实验配置")
    print("=" * 60)
    
    # 创建配置目录
    config_dir = Path("config/projects")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    total_configs = 0
    successful_projects = 0
    
    # 生成100个Project的配置
    for project_id in range(1, 101):  # Project001 到 Project100
        try:
            print(f"📝 生成 Project{project_id:03d} 配置...")
            configs = generate_project_configs(project_id)
            count = save_project_configs(project_id, configs)
            total_configs += count
            successful_projects += 1
        except Exception as e:
            print(f"❌ Project{project_id:03d} 配置生成失败: {e}")
    
    # 生成全局索引
    global_index = {
        'total_projects': successful_projects,
        'total_configs': total_configs,
        'projects': []
    }
    
    for project_id in range(1, 101):
        project_dir = config_dir / f"Project{project_id:03d}"
        if project_dir.exists():
            global_index['projects'].append({
                'project_id': f"Project{project_id:03d}",
                'config_dir': str(project_dir),
                'config_count': 360
            })
    
    global_index_file = config_dir / "global_index.yaml"
    with open(global_index_file, 'w', encoding='utf-8') as f:
        yaml.dump(global_index, f, default_flow_style=False, allow_unicode=True)
    
    print("\n" + "=" * 60)
    print("🎉 配置生成完成!")
    print(f"✅ 成功生成 {successful_projects} 个Project的配置")
    print(f"📊 总配置数量: {total_configs}")
    print(f"📁 配置保存在: {config_dir}")
    print(f"📋 全局索引: {global_index_file}")

if __name__ == "__main__":
    main()
