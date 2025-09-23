#!/usr/bin/env python3
"""
Season and Hour Analysis实验配置生成器
为100个厂生成8个实验的配置：Linear/LSR进行24 hour look back, noTE, 80%dataset, NWP实验，
其他7个模型进行24 hours look back, low complexity, no TE, 80%dataset, PV+NWP的设置
"""

import os
import yaml
from pathlib import Path
import re

# 全局变量存储数据目录路径
DATA_DIR = None

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
            'level1': 30,
            'level2': 50,
            'level3': 65,
            'level4': 80
        },
        'save_options': {
            'save_model': False,
            'save_summary': True,
            'save_predictions': True,
            'save_training_log': False,
            'save_excel_results': False
        }
    }

def create_linear_config():
    """创建Linear/LSR配置 - 24 hour look back, noTE, 80%dataset, NWP实验"""
    config = {
        'model': 'Linear',
        'model_complexity': 'level2',
        'epochs': 1,
        'past_hours': 24,
        'past_days': 1,
        'use_pv': True,
        'use_hist_weather': False,
        'use_forecast': True,
        'use_ideal_nwp': False,
        'use_time_encoding': False,
        'weather_category': 'nwp_only',
        'selected_weather_features': [
            'global_tilted_irradiance_pred', 'vapour_pressure_deficit_pred', 'relative_humidity_2m_pred',
            'temperature_2m_pred', 'wind_gusts_10m_pred', 'cloud_cover_low_pred', 'wind_speed_100m_pred',
            'snow_depth_pred', 'dew_point_2m_pred', 'surface_pressure_pred', 'precipitation_pred'
        ],
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'model_params': {
            'ml_level1': {'learning_rate': 0.001},
            'ml_level2': {'learning_rate': 0.001},
            'ml_level3': {'learning_rate': 0.001},
            'ml_level4': {'learning_rate': 0.001}
        }
    }
    return config

def create_other_model_config(model_name):
    """创建其他7个模型的配置 - 24 hours look back, low complexity, no TE, 80%dataset, PV+NWP"""
    config = {
        'model': model_name,
        'model_complexity': 'level2',  # low complexity
        'past_hours': 24,
        'past_days': 1,
        'use_pv': True,
        'use_hist_weather': False,
        'use_forecast': True,
        'use_ideal_nwp': False,
        'use_time_encoding': False,
        'weather_category': 'pv_nwp',
        'selected_weather_features': [
            'global_tilted_irradiance_pred', 'vapour_pressure_deficit_pred', 'relative_humidity_2m_pred',
            'temperature_2m_pred', 'wind_gusts_10m_pred', 'cloud_cover_low_pred', 'wind_speed_100m_pred',
            'snow_depth_pred', 'dew_point_2m_pred', 'surface_pressure_pred', 'precipitation_pred'
        ],
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1
    }
    
    # 设置epochs
    config['epochs'] = 50  # level2对应50个epoch
    
    # 根据模型类型设置参数
    if model_name in ['RF', 'XGB', 'LGBM']:
        config['model_params'] = {
            'ml_level1': {'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.2},
            'ml_level2': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1},
            'ml_level3': {'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.05},
            'ml_level4': {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.01}
        }
        config['train_params'] = {
            'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1
        }
    elif model_name == 'TCN':
        config['model_params'] = {
            'level1': {'tcn_channels': [16, 32], 'kernel_size': 2, 'dropout': 0.05},
            'level2': {'tcn_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.1},
            'level3': {'tcn_channels': [48, 96], 'kernel_size': 4, 'dropout': 0.2},
            'level4': {'tcn_channels': [64, 128, 256], 'kernel_size': 5, 'dropout': 0.3}
        }
        config['train_params'] = {
            'batch_size': 64,
            'learning_rate': 0.001
        }
    else:  # LSTM, GRU, Transformer
        config['model_params'] = {
            'level1': {'d_model': 32, 'num_heads': 2, 'num_layers': 3, 'hidden_dim': 16, 'dropout': 0.05},
            'level2': {'d_model': 64, 'num_heads': 4, 'num_layers': 6, 'hidden_dim': 32, 'dropout': 0.1},
            'level3': {'d_model': 128, 'num_heads': 8, 'num_layers': 12, 'hidden_dim': 64, 'dropout': 0.2},
            'level4': {'d_model': 256, 'num_heads': 16, 'num_layers': 18, 'hidden_dim': 128, 'dropout': 0.3}
        }
        config['train_params'] = {
            'batch_size': 64,
            'learning_rate': 0.001
        }
    
    return config

def generate_season_hour_configs(project_id):
    """为单个Project生成所有season and hour analysis配置"""
    configs = []
    
    # 8个实验配置
    experiments = [
        ('Linear', create_linear_config),
        ('RF', create_other_model_config),
        ('XGB', create_other_model_config),
        ('LGBM', create_other_model_config),
        ('LSTM', create_other_model_config),
        ('GRU', create_other_model_config),
        ('TCN', create_other_model_config),
        ('Transformer', create_other_model_config)
    ]
    
    config_count = 0
    
    for model_name, config_func in experiments:
        config_count += 1
        
        # 创建配置名称
        config_name = f"season_hour_{model_name.lower()}"
        
        # 生成基础配置
        base_config = generate_base_config()
        # 使用检测到的数据目录路径
        if DATA_DIR:
            base_config['data_path'] = f"{DATA_DIR}/Project{project_id}.csv"
        else:
            base_config['data_path'] = f"data/Project{project_id}.csv"
        base_config['save_dir'] = f"season_and_hour_analysis/results/{project_id}/{config_name}"
        
        # 添加模型特定配置
        if model_name == 'Linear':
            model_config = create_linear_config()
        else:
            model_config = create_other_model_config(model_name)
        
        base_config.update(model_config)
        
        # 添加实验类型标识
        base_config['experiment_type'] = 'season_hour_analysis'
        
        # 保存配置信息
        config_info = {
            'name': config_name,
            'config_id': config_count,
            'config': base_config,
            'experiment_type': 'season_hour_analysis',
            'model': model_name,
            'lookback_hours': 24,
            'complexity_level': 2,
            'dataset_scale': '80%',
            'weather_setup': 'nwp_only' if model_name == 'Linear' else 'pv_nwp'
        }
        configs.append(config_info)
    
    return configs

def save_season_hour_configs(project_id, configs):
    """保存season and hour analysis配置到文件"""
    project_dir = Path("season_and_hour_analysis/configs") / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个配置文件
    for config_info in configs:
        config_file = project_dir / f"{config_info['name']}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_info['config'], f, default_flow_style=False, allow_unicode=True)
    
    # 保存配置索引
    index_data = {
        'project_id': project_id,
        'experiment_type': 'season_hour_analysis',
        'total_configs': len(configs),
        'configs': [
            {
                'name': c['name'],
                'config_id': c['config_id'],
                'file': f"{c['name']}.yaml",
                'model': c['model'],
                'lookback_hours': c['lookback_hours'],
                'complexity_level': c['complexity_level'],
                'dataset_scale': c['dataset_scale'],
                'weather_setup': c['weather_setup']
            }
            for c in configs
        ]
    }
    
    index_file = project_dir / "season_hour_index.yaml"
    with open(index_file, 'w', encoding='utf-8') as f:
        yaml.dump(index_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ {project_id}: 生成 {len(configs)} 个season and hour analysis配置文件")
    return len(configs)

def detect_project_files():
    """检测data目录中的Project文件"""
    global DATA_DIR
    
    # 尝试多个可能的data目录路径
    possible_data_dirs = [
        "data",  # 本地路径
        "/content/SolarPV-Prediction/data",  # Colab路径
        "/content/drive/MyDrive/Solar PV electricity/data",  # Google Drive路径
    ]
    
    data_dir = None
    for dir_path in possible_data_dirs:
        if Path(dir_path).exists():
            data_dir = Path(dir_path)
            DATA_DIR = str(data_dir)  # 设置全局变量
            print(f"📁 找到数据目录: {dir_path}")
            break
    
    if data_dir is None:
        print("❌ 未找到数据目录，尝试的路径:")
        for dir_path in possible_data_dirs:
            print(f"   - {dir_path}")
        return []
    
    # 查找所有Project*.csv文件
    csv_files = list(data_dir.glob("Project*.csv"))
    print(f"📊 在 {data_dir} 中找到 {len(csv_files)} 个Project文件")
    
    # 提取Project ID
    project_ids = []
    for csv_file in csv_files:
        # 使用正则表达式提取Project ID
        match = re.match(r'Project(\d+)\.csv', csv_file.name)
        if match:
            project_id = match.group(1)
            project_ids.append(project_id)
    
    # 按Project ID排序（数字排序）
    project_ids = sorted(project_ids, key=int)
    print(f"📋 检测到的Project ID: {project_ids[:10]}{'...' if len(project_ids) > 10 else ''}")
    
    return project_ids

def main():
    """主函数"""
    print("🚀 开始生成Season and Hour Analysis实验配置")
    print("=" * 60)
    
    # 检测实际的Project文件
    project_ids = detect_project_files()
    
    if not project_ids:
        print("❌ 未找到任何Project CSV文件")
        print("请确保data/目录下有Project*.csv文件")
        return
    
    print(f"📊 检测到 {len(project_ids)} 个Project文件")
    
    # 处理前100个厂
    target_projects = project_ids[:100]
    print(f"🎯 将处理前100个厂: {target_projects[:10]}{'...' if len(target_projects) > 10 else ''}")
    
    # 创建配置目录
    config_dir = Path("season_and_hour_analysis/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    total_configs = 0
    successful_projects = 0
    
    # 为每个目标Project生成配置
    for project_id in target_projects:
        try:
            print(f"📝 生成 Project{project_id} season and hour analysis配置...")
            configs = generate_season_hour_configs(project_id)
            count = save_season_hour_configs(project_id, configs)
            total_configs += count
            successful_projects += 1
        except Exception as e:
            print(f"❌ Project{project_id} 配置生成失败: {e}")
    
    # 生成全局索引
    global_index = {
        'experiment_type': 'season_hour_analysis',
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
                'index_file': str(project_dir / "season_hour_index.yaml")
            })
    
    # 保存全局索引
    global_index_file = config_dir / "season_hour_global_index.yaml"
    with open(global_index_file, 'w', encoding='utf-8') as f:
        yaml.dump(global_index, f, default_flow_style=False, allow_unicode=True)
    
    print("=" * 60)
    print("🎉 Season and Hour Analysis配置生成完成!")
    print(f"✅ 成功生成 {successful_projects} 个Project的配置")
    print(f"📊 总配置数量: {total_configs}")
    print(f"📁 配置保存在: season_and_hour_analysis/configs")
    print(f"📋 全局索引: season_and_hour_analysis/configs/season_hour_global_index.yaml")

if __name__ == "__main__":
    main()
