#!/usr/bin/env python3
"""
动态生成Project消融实验配置
自动检测data目录中的实际Project文件名
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
    
    # 输入特征配置
    if input_category == 'PV':
        config['use_pv'] = True
        config['use_hist_weather'] = False
        config['use_forecast'] = False
        config['weather_category'] = 'none'
    elif input_category == 'PV_plus_NWP':
        config['use_pv'] = True
        config['use_hist_weather'] = False
        config['use_forecast'] = True
        config['weather_category'] = 'forecast'
    elif input_category == 'PV_plus_NWP_plus':
        config['use_pv'] = True
        config['use_hist_weather'] = False
        config['use_forecast'] = True
        config['weather_category'] = 'forecast_ideal'
    elif input_category == 'PV_plus_HW':
        config['use_pv'] = True
        config['use_hist_weather'] = True
        config['use_forecast'] = False
        config['weather_category'] = 'historical'
    elif input_category == 'NWP':
        config['use_pv'] = False
        config['use_hist_weather'] = False
        config['use_forecast'] = True
        config['weather_category'] = 'forecast'
    elif input_category == 'NWP_plus':
        config['use_pv'] = False
        config['use_hist_weather'] = False
        config['use_forecast'] = True
        config['weather_category'] = 'forecast_ideal'
    
    # 回看窗口配置
    config['past_hours'] = lookback_hours
    config['past_days'] = lookback_hours // 24
    
    # 时间编码配置
    config['use_time_encoding'] = use_time_encoding
    
    return config

def create_model_config(model, complexity):
    """创建模型配置"""
    config = {'model': model}
    
    # 初始化model_params和train_params
    config['model_params'] = {}
    config['train_params'] = {}
    
    if model == 'LSR':
        config['model_complexity'] = 'baseline'
        config['epochs'] = 1
        config['model_params'] = {
            'ml_low': {
                'learning_rate': 0.001
            }
        }
    elif model in ['RF', 'XGB', 'LGBM']:
        config['model_complexity'] = complexity
        config['epochs'] = 15 if complexity == 'low' else 50
        if complexity == 'low':
            config['model_params'] = {
                'ml_low': {
                    'n_estimators': 50,
                    'max_depth': 5,
                    'learning_rate': 0.1
                }
            }
            config['train_params'].update({
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1
            })
        else:  # high
            config['model_params'] = {
                'ml_high': {
                    'n_estimators': 200,
                    'max_depth': 12,
                    'learning_rate': 0.01
                }
            }
            config['train_params'].update({
                'n_estimators': 200,
                'max_depth': 12,
                'learning_rate': 0.01
            })
    else:  # 深度学习模型
        config['model_complexity'] = complexity
        config['epochs'] = 15 if complexity == 'low' else 50
        if complexity == 'low':
            config['model_params'] = {
                'low': {
                    'd_model': 64,
                    'num_heads': 4,
                    'num_layers': 6,
                    'hidden_dim': 32,
                    'dropout': 0.1
                }
            }
            config['train_params'].update({
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 6,
                'hidden_dim': 32,
                'dropout': 0.1
            })
        else:  # high
            config['model_params'] = {
                'high': {
                    'd_model': 256,
                    'num_heads': 16,
                    'num_layers': 18,
                    'hidden_dim': 128,
                    'dropout': 0.3
                }
            }
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
                        
                        # 创建配置名称
                        te_suffix = 'TE' if te else 'noTE'
                        config_name = f"{model}_{complexity}_{input_cat}_{lookback}h_{te_suffix}"
                        
                        # 生成基础配置
                        base_config = generate_base_config()
                        
                        # 设置数据路径
                        base_config['data_path'] = f"data/Project{project_id}.csv"
                        base_config['save_dir'] = f"temp_results/{project_id}/{config_name}"
                        
                        # 添加特征配置
                        feature_config = create_feature_config(input_cat, lookback, te)
                        base_config.update(feature_config)
                        
                        # 添加模型配置
                        model_config = create_model_config(model, complexity)
                        base_config.update(model_config)
                        
                        # 保存配置信息
                        config_info = {
                            'name': config_name,
                            'config_id': config_count,
                            'config': base_config
                        }
                        configs.append(config_info)
    
    return configs

def save_project_configs(project_id, configs):
    """保存Project配置到文件"""
    project_dir = Path("config/projects") / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个配置文件
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
    
    print(f"✅ {project_id}: 生成 {len(configs)} 个配置文件")
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
    print("🚀 开始动态生成Project消融实验配置")
    print("=" * 60)
    
    # 检测实际的Project文件
    project_ids = detect_project_files()
    
    if not project_ids:
        print("❌ 未找到任何Project CSV文件")
        print("请确保data/目录下有Project*.csv文件")
        return
    
    print(f"📊 检测到 {len(project_ids)} 个Project文件:")
    for i, project_id in enumerate(project_ids[:10]):  # 显示前10个
        print(f"   {i+1}. Project{project_id}.csv")
    if len(project_ids) > 10:
        print(f"   ... 还有 {len(project_ids) - 10} 个文件")
    
    # 创建配置目录
    config_dir = Path("config/projects")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    total_configs = 0
    successful_projects = 0
    
    # 为每个检测到的Project生成配置
    for project_id in project_ids:
        try:
            print(f"📝 生成 Project{project_id} 配置...")
            configs = generate_project_configs(project_id)
            count = save_project_configs(project_id, configs)
            total_configs += count
            successful_projects += 1
        except Exception as e:
            print(f"❌ Project{project_id} 配置生成失败: {e}")
    
    # 生成全局索引
    global_index = {
        'total_projects': successful_projects,
        'total_configs': total_configs,
        'projects': []
    }
    
    for project_id in project_ids:
        project_dir = config_dir / project_id
        if project_dir.exists():
            global_index['projects'].append({
                'project_id': project_id,
                'config_dir': str(project_dir),
                'index_file': str(project_dir / "config_index.yaml")
            })
    
    # 保存全局索引
    global_index_file = config_dir / "global_index.yaml"
    with open(global_index_file, 'w', encoding='utf-8') as f:
        yaml.dump(global_index, f, default_flow_style=False, allow_unicode=True)
    
    print("=" * 60)
    print("🎉 配置生成完成!")
    print(f"✅ 成功生成 {successful_projects} 个Project的配置")
    print(f"📊 总配置数量: {total_configs}")
    print(f"📁 配置保存在: config/projects")
    print(f"📋 全局索引: config/projects/global_index.yaml")

if __name__ == "__main__":
    main()
