#!/usr/bin/env python3
"""
Colab专用配置生成脚本
确保生成正确的配置文件
"""

import os
import yaml
import glob

def create_model_config(model, complexity):
    """创建模型配置"""
    config = {'model': model}
    
    # 初始化model_params和train_params
    config['model_params'] = {}
    config['train_params'] = {}
    
    if model == 'LSR':
        config['model'] = 'Linear'
        config['model_complexity'] = 'low'
        config['epochs'] = 1
        config['model_params'] = {
            'ml_low': {'learning_rate': 0.001},
            'ml_high': {'learning_rate': 0.001}
        }
        config['train_params'] = {
            'batch_size': 32,
            'learning_rate': 0.001
        }
    elif model in ['RF', 'XGB', 'LGBM']:
        config['model_complexity'] = complexity
        config['epochs'] = 15 if complexity == 'low' else 50
        if complexity == 'low':
            config['model_params'] = {
                'ml_low': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1},
                'ml_high': {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.01}
            }
            config['train_params'] = {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1
            }
        else:  # high
            config['model_params'] = {
                'ml_low': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1},
                'ml_high': {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.01}
            }
            config['train_params'] = {
                'n_estimators': 200,
                'max_depth': 12,
                'learning_rate': 0.01
            }
    else:  # 深度学习模型
        config['model_complexity'] = complexity
        config['epochs'] = 15 if complexity == 'low' else 50
        if complexity == 'low':
            if model == 'TCN':
                config['model_params'] = {
                    'low': {'tcn_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.1},
                    'high': {'tcn_channels': [64, 128, 256], 'kernel_size': 5, 'dropout': 0.3}
                }
            else:
                config['model_params'] = {
                    'low': {'d_model': 64, 'num_heads': 4, 'num_layers': 6, 'hidden_dim': 32, 'dropout': 0.1},
                    'high': {'d_model': 256, 'num_heads': 16, 'num_layers': 18, 'hidden_dim': 128, 'dropout': 0.3}
                }
            config['train_params'] = {
                'batch_size': 32,
                'learning_rate': 0.001
            }
        else:  # high
            if model == 'TCN':
                config['model_params'] = {
                    'low': {'tcn_channels': [32, 64], 'kernel_size': 3, 'dropout': 0.1},
                    'high': {'tcn_channels': [64, 128, 256], 'kernel_size': 5, 'dropout': 0.3}
                }
            else:
                config['model_params'] = {
                    'low': {'d_model': 64, 'num_heads': 4, 'num_layers': 6, 'hidden_dim': 32, 'dropout': 0.1},
                    'high': {'d_model': 256, 'num_heads': 16, 'num_layers': 18, 'hidden_dim': 128, 'dropout': 0.3}
                }
            config['train_params'] = {
                'batch_size': 32,
                'learning_rate': 0.001
            }
    
    return config

def generate_project_configs(project_id):
    """为单个Project生成所有配置"""
    configs = []
    
    # 实验参数
    input_categories = ['PV', 'PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW', 'NWP', 'NWP_plus']
    lookback_hours = [24, 72]
    time_encodings = [False, True]  # noTE, TE
    models = ['GRU', 'LSTM', 'Transformer', 'TCN', 'RF', 'XGB', 'LGBM', 'LSR']
    complexities = ['low', 'high']
    
    for model in models:
        for complexity in complexities:
            for input_category in input_categories:
                for lookback_hour in lookback_hours:
                    for time_encoding in time_encodings:
                        # 创建基础配置
                        config = create_model_config(model, complexity)
                        
                        # 添加实验参数
                        config.update({
                            'data_path': f'data/Project{project_id}.csv',
                            'save_dir': f'temp_results/{project_id}',
                            'past_days': 1,
                            'past_hours': lookback_hour,
                            'future_hours': 24,
                            'train_ratio': 0.8,
                            'val_ratio': 0.1,
                            'start_date': '2022-01-01',
                            'end_date': '2024-09-28',
                            'plot_days': 7,
                            'use_pv': input_category in ['PV', 'PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW'],
                            'use_hist_weather': input_category in ['PV_plus_HW'],
                            'use_forecast': input_category in ['PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW', 'NWP', 'NWP_plus'],
                            'use_time_encoding': time_encoding,
                            'weather_category': 'all_weather',
                            'save_options': {
                                'save_excel_results': True,
                                'save_model': False,
                                'save_predictions': False,
                                'save_summary': False,
                                'save_training_log': False
                            }
                        })
                        
                        # 生成文件名
                        time_str = "TE" if time_encoding else "noTE"
                        filename = f"{model}_{complexity}_{input_category}_{lookback_hour}h_{time_str}.yaml"
                        
                        configs.append((filename, config))
    
    return configs

def main():
    """主函数"""
    print("🔧 生成所有项目的配置文件...")
    
    # 获取所有项目
    data_files = glob.glob("data/Project*.csv")
    projects = []
    for data_file in data_files:
        project_id = os.path.basename(data_file).replace("Project", "").replace(".csv", "")
        projects.append(project_id)
    
    print(f"📊 找到 {len(projects)} 个项目: {projects}")
    
    # 为每个项目生成配置
    total_configs = 0
    for project_id in projects:
        print(f"📁 生成项目 {project_id} 的配置...")
        
        # 创建项目目录
        project_dir = f"config/projects/{project_id}"
        os.makedirs(project_dir, exist_ok=True)
        
        # 生成配置
        configs = generate_project_configs(project_id)
        
        # 保存配置
        for filename, config in configs:
            config_path = os.path.join(project_dir, filename)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"  ✅ 生成了 {len(configs)} 个配置文件")
        total_configs += len(configs)
    
    print(f"🎉 总共生成了 {total_configs} 个配置文件")
    print("✅ 配置生成完成!")

if __name__ == "__main__":
    main()