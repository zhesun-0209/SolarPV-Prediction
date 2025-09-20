#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为多个Plant生成配置文件
支持Plant 171、172、186
"""

import os
import yaml
from pathlib import Path

def get_scenario_configs():
    """获取所有场景配置"""
    scenarios = {
        'PV': {
            'use_pv': True,
            'use_forecast': False,
            'use_hist_weather': False,
            'use_ideal_nwp': False,
            'weather_category': 'none'
        },
        'PV_plus_NWP': {
            'use_pv': True,
            'use_forecast': True,
            'use_hist_weather': False,
            'use_ideal_nwp': False,
            'weather_category': 'nwp'
        },
        'PV_plus_NWP_plus': {
            'use_pv': True,
            'use_forecast': True,
            'use_hist_weather': False,
            'use_ideal_nwp': True,
            'weather_category': 'nwp_plus'
        },
        'PV_plus_HW': {
            'use_pv': True,
            'use_forecast': False,
            'use_hist_weather': True,
            'use_ideal_nwp': False,
            'weather_category': 'hist_weather'
        },
        'NWP': {
            'use_pv': False,
            'use_forecast': True,
            'use_hist_weather': False,
            'use_ideal_nwp': False,
            'weather_category': 'nwp'
        },
        'NWP_plus': {
            'use_pv': False,
            'use_forecast': True,
            'use_hist_weather': False,
            'use_ideal_nwp': True,
            'weather_category': 'nwp_plus'
        }
    }
    return scenarios

def get_model_configs():
    """获取模型配置"""
    models = {
        'LSTM': {
            'model_params': {
                'high': {
                    'd_model': 256,
                    'dropout': 0.3,
                    'hidden_dim': 128,
                    'num_heads': 16,
                    'num_layers': 18
                },
                'low': {
                    'd_model': 64,
                    'dropout': 0.1,
                    'hidden_dim': 32,
                    'num_heads': 4,
                    'num_layers': 6
                }
            }
        },
        'GRU': {
            'model_params': {
                'high': {
                    'd_model': 256,
                    'dropout': 0.3,
                    'hidden_dim': 128,
                    'num_heads': 16,
                    'num_layers': 18
                },
                'low': {
                    'd_model': 64,
                    'dropout': 0.1,
                    'hidden_dim': 32,
                    'num_heads': 4,
                    'num_layers': 6
                }
            }
        },
        'TCN': {
            'model_params': {
                'high': {
                    'd_model': 256,
                    'dropout': 0.3,
                    'hidden_dim': 128,
                    'num_heads': 16,
                    'num_layers': 18
                },
                'low': {
                    'd_model': 64,
                    'dropout': 0.1,
                    'hidden_dim': 32,
                    'num_heads': 4,
                    'num_layers': 6
                }
            }
        },
        'Transformer': {
            'model_params': {
                'high': {
                    'd_model': 256,
                    'dropout': 0.3,
                    'hidden_dim': 128,
                    'num_heads': 16,
                    'num_layers': 18
                },
                'low': {
                    'd_model': 64,
                    'dropout': 0.1,
                    'hidden_dim': 32,
                    'num_heads': 4,
                    'num_layers': 6
                }
            }
        },
        'RF': {
            'model_params': {
                'high': {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                },
                'low': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            }
        },
        'XGB': {
            'model_params': {
                'high': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'low': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'random_state': 42
                }
            }
        },
        'LGBM': {
            'model_params': {
                'high': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'low': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'random_state': 42
                }
            }
        },
        'Linear': {
            'model_params': {
                'high': {
                    'fit_intercept': True,
                    'normalize': False
                },
                'low': {
                    'fit_intercept': True,
                    'normalize': False
                }
            }
        }
    }
    return models

def generate_configs_for_plant(plant_id):
    """为指定Plant生成所有配置文件"""
    print(f"🚀 为Plant {plant_id}生成配置文件...")
    
    # 创建配置目录
    config_dir = f"config/projects/{plant_id}"
    os.makedirs(config_dir, exist_ok=True)
    
    # 获取配置
    models = get_model_configs()
    scenarios = get_scenario_configs()
    lookbacks = [24, 72]  # 24小时和72小时
    te_options = [True, False]  # 时间编码
    complexities = ['low', 'high']  # 复杂度
    
    config_count = 0
    
    for model_name, model_config in models.items():
        for scenario_name, scenario_config in scenarios.items():
            for lookback in lookbacks:
                for te in te_options:
                    for complexity in complexities:
                        # 生成配置文件名
                        te_str = 'TE' if te else 'noTE'
                        config_filename = f"{model_name}_{complexity}_{scenario_name}_{lookback}h_{te_str}.yaml"
                        config_path = os.path.join(config_dir, config_filename)
                        
                        # 生成配置内容
                        config = {
                            'data_path': f'data/Project{plant_id}.csv',
                            'end_date': '2024-09-28',
                            'epoch_params': {
                                'high': 80,
                                'low': 50
                            },
                            'epochs': 50,
                            'future_hours': 24,
                            'model': model_name,
                            'model_complexity': complexity,
                            'model_params': model_config['model_params'],
                            'past_days': 3,
                            'past_hours': lookback,
                            'plot_days': 7,
                            'save_dir': f'temp_results/{plant_id}/{model_name}_{complexity}_{scenario_name}_{lookback}h_{te_str}',
                            'save_options': {
                                'save_excel_results': False,
                                'save_model': False,
                                'save_predictions': False,
                                'save_summary': False,
                                'save_training_log': False
                            },
                            'start_date': '2022-01-01',
                            'train_params': {
                                'batch_size': 64,
                                'learning_rate': 0.001
                            },
                            'train_ratio': 0.8,
                            'use_forecast': scenario_config['use_forecast'],
                            'use_hist_weather': scenario_config['use_hist_weather'],
                            'use_ideal_nwp': scenario_config['use_ideal_nwp'],
                            'use_pv': scenario_config['use_pv'],
                            'use_time_encoding': te,
                            'val_ratio': 0.1,
                            'weather_category': scenario_config['weather_category']
                        }
                        
                        # 保存配置文件
                        with open(config_path, 'w', encoding='utf-8') as f:
                            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                        
                        config_count += 1
    
    print(f"✅ Plant {plant_id} 配置文件生成完成，共 {config_count} 个文件")
    return config_count

def main():
    """主函数"""
    print("🚀 为多个Plant生成配置文件...")
    
    # 要生成的Plant列表
    plant_ids = [171, 172, 186]
    
    total_configs = 0
    for plant_id in plant_ids:
        config_count = generate_configs_for_plant(plant_id)
        total_configs += config_count
    
    print(f"\n🎉 所有Plant配置文件生成完成！")
    print(f"📊 总计生成 {total_configs} 个配置文件")
    print(f"📁 每个Plant包含 {total_configs // len(plant_ids)} 个配置文件")

if __name__ == "__main__":
    main()
