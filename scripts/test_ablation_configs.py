#!/usr/bin/env python3
"""
测试消融实验配置
运行几个示例配置来验证实验设计是否正确
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path

def test_config_loading():
    """测试配置文件加载"""
    config_dir = Path("config/ablation")
    
    if not config_dir.exists():
        print("❌ 配置文件目录不存在，请先运行 generate_ablation_configs.py")
        return False
    
    config_files = list(config_dir.glob("*.yaml"))
    config_files = [f for f in config_files if f.name != "config_index.yaml"]
    
    print(f"📁 找到 {len(config_files)} 个配置文件")
    
    # 测试几个不同的配置
    test_configs = [
        "LSR_baseline_PV_24h_noTE.yaml",
        "Transformer_high_PV_plus_NWP_72h_TE.yaml",
        "XGB_low_NWP_24h_noTE.yaml"
    ]
    
    for config_name in test_configs:
        config_file = config_dir / config_name
        if config_file.exists():
            print(f"\n✅ 测试配置: {config_name}")
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"   模型: {config.get('model', 'N/A')}")
            print(f"   复杂度: {config.get('model_complexity', 'N/A')}")
            print(f"   输入类别: {config.get('input_category', 'N/A')}")
            print(f"   回看窗口: {config.get('past_hours', 'N/A')} 小时")
            print(f"   时间编码: {config.get('use_time_encoding', 'N/A')}")
            print(f"   PV输入: {config.get('use_pv', 'N/A')}")
            print(f"   历史天气: {config.get('use_hist_weather', 'N/A')}")
            print(f"   预测天气: {config.get('use_forecast', 'N/A')}")
        else:
            print(f"❌ 配置文件不存在: {config_name}")
    
    return True

def test_data_compatibility():
    """测试数据兼容性"""
    data_file = "data/Project1140.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    print(f"\n📊 测试数据文件: {data_file}")
    
    try:
        # 读取数据头部
        df = pd.read_csv(data_file, nrows=100)
        print(f"   数据形状: {df.shape}")
        print(f"   列数: {len(df.columns)}")
        
        # 检查关键列
        required_cols = ['Capacity Factor', 'DateTime']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ 缺少必需列: {missing_cols}")
            return False
        
        # 检查天气特征 (与_pred后缀对应的历史天气特征)
        weather_features = [
            'global_tilted_irradiance', 'vapour_pressure_deficit', 'relative_humidity_2m',
            'temperature_2m', 'wind_gusts_10m', 'cloud_cover_low', 'wind_speed_100m',
            'snow_depth', 'dew_point_2m', 'precipitation', 'surface_pressure'
        ]
        
        available_hw = [col for col in weather_features if col in df.columns]
        available_nwp = [col for col in weather_features if f"{col}_pred" in df.columns]
        
        print(f"   历史天气特征: {len(available_hw)}/{len(weather_features)}")
        print(f"   预测天气特征: {len(available_nwp)}/{len(weather_features)}")
        
        if len(available_hw) < len(weather_features):
            missing_hw = [col for col in weather_features if col not in df.columns]
            print(f"   缺少历史天气特征: {missing_hw}")
        
        if len(available_nwp) < len(weather_features):
            missing_nwp = [col for col in weather_features if f"{col}_pred" not in df.columns]
            print(f"   缺少预测天气特征: {missing_nwp}")
        
        print("✅ 数据兼容性检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据读取错误: {e}")
        return False

def test_experiment_matrix():
    """测试实验矩阵完整性"""
    config_dir = Path("config/ablation")
    
    if not config_dir.exists():
        print("❌ 配置文件目录不存在")
        return False
    
    config_files = list(config_dir.glob("*.yaml"))
    config_files = [f for f in config_files if f.name != "config_index.yaml"]
    
    print(f"\n🧮 测试实验矩阵完整性")
    print(f"   总配置文件数: {len(config_files)}")
    
    # 分析配置类型
    input_categories = set()
    models = set()
    complexities = set()
    lookback_hours = set()
    time_encoding = set()
    
    for config_file in config_files:
        config_name = config_file.name
        
        # 解析配置名称
        parts = config_name.replace('.yaml', '').split('_')
        
        if len(parts) >= 4:
            model = parts[0]
            complexity = parts[1]
            
            models.add(model)
            complexities.add(complexity)
            
            # 提取其他信息
            if '24h' in config_name:
                lookback_hours.add(24)
            if '72h' in config_name:
                lookback_hours.add(72)
            
            if 'TE' in config_name:
                time_encoding.add(True)
            if 'noTE' in config_name:
                time_encoding.add(False)
            
            # 提取输入类别
            if 'PV_plus_NWP_plus' in config_name:
                input_categories.add('PV+NWP+')
            elif 'PV_plus_NWP' in config_name:
                input_categories.add('PV+NWP')
            elif 'PV_plus_HW' in config_name:
                input_categories.add('PV+HW')
            elif 'NWP_plus' in config_name and 'PV' not in config_name:
                input_categories.add('NWP+')
            elif 'NWP' in config_name and 'PV' not in config_name:
                input_categories.add('NWP')
            elif 'PV' in config_name and 'plus' not in config_name:
                input_categories.add('PV')
    
    print(f"   模型类型: {sorted(models)}")
    print(f"   复杂度: {sorted(complexities)}")
    print(f"   输入类别: {sorted(input_categories)}")
    print(f"   回看窗口: {sorted(lookback_hours)} 小时")
    print(f"   时间编码: {sorted(time_encoding)}")
    
    # 计算期望配置数
    # 主要模型配置 (除了LSR)
    main_models = [m for m in models if m != 'LSR']
    expected_configs = len(input_categories) * len(lookback_hours) * len(time_encoding) * len([c for c in complexities if c != 'baseline']) * len(main_models)
    
    # LSR基线配置 (不区分复杂度)
    expected_configs += len(input_categories) * len(lookback_hours) * len(time_encoding)
    
    print(f"   期望配置数: {expected_configs}")
    print(f"   实际配置数: {len(config_files)}")
    
    if len(config_files) == expected_configs:
        print("✅ 实验矩阵完整性检查通过")
        return True
    else:
        print("❌ 实验矩阵不完整")
        return False

def main():
    """主测试函数"""
    print("🧪 开始测试消融实验配置...")
    
    tests = [
        ("配置文件加载", test_config_loading),
        ("数据兼容性", test_data_compatibility),
        ("实验矩阵完整性", test_experiment_matrix)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print(f"\n{'='*50}")
    print("测试总结")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n通过率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！可以开始运行消融实验。")
        print("\n运行命令示例:")
        print("python run_ablation_experiments.py --max-configs 5 --dry-run  # 测试运行5个配置")
        print("python run_ablation_experiments.py --model-filter LSR,Transformer  # 只运行LSR和Transformer")
    else:
        print("\n⚠️  部分测试失败，请检查配置后再运行实验。")

if __name__ == "__main__":
    main()
