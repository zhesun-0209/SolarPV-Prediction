#!/usr/bin/env python3
"""
测试敏感性分析配置生成
验证配置生成是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from generate_sensitivity_configs import generate_sensitivity_configs, save_sensitivity_configs

def test_config_generation():
    """测试配置生成"""
    print("🧪 测试敏感性分析配置生成")
    print("=" * 50)
    
    # 测试单个项目的配置生成
    test_project_id = "171"
    
    try:
        print(f"📝 生成 Project{test_project_id} 的敏感性分析配置...")
        configs = generate_sensitivity_configs(test_project_id)
        
        print(f"✅ 成功生成 {len(configs)} 个配置")
        
        # 显示前5个配置的详细信息
        print("\n📋 前5个配置示例:")
        for i, config_info in enumerate(configs[:5]):
            print(f"\n配置 {i+1}: {config_info['name']}")
            print(f"  模型: {config_info['model']}")
            print(f"  天气级别: {config_info['weather_level']}")
            print(f"  回看小时: {config_info['lookback_hours']}")
            print(f"  复杂度级别: {config_info['complexity_level']}")
            print(f"  数据集规模: {config_info['dataset_scale']}")
        
        # 统计各类型的配置数量
        print(f"\n📊 配置统计:")
        weather_counts = {}
        lookback_counts = {}
        complexity_counts = {}
        dataset_counts = {}
        model_counts = {}
        
        for config in configs:
            weather = config['weather_level']
            lookback = config['lookback_hours']
            complexity = config['complexity_level']
            dataset = config['dataset_scale']
            model = config['model']
            
            weather_counts[weather] = weather_counts.get(weather, 0) + 1
            lookback_counts[lookback] = lookback_counts.get(lookback, 0) + 1
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            model_counts[model] = model_counts.get(model, 0) + 1
        
        print(f"天气级别分布: {weather_counts}")
        print(f"回看小时分布: {lookback_counts}")
        print(f"复杂度级别分布: {complexity_counts}")
        print(f"数据集规模分布: {dataset_counts}")
        print(f"模型分布: {model_counts}")
        
        # 验证LSR模型的配置
        lsr_configs = [c for c in configs if c['model'] == 'LSR']
        print(f"\n🔍 LSR模型配置验证:")
        print(f"LSR配置数量: {len(lsr_configs)}")
        if lsr_configs:
            lsr_config = lsr_configs[0]
            print(f"LSR配置示例: {lsr_config['name']}")
            print(f"  天气级别: {lsr_config['weather_level']} (应为L)")
            print(f"  回看小时: {lsr_config['lookback_hours']} (应为24)")
            print(f"  复杂度级别: {lsr_config['complexity_level']} (应为2)")
            print(f"  数据集规模: {lsr_config['dataset_scale']}")
        
        # 验证非LSR模型的配置
        non_lsr_configs = [c for c in configs if c['model'] != 'LSR']
        print(f"\n🔍 非LSR模型配置验证:")
        print(f"非LSR配置数量: {len(non_lsr_configs)}")
        print(f"预期数量: 4 × 4 × 4 × 4 × 7 = 1,792")
        print(f"实际数量: {len(non_lsr_configs)}")
        
        # 验证总配置数量
        # Weather adoption: 4 × 8 = 32
        # Lookback length: 4 × 7 = 28 (排除LSR)
        # Model complexity: 4 × 7 = 28 (排除LSR)
        # Dataset scale: 4 × 8 = 32
        total_expected = 32 + 28 + 28 + 32  # 120
        print(f"\n📊 总配置数量验证:")
        print(f"预期总数: {total_expected}")
        print(f"实际总数: {len(configs)}")
        print(f"验证结果: {'✅ 通过' if len(configs) == total_expected else '❌ 失败'}")
        
        # 验证各实验类型的配置数量
        experiment_types = {}
        for config in configs:
            exp_type = config['experiment_type']
            experiment_types[exp_type] = experiment_types.get(exp_type, 0) + 1
        
        print(f"\n📊 各实验类型配置数量:")
        print(f"Weather adoption: {experiment_types.get('weather_adoption', 0)} (预期32)")
        print(f"Lookback length: {experiment_types.get('lookback_length', 0)} (预期28)")
        print(f"Model complexity: {experiment_types.get('model_complexity', 0)} (预期28)")
        print(f"Dataset scale: {experiment_types.get('dataset_scale', 0)} (预期32)")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置生成测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def test_config_saving():
    """测试配置保存"""
    print("\n🧪 测试配置保存")
    print("=" * 50)
    
    test_project_id = "171"
    
    try:
        # 生成配置
        configs = generate_sensitivity_configs(test_project_id)
        
        # 保存配置
        count = save_sensitivity_configs(test_project_id, configs)
        
        print(f"✅ 成功保存 {count} 个配置")
        
        # 检查保存的文件
        config_dir = f"sensitivity_analysis/configs/{test_project_id}"
        if os.path.exists(config_dir):
            yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
            print(f"📁 保存的YAML文件数量: {len(yaml_files)}")
            print(f"📁 配置文件目录: {config_dir}")
            
            # 检查索引文件
            index_file = f"{config_dir}/sensitivity_index.yaml"
            if os.path.exists(index_file):
                print(f"✅ 索引文件已创建: {index_file}")
            else:
                print(f"❌ 索引文件未创建: {index_file}")
        else:
            print(f"❌ 配置目录未创建: {config_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置保存测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def main():
    """主函数"""
    print("🚀 敏感性分析配置测试")
    print("=" * 60)
    
    # 测试配置生成
    config_gen_success = test_config_generation()
    
    # 测试配置保存
    config_save_success = test_config_saving()
    
    # 总结
    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print(f"配置生成: {'✅ 通过' if config_gen_success else '❌ 失败'}")
    print(f"配置保存: {'✅ 通过' if config_save_success else '❌ 失败'}")
    
    if config_gen_success and config_save_success:
        print("🎊 所有测试通过！敏感性分析配置系统工作正常。")
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
