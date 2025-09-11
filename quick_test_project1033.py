#!/usr/bin/env python3
"""
Project1033快速测试脚本
只测试几个关键实验组合，用于验证配置是否正确
"""

import os
import sys
import subprocess
import time

def quick_test_project1033():
    """运行Project1033的快速测试"""
    
    print("🚀 开始Project1033快速测试")
    print("=" * 60)
    
    # 检查数据文件
    data_file = "data/Project1033.csv"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    print(f"✅ 数据文件存在: {data_file}")
    
    # 测试配置
    test_configs = [
        {
            'model': 'Transformer',
            'use_pv': 'true',
            'use_hist_weather': 'false',
            'use_forecast': 'false',
            'weather_category': 'irradiance',
            'use_time_encoding': 'true',
            'past_days': '1',
            'model_complexity': 'low'
        },
        {
            'model': 'RF',
            'use_pv': 'true',
            'use_hist_weather': 'true',
            'use_forecast': 'false',
            'weather_category': 'all_weather',
            'use_time_encoding': 'false',
            'past_days': '3',
            'model_complexity': 'high'
        },
        {
            'model': 'Linear',
            'use_pv': 'false',
            'use_hist_weather': 'false',
            'use_forecast': 'true',
            'weather_category': 'irradiance',
            'use_time_encoding': 'true',
            'past_days': '0',
            'model_complexity': 'low'
        }
    ]
    
    print(f"📊 将测试 {len(test_configs)} 个实验组合")
    
    success_count = 0
    total_time = 0
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 测试 {i}/{len(test_configs)}: {config['model']} + {config['weather_category']} + {config['use_time_encoding']}")
        
        # 构建命令
        cmd = [
            sys.executable, 'main.py',
            '--config', 'config/project1033_test.yaml',
            '--model', config['model'],
            '--use_pv', config['use_pv'],
            '--use_hist_weather', config['use_hist_weather'],
            '--use_forecast', config['use_forecast'],
            '--weather_category', config['weather_category'],
            '--use_time_encoding', config['use_time_encoding'],
            '--data_path', data_file,
            '--plant_id', 'Project1033',
            '--save_dir', '/content/drive/MyDrive/Solar PV electricity/result_new/Project1033'
        ]
        
        # 添加past_days参数（如果不是0）
        if config['past_days'] != '0':
            cmd.extend(['--past_days', config['past_days']])
        
        # 添加model_complexity参数（如果不是Linear）
        if config['model'] != 'Linear':
            cmd.extend(['--model_complexity', config['model_complexity']])
        
        # 添加no_hist_power参数（如果use_pv为false）
        if config['use_pv'] == 'false':
            cmd.extend(['--no_hist_power', 'true'])
        
        print(f"   命令: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10分钟超时
            
            end_time = time.time()
            duration = end_time - start_time
            total_time += duration
            
            if result.returncode == 0:
                print(f"   ✅ 成功 (耗时: {duration:.1f}秒)")
                success_count += 1
            else:
                print(f"   ❌ 失败 (耗时: {duration:.1f}秒)")
                print(f"   错误: {result.stderr[:200]}...")
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ 超时 (10分钟)")
        except Exception as e:
            print(f"   ❌ 异常: {e}")
    
    print(f"\n📊 快速测试完成:")
    print(f"   成功: {success_count}/{len(test_configs)}")
    print(f"   总耗时: {total_time/60:.1f}分钟")
    print(f"   平均每实验: {total_time/len(test_configs):.1f}秒")
    
    return success_count == len(test_configs)

def main():
    """主函数"""
    print("🔬 Project1033快速测试工具")
    print("=" * 60)
    print("📊 将测试3个关键实验组合:")
    print("   1. Transformer + 仅历史PV + 时间编码 + 1天 + Low")
    print("   2. RF + 历史PV+历史天气 + 全部天气 + 无时间编码 + 3天 + High")
    print("   3. Linear + 仅预测天气 + 太阳辐射 + 时间编码 + 无历史数据")
    print("=" * 60)
    
    # 确认运行
    response = input("\n是否开始快速测试? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 测试已取消")
        return
    
    # 运行测试
    success = quick_test_project1033()
    
    if success:
        print("\n🎉 快速测试全部成功!")
        print("   配置正确，可以运行完整测试")
    else:
        print("\n❌ 快速测试有失败!")
        print("   请检查配置和错误信息")

if __name__ == "__main__":
    main()
