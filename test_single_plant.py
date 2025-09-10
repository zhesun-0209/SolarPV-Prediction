#!/usr/bin/env python3
"""
测试单个厂的实验
用于验证配置和调试
"""

import os
import sys
import subprocess

def test_single_plant(plant_id, data_file):
    """测试单个厂的实验"""
    
    print(f"🧪 测试厂: {plant_id}")
    print(f"   数据文件: {data_file}")
    print("=" * 60)
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    # 运行单个实验
    cmd = [
        'python', 'main.py',
        '--config', 'config/default.yaml',
        '--model', 'RF',
        '--use_hist_weather', 'false',
        '--use_forecast', 'false',
        '--model_complexity', 'medium',
        '--past_days', '1',
        '--data_path', data_file
    ]
    
    print(f"🚀 运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 测试成功")
            print("输出:")
            print(result.stdout)
            return True
        else:
            print("❌ 测试失败")
            print("错误:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 测试超时（5分钟）")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试单个厂的实验')
    parser.add_argument('plant_id', help='厂ID')
    parser.add_argument('data_file', help='数据文件路径')
    
    args = parser.parse_args()
    
    test_single_plant(args.plant_id, args.data_file)
