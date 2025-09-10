#!/usr/bin/env python3
"""
测试单个实验
"""

import os
import sys
import subprocess
import time

def test_single_experiment():
    """测试单个实验"""
    
    print("🧪 测试单个实验")
    print("   结果保存到: /content/drive/MyDrive/Solar PV electricity/results")
    print("=" * 60)
    
    # 检查数据文件
    data_file = 'data/Project1033.csv'
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    print(f"✅ 数据文件存在: {data_file}")
    
    # 构建命令 - 运行一个简单的实验
    cmd = [
        sys.executable, 'main.py',
        '--config', 'config/default.yaml',
        '--model', 'RF',
        '--use_hist_weather', 'false',
        '--use_forecast', 'false',
        '--model_complexity', 'low',
        '--past_days', '1',
        '--epochs', '15',
        '--data_path', data_file
    ]
    
    print(f"🚀 运行命令: {' '.join(cmd)}")
    
    # 运行实验
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5分钟超时
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️ 实验耗时: {duration:.1f}秒")
        
        if result.returncode == 0:
            print("✅ 实验成功!")
            print("输出:")
            print(result.stdout)
            return True
        else:
            print("❌ 实验失败!")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 实验超时 (5分钟)")
        return False
    except Exception as e:
        print(f"❌ 实验异常: {e}")
        return False

if __name__ == "__main__":
    success = test_single_experiment()
    sys.exit(0 if success else 1)
