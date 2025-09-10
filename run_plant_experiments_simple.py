#!/usr/bin/env python3
"""
运行单个厂的简化实验（只运行几个测试实验）
"""

import os
import sys
import subprocess
import time

def run_plant_experiments_simple(plant_id, data_file):
    """运行单个厂的简化实验"""
    
    print(f"🏭 开始运行厂 {plant_id} 的简化实验")
    print(f"   数据文件: {data_file}")
    print("=" * 80)
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    # 定义简化的实验组合（只运行几个测试实验）
    test_experiments = [
        # (model, hist_weather, forecast, complexity, past_days)
        ('RF', False, False, 'low', 1),
        ('RF', True, False, 'low', 1),
        ('LSTM', False, False, 'low', 1),
        ('LSTM', True, False, 'low', 1),
    ]
    
    # 根据复杂度设置epoch数
    epoch_map = {'low': 15, 'medium': 30, 'high': 50}
    
    total_experiments = len(test_experiments)
    print(f"📊 测试实验数: {total_experiments}")
    
    completed = 0
    failed = 0
    
    start_time = time.time()
    
    for i, (model, hist_weather, forecast, complexity, past_days) in enumerate(test_experiments, 1):
        # 生成实验ID
        feat_str = f"feat{str(hist_weather).lower()}_fcst{str(forecast).lower()}_days{past_days}_comp{complexity}"
        exp_id = f"{model}_{feat_str}"
        
        print(f"\n🚀 运行实验 {i}/{total_experiments}: {exp_id}")
        
        # 构建命令
        epochs = epoch_map[complexity]
        
        cmd = [
            sys.executable, 'main.py',
            '--config', 'config/default.yaml',
            '--model', model,
            '--use_hist_weather', str(hist_weather).lower(),
            '--use_forecast', str(forecast).lower(),
            '--model_complexity', complexity,
            '--past_days', str(past_days),
            '--epochs', str(epochs),
            '--data_path', data_file
        ]
        
        # 运行实验
        exp_start = time.time()
        try:
            print(f"   命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10分钟超时
            exp_end = time.time()
            exp_duration = exp_end - exp_start
            
            if result.returncode == 0:
                print(f"✅ 实验完成 (耗时: {exp_duration:.1f}秒)")
                completed += 1
            else:
                print(f"❌ 实验失败")
                print("错误输出:")
                print(result.stderr)
                failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"❌ 实验超时 (10分钟)")
            failed += 1
        except Exception as e:
            print(f"❌ 实验异常: {e}")
            failed += 1
        
        # 显示进度
        print(f"📈 进度: {i}/{total_experiments} ({i/total_experiments*100:.1f}%)")
    
    # 最终统计
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n🎉 厂 {plant_id} 简化实验完成!")
    print("=" * 80)
    print(f"总实验数: {total_experiments}")
    print(f"成功: {completed}")
    print(f"失败: {failed}")
    print(f"总耗时: {total_duration/60:.1f}分钟")
    
    return completed > 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行单个厂的简化实验')
    parser.add_argument('plant_id', help='厂ID')
    parser.add_argument('data_file', help='数据文件路径')
    
    args = parser.parse_args()
    
    success = run_plant_experiments_simple(args.plant_id, args.data_file)
    sys.exit(0 if success else 1)
