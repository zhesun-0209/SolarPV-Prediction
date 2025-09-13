#!/usr/bin/env python3
"""
Colab快速测试脚本 - 测试几个关键模型
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def test_key_models():
    """测试关键模型"""
    # 选择几个代表性的模型进行测试
    test_configs = [
        "config/projects/1140/LSTM_low_PV_24h_TE.yaml",
        "config/projects/1140/Transformer_low_PV_24h_TE.yaml", 
        "config/projects/1140/RF_low_PV_24h_TE.yaml",
        "config/projects/1140/XGB_low_PV_24h_TE.yaml",
        "config/projects/1140/LGBM_low_PV_24h_TE.yaml"
    ]
    
    print("🧪 快速测试关键模型...")
    print(f"📁 测试 {len(test_configs)} 个模型")
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'='*50}")
        print(f"🔥 测试 {i}/{len(test_configs)}: {os.path.basename(config)}")
        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "main.py", "--config", config
            ], capture_output=True, text=True, timeout=600)  # 10分钟超时
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ 成功! 用时: {duration:.1f}秒")
                
                # 提取结果
                if "mse=" in result.stdout:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "mse=" in line and "rmse=" in line and "mae=" in line:
                            print(f"📊 {line.strip()}")
                            results.append({
                                'model': os.path.basename(config),
                                'status': 'success',
                                'duration': duration,
                                'metrics': line.strip()
                            })
                            break
                else:
                    results.append({
                        'model': os.path.basename(config),
                        'status': 'success',
                        'duration': duration,
                        'metrics': 'No metrics found'
                    })
            else:
                print(f"❌ 失败! 返回码: {result.returncode}")
                print(f"错误: {result.stderr[-200:]}")
                results.append({
                    'model': os.path.basename(config),
                    'status': 'failed',
                    'duration': duration,
                    'error': result.stderr[-200:]
                })
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 超时 (10分钟)")
            results.append({
                'model': os.path.basename(config),
                'status': 'timeout',
                'duration': 600,
                'error': 'Timeout'
            })
        except Exception as e:
            print(f"💥 异常: {str(e)}")
            results.append({
                'model': os.path.basename(config),
                'status': 'error',
                'duration': 0,
                'error': str(e)
            })
    
    # 显示总结
    print(f"\n🎉 测试完成!")
    print(f"{'='*50}")
    print(f"📊 测试总结:")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print(f"  成功: {success_count}/{len(results)}")
    print(f"  失败: {failed_count}/{len(results)}")
    
    if success_count > 0:
        avg_duration = sum(r['duration'] for r in results if r['status'] == 'success') / success_count
        print(f"  平均用时: {avg_duration:.1f}秒")
    
    print(f"\n📋 详细结果:")
    for result in results:
        status_emoji = "✅" if result['status'] == 'success' else "❌"
        print(f"  {status_emoji} {result['model']}: {result['status']}")
        if result['status'] == 'success' and 'metrics' in result:
            print(f"      {result['metrics']}")
        elif 'error' in result:
            print(f"      错误: {result['error'][:100]}...")
    
    return results

if __name__ == "__main__":
    test_key_models()
