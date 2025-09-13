#!/usr/bin/env python3
"""
测试所有模型类型 - 确保每个模型都能正常运行并保存结果
"""

import os
import sys
import subprocess
import time
import yaml
import glob
from datetime import datetime

def test_model_type(model_name, config_pattern):
    """测试特定类型的模型"""
    print(f"\n{'='*60}")
    print(f"🧪 测试 {model_name} 模型")
    print(f"{'='*60}")
    
    # 找到对应的配置文件
    config_dir = "config/projects/1140"
    yaml_files = glob.glob(os.path.join(config_dir, f"*{config_pattern}*.yaml"))
    
    if not yaml_files:
        print(f"❌ 未找到 {model_name} 配置文件")
        return False, "No config found"
    
    # 选择第一个配置文件进行测试
    config_file = yaml_files[0]
    print(f"📁 使用配置: {os.path.basename(config_file)}")
    
    try:
        # 运行训练
        start_time = time.time()
        result = subprocess.run([
            sys.executable, "main.py", "--config", config_file
        ], capture_output=True, text=True, timeout=600)  # 10分钟超时
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} 训练成功! 用时: {duration:.1f}秒")
            
            # 提取结果指标
            if "mse=" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "mse=" in line and "rmse=" in line and "mae=" in line:
                        print(f"📊 结果: {line.strip()}")
                        break
            
            # 检查结果文件是否保存
            check_result_files(model_name, config_file)
            
            return True, result.stdout
        else:
            print(f"❌ {model_name} 训练失败! 返回码: {result.returncode}")
            print(f"错误: {result.stderr[-300:]}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} 训练超时 (10分钟)")
        return False, "Timeout"
    except Exception as e:
        print(f"💥 {model_name} 训练异常: {str(e)}")
        return False, str(e)

def check_result_files(model_name, config_file):
    """检查结果文件是否保存"""
    print(f"🔍 检查 {model_name} 结果文件...")
    
    # 从配置文件中获取保存目录
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    save_dir = config.get('save_dir', 'temp_results/1140')
    
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        print(f"📁 结果目录: {save_dir}")
        print(f"📄 文件数量: {len(files)}")
        
        # 检查是否有结果文件
        result_files = []
        for file in files:
            if file.endswith(('.csv', '.json', '.pkl', '.pth')):
                result_files.append(file)
        
        if result_files:
            print(f"✅ 找到结果文件: {result_files}")
        else:
            print(f"⚠️ 未找到结果文件")
            
        # 显示目录内容
        print(f"📋 目录内容:")
        for file in sorted(files)[:10]:  # 显示前10个文件
            file_path = os.path.join(save_dir, file)
            if os.path.isdir(file_path):
                print(f"  📁 {file}/")
            else:
                size = os.path.getsize(file_path)
                print(f"  📄 {file} ({size} bytes)")
    else:
        print(f"❌ 结果目录不存在: {save_dir}")

def main():
    """主函数"""
    print("🌟 SolarPV项目 - 全模型类型测试")
    print("=" * 80)
    
    # 定义要测试的模型类型
    model_tests = [
        ("LSTM", "LSTM_low_PV_24h_TE"),
        ("GRU", "GRU_low_PV_24h_TE"), 
        ("Transformer", "Transformer_low_PV_24h_TE"),
        ("TCN", "TCN_low_PV_24h_TE"),
        ("RF", "RF_low_PV_24h_TE"),
        ("XGB", "XGB_low_PV_24h_TE"),
        ("LGBM", "LGBM_low_PV_24h_TE"),
        ("LSR", "LSR_low_PV_24h_TE")
    ]
    
    results = {}
    total_tests = len(model_tests)
    successful = 0
    failed = 0
    
    print(f"📊 将测试 {total_tests} 种模型类型")
    
    for i, (model_name, config_pattern) in enumerate(model_tests, 1):
        print(f"\n🔄 进度: {i}/{total_tests}")
        
        success, output = test_model_type(model_name, config_pattern)
        results[model_name] = {'success': success, 'output': output}
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # 显示当前统计
        print(f"📈 当前统计: 成功 {successful}, 失败 {failed}")
    
    # 最终统计
    print(f"\n🎉 测试完成!")
    print("=" * 80)
    print(f"📊 最终统计:")
    print(f"  总测试: {total_tests}")
    print(f"  成功: {successful} ({successful/total_tests*100:.1f}%)")
    print(f"  失败: {failed} ({failed/total_tests*100:.1f}%)")
    
    print(f"\n📋 详细结果:")
    for model_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {model_name}: {'成功' if result['success'] else '失败'}")
    
    # 检查所有结果目录
    print(f"\n📁 检查所有结果目录:")
    results_dir = "temp_results/1140"
    if os.path.exists(results_dir):
        all_dirs = os.listdir(results_dir)
        print(f"  找到 {len(all_dirs)} 个结果目录")
        for dir_name in sorted(all_dirs)[:10]:  # 显示前10个
            print(f"    📁 {dir_name}")
        if len(all_dirs) > 10:
            print(f"    ... 还有 {len(all_dirs) - 10} 个目录")
    else:
        print(f"  ❌ 结果目录不存在: {results_dir}")
    
    return results

if __name__ == "__main__":
    results = main()
