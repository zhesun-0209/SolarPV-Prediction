#!/usr/bin/env python3
"""
Colab测试脚本 - 测试8类模型，确保都能正常运行并输出指标
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
            
            # 检查CSV结果文件是否保存
            check_csv_results(model_name, config_file)
            
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

def check_csv_results(model_name, config_file):
    """检查CSV结果文件是否保存"""
    print(f"🔍 检查 {model_name} CSV结果文件...")
    
    # 从配置文件中获取保存目录
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    save_dir = config.get('save_dir', 'temp_results/1140')
    
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        print(f"📁 结果目录: {save_dir}")
        
        # 检查是否有CSV文件
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if csv_files:
            print(f"✅ 找到CSV结果文件: {csv_files}")
            for file in csv_files:
                file_path = os.path.join(save_dir, file)
                size = os.path.getsize(file_path)
                print(f"  📊 {file} ({size} bytes)")
                
                # 显示CSV文件内容预览
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    print(f"    📋 行数: {len(df)}, 列数: {len(df.columns)}")
                    if len(df) > 0:
                        print(f"    📋 列名: {list(df.columns)[:10]}...")  # 显示前10列
                except Exception as e:
                    print(f"    ❌ 读取CSV文件失败: {e}")
        else:
            print(f"⚠️ 未找到CSV结果文件")
            
        # 显示目录内容
        print(f"📋 目录内容:")
        for file in sorted(files)[:5]:  # 显示前5个文件
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
    print("🌟 SolarPV项目 - 8类模型测试")
    print("=" * 80)
    
    # 定义要测试的8类模型
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
    
    print(f"📊 将测试 {total_tests} 类模型")
    print(f"🎯 目标: 确保每类模型都能正常运行并输出指标")
    print(f"💾 只保存CSV结果文件")
    
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
    
    # 提供下一步建议
    if successful == total_tests:
        print(f"\n🎊 所有8类模型测试成功！")
        print(f"💡 建议: 现在可以运行完整的训练脚本 colab_fixed_training.py")
    elif successful > 0:
        print(f"\n⚠️ 部分模型测试成功")
        print(f"💡 建议: 检查失败的模型，可能需要调整配置或环境")
    else:
        print(f"\n❌ 所有模型测试失败")
        print(f"💡 建议: 检查环境配置和依赖安装")
    
    return results

if __name__ == "__main__":
    results = main()
