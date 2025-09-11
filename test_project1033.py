#!/usr/bin/env python3
"""
Project1033单厂测试脚本
测试所有300个实验组合，保存到Drive的result_new文件夹
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_project1033():
    """运行Project1033的所有300个实验"""
    
    print("🚀 开始Project1033单厂测试")
    print("=" * 60)
    
    # 检查数据文件
    data_file = "data/Project1033.csv"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    print(f"✅ 数据文件存在: {data_file}")
    
    # 检查配置文件
    config_file = "config/project1033_test.yaml"
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    print(f"✅ 配置文件存在: {config_file}")
    
    # 运行单厂实验
    print("\n🏭 开始运行Project1033的所有实验")
    print(f"   数据文件: {data_file}")
    print(f"   结果保存到: /content/drive/MyDrive/Solar PV electricity/result_new")
    print("=" * 60)
    
    cmd = [
        sys.executable, 'run_plant_experiments.py',
        'Project1033', data_file
    ]
    
    print(f"🚀 运行命令: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=18000)  # 5小时超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\n✅ Project1033测试完成!")
            print(f"   总耗时: {duration/3600:.2f}小时")
            print(f"   平均每实验: {duration/300:.1f}秒")
            
            # 检查结果文件
            result_dir = "/content/drive/MyDrive/Solar PV electricity/result_new/Project1033"
            excel_file = f"{result_dir}/Project1033_results.xlsx"
            
            if os.path.exists(excel_file):
                print(f"✅ 结果文件已保存: {excel_file}")
                
                # 检查结果文件大小
                file_size = os.path.getsize(excel_file)
                print(f"   文件大小: {file_size/1024:.1f} KB")
                
                # 检查实验数量
                try:
                    import pandas as pd
                    df = pd.read_excel(excel_file)
                    print(f"   实验数量: {len(df)}")
                    print(f"   列数: {len(df.columns)}")
                except Exception as e:
                    print(f"⚠️  无法读取Excel文件: {e}")
            else:
                print(f"❌ 结果文件未找到: {excel_file}")
            
            return True
            
        else:
            print(f"\n❌ Project1033测试失败!")
            print(f"   返回码: {result.returncode}")
            print(f"   耗时: {duration/3600:.2f}小时")
            print("\n错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ Project1033测试超时 (5小时)")
        return False
    except Exception as e:
        print(f"\n❌ Project1033测试异常: {e}")
        return False

def main():
    """主函数"""
    print("🔬 Project1033单厂测试工具")
    print("=" * 60)
    print("📊 实验规模:")
    print("   模型: 8种 (Transformer, LSTM, GRU, TCN, RF, XGB, LGBM, Linear)")
    print("   特征组合: 20种")
    print("   复杂度: 2种 (Low, High)")
    print("   总实验数: 300个")
    print("   预计时间: 2.5小时 (假设每实验30秒)")
    print("   保存位置: /content/drive/MyDrive/Solar PV electricity/result_new")
    print("=" * 60)
    
    # 确认运行
    response = input("\n是否开始测试? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 测试已取消")
        return
    
    # 运行测试
    success = test_project1033()
    
    if success:
        print("\n🎉 Project1033测试成功完成!")
        print("   结果已保存到Drive的result_new文件夹")
    else:
        print("\n❌ Project1033测试失败!")
        print("   请检查错误信息并重试")

if __name__ == "__main__":
    main()
