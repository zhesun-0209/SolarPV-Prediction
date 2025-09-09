#!/usr/bin/env python3
"""
Colab快速设置和运行脚本
"""

import os
import subprocess
import sys
import yaml
from google.colab import drive

def setup_environment():
    """设置环境"""
    print("🔧 设置Colab环境...")
    
    # 挂载Google Drive
    try:
        drive.mount('/content/drive')
        print("✅ Google Drive已挂载")
    except Exception as e:
        print(f"❌ 挂载Google Drive失败: {e}")
        return False
    
    # 克隆项目
    if not os.path.exists('SolarPV-Prediction'):
        print("📥 克隆项目...")
        result = subprocess.run(['git', 'clone', 'https://github.com/zhesun-0209/SolarPV-Prediction.git'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 克隆项目失败: {result.stderr}")
            return False
        print("✅ 项目已克隆")
    
    # 进入项目目录
    os.chdir('SolarPV-Prediction')
    print(f"📁 当前目录: {os.getcwd()}")
    
    # 安装依赖
    print("📦 安装依赖...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 安装依赖失败: {result.stderr}")
        return False
    print("✅ 依赖已安装")
    
    return True

def configure_paths():
    """配置路径"""
    print("⚙️ 配置路径...")
    
    # 修改配置文件
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置保存路径
    config['save_dir'] = '/content/drive/MyDrive/Solar PV electricity/results'
    
    # 保存配置
    with open('config/default.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ 配置已更新")
    print(f"   数据路径: {config['data_path']}")
    print(f"   保存路径: {config['save_dir']}")

def check_data():
    """检查数据"""
    print("🔍 检查数据...")
    
    if not os.path.exists('data/Project1033.csv'):
        print("❌ 数据文件不存在: data/Project1033.csv")
        return False
    
    import pandas as pd
    df = pd.read_csv('data/Project1033.csv')
    print(f"✅ 数据文件检查通过")
    print(f"   数据形状: {df.shape}")
    print(f"   列数: {len(df.columns)}")
    print(f"   前5列: {df.columns[:5].tolist()}")
    
    return True

def run_quick_test():
    """运行快速测试"""
    print("🚀 运行快速测试...")
    
    # 运行一个简单的实验
    cmd = [
        sys.executable, 'main.py',
        '--model', 'LSTM',
        '--use_hist_weather', 'true',
        '--use_forecast', 'false',
        '--model_complexity', 'low',
        '--past_days', '1'
    ]
    
    print(f"命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 快速测试成功")
        print("输出:")
        print(result.stdout[-500:])  # 显示最后500个字符
    else:
        print("❌ 快速测试失败")
        print("错误:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction Colab快速设置")
    print("=" * 50)
    
    # 设置环境
    if not setup_environment():
        print("❌ 环境设置失败")
        return
    
    # 配置路径
    configure_paths()
    
    # 检查数据
    if not check_data():
        print("❌ 数据检查失败")
        return
    
    # 运行快速测试
    if run_quick_test():
        print("\n🎉 设置完成！现在可以运行:")
        print("   !python colab_run.py          # 快速模型对比")
        print("   !python colab_full_experiments.py  # 全参数实验")
    else:
        print("\n❌ 设置失败，请检查错误信息")

if __name__ == "__main__":
    main()
