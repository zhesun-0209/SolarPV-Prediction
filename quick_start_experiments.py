#!/usr/bin/env python3
"""
快速启动100个Project消融实验
包含完整的实验运行和Google Drive结果保存
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """设置实验环境"""
    print("🔧 设置实验环境...")
    
    # 挂载Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive已挂载")
    except ImportError:
        print("⚠️ 不在Colab环境中，跳过Drive挂载")
    except Exception as e:
        print(f"❌ Drive挂载失败: {e}")
        return False
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['PYTORCH_CUDNN_V8_API_ENABLED'] = '1'
    os.environ['FORCE_GPU'] = '1'
    os.environ['USE_GPU'] = '1'
    
    print("✅ 环境变量已设置")
    return True

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    required_packages = ['pandas', 'numpy', 'sklearn', 'torch', 'xgboost', 'lightgbm', 'yaml']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"⚠️ 缺少依赖: {missing_packages}")
        print("正在安装...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_properties(0).name
            print(f"✅ GPU检测: {gpu_name} (数量: {gpu_count})")
            return True
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
            return False
    except:
        print("❌ PyTorch未安装")
        return False

def prepare_data():
    """准备数据"""
    print("📁 准备数据...")
    
    data_dir = Path("./data")
    if not data_dir.exists():
        print("❌ data目录不存在")
        return False
    
    csv_files = list(data_dir.glob("Project*.csv"))
    print(f"📊 发现 {len(csv_files)} 个Project数据文件")
    
    if len(csv_files) == 0:
        print("❌ 未找到Project数据文件")
        print("请确保文件命名为Project001.csv, Project002.csv, ...")
        return False
    
    return True

def generate_configs():
    """生成配置"""
    print("📝 生成配置...")
    
    config_dir = Path("./config/projects")
    if config_dir.exists():
        print("✅ 配置已存在，跳过生成")
        return True
    
    try:
        result = subprocess.run([
            sys.executable, "scripts/generate_multi_project_configs.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 配置生成成功")
            return True
        else:
            print(f"❌ 配置生成失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 配置生成异常: {e}")
        return False

def run_experiments():
    """运行实验"""
    print("🚀 开始运行实验...")
    
    # 创建结果目录
    drive_results_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(drive_results_dir, exist_ok=True)
    
    # 选择运行策略
    print("\n选择运行策略:")
    print("1. GPU专用版 (推荐) - 所有模型都使用GPU，10倍加速")
    print("2. 高性能版 - GPU+CPU混合，5倍加速")
    print("3. 标准版 - 传统方式")
    
    try:
        choice = input("请选择 (1/2/3): ").strip()
    except KeyboardInterrupt:
        print("\n🛑 用户取消")
        return False
    
    if choice == "1":
        # GPU专用版
        print("🎯 启动GPU专用版实验...")
        cmd = [
            sys.executable, "scripts/run_gpu_only_experiments.py",
            "--drive-path", drive_results_dir,
            "--max-gpu-experiments", "24",
            "--batch-size", "30"
        ]
    elif choice == "2":
        # 高性能版
        print("🚀 启动高性能版实验...")
        cmd = [
            sys.executable, "scripts/run_high_performance_experiments.py",
            "--drive-path", drive_results_dir,
            "--max-gpu-experiments", "16",
            "--max-cpu-experiments", "32",
            "--batch-size", "25"
        ]
    else:
        # 标准版
        print("📊 启动标准版实验...")
        cmd = [
            sys.executable, "scripts/run_multi_project_experiments.py",
            "--drive-path", drive_results_dir,
            "--max-workers", "8",
            "--batch-size", "20"
        ]
    
    # 运行实验
    try:
        print(f"执行命令: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # 实时显示输出
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("🎉 实验完成!")
            return True
        else:
            print(f"❌ 实验失败，返回码: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断实验")
        process.terminate()
        return False
    except Exception as e:
        print(f"💥 实验异常: {e}")
        return False

def monitor_progress():
    """监控进度"""
    print("📊 监控实验进度...")
    
    try:
        from utils.checkpoint_manager import CheckpointManager
        manager = CheckpointManager()
        progress_df = manager.get_all_projects_progress()
        
        if not progress_df.empty:
            completed = len(progress_df[progress_df['is_complete'] == True])
            total = len(progress_df)
            print(f"📈 当前进度: {completed}/{total} ({completed/total*100:.1f}%)")
            
            if completed > 0:
                print("✅ 最近完成的Project:")
                recent_completed = progress_df[progress_df['is_complete'] == True].tail(5)
                for _, row in recent_completed.iterrows():
                    print(f"   - {row['project_id']}")
        else:
            print("📊 暂无进度信息")
    except Exception as e:
        print(f"⚠️ 进度查询失败: {e}")

def show_results():
    """显示结果"""
    print("📊 实验结果...")
    
    drive_results_dir = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    
    if os.path.exists(drive_results_dir):
        csv_files = list(Path(drive_results_dir).glob("Project*.csv"))
        print(f"📁 结果文件数量: {len(csv_files)}")
        
        if csv_files:
            print("📋 结果文件列表:")
            for csv_file in sorted(csv_files)[:10]:  # 显示前10个
                file_size = csv_file.stat().st_size / 1024  # KB
                print(f"   - {csv_file.name} ({file_size:.1f}KB)")
            
            if len(csv_files) > 10:
                print(f"   ... 还有 {len(csv_files) - 10} 个文件")
        else:
            print("⚠️ 暂无结果文件")
    else:
        print("❌ 结果目录不存在")

def main():
    """主函数"""
    print("🚀 Project1140 100个Project消融实验快速启动")
    print("=" * 60)
    
    # 设置环境
    if not setup_environment():
        print("❌ 环境设置失败")
        return
    
    # 检查依赖
    gpu_available = check_dependencies()
    
    # 准备数据
    if not prepare_data():
        print("❌ 数据准备失败")
        return
    
    # 生成配置
    if not generate_configs():
        print("❌ 配置生成失败")
        return
    
    print("\n" + "=" * 60)
    print("🎯 准备开始实验!")
    print("📊 结果将实时保存到Google Drive")
    print("🔄 支持断点续训")
    
    if gpu_available:
        print("🚀 检测到GPU，推荐使用GPU专用版")
    else:
        print("💻 未检测到GPU，将使用CPU模式")
    
    print("=" * 60)
    
    # 用户确认
    try:
        response = input("\n是否开始实验? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("🛑 用户取消实验")
            return
    except KeyboardInterrupt:
        print("\n🛑 用户取消实验")
        return
    
    # 运行实验
    start_time = time.time()
    success = run_experiments()
    end_time = time.time()
    
    # 显示结果
    if success:
        print(f"\n⏱️ 总运行时间: {(end_time - start_time)/3600:.1f} 小时")
        monitor_progress()
        show_results()
        print("\n🎉 实验完成!")
    else:
        print("\n⚠️ 实验未完全完成")
        monitor_progress()
        show_results()
    
    print("\n📊 查看结果:")
    print("   Google Drive: /content/drive/MyDrive/Solar PV electricity/ablation results")
    print("   每个Project一个CSV文件: Project001.csv, Project002.csv, ...")

if __name__ == "__main__":
    main()
