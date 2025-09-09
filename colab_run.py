#!/usr/bin/env python3
"""
Colab运行脚本
在Colab中运行实验，结果保存到result文件夹
"""

import os
import subprocess
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

def check_environment():
    """检查环境"""
    print("🔍 检查环境...")
    
    # 检查CUDA
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not available")
    
    # 检查数据文件
    if os.path.exists('data/Project1033.csv'):
        print("✅ 数据文件存在")
    else:
        print("❌ 数据文件不存在")
        return False
    
    return True

def setup_a100():
    """设置A100优化"""
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.9)
        print("✅ A100优化设置完成")
    except ImportError:
        print("PyTorch not available, skipping A100 optimization")

def run_experiment(model, description):
    """运行实验"""
    print(f"\n🚀 运行 {description}")
    print("-" * 50)
    
    cmd = [
        sys.executable, 'main.py',
        '--config', 'config/default.yaml',
        '--model', model,
        '--use_hist_weather', 'true',
        '--use_forecast', 'true',
        '--model_complexity', 'medium',
        '--past_days', '3'
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} 完成 (耗时: {end_time - start_time:.2f}秒)")
        return True
    else:
        print(f"❌ {description} 失败")
        print(f"错误: {result.stderr}")
        return False

def analyze_results():
    """分析结果"""
    print("\n📊 分析结果...")
    
    try:
        import glob
        
        # 查找所有summary.csv文件
        summary_files = glob.glob('result/**/summary.csv', recursive=True)
        
        if not summary_files:
            print("❌ 未找到结果文件")
            return
        
        print(f"✅ 找到 {len(summary_files)} 个结果文件")
        
        # 合并所有结果
        all_results = []
        for file in summary_files:
            try:
                df = pd.read_csv(file)
                all_results.append(df)
            except Exception as e:
                print(f"❌ 读取文件失败 {file}: {e}")
        
        if not all_results:
            print("❌ 没有有效的结果文件")
            return
        
        # 合并结果
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # 保存合并结果
        combined_df.to_csv('result/combined_results.csv', index=False)
        print("✅ 合并结果保存到 result/combined_results.csv")
        
        # 显示结果
        print("\n=== 模型性能对比 ===")
        display_df = combined_df[['model', 'test_loss', 'rmse', 'mae', 'train_time_sec', 'inference_time_sec']].round(4)
        print(display_df)
        
        # 可视化结果
        plt.figure(figsize=(15, 10))
        
        # RMSE对比
        plt.subplot(2, 3, 1)
        plt.bar(combined_df['model'], combined_df['rmse'])
        plt.title('RMSE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('RMSE')
        
        # MAE对比
        plt.subplot(2, 3, 2)
        plt.bar(combined_df['model'], combined_df['mae'])
        plt.title('MAE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('MAE')
        
        # 训练时间对比
        plt.subplot(2, 3, 3)
        plt.bar(combined_df['model'], combined_df['train_time_sec'])
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Time (s)')
        
        # 推理时间对比
        plt.subplot(2, 3, 4)
        plt.bar(combined_df['model'], combined_df['inference_time_sec'])
        plt.title('Inference Time Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Time (s)')
        
        # 参数数量对比
        plt.subplot(2, 3, 5)
        plt.bar(combined_df['model'], combined_df['param_count'])
        plt.title('Parameter Count Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Parameters')
        
        # 性能vs时间散点图
        plt.subplot(2, 3, 6)
        plt.scatter(combined_df['train_time_sec'], combined_df['rmse'], s=100)
        for i, model in enumerate(combined_df['model']):
            plt.annotate(model, (combined_df['train_time_sec'].iloc[i], combined_df['rmse'].iloc[i]))
        plt.xlabel('Training Time (s)')
        plt.ylabel('RMSE')
        plt.title('Performance vs Training Time')
        
        plt.tight_layout()
        plt.savefig('result/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 找出最佳模型
        best_model = combined_df.loc[combined_df['rmse'].idxmin()]
        print(f"\n🏆 最佳模型: {best_model['model']}")
        print(f"   RMSE: {best_model['rmse']:.4f}")
        print(f"   MAE: {best_model['mae']:.4f}")
        print(f"   训练时间: {best_model['train_time_sec']:.2f}秒")
        print(f"   推理时间: {best_model['inference_time_sec']:.2f}秒")
        
    except Exception as e:
        print(f"❌ 分析结果时出错: {e}")

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction Colab运行器")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        return
    
    # 设置A100优化
    setup_a100()
    
    # 检查Drive目录并修改配置
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    if os.path.exists(drive_dir):
        print(f"✅ Drive目录存在: {drive_dir}")
        # 修改配置文件，保存到Drive目录
        import yaml
        with open('config/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        config['save_dir'] = drive_dir
        with open('config/default.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✅ 配置已更新，结果将保存到: {drive_dir}")
    else:
        print(f"⚠️ Drive目录不存在: {drive_dir}")
        print("💡 请确保已挂载Google Drive")
        # 创建本地结果目录
        if not os.path.exists('result'):
            os.makedirs('result')
            print("✅ 创建本地result目录")
    
    # 运行实验
    models = [
        ('TCN', 'TCN模型'),
        ('LSTM', 'LSTM模型'),
        ('Transformer', 'Transformer模型'),
        ('XGB', 'XGBoost模型')
    ]
    
    success_count = 0
    for model, description in models:
        if run_experiment(model, description):
            success_count += 1
    
    # 分析结果
    analyze_results()
    
    print(f"\n🎉 实验完成！")
    print(f"📊 成功: {success_count}/{len(models)} 个模型")
    print(f"📁 结果保存在: result/ 目录")
    print(f"📊 性能对比图: result/performance_comparison.png")

if __name__ == "__main__":
    main()
