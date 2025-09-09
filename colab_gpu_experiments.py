#!/usr/bin/env python3
"""
Colab GPU版本全参数组合实验脚本
支持断点续传和GPU加速的RF/GBR
"""

import os
import subprocess
import sys
import time
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def setup_gpu_environment():
    """设置GPU环境"""
    print("🔧 设置GPU环境...")
    
    # 安装cuML (GPU版本的RF/GBR)
    print("📦 安装cuML...")
    try:
        import cuml
        print(f"✅ cuML已安装，版本: {cuml.__version__}")
    except ImportError:
        print("📥 安装cuML...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'cuml-cu11', '--extra-index-url=https://pypi.nvidia.com'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ cuML安装成功")
        else:
            print("❌ cuML安装失败，将使用CPU版本")
            print("错误:", result.stderr)
    
    # 设置A100优化
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.9)
        print("✅ A100优化设置完成")
    except ImportError:
        print("⚠️ PyTorch未安装，跳过A100优化")

def check_existing_results():
    """检查现有结果"""
    print("🔍 检查现有结果...")
    
    # 查找所有summary.csv文件
    summary_files = glob.glob('result/**/summary.csv', recursive=True)
    
    if not summary_files:
        print("📝 未找到现有结果，将从头开始")
        return set()
    
    # 读取现有结果
    existing_experiments = set()
    for file in summary_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                # 创建实验标识
                exp_id = f"{df.iloc[0]['model']}_{df.iloc[0]['use_hist_weather']}_{df.iloc[0]['use_forecast']}_{df.iloc[0].get('model_complexity', 'medium')}"
                existing_experiments.add(exp_id)
        except Exception as e:
            print(f"⚠️ 读取结果文件失败 {file}: {e}")
    
    print(f"📊 找到 {len(existing_experiments)} 个已完成实验")
    return existing_experiments

def get_experiment_id(model, hist_weather, forecast, complexity, past_days):
    """生成实验标识"""
    return f"{model}_{hist_weather}_{forecast}_{complexity}_{past_days}"

def run_experiment(model, hist_weather, forecast, complexity, past_days, description):
    """运行单个实验"""
    print(f"\n🚀 {description}")
    print("-" * 60)
    
    cmd = [
        sys.executable, 'main.py',
        '--config', 'config/default.yaml',
        '--model', model,
        '--use_hist_weather', hist_weather,
        '--use_forecast', forecast,
        '--model_complexity', complexity,
        '--past_days', str(past_days)
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} 完成 (耗时: {end_time - start_time:.2f}秒)")
        return True
    else:
        print(f"❌ {description} 失败")
        print("错误:", result.stderr)
        return False

def run_gpu_experiments():
    """运行GPU版本实验"""
    print("\n🔬 GPU版本全参数组合实验")
    print("=" * 80)
    
    # 模型列表 (已移除GBR)
    models = ['Transformer', 'LSTM', 'GRU', 'TCN', 'RF', 'XGB', 'LGBM']
    
    # 特征组合
    feature_configs = [
        ('hist_only', 'true', 'false'),
        ('fcst_only', 'false', 'true'),
        ('both', 'true', 'true'),
        ('none', 'false', 'false')
    ]
    
    # 复杂度组合
    complexities = ['low', 'medium', 'high']
    
    # 时间窗口组合
    past_days_options = [1, 3, 7]
    
    # 检查现有结果
    existing_experiments = check_existing_results()
    
    # 统计信息
    total_experiments = len(models) * len(feature_configs) * len(complexities) * len(past_days_options)
    completed = 0
    failed = 0
    skipped = 0
    
    print(f"📊 总实验数: {total_experiments}")
    print(f"   已完成: {len(existing_experiments)}")
    print(f"   待完成: {total_experiments - len(existing_experiments)}")
    
    start_time = time.time()
    
    for model in models:
        print(f"\n🎯 开始 {model} 模型实验")
        print("-" * 60)
        
        for feat_desc, hist_weather, forecast in feature_configs:
            print(f"\n📋 特征组合: {feat_desc}")
            
            for complexity in complexities:
                for past_days in past_days_options:
                    exp_id = get_experiment_id(model, hist_weather, forecast, complexity, past_days)
                    description = f"{model} - {feat_desc} - {complexity} - {past_days}天"
                    
                    if exp_id in existing_experiments:
                        print(f"⏭️ {description} - 跳过 (已完成)")
                        skipped += 1
                        continue
                    
                    if run_experiment(model, hist_weather, forecast, complexity, past_days, description):
                        completed += 1
                    else:
                        failed += 1
                    
                    # 显示进度
                    total_done = completed + failed + skipped
                    progress = total_done / total_experiments * 100
                    elapsed = time.time() - start_time
                    eta = elapsed / total_done * (total_experiments - total_done) if total_done > 0 else 0
                    
                    print(f"📈 进度: {total_done}/{total_experiments} ({progress:.1f}%)")
                    print(f"⏱️  已用时间: {elapsed/60:.1f}分钟, 预计剩余: {eta/60:.1f}分钟")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 GPU版本实验完成！")
    print(f"📊 成功: {completed}")
    print(f"❌ 失败: {failed}")
    print(f"⏭️ 跳过: {skipped}")
    print(f"⏱️  总耗时: {total_time/60:.1f}分钟")
    
    return completed, failed, skipped

def analyze_results():
    """分析结果"""
    print("\n📊 分析结果...")
    
    try:
        # 检查Drive目录
        drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
        if os.path.exists(drive_dir):
            result_dir = drive_dir
        else:
            result_dir = 'result'
        
        # 查找所有summary.csv文件
        summary_files = glob.glob(f'{result_dir}/**/summary.csv', recursive=True)
        
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
        combined_df.to_csv(f'{result_dir}/all_experiments_results.csv', index=False)
        print(f"✅ 合并结果保存到 {result_dir}/all_experiments_results.csv")
        
        # 创建可视化
        create_visualizations(combined_df)
        
        # 显示统计结果
        show_statistics(combined_df)
        
    except Exception as e:
        print(f"❌ 分析结果时出错: {e}")

def create_visualizations(df):
    """创建可视化图表"""
    print("\n📊 创建可视化图表...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. 模型性能对比
    ax1 = axes[0, 0]
    model_perf = df.groupby('model')['rmse'].mean().sort_values()
    model_perf.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Model Performance (RMSE)', fontsize=12)
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 特征组合影响
    ax2 = axes[0, 1]
    feature_perf = df.groupby(['use_hist_weather', 'use_forecast'])['rmse'].mean()
    feature_perf.plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_title('Feature Combination Impact', fontsize=12)
    ax2.set_ylabel('RMSE')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 复杂度影响
    ax3 = axes[0, 2]
    complexity_perf = df.groupby('model_complexity')['rmse'].mean()
    complexity_perf.plot(kind='bar', ax=ax3, color='lightgreen')
    ax3.set_title('Complexity Impact', fontsize=12)
    ax3.set_ylabel('RMSE')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 训练时间对比
    ax4 = axes[1, 0]
    train_time = df.groupby('model')['train_time_sec'].mean().sort_values()
    train_time.plot(kind='bar', ax=ax4, color='orange')
    ax4.set_title('Training Time Comparison', fontsize=12)
    ax4.set_ylabel('Time (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. 参数数量对比
    ax5 = axes[1, 1]
    param_count = df.groupby('model')['param_count'].mean().sort_values()
    param_count.plot(kind='bar', ax=ax5, color='purple')
    ax5.set_title('Parameter Count Comparison', fontsize=12)
    ax5.set_ylabel('Parameters')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. 性能vs时间散点图
    ax6 = axes[1, 2]
    scatter = ax6.scatter(df['train_time_sec'], df['rmse'], c=df['param_count'], cmap='viridis', alpha=0.7)
    ax6.set_xlabel('Training Time (s)')
    ax6.set_ylabel('RMSE')
    ax6.set_title('Performance vs Training Time')
    plt.colorbar(scatter, ax=ax6, label='Parameter Count')
    
    plt.tight_layout()
    # 检查Drive目录
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    if os.path.exists(drive_dir):
        result_dir = drive_dir
    else:
        result_dir = 'result'
    
    plt.savefig(f'{result_dir}/gpu_experiments_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 可视化图表保存到 {result_dir}/gpu_experiments_analysis.png")

def show_statistics(df):
    """显示统计结果"""
    print("\n=== 详细统计结果 ===")
    
    # 按模型统计
    print("\n📊 按模型统计:")
    model_stats = df.groupby('model').agg({
        'test_loss': ['mean', 'std', 'min'],
        'rmse': ['mean', 'std', 'min'],
        'mae': ['mean', 'std', 'min'],
        'train_time_sec': ['mean', 'std'],
        'param_count': ['mean', 'std']
    }).round(4)
    print(model_stats)
    
    # 找出最佳配置
    best_overall = df.loc[df['rmse'].idxmin()]
    print(f"\n🏆 最佳整体配置:")
    print(f"   模型: {best_overall['model']}")
    print(f"   历史天气: {best_overall['use_hist_weather']}")
    print(f"   预测天气: {best_overall['use_forecast']}")
    print(f"   复杂度: {best_overall.get('model_complexity', 'N/A')}")
    print(f"   RMSE: {best_overall['rmse']:.4f}")
    print(f"   MAE: {best_overall['mae']:.4f}")
    print(f"   训练时间: {best_overall['train_time_sec']:.2f}秒")

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction GPU版本全参数组合实验")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 设置GPU环境
    setup_gpu_environment()
    
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
    
    # 检查数据文件
    if not os.path.exists('data/Project1033.csv'):
        print("❌ 数据文件不存在: data/Project1033.csv")
        return
    
    print("✅ 数据文件检查通过")
    
    # 运行GPU实验
    completed, failed, skipped = run_gpu_experiments()
    
    # 分析结果
    analyze_results()
    
    print(f"\n🎉 所有实验完成！")
    print(f"📊 成功: {completed}")
    print(f"❌ 失败: {failed}")
    print(f"⏭️ 跳过: {skipped}")
    # 检查Drive目录
    drive_dir = '/content/drive/MyDrive/Solar PV electricity/results'
    if os.path.exists(drive_dir):
        result_dir = drive_dir
        print(f"📁 结果保存在: {result_dir} (Google Drive)")
    else:
        result_dir = 'result'
        print(f"📁 结果保存在: {result_dir} (本地)")
    
    print(f"📊 合并结果: {result_dir}/all_experiments_results.csv")
    print(f"📊 可视化: {result_dir}/gpu_experiments_analysis.png")

if __name__ == "__main__":
    main()
