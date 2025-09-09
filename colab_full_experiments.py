#!/usr/bin/env python3
"""
Colab全参数组合实验脚本
在Colab中运行所有厂、所有模型、所有参数组合
"""

import os
import subprocess
import sys
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def run_experiment(model, hist_weather, forecast, complexity, description):
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

def run_full_experiments():
    """运行全参数组合实验"""
    print("\n🔬 全参数组合实验")
    print("=" * 80)
    
    # 模型列表
    models = ['Transformer', 'LSTM', 'GRU', 'TCN', 'RF', 'GBR', 'XGB', 'LGBM']
    
    # 特征组合
    feature_configs = [
        ('hist_only', 'true', 'false'),
        ('fcst_only', 'false', 'true'),
        ('both', 'true', 'true'),
        ('none', 'false', 'false')
    ]
    
    # 复杂度组合
    complexities = ['low', 'medium', 'high']
    
    # 统计信息
    total_experiments = len(models) * len(feature_configs) * len(complexities)
    completed = 0
    failed = 0
    
    print(f"📊 总实验数: {total_experiments}")
    print(f"   模型数: {len(models)}")
    print(f"   特征组合: {len(feature_configs)}")
    print(f"   复杂度: {len(complexities)}")
    
    start_time = time.time()
    
    for model in models:
        print(f"\n🎯 开始 {model} 模型实验")
        print("-" * 60)
        
        for feat_desc, hist_weather, forecast in feature_configs:
            print(f"\n📋 特征组合: {feat_desc}")
            
            for complexity in complexities:
                description = f"{model} - {feat_desc} - {complexity}"
                
                if run_experiment(model, hist_weather, forecast, complexity, description):
                    completed += 1
                else:
                    failed += 1
                
                # 显示进度
                progress = (completed + failed) / total_experiments * 100
                elapsed = time.time() - start_time
                eta = elapsed / (completed + failed) * (total_experiments - completed - failed) if (completed + failed) > 0 else 0
                
                print(f"📈 进度: {completed + failed}/{total_experiments} ({progress:.1f}%)")
                print(f"⏱️  已用时间: {elapsed/60:.1f}分钟, 预计剩余: {eta/60:.1f}分钟")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 全参数组合实验完成！")
    print(f"📊 成功: {completed}/{total_experiments}")
    print(f"❌ 失败: {failed}/{total_experiments}")
    print(f"⏱️  总耗时: {total_time/60:.1f}分钟")
    
    return completed, failed

def analyze_all_results():
    """分析所有结果"""
    print("\n📊 分析所有结果...")
    
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
        combined_df.to_csv('result/all_experiments_results.csv', index=False)
        print("✅ 合并结果保存到 result/all_experiments_results.csv")
        
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
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
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
    
    # 7. 模型类型性能对比
    ax7 = axes[2, 0]
    df['model_type'] = df['model'].apply(lambda x: 'DL' if x in ['Transformer', 'LSTM', 'GRU', 'TCN'] else 'ML')
    type_perf = df.groupby('model_type')['rmse'].mean()
    type_perf.plot(kind='bar', ax=ax7, color=['red', 'blue'])
    ax7.set_title('DL vs ML Performance', fontsize=12)
    ax7.set_ylabel('RMSE')
    ax7.tick_params(axis='x', rotation=0)
    
    # 8. 特征重要性热图
    ax8 = axes[2, 1]
    feature_matrix = df.pivot_table(values='rmse', index='use_hist_weather', columns='use_forecast', aggfunc='mean')
    sns.heatmap(feature_matrix, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax8)
    ax8.set_title('Feature Combination Heatmap', fontsize=12)
    ax8.set_xlabel('Use Forecast')
    ax8.set_ylabel('Use Hist Weather')
    
    # 9. 复杂度vs性能
    ax9 = axes[2, 2]
    complexity_rmse = df.groupby(['model', 'model_complexity'])['rmse'].mean().unstack()
    complexity_rmse.plot(kind='bar', ax=ax9, width=0.8)
    ax9.set_title('Complexity vs Performance by Model', fontsize=12)
    ax9.set_ylabel('RMSE')
    ax9.tick_params(axis='x', rotation=45)
    ax9.legend(title='Complexity')
    
    plt.tight_layout()
    plt.savefig('result/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 可视化图表保存到 result/comprehensive_analysis.png")

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
    
    # 按特征组合统计
    print("\n📊 按特征组合统计:")
    feature_stats = df.groupby(['use_hist_weather', 'use_forecast']).agg({
        'test_loss': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std']
    }).round(4)
    print(feature_stats)
    
    # 按复杂度统计
    print("\n📊 按复杂度统计:")
    complexity_stats = df.groupby('model_complexity').agg({
        'test_loss': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'train_time_sec': ['mean', 'std'],
        'param_count': ['mean', 'std']
    }).round(4)
    print(complexity_stats)
    
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
    
    # 按模型找出最佳
    print(f"\n🏆 各模型最佳配置:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        best_model = model_df.loc[model_df['rmse'].idxmin()]
        print(f"   {model}: RMSE={best_model['rmse']:.4f}, 特征={best_model['use_hist_weather']}/{best_model['use_forecast']}")

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction 全参数组合实验 (Colab版)")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 检查环境
    if not check_environment():
        return
    
    # 设置A100优化
    setup_a100()
    
    # 创建结果目录
    if not os.path.exists('result'):
        os.makedirs('result')
        print("✅ 创建result目录")
    
    # 运行全参数组合实验
    completed, failed = run_full_experiments()
    
    # 分析结果
    analyze_all_results()
    
    print(f"\n🎉 所有实验完成！")
    print(f"📊 成功: {completed}")
    print(f"❌ 失败: {failed}")
    print(f"📁 结果保存在: result/ 目录")
    print(f"📊 合并结果: result/all_experiments_results.csv")
    print(f"📊 可视化: result/comprehensive_analysis.png")

if __name__ == "__main__":
    main()
