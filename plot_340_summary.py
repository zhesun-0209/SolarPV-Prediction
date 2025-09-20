#!/usr/bin/env python3
"""
340个组合结果汇总展示
按照6种情况 × 8个模型的方式展示最佳结果
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 18

def load_results_from_configs(config_dir="config/ablation"):
    """从配置文件加载结果（模拟）"""
    print("📊 加载340个组合的配置信息...")
    
    config_files = []
    for file in os.listdir(config_dir):
        if file.endswith('.yaml'):
            config_files.append(os.path.join(config_dir, file))
    
    results = []
    
    for config_path in config_files:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 获取场景名称
            scenario = get_scenario_name(config)
            
            # 获取模型信息
            model_name = config['model']
            lookback = config['past_hours']
            te = config.get('use_time_encoding', False)
            complexity = config.get('model_complexity', 'low')
            
            # 模拟结果（实际使用时需要从实验结果中加载）
            results.append({
                'model': model_name,
                'scenario': scenario,
                'lookback': lookback,
                'te': te,
                'complexity': complexity,
                'config_path': config_path,
                'mse': np.random.uniform(0.1, 2.0),  # 模拟MSE
                'rmse': np.random.uniform(0.3, 1.4),  # 模拟RMSE
                'mae': np.random.uniform(0.2, 1.0),   # 模拟MAE
                'r_square': np.random.uniform(0.6, 0.95)  # 模拟R²
            })
            
        except Exception as e:
            print(f"⚠️ 无法加载配置文件 {config_path}: {e}")
            continue
    
    print(f"✅ 加载了 {len(results)} 个配置")
    return results

def get_scenario_name(config):
    """根据配置获取场景名称"""
    use_pv = config.get('use_pv', False)
    use_forecast = config.get('use_forecast', False)
    use_hist_weather = config.get('use_hist_weather', False)
    use_ideal_nwp = config.get('use_ideal_nwp', False)
    
    if use_pv and use_hist_weather:
        return 'PV+HW'
    elif use_pv and use_forecast and use_ideal_nwp:
        return 'PV+NWP+'
    elif use_pv and use_forecast:
        return 'PV+NWP'
    elif use_pv:
        return 'PV'
    elif use_forecast and use_ideal_nwp:
        return 'NWP+'
    elif use_forecast:
        return 'NWP'
    else:
        return 'Unknown'

def plot_best_results_summary(results, output_dir):
    """绘制最佳结果汇总图"""
    print("🎨 绘制最佳结果汇总图...")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 按场景和模型分组，找到最佳结果（最小MSE）
    best_results = df.loc[df.groupby(['scenario', 'model'])['mse'].idxmin()]
    
    # 创建子图：2行3列（6个场景）
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    scenarios = ['PV', 'PV+NWP', 'PV+NWP+', 'PV+HW', 'NWP', 'NWP+']
    models = ['LSTM', 'GRU', 'TCN', 'Transformer', 'RF', 'XGB', 'LGBM', 'Linear', 'LSR']
    
    # 定义颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    model_colors = dict(zip(models, colors))
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        
        # 获取该场景的最佳结果
        scenario_results = best_results[best_results['scenario'] == scenario]
        
        if len(scenario_results) == 0:
            ax.text(0.5, 0.5, f'{scenario}\n无数据', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{scenario}', fontweight='bold')
            continue
        
        # 按模型排序
        scenario_results = scenario_results.sort_values('model')
        
        # 绘制柱状图
        x_pos = np.arange(len(scenario_results))
        bars = ax.bar(x_pos, scenario_results['mse'], 
                     color=[model_colors.get(model, 'gray') for model in scenario_results['model']],
                     alpha=0.7)
        
        # 设置标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenario_results['model'], rotation=45, ha='right')
        ax.set_ylabel('MSE')
        ax.set_title(f'{scenario} - Best Results by Model', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for j, (bar, mse) in enumerate(zip(bars, scenario_results['mse'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{mse:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 隐藏多余的子图
    for i in range(len(scenarios), len(axes)):
        axes[i].set_visible(False)
    
    # 设置总标题
    plt.suptitle('340 Combinations - Best Results Summary by Scenario', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'best_results_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图片已保存: {output_path}")
    
    plt.close()

def plot_model_performance_heatmap(results, output_dir):
    """绘制模型性能热力图"""
    print("🎨 绘制模型性能热力图...")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 按场景和模型分组，找到最佳结果
    best_results = df.loc[df.groupby(['scenario', 'model'])['mse'].idxmin()]
    
    # 创建透视表
    pivot_table = best_results.pivot_table(values='mse', index='model', columns='scenario', aggfunc='mean')
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
    
    # 设置标签
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    
    # 添加数值标签
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            text = ax.text(j, i, f'{pivot_table.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # 设置标题和标签
    ax.set_title('Model Performance Heatmap (MSE)', fontweight='bold', fontsize=16)
    ax.set_xlabel('Scenario', fontsize=14)
    ax.set_ylabel('Model', fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MSE', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'model_performance_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图片已保存: {output_path}")
    
    plt.close()

def plot_parameter_analysis(results, output_dir):
    """绘制参数分析图"""
    print("🎨 绘制参数分析图...")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 创建子图：2行2列
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Lookback分析
    ax1 = axes[0, 0]
    lookback_analysis = df.groupby('lookback')['mse'].mean()
    ax1.bar(lookback_analysis.index, lookback_analysis.values, alpha=0.7)
    ax1.set_title('MSE by Lookback Hours', fontweight='bold')
    ax1.set_xlabel('Lookback Hours')
    ax1.set_ylabel('Average MSE')
    ax1.grid(True, alpha=0.3)
    
    # 2. Time Encoding分析
    ax2 = axes[0, 1]
    te_analysis = df.groupby('te')['mse'].mean()
    ax2.bar(['No TE', 'TE'], te_analysis.values, alpha=0.7)
    ax2.set_title('MSE by Time Encoding', fontweight='bold')
    ax2.set_xlabel('Time Encoding')
    ax2.set_ylabel('Average MSE')
    ax2.grid(True, alpha=0.3)
    
    # 3. Complexity分析
    ax3 = axes[1, 0]
    complexity_analysis = df.groupby('complexity')['mse'].mean()
    ax3.bar(complexity_analysis.index, complexity_analysis.values, alpha=0.7)
    ax3.set_title('MSE by Model Complexity', fontweight='bold')
    ax3.set_xlabel('Complexity')
    ax3.set_ylabel('Average MSE')
    ax3.grid(True, alpha=0.3)
    
    # 4. 模型性能对比
    ax4 = axes[1, 1]
    model_analysis = df.groupby('model')['mse'].mean().sort_values()
    ax4.bar(range(len(model_analysis)), model_analysis.values, alpha=0.7)
    ax4.set_title('Average MSE by Model', fontweight='bold')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Average MSE')
    ax4.set_xticks(range(len(model_analysis)))
    ax4.set_xticklabels(model_analysis.index, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Parameter Analysis - 340 Combinations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'parameter_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图片已保存: {output_path}")
    
    plt.close()

def main():
    """主函数"""
    print("🚀 绘制340个组合结果汇总...")
    
    # 加载结果
    results = load_results_from_configs()
    
    if len(results) == 0:
        print("❌ 没有找到结果数据")
        return
    
    # 创建输出目录
    output_dir = '340_combinations_summary'
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制各种汇总图
    plot_best_results_summary(results, output_dir)
    plot_model_performance_heatmap(results, output_dir)
    plot_parameter_analysis(results, output_dir)
    
    print(f"\n✅ 所有汇总图生成完成！")
    print(f"📁 图片保存在: {os.path.abspath(output_dir)}")
    
    # 列出生成的文件
    files = os.listdir(output_dir)
    files.sort()
    print(f"\n📋 生成的文件列表:")
    for i, file in enumerate(files, 1):
        print(f"{i:2d}. {file}")

if __name__ == "__main__":
    main()
