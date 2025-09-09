#!/usr/bin/env python3
"""
全参数组合实验脚本
运行所有厂、所有模型、所有参数组合
"""

import os
import subprocess
import sys
import time
from datetime import datetime
import pandas as pd

def run_command(cmd, description):
    """运行命令并记录结果"""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} - 成功")
        print(f"⏱️  耗时: {end_time - start_time:.2f}秒")
        if result.stdout:
            # 只显示最后几行输出
            lines = result.stdout.strip().split('\n')
            for line in lines[-3:]:
                if line.strip():
                    print(f"   {line}")
    else:
        print(f"❌ {description} - 失败")
        print(f"错误: {result.stderr}")
    
    return result.returncode == 0

def create_result_dir():
    """创建结果目录"""
    if not os.path.exists('result'):
        os.makedirs('result')
        print("✅ 创建result目录")
    else:
        print("✅ result目录已存在")

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
                
                cmd = [
                    sys.executable, 'main.py',
                    '--config', 'config/default.yaml',
                    '--model', model,
                    '--use_hist_weather', hist_weather,
                    '--use_forecast', forecast,
                    '--model_complexity', complexity,
                    '--past_days', '3'
                ]
                
                if run_command(cmd, description):
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
        
        # 按模型分组统计
        print("\n=== 按模型统计 ===")
        model_stats = combined_df.groupby('model').agg({
            'test_loss': ['mean', 'std', 'min'],
            'rmse': ['mean', 'std', 'min'],
            'mae': ['mean', 'std', 'min'],
            'train_time_sec': ['mean', 'std'],
            'param_count': ['mean', 'std']
        }).round(4)
        print(model_stats)
        
        # 按特征组合统计
        print("\n=== 按特征组合统计 ===")
        feature_stats = combined_df.groupby(['use_hist_weather', 'use_forecast']).agg({
            'test_loss': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std']
        }).round(4)
        print(feature_stats)
        
        # 按复杂度统计
        print("\n=== 按复杂度统计 ===")
        complexity_stats = combined_df.groupby('model_complexity').agg({
            'test_loss': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'train_time_sec': ['mean', 'std'],
            'param_count': ['mean', 'std']
        }).round(4)
        print(complexity_stats)
        
        # 找出最佳配置
        best_overall = combined_df.loc[combined_df['rmse'].idxmin()]
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
        for model in combined_df['model'].unique():
            model_df = combined_df[combined_df['model'] == model]
            best_model = model_df.loc[model_df['rmse'].idxmin()]
            print(f"   {model}: RMSE={best_model['rmse']:.4f}, 特征={best_model['use_hist_weather']}/{best_model['use_forecast']}")
        
    except Exception as e:
        print(f"❌ 分析结果时出错: {e}")

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction 全参数组合实验")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 创建结果目录
    create_result_dir()
    
    # 检查数据文件
    if not os.path.exists('data/Project1033.csv'):
        print("❌ 数据文件不存在: data/Project1033.csv")
        return
    
    print("✅ 数据文件检查通过")
    
    # 运行全参数组合实验
    completed, failed = run_full_experiments()
    
    # 分析结果
    analyze_all_results()
    
    print(f"\n🎉 所有实验完成！")
    print(f"📊 成功: {completed}")
    print(f"❌ 失败: {failed}")
    print(f"📁 结果保存在: result/ 目录")
    print(f"📊 合并结果: result/all_experiments_results.csv")

if __name__ == "__main__":
    main()
