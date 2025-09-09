#!/usr/bin/env python3
"""
运行实验脚本
自动运行所有模型和消融实验，结果保存到result文件夹
"""

import os
import subprocess
import sys
import time
from datetime import datetime

def run_command(cmd, description):
    """运行命令并记录结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} - 成功")
        print(f"⏱️  耗时: {end_time - start_time:.2f}秒")
        if result.stdout:
            print("输出:")
            print(result.stdout[-500:])  # 只显示最后500个字符
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

def run_quick_test():
    """运行快速测试"""
    print("\n🧪 阶段1: 快速测试")
    
    # 测试TCN模型
    cmd = [
        sys.executable, 'main.py',
        '--config', 'config/default.yaml',
        '--model', 'TCN',
        '--use_hist_weather', 'true',
        '--use_forecast', 'false',
        '--model_complexity', 'low',
        '--past_days', '1'
    ]
    
    return run_command(cmd, "TCN快速测试")

def run_model_comparison():
    """运行模型对比实验"""
    print("\n🔄 阶段2: 模型对比实验")
    
    models = [
        ('Transformer', 'medium'),
        ('LSTM', 'medium'),
        ('GRU', 'medium'),
        ('TCN', 'medium'),
        ('RF', 'medium'),
        ('GBR', 'medium'),
        ('XGB', 'medium'),
        ('LGBM', 'medium')
    ]
    
    success_count = 0
    total_count = len(models)
    
    for model, complexity in models:
        cmd = [
            sys.executable, 'main.py',
            '--config', 'config/default.yaml',
            '--model', model,
            '--use_hist_weather', 'true',
            '--use_forecast', 'true',
            '--model_complexity', complexity,
            '--past_days', '3'
        ]
        
        if run_command(cmd, f"{model}模型训练"):
            success_count += 1
    
    print(f"\n📊 模型对比完成: {success_count}/{total_count} 成功")
    return success_count == total_count

def run_feature_ablation():
    """运行特征消融实验"""
    print("\n🔬 阶段3: 特征消融实验")
    
    # 使用最佳模型进行特征消融
    best_model = 'Transformer'
    complexity = 'medium'
    
    feature_configs = [
        ('全部特征', 'true', 'true'),
        ('仅历史天气', 'true', 'false'),
        ('仅预测天气', 'false', 'true'),
        ('仅时间编码', 'false', 'false')
    ]
    
    success_count = 0
    total_count = len(feature_configs)
    
    for desc, hist_weather, forecast in feature_configs:
        cmd = [
            sys.executable, 'main.py',
            '--config', 'config/default.yaml',
            '--model', best_model,
            '--use_hist_weather', hist_weather,
            '--use_forecast', forecast,
            '--model_complexity', complexity,
            '--past_days', '3'
        ]
        
        if run_command(cmd, f"{best_model} - {desc}"):
            success_count += 1
    
    print(f"\n📊 特征消融完成: {success_count}/{total_count} 成功")
    return success_count == total_count

def run_time_window_test():
    """运行时间窗口测试"""
    print("\n⏰ 阶段4: 时间窗口测试")
    
    model = 'Transformer'
    complexity = 'medium'
    
    time_windows = [1, 3, 7]
    
    success_count = 0
    total_count = len(time_windows)
    
    for past_days in time_windows:
        cmd = [
            sys.executable, 'main.py',
            '--config', 'config/default.yaml',
            '--model', model,
            '--use_hist_weather', 'true',
            '--use_forecast', 'true',
            '--model_complexity', complexity,
            '--past_days', str(past_days)
        ]
        
        if run_command(cmd, f"{model} - {past_days}天历史数据"):
            success_count += 1
    
    print(f"\n📊 时间窗口测试完成: {success_count}/{total_count} 成功")
    return success_count == total_count

def run_complexity_test():
    """运行复杂度测试"""
    print("\n⚙️ 阶段5: 模型复杂度测试")
    
    model = 'Transformer'
    
    complexities = ['low', 'medium', 'high']
    
    success_count = 0
    total_count = len(complexities)
    
    for complexity in complexities:
        cmd = [
            sys.executable, 'main.py',
            '--config', 'config/default.yaml',
            '--model', model,
            '--use_hist_weather', 'true',
            '--use_forecast', 'true',
            '--model_complexity', complexity,
            '--past_days', '3'
        ]
        
        if run_command(cmd, f"{model} - {complexity}复杂度"):
            success_count += 1
    
    print(f"\n📊 复杂度测试完成: {success_count}/{total_count} 成功")
    return success_count == total_count

def analyze_results():
    """分析结果"""
    print("\n📊 分析结果...")
    
    try:
        import pandas as pd
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
        
        # 显示结果摘要
        print("\n=== 模型性能对比 ===")
        print(combined_df[['model', 'use_hist_weather', 'use_forecast', 'past_hours', 'test_loss', 'rmse', 'mae', 'train_time_sec']].round(4))
        
        # 找出最佳模型
        best_model = combined_df.loc[combined_df['rmse'].idxmin()]
        print(f"\n🏆 最佳模型: {best_model['model']}")
        print(f"   RMSE: {best_model['rmse']:.4f}")
        print(f"   MAE: {best_model['mae']:.4f}")
        print(f"   训练时间: {best_model['train_time_sec']:.2f}秒")
        
    except ImportError:
        print("❌ 需要安装pandas来分析结果")
    except Exception as e:
        print(f"❌ 分析结果时出错: {e}")

def main():
    """主函数"""
    print("🚀 SolarPV-Prediction 实验运行器")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 创建结果目录
    create_result_dir()
    
    # 检查数据文件
    if not os.path.exists('data/Project1033.csv'):
        print("❌ 数据文件不存在: data/Project1033.csv")
        return
    
    print("✅ 数据文件检查通过")
    
    # 运行实验
    start_time = time.time()
    
    # 阶段1: 快速测试
    if not run_quick_test():
        print("❌ 快速测试失败，停止实验")
        return
    
    # 阶段2: 模型对比
    run_model_comparison()
    
    # 阶段3: 特征消融
    run_feature_ablation()
    
    # 阶段4: 时间窗口测试
    run_time_window_test()
    
    # 阶段5: 复杂度测试
    run_complexity_test()
    
    # 分析结果
    analyze_results()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 所有实验完成！")
    print(f"⏱️  总耗时: {total_time/60:.2f}分钟")
    print(f"📁 结果保存在: result/ 目录")
    print(f"📊 合并结果: result/combined_results.csv")

if __name__ == "__main__":
    main()
