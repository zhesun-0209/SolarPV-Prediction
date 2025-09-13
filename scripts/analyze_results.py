#!/usr/bin/env python3
"""
消融实验结果分析脚本
用于分析360个消融实验的结果，生成统计报告和可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import yaml

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AblationResultsAnalyzer:
    """消融实验结果分析器"""
    
    def __init__(self, results_dir="results/ablation"):
        self.results_dir = Path(results_dir)
        self.summary_file = self.results_dir / "experiment_summary.csv"
        self.failed_file = self.results_dir / "failed_configs.csv"
        
    def load_results(self):
        """加载实验结果"""
        if not self.summary_file.exists():
            print(f"❌ 结果文件不存在: {self.summary_file}")
            return None
        
        df = pd.read_csv(self.summary_file)
        print(f"✅ 加载了 {len(df)} 个实验结果")
        return df
    
    def load_failed_configs(self):
        """加载失败配置"""
        if not self.failed_file.exists():
            print("✅ 没有失败的配置")
            return pd.DataFrame()
        
        df = pd.read_csv(self.failed_file)
        print(f"⚠️  发现 {len(df)} 个失败的配置")
        return df
    
    def analyze_overall_performance(self, df):
        """分析总体性能"""
        print("\n📊 总体性能分析")
        print("=" * 50)
        
        completed_df = df[df['status'] == 'completed']
        print(f"成功完成的实验: {len(completed_df)}")
        print(f"失败的实验: {len(df) - len(completed_df)}")
        print(f"成功率: {len(completed_df)/len(df)*100:.1f}%")
        
        if len(completed_df) > 0:
            metrics = ['mae', 'rmse', 'r2']
            for metric in metrics:
                if metric in completed_df.columns:
                    values = completed_df[metric].dropna()
                    if len(values) > 0:
                        print(f"\n{metric.upper()}:")
                        print(f"  平均值: {values.mean():.4f}")
                        print(f"  标准差: {values.std():.4f}")
                        print(f"  最小值: {values.min():.4f}")
                        print(f"  最大值: {values.max():.4f}")
    
    def analyze_model_performance(self, df):
        """分析模型性能"""
        print("\n🤖 模型性能分析")
        print("=" * 50)
        
        completed_df = df[df['status'] == 'completed'].copy()
        if len(completed_df) == 0:
            print("没有成功完成的实验")
            return
        
        # 提取模型名称
        completed_df['model'] = completed_df['config_name'].str.split('_').str[0]
        
        # 按模型分组统计
        model_stats = completed_df.groupby('model').agg({
            'mae': ['mean', 'std', 'count'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'duration': 'mean'
        }).round(4)
        
        print("模型性能统计:")
        print(model_stats)
        
        # 找出最佳模型
        best_models = {}
        for metric in ['mae', 'rmse']:
            if metric in completed_df.columns:
                best_idx = completed_df[metric].idxmin()
                best_model = completed_df.loc[best_idx, 'model']
                best_value = completed_df.loc[best_idx, metric]
                best_models[metric] = (best_model, best_value)
        
        print(f"\n最佳模型:")
        for metric, (model, value) in best_models.items():
            print(f"  {metric.upper()}: {model} ({value:.4f})")
    
    def analyze_input_features(self, df):
        """分析输入特征效果"""
        print("\n🔍 输入特征效果分析")
        print("=" * 50)
        
        completed_df = df[df['status'] == 'completed'].copy()
        if len(completed_df) == 0:
            print("没有成功完成的实验")
            return
        
        # 解析输入特征类别
        def extract_input_category(config_name):
            if 'PV_plus_NWP_plus' in config_name:
                return 'PV+NWP+'
            elif 'PV_plus_NWP' in config_name:
                return 'PV+NWP'
            elif 'PV_plus_HW' in config_name:
                return 'PV+HW'
            elif 'NWP_plus' in config_name and 'PV' not in config_name:
                return 'NWP+'
            elif 'NWP' in config_name and 'PV' not in config_name:
                return 'NWP'
            elif 'PV' in config_name and 'plus' not in config_name:
                return 'PV'
            else:
                return 'Unknown'
        
        completed_df['input_category'] = completed_df['config_name'].apply(extract_input_category)
        
        # 按输入特征分组统计
        input_stats = completed_df.groupby('input_category').agg({
            'mae': ['mean', 'std', 'count'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std']
        }).round(4)
        
        print("输入特征效果统计:")
        print(input_stats)
        
        # 找出最佳输入特征组合
        best_inputs = {}
        for metric in ['mae', 'rmse']:
            if metric in completed_df.columns:
                best_idx = completed_df[metric].idxmin()
                best_input = completed_df.loc[best_idx, 'input_category']
                best_value = completed_df.loc[best_idx, metric]
                best_inputs[metric] = (best_input, best_value)
        
        print(f"\n最佳输入特征组合:")
        for metric, (input_cat, value) in best_inputs.items():
            print(f"  {metric.upper()}: {input_cat} ({value:.4f})")
    
    def analyze_hyperparameters(self, df):
        """分析超参数效果"""
        print("\n⚙️ 超参数效果分析")
        print("=" * 50)
        
        completed_df = df[df['status'] == 'completed'].copy()
        if len(completed_df) == 0:
            print("没有成功完成的实验")
            return
        
        # 解析超参数
        def extract_hyperparams(config_name):
            parts = config_name.split('_')
            model = parts[0]
            complexity = parts[1]
            
            # 提取回看窗口
            lookback = None
            for part in parts:
                if part.endswith('h'):
                    lookback = int(part[:-1])
                    break
            
            # 提取时间编码
            te = 'TE' in config_name
            
            return {
                'model': model,
                'complexity': complexity,
                'lookback': lookback,
                'time_encoding': te
            }
        
        # 添加超参数列
        hyperparams_df = completed_df['config_name'].apply(extract_hyperparams)
        for col in ['model', 'complexity', 'lookback', 'time_encoding']:
            completed_df[col] = hyperparams_df.apply(lambda x: x[col])
        
        # 分析复杂度效果
        complexity_stats = completed_df.groupby('complexity')['mae'].agg(['mean', 'std', 'count']).round(4)
        print("模型复杂度效果:")
        print(complexity_stats)
        
        # 分析回看窗口效果
        lookback_stats = completed_df.groupby('lookback')['mae'].agg(['mean', 'std', 'count']).round(4)
        print("\n回看窗口效果:")
        print(lookback_stats)
        
        # 分析时间编码效果
        te_stats = completed_df.groupby('time_encoding')['mae'].agg(['mean', 'std', 'count']).round(4)
        print("\n时间编码效果:")
        print(te_stats)
    
    def create_visualizations(self, df):
        """创建可视化图表"""
        print("\n📈 生成可视化图表")
        print("=" * 50)
        
        completed_df = df[df['status'] == 'completed'].copy()
        if len(completed_df) == 0:
            print("没有成功完成的实验，跳过可视化")
            return
        
        # 提取模型名称
        completed_df['model'] = completed_df['config_name'].str.split('_').str[0]
        
        # 创建图表目录
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. 模型性能对比箱线图
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=completed_df, x='model', y='mae')
        plt.title('不同模型的MAE分布')
        plt.xlabel('模型')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "model_mae_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 模型性能对比散点图
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=completed_df, x='rmse', y='mae', hue='model', alpha=0.7)
        plt.title('RMSE vs MAE (按模型着色)')
        plt.xlabel('RMSE')
        plt.ylabel('MAE')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(plots_dir / "rmse_vs_mae.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 训练时间 vs 性能
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=completed_df, x='duration', y='mae', hue='model', alpha=0.7)
        plt.title('训练时间 vs MAE')
        plt.xlabel('训练时间 (秒)')
        plt.ylabel('MAE')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(plots_dir / "training_time_vs_mae.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 图表已保存到: {plots_dir}")
    
    def generate_report(self, df, failed_df):
        """生成分析报告"""
        print("\n📝 生成分析报告")
        print("=" * 50)
        
        report_file = self.results_dir / "analysis_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Project1140 消融实验结果分析报告\n\n")
            f.write(f"**分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 总体统计
            f.write("## 总体统计\n\n")
            completed_df = df[df['status'] == 'completed']
            f.write(f"- **总实验数**: {len(df)}\n")
            f.write(f"- **成功完成**: {len(completed_df)}\n")
            f.write(f"- **失败实验**: {len(df) - len(completed_df)}\n")
            f.write(f"- **成功率**: {len(completed_df)/len(df)*100:.1f}%\n\n")
            
            if len(completed_df) > 0:
                # 性能统计
                f.write("## 性能指标统计\n\n")
                metrics = ['mae', 'rmse', 'r2']
                for metric in metrics:
                    if metric in completed_df.columns:
                        values = completed_df[metric].dropna()
                        if len(values) > 0:
                            f.write(f"### {metric.upper()}\n")
                            f.write(f"- **平均值**: {values.mean():.4f}\n")
                            f.write(f"- **标准差**: {values.std():.4f}\n")
                            f.write(f"- **最小值**: {values.min():.4f}\n")
                            f.write(f"- **最大值**: {values.max():.4f}\n\n")
                
                # 最佳结果
                f.write("## 最佳结果\n\n")
                for metric in ['mae', 'rmse']:
                    if metric in completed_df.columns:
                        best_idx = completed_df[metric].idxmin()
                        best_config = completed_df.loc[best_idx, 'config_name']
                        best_value = completed_df.loc[best_idx, metric]
                        f.write(f"- **最佳{metric.upper()}**: {best_config} ({best_value:.4f})\n")
            
            # 失败分析
            if len(failed_df) > 0:
                f.write("\n## 失败分析\n\n")
                status_counts = failed_df['status'].value_counts()
                f.write("### 失败类型分布\n\n")
                for status, count in status_counts.items():
                    f.write(f"- **{status}**: {count}\n")
        
        print(f"✅ 分析报告已保存: {report_file}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("🔍 开始分析消融实验结果")
        print("=" * 50)
        
        # 加载数据
        df = self.load_results()
        if df is None:
            return
        
        failed_df = self.load_failed_configs()
        
        # 运行分析
        self.analyze_overall_performance(df)
        self.analyze_model_performance(df)
        self.analyze_input_features(df)
        self.analyze_hyperparameters(df)
        self.create_visualizations(df)
        self.generate_report(df, failed_df)
        
        print("\n🎉 分析完成！")
        print(f"📁 结果保存在: {self.results_dir}")

def main():
    parser = argparse.ArgumentParser(description="分析消融实验结果")
    parser.add_argument("--results-dir", default="results/ablation", help="结果目录")
    
    args = parser.parse_args()
    
    analyzer = AblationResultsAnalyzer(args.results_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
