#!/usr/bin/env python3
"""
运行Project1140消融实验
执行所有360个实验配置，生成完整的消融实验结果
"""

import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ablation_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AblationExperimentRunner:
    """消融实验运行器"""
    
    def __init__(self, config_dir="config/ablation", results_dir="results/ablation", max_workers=4):
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.max_workers = max_workers
        
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化结果记录
        self.results_summary = []
        self.failed_configs = []
        
    def load_configs(self):
        """加载所有配置文件"""
        config_files = list(self.config_dir.glob("*.yaml"))
        # 排除索引文件
        config_files = [f for f in config_files if f.name != "config_index.yaml"]
        
        logger.info(f"找到 {len(config_files)} 个配置文件")
        return sorted(config_files)
    
    def run_single_experiment(self, config_file):
        """运行单个实验配置"""
        try:
            logger.info(f"开始运行实验: {config_file.name}")
            
            # 加载配置
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 创建实验特定的结果目录
            exp_name = config_file.stem
            exp_results_dir = self.results_dir / exp_name
            exp_results_dir.mkdir(exist_ok=True)
            
            # 更新配置中的保存路径
            config['save_dir'] = str(exp_results_dir)
            config['save_options']['save_model'] = True
            config['save_options']['save_predictions'] = True
            config['save_options']['save_excel_results'] = True
            
            # 保存更新的配置
            updated_config_file = exp_results_dir / "config.yaml"
            with open(updated_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 运行实验
            start_time = time.time()
            
            # 调用主程序
            cmd = [sys.executable, "main.py", "--config", str(updated_config_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ 实验完成: {exp_name} (耗时: {duration:.1f}秒)")
                
                # 收集结果
                result_info = self.collect_experiment_results(exp_name, exp_results_dir, duration)
                return result_info
            else:
                logger.error(f"❌ 实验失败: {exp_name}")
                logger.error(f"错误输出: {result.stderr}")
                return {
                    'config_name': exp_name,
                    'status': 'failed',
                    'error': result.stderr,
                    'duration': duration
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ 实验超时: {exp_name}")
            return {
                'config_name': exp_name,
                'status': 'timeout',
                'error': 'Experiment timeout (1 hour)',
                'duration': 3600
            }
        except Exception as e:
            logger.error(f"💥 实验异常: {exp_name} - {str(e)}")
            return {
                'config_name': exp_name,
                'status': 'error',
                'error': str(e),
                'duration': 0
            }
    
    def collect_experiment_results(self, exp_name, exp_results_dir, duration):
        """收集单个实验的结果"""
        result_info = {
            'config_name': exp_name,
            'status': 'completed',
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        # 尝试读取结果文件
        excel_file = exp_results_dir / "results.xlsx"
        if excel_file.exists():
            try:
                df = pd.read_excel(excel_file)
                if not df.empty:
                    # 提取关键指标
                    result_info.update({
                        'mae': df.get('MAE', [np.nan])[0] if 'MAE' in df.columns else np.nan,
                        'rmse': df.get('RMSE', [np.nan])[0] if 'RMSE' in df.columns else np.nan,
                        'r2': df.get('R²', [np.nan])[0] if 'R²' in df.columns else np.nan,
                        'mape': df.get('MAPE', [np.nan])[0] if 'MAPE' in df.columns else np.nan
                    })
            except Exception as e:
                logger.warning(f"无法读取结果文件 {excel_file}: {e}")
        
        return result_info
    
    def run_parallel_experiments(self, config_files, max_workers=None):
        """并行运行实验"""
        if max_workers is None:
            max_workers = self.max_workers
            
        logger.info(f"开始并行运行 {len(config_files)} 个实验 (最大并发: {max_workers})")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(self.run_single_experiment, config_file): config_file
                for config_file in config_files
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_config):
                config_file = future_to_config[future]
                try:
                    result = future.result()
                    self.results_summary.append(result)
                    
                    if result['status'] != 'completed':
                        self.failed_configs.append(result)
                    
                    completed += 1
                    logger.info(f"进度: {completed}/{len(config_files)} ({completed/len(config_files)*100:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"实验 {config_file.name} 执行异常: {e}")
                    self.failed_configs.append({
                        'config_name': config_file.name,
                        'status': 'exception',
                        'error': str(e),
                        'duration': 0
                    })
        
        logger.info(f"所有实验完成! 成功: {len(self.results_summary)-len(self.failed_configs)}, 失败: {len(self.failed_configs)}")
    
    def save_results_summary(self):
        """保存实验结果摘要"""
        # 保存详细结果
        summary_file = self.results_dir / "experiment_summary.csv"
        df_summary = pd.DataFrame(self.results_summary)
        df_summary.to_csv(summary_file, index=False, encoding='utf-8')
        logger.info(f"实验结果摘要已保存: {summary_file}")
        
        # 保存失败配置
        if self.failed_configs:
            failed_file = self.results_dir / "failed_configs.csv"
            df_failed = pd.DataFrame(self.failed_configs)
            df_failed.to_csv(failed_file, index=False, encoding='utf-8')
            logger.info(f"失败配置已保存: {failed_file}")
        
        # 生成统计报告
        self.generate_statistics_report()
    
    def generate_statistics_report(self):
        """生成统计报告"""
        report_file = self.results_dir / "statistics_report.md"
        
        df = pd.DataFrame(self.results_summary)
        completed_df = df[df['status'] == 'completed']
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Project1140 消融实验结果统计报告\n\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 总体统计
            f.write("## 总体统计\n\n")
            f.write(f"- **总配置数**: {len(df)}\n")
            f.write(f"- **成功完成**: {len(completed_df)}\n")
            f.write(f"- **失败配置**: {len(df) - len(completed_df)}\n")
            f.write(f"- **成功率**: {len(completed_df)/len(df)*100:.1f}%\n\n")
            
            if len(completed_df) > 0:
                # 性能统计
                f.write("## 性能指标统计\n\n")
                metrics = ['mae', 'rmse', 'r2', 'mape']
                for metric in metrics:
                    if metric in completed_df.columns:
                        values = completed_df[metric].dropna()
                        if len(values) > 0:
                            f.write(f"### {metric.upper()}\n")
                            f.write(f"- **平均值**: {values.mean():.4f}\n")
                            f.write(f"- **标准差**: {values.std():.4f}\n")
                            f.write(f"- **最小值**: {values.min():.4f}\n")
                            f.write(f"- **最大值**: {values.max():.4f}\n\n")
                
                # 模型比较
                f.write("## 模型性能比较\n\n")
                if 'config_name' in completed_df.columns:
                    # 提取模型名称
                    completed_df['model'] = completed_df['config_name'].str.split('_').str[0]
                    
                    for metric in metrics:
                        if metric in completed_df.columns:
                            model_stats = completed_df.groupby('model')[metric].agg(['mean', 'std', 'count']).round(4)
                            f.write(f"### {metric.upper()} 按模型分组\n\n")
                            f.write(model_stats.to_string())
                            f.write("\n\n")
            
            # 失败分析
            if len(self.failed_configs) > 0:
                f.write("## 失败配置分析\n\n")
                failed_df = pd.DataFrame(self.failed_configs)
                status_counts = failed_df['status'].value_counts()
                f.write("### 失败类型分布\n\n")
                for status, count in status_counts.items():
                    f.write(f"- **{status}**: {count}\n")
                f.write("\n")
        
        logger.info(f"统计报告已生成: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="运行Project1140消融实验")
    parser.add_argument("--config-dir", default="config/ablation", help="配置文件目录")
    parser.add_argument("--results-dir", default="results/ablation", help="结果保存目录")
    parser.add_argument("--max-workers", type=int, default=4, help="最大并发数")
    parser.add_argument("--start-from", type=int, default=0, help="从第几个配置开始运行")
    parser.add_argument("--max-configs", type=int, help="最大运行配置数")
    parser.add_argument("--model-filter", help="只运行指定模型 (如: Transformer,LSTM)")
    parser.add_argument("--dry-run", action="store_true", help="只列出配置，不实际运行")
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = AblationExperimentRunner(
        config_dir=args.config_dir,
        results_dir=args.results_dir,
        max_workers=args.max_workers
    )
    
    # 加载配置
    config_files = runner.load_configs()
    
    # 应用过滤器
    if args.model_filter:
        models = args.model_filter.split(',')
        config_files = [f for f in config_files if any(model in f.name for model in models)]
        logger.info(f"应用模型过滤器后剩余 {len(config_files)} 个配置")
    
    # 应用范围限制
    if args.start_from > 0:
        config_files = config_files[args.start_from:]
        logger.info(f"从第 {args.start_from} 个配置开始运行")
    
    if args.max_configs:
        config_files = config_files[:args.max_configs]
        logger.info(f"限制运行 {args.max_configs} 个配置")
    
    if args.dry_run:
        logger.info("Dry run模式 - 只列出配置:")
        for i, config_file in enumerate(config_files):
            logger.info(f"{i+1:3d}. {config_file.name}")
        return
    
    # 运行实验
    start_time = time.time()
    runner.run_parallel_experiments(config_files)
    end_time = time.time()
    
    # 保存结果
    runner.save_results_summary()
    
    total_duration = end_time - start_time
    logger.info(f"🎉 所有实验完成! 总耗时: {total_duration/3600:.1f} 小时")

if __name__ == "__main__":
    main()
