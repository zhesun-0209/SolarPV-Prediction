#!/usr/bin/env python3
"""
高性能100个Project消融实验运行器
针对A100 GPU优化的并行策略
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import signal
import threading
import multiprocessing as mp
import torch
import queue
import psutil

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.drive_results_saver import DriveResultsSaver
from utils.checkpoint_manager import CheckpointManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('high_performance_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HighPerformanceExperimentRunner:
    """高性能实验运行器 - 针对A100优化"""
    
    def __init__(self, 
                 drive_path: str = "/content/drive/MyDrive/Solar PV electricity/ablation results",
                 max_gpu_experiments: int = 8,  # GPU并行实验数
                 max_cpu_experiments: int = 16,  # CPU并行实验数
                 batch_size: int = 20):
        self.drive_saver = DriveResultsSaver(drive_path)
        self.checkpoint_manager = CheckpointManager(drive_path)
        
        # 并行策略参数
        self.max_gpu_experiments = max_gpu_experiments
        self.max_cpu_experiments = max_cpu_experiments
        self.batch_size = batch_size
        
        # 创建临时结果目录
        self.temp_results_dir = Path("./temp_results")
        self.temp_results_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行状态
        self.running = True
        self.stats = {
            'total_experiments': 0,
            'completed_experiments': 0,
            'failed_experiments': 0,
            'gpu_experiments': 0,
            'cpu_experiments': 0,
            'start_time': None
        }
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 检测系统资源
        self._detect_system_resources()
        
        logger.info(f"高性能实验运行器初始化完成")
        logger.info(f"GPU并行数: {self.max_gpu_experiments}")
        logger.info(f"CPU并行数: {self.max_cpu_experiments}")
        logger.info(f"批次大小: {self.batch_size}")
    
    def _detect_system_resources(self):
        """检测系统资源"""
        # 检测GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            logger.info(f"🚀 检测到 {gpu_count} 个GPU，总内存: {gpu_memory:.1f}GB")
            
            # 根据GPU内存调整并行数
            if gpu_memory >= 80:  # A100 80GB
                self.max_gpu_experiments = min(16, gpu_count * 4)
                logger.info(f"🎯 A100检测到，优化GPU并行数为: {self.max_gpu_experiments}")
            elif gpu_memory >= 40:  # A100 40GB 或其他高端GPU
                self.max_gpu_experiments = min(12, gpu_count * 3)
                logger.info(f"🎯 高端GPU检测到，优化GPU并行数为: {self.max_gpu_experiments}")
            else:
                self.max_gpu_experiments = min(8, gpu_count * 2)
        else:
            logger.warning("⚠️ 未检测到GPU，将使用CPU模式")
            self.max_gpu_experiments = 0
        
        # 检测CPU
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024**3
        
        # 根据CPU和内存调整CPU并行数
        if memory_gb >= 128:  # 大内存系统
            self.max_cpu_experiments = min(cpu_count * 2, 32)
        elif memory_gb >= 64:  # 中等内存系统
            self.max_cpu_experiments = min(cpu_count, 24)
        else:  # 小内存系统
            self.max_cpu_experiments = min(cpu_count // 2, 16)
        
        logger.info(f"💻 CPU核心数: {cpu_count}, 内存: {memory_gb:.1f}GB")
        logger.info(f"🎯 优化CPU并行数为: {self.max_cpu_experiments}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器，用于优雅关闭"""
        logger.info(f"收到信号 {signum}，开始优雅关闭...")
        self.running = False
    
    def _classify_experiment_type(self, config):
        """分类实验类型（GPU vs CPU）"""
        model = config.get('model', '')
        
        # 深度学习模型使用GPU
        gpu_models = {'LSTM', 'GRU', 'TCN', 'Transformer'}
        if model in gpu_models:
            return 'gpu'
        else:
            return 'cpu'
    
    def _run_gpu_experiment_batch(self, experiments):
        """批量运行GPU实验"""
        if not experiments:
            return []
        
        results = []
        
        # 使用ThreadPoolExecutor进行GPU实验的并行
        with ThreadPoolExecutor(max_workers=self.max_gpu_experiments) as executor:
            future_to_exp = {
                executor.submit(self._run_single_experiment, exp): exp
                for exp in experiments
            }
            
            for future in as_completed(future_to_exp):
                if not self.running:
                    break
                
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'completed':
                        self.stats['gpu_experiments'] += 1
                        logger.info(f"🚀 GPU实验完成: {exp['project_id']} - {exp['config_name']}")
                    else:
                        logger.error(f"❌ GPU实验失败: {exp['project_id']} - {exp['config_name']}")
                    
                except Exception as e:
                    logger.error(f"💥 GPU实验异常: {exp['project_id']} - {exp['config_name']}: {e}")
                    results.append({
                        'project_id': exp['project_id'],
                        'config_name': exp['config_name'],
                        'status': 'error',
                        'error': str(e)
                    })
        
        return results
    
    def _run_cpu_experiment_batch(self, experiments):
        """批量运行CPU实验"""
        if not experiments:
            return []
        
        results = []
        
        # 使用ProcessPoolExecutor进行CPU实验的并行
        with ProcessPoolExecutor(max_workers=self.max_cpu_experiments) as executor:
            future_to_exp = {
                executor.submit(self._run_single_experiment, exp): exp
                for exp in experiments
            }
            
            for future in as_completed(future_to_exp):
                if not self.running:
                    break
                
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'completed':
                        self.stats['cpu_experiments'] += 1
                        logger.info(f"💻 CPU实验完成: {exp['project_id']} - {exp['config_name']}")
                    else:
                        logger.error(f"❌ CPU实验失败: {exp['project_id']} - {exp['config_name']}")
                    
                except Exception as e:
                    logger.error(f"💥 CPU实验异常: {exp['project_id']} - {exp['config_name']}: {e}")
                    results.append({
                        'project_id': exp['project_id'],
                        'config_name': exp['config_name'],
                        'status': 'error',
                        'error': str(e)
                    })
        
        return results
    
    def _run_single_experiment(self, experiment_info):
        """运行单个实验（优化版本）"""
        try:
            project_id = experiment_info['project_id']
            config_name = experiment_info['config_name']
            config = experiment_info['config']
            config_file = experiment_info['file_path']
            
            # 创建实验特定的结果目录
            exp_results_dir = self.temp_results_dir / project_id / config_name
            exp_results_dir.mkdir(parents=True, exist_ok=True)
            
            # 更新配置中的保存路径
            config['save_dir'] = str(exp_results_dir)
            config['save_options']['save_model'] = False
            config['save_options']['save_summary'] = False
            config['save_options']['save_predictions'] = False
            config['save_options']['save_training_log'] = False
            config['save_options']['save_excel_results'] = False
            
            # 优化训练参数（针对A100）
            if self._classify_experiment_type(config) == 'gpu':
                # GPU实验优化
                config['train_params']['batch_size'] = min(64, config['train_params'].get('batch_size', 32) * 2)
                config['train_params']['learning_rate'] = config['train_params'].get('learning_rate', 5e-4) * 1.5
                # 增加混合精度训练
                config['train_params']['use_amp'] = True
            
            # 保存更新的配置
            updated_config_file = exp_results_dir / "config.yaml"
            with open(updated_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 运行实验
            start_time = time.time()
            
            # 调用主程序
            cmd = [sys.executable, "main.py", "--config", str(updated_config_file)]
            
            # 设置环境变量优化
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=1800,  # 30分钟超时（优化后应该更快）
                env=env
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                # 收集结果
                result_data = self._collect_experiment_results(
                    project_id, config_name, config, duration, exp_results_dir
                )
                
                # 保存到Drive
                self.drive_saver.save_experiment_result(project_id, result_data)
                
                return {
                    'project_id': project_id,
                    'config_name': config_name,
                    'status': 'completed',
                    'duration': duration,
                    'result_data': result_data
                }
            else:
                # 保存失败结果
                result_data = {
                    'config_name': config_name,
                    'status': 'failed',
                    'duration': duration,
                    'error_message': result.stderr
                }
                
                self.drive_saver.save_experiment_result(project_id, result_data)
                
                return {
                    'project_id': project_id,
                    'config_name': config_name,
                    'status': 'failed',
                    'duration': duration,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            result_data = {
                'config_name': config_name,
                'status': 'timeout',
                'duration': 1800,
                'error_message': 'Experiment timeout (30 minutes)'
            }
            
            self.drive_saver.save_experiment_result(project_id, result_data)
            
            return {
                'project_id': project_id,
                'config_name': config_name,
                'status': 'timeout',
                'duration': 1800,
                'error': 'Timeout'
            }
        except Exception as e:
            result_data = {
                'config_name': config_name,
                'status': 'error',
                'duration': 0,
                'error_message': str(e)
            }
            
            self.drive_saver.save_experiment_result(project_id, result_data)
            
            return {
                'project_id': project_id,
                'config_name': config_name,
                'status': 'error',
                'duration': 0,
                'error': str(e)
            }
    
    def _collect_experiment_results(self, project_id: str, config_name: str, 
                                  config: dict, duration: float, 
                                  exp_results_dir: Path) -> dict:
        """收集实验结果"""
        result_data = {
            'config_name': config_name,
            'status': 'completed',
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            
            # 配置信息
            'model': config.get('model', ''),
            'model_complexity': config.get('model_complexity', ''),
            'input_category': self._extract_input_category(config_name),
            'lookback_hours': config.get('past_hours', 24),
            'use_time_encoding': config.get('use_time_encoding', False),
            
            # 默认值
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'mape': np.nan,
            'train_time_sec': np.nan,
            'inference_time_sec': np.nan,
            'param_count': np.nan,
            'samples_count': np.nan
        }
        
        # 尝试从结果文件中读取指标
        try:
            results_files = list(exp_results_dir.glob("*.csv"))
            if results_files:
                df = pd.read_csv(results_files[0])
                
                if 'MAE' in df.columns:
                    result_data['mae'] = df['MAE'].iloc[0] if len(df) > 0 else np.nan
                if 'RMSE' in df.columns:
                    result_data['rmse'] = df['RMSE'].iloc[0] if len(df) > 0 else np.nan
                if 'R²' in df.columns:
                    result_data['r2'] = df['R²'].iloc[0] if len(df) > 0 else np.nan
                if 'MAPE' in df.columns:
                    result_data['mape'] = df['MAPE'].iloc[0] if len(df) > 0 else np.nan
                
        except Exception as e:
            logger.warning(f"无法读取结果文件: {e}")
        
        return result_data
    
    def _extract_input_category(self, config_name: str) -> str:
        """从配置名称中提取输入特征类别"""
        if 'PV_plus_NWP_plus' in config_name:
            return 'PV_plus_NWP_plus'
        elif 'PV_plus_NWP' in config_name:
            return 'PV_plus_NWP'
        elif 'PV_plus_HW' in config_name:
            return 'PV_plus_HW'
        elif 'NWP_plus' in config_name and 'PV' not in config_name:
            return 'NWP_plus'
        elif 'NWP' in config_name and 'PV' not in config_name:
            return 'NWP'
        elif 'PV' in config_name and 'plus' not in config_name:
            return 'PV'
        else:
            return 'Unknown'
    
    def run_high_performance_experiments(self, project_ids: list = None):
        """运行高性能实验"""
        if project_ids is None:
            project_ids = self.checkpoint_manager.get_incomplete_projects()
        
        if not project_ids:
            logger.info("🎉 所有Project实验已完成!")
            return
        
        logger.info(f"🚀 开始高性能运行 {len(project_ids)} 个Project的实验")
        
        self.stats['start_time'] = datetime.now()
        
        # 收集所有待执行的实验
        all_experiments = []
        for project_id in project_ids:
            pending_configs = self.checkpoint_manager.get_pending_experiments(project_id)
            for config_info in pending_configs:
                all_experiments.append({
                    'project_id': project_id,
                    'config_name': config_info['name'],
                    'config': config_info['config'],
                    'file_path': config_info['file_path']
                })
        
        self.stats['total_experiments'] = len(all_experiments)
        logger.info(f"📊 总实验数: {self.stats['total_experiments']}")
        
        # 分类实验
        gpu_experiments = []
        cpu_experiments = []
        
        for exp in all_experiments:
            exp_type = self._classify_experiment_type(exp['config'])
            if exp_type == 'gpu':
                gpu_experiments.append(exp)
            else:
                cpu_experiments.append(exp)
        
        logger.info(f"🎯 GPU实验: {len(gpu_experiments)} 个")
        logger.info(f"💻 CPU实验: {len(cpu_experiments)} 个")
        
        # 创建两个线程分别处理GPU和CPU实验
        gpu_thread = threading.Thread(target=self._run_gpu_experiments_loop, args=(gpu_experiments,))
        cpu_thread = threading.Thread(target=self._run_cpu_experiments_loop, args=(cpu_experiments,))
        
        # 启动线程
        gpu_thread.start()
        cpu_thread.start()
        
        # 等待完成
        gpu_thread.join()
        cpu_thread.join()
        
        # 最终统计
        self._print_final_statistics()
    
    def _run_gpu_experiments_loop(self, experiments):
        """GPU实验循环"""
        for i in range(0, len(experiments), self.batch_size):
            if not self.running:
                break
            
            batch = experiments[i:i + self.batch_size]
            logger.info(f"🚀 处理GPU批次 {i//self.batch_size + 1}: {len(batch)} 个实验")
            
            results = self._run_gpu_experiment_batch(batch)
            
            # 更新统计
            for result in results:
                self.stats['completed_experiments'] += 1
                if result['status'] != 'completed':
                    self.stats['failed_experiments'] += 1
            
            # 批次间同步
            self.drive_saver.sync_to_drive()
            
            logger.info(f"📊 GPU进度: {self.stats['gpu_experiments']} 完成")
    
    def _run_cpu_experiments_loop(self, experiments):
        """CPU实验循环"""
        for i in range(0, len(experiments), self.batch_size):
            if not self.running:
                break
            
            batch = experiments[i:i + self.batch_size]
            logger.info(f"💻 处理CPU批次 {i//self.batch_size + 1}: {len(batch)} 个实验")
            
            results = self._run_cpu_experiment_batch(batch)
            
            # 更新统计
            for result in results:
                self.stats['completed_experiments'] += 1
                if result['status'] != 'completed':
                    self.stats['failed_experiments'] += 1
            
            # 批次间同步
            self.drive_saver.sync_to_drive()
            
            logger.info(f"📊 CPU进度: {self.stats['cpu_experiments']} 完成")
    
    def _print_final_statistics(self):
        """打印最终统计信息"""
        if self.stats['start_time']:
            total_duration = (datetime.now() - self.stats['start_time']).total_seconds()
            logger.info("\n" + "=" * 60)
            logger.info("🎉 高性能实验完成!")
            logger.info(f"📊 总实验数: {self.stats['total_experiments']}")
            logger.info(f"🚀 GPU实验: {self.stats['gpu_experiments']}")
            logger.info(f"💻 CPU实验: {self.stats['cpu_experiments']}")
            logger.info(f"✅ 成功完成: {self.stats['completed_experiments']}")
            logger.info(f"❌ 失败实验: {self.stats['failed_experiments']}")
            logger.info(f"⏱️ 总耗时: {total_duration/3600:.1f} 小时")
            logger.info(f"📈 成功率: {self.stats['completed_experiments']/self.stats['total_experiments']*100:.1f}%")
            logger.info(f"🚀 平均速度: {self.stats['completed_experiments']/total_duration*3600:.1f} 实验/小时")
            logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="高性能运行100个Project的消融实验")
    parser.add_argument("--drive-path", default="/content/drive/MyDrive/Solar PV electricity/ablation results",
                       help="Google Drive结果保存路径")
    parser.add_argument("--max-gpu-experiments", type=int, default=16,
                       help="GPU并行实验数")
    parser.add_argument("--max-cpu-experiments", type=int, default=24,
                       help="CPU并行实验数")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="批次大小")
    parser.add_argument("--project-ids", nargs="+",
                       help="指定要运行的Project ID列表")
    
    args = parser.parse_args()
    
    runner = HighPerformanceExperimentRunner(
        drive_path=args.drive_path,
        max_gpu_experiments=args.max_gpu_experiments,
        max_cpu_experiments=args.max_cpu_experiments,
        batch_size=args.batch_size
    )
    
    try:
        runner.run_high_performance_experiments(args.project_ids)
    except KeyboardInterrupt:
        logger.info("🛑 用户中断，开始优雅关闭...")
        runner.running = False
    except Exception as e:
        logger.error(f"💥 运行异常: {e}")
        raise

if __name__ == "__main__":
    main()
