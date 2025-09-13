#!/usr/bin/env python3
"""
GPU专用实验运行器 - 所有模型都强制使用GPU
包括传统ML模型的GPU版本
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import signal
import threading
import torch
import queue
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.drive_results_saver import DriveResultsSaver
from utils.checkpoint_manager import CheckpointManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_only_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPUOnlyExperimentRunner:
    """GPU专用实验运行器 - 所有模型都使用GPU"""
    
    def __init__(self, 
                 drive_path: str = "/content/drive/MyDrive/Solar PV electricity/ablation results",
                 max_gpu_experiments: int = 24,  # 大幅增加GPU并行数
                 batch_size: int = 30):
        self.drive_saver = DriveResultsSaver(drive_path)
        self.checkpoint_manager = CheckpointManager(drive_path)
        
        # GPU专用参数
        self.max_gpu_experiments = max_gpu_experiments
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
            'start_time': None
        }
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 检测GPU资源
        self._detect_gpu_resources()
        
        logger.info(f"GPU专用实验运行器初始化完成")
        logger.info(f"GPU并行数: {self.max_gpu_experiments}")
        logger.info(f"批次大小: {self.batch_size}")
    
    def _detect_gpu_resources(self):
        """检测GPU资源"""
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            self.gpu_memory_total = []
            
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3  # GB
                self.gpu_memory_total.append(total_memory)
                logger.info(f"GPU {i}: {props.name} ({total_memory:.1f}GB)")
            
            # 根据GPU内存调整并行数
            max_memory = max(self.gpu_memory_total)
            if max_memory >= 80:  # A100 80GB
                self.max_gpu_experiments = min(32, self.gpu_count * 8)
                logger.info(f"🎯 A100 80GB检测到，优化GPU并行数为: {self.max_gpu_experiments}")
            elif max_memory >= 40:  # A100 40GB
                self.max_gpu_experiments = min(24, self.gpu_count * 6)
                logger.info(f"🎯 A100 40GB检测到，优化GPU并行数为: {self.max_gpu_experiments}")
            elif max_memory >= 24:  # RTX 3090/4090
                self.max_gpu_experiments = min(16, self.gpu_count * 4)
                logger.info(f"🎯 高端GPU检测到，优化GPU并行数为: {self.max_gpu_experiments}")
            else:
                self.max_gpu_experiments = min(8, self.gpu_count * 2)
                logger.info(f"💻 标准GPU，设置并行数为: {self.max_gpu_experiments}")
        else:
            logger.error("❌ 未检测到GPU，无法运行GPU专用实验")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """信号处理器，用于优雅关闭"""
        logger.info(f"收到信号 {signum}，开始优雅关闭...")
        self.running = False
    
    def _optimize_config_for_gpu(self, config):
        """为GPU优化配置"""
        model = config.get('model', '')
        
        # 强制设置GPU相关参数
        config['use_gpu'] = True
        config['device'] = 'cuda'
        
        # 根据模型类型设置GPU特定参数
        if model in ['LSTM', 'GRU', 'TCN', 'Transformer']:
            # 深度学习模型
            config['train_params']['use_amp'] = True
            config['train_params']['batch_size'] = min(128, config['train_params'].get('batch_size', 32) * 4)
            config['train_params']['gradient_accumulation_steps'] = 2
        elif model in ['RF', 'XGB', 'LGBM']:
            # 树模型GPU版本
            config['train_params']['tree_method'] = 'gpu_hist'  # XGBoost GPU
            config['train_params']['gpu_id'] = 0
            config['train_params']['predictor'] = 'gpu_predictor'
            
            # LightGBM GPU
            if model == 'LGBM':
                config['train_params']['device'] = 'gpu'
                config['train_params']['gpu_platform_id'] = 0
                config['train_params']['gpu_device_id'] = 0
            
            # 增加样本数量以充分利用GPU
            config['train_params']['n_estimators'] = config['train_params'].get('n_estimators', 100) * 2
            config['train_params']['batch_size'] = 1024  # 大批次处理
            
        elif model == 'LSR':
            # 线性回归也可以使用GPU
            config['train_params']['batch_size'] = 2048  # 大批次处理
        
        # 通用GPU优化
        config['train_params']['num_workers'] = min(8, os.cpu_count())
        config['train_params']['pin_memory'] = True
        
        return config
    
    def _run_single_experiment(self, experiment_info):
        """运行单个GPU实验"""
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
            
            # GPU优化配置
            config = self._optimize_config_for_gpu(config)
            
            # 保存更新的配置
            updated_config_file = exp_results_dir / "config.yaml"
            with open(updated_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 运行实验
            start_time = time.time()
            
            # 调用主程序
            cmd = [sys.executable, "main.py", "--config", str(updated_config_file)]
            
            # 设置GPU环境变量
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            env['PYTORCH_CUDNN_V8_API_ENABLED'] = '1'
            
            # 强制使用GPU
            env['FORCE_GPU'] = '1'
            env['USE_GPU'] = '1'
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=1200,  # 20分钟超时（GPU应该更快）
                env=env
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ GPU实验完成: {project_id} - {config_name} (耗时: {duration:.1f}秒)")
                
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
                logger.error(f"❌ GPU实验失败: {project_id} - {config_name}")
                logger.error(f"错误输出: {result.stderr}")
                
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
            logger.error(f"⏰ GPU实验超时: {project_id} - {config_name}")
            
            result_data = {
                'config_name': config_name,
                'status': 'timeout',
                'duration': 1200,
                'error_message': 'GPU experiment timeout (20 minutes)'
            }
            
            self.drive_saver.save_experiment_result(project_id, result_data)
            
            return {
                'project_id': project_id,
                'config_name': config_name,
                'status': 'timeout',
                'duration': 1200,
                'error': 'Timeout'
            }
        except Exception as e:
            logger.error(f"💥 GPU实验异常: {project_id} - {config_name}: {e}")
            
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
            'use_gpu': True,  # 标记为GPU实验
            
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
            # 优先查找Excel文件
            excel_files = list(exp_results_dir.glob("*.xlsx"))
            if excel_files:
                df = pd.read_excel(excel_files[0])
                
                # 使用小写列名（Excel文件中使用的）
                if 'mae' in df.columns:
                    result_data['mae'] = df['mae'].iloc[0] if len(df) > 0 else np.nan
                if 'rmse' in df.columns:
                    result_data['rmse'] = df['rmse'].iloc[0] if len(df) > 0 else np.nan
                if 'r_square' in df.columns:
                    result_data['r2'] = df['r_square'].iloc[0] if len(df) > 0 else np.nan
                if 'smape' in df.columns:
                    result_data['mape'] = df['smape'].iloc[0] if len(df) > 0 else np.nan
                if 'train_time_sec' in df.columns:
                    result_data['train_time_sec'] = df['train_time_sec'].iloc[0] if len(df) > 0 else np.nan
                if 'inference_time_sec' in df.columns:
                    result_data['inference_time_sec'] = df['inference_time_sec'].iloc[0] if len(df) > 0 else np.nan
                if 'param_count' in df.columns:
                    result_data['param_count'] = df['param_count'].iloc[0] if len(df) > 0 else np.nan
                if 'samples_count' in df.columns:
                    result_data['samples_count'] = df['samples_count'].iloc[0] if len(df) > 0 else np.nan
                    
            else:
                # 回退到CSV文件
                csv_files = list(exp_results_dir.glob("*.csv"))
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                    
                    # 尝试多种可能的列名
                    if 'mae' in df.columns:
                        result_data['mae'] = df['mae'].iloc[0] if len(df) > 0 else np.nan
                    elif 'MAE' in df.columns:
                        result_data['mae'] = df['MAE'].iloc[0] if len(df) > 0 else np.nan
                        
                    if 'rmse' in df.columns:
                        result_data['rmse'] = df['rmse'].iloc[0] if len(df) > 0 else np.nan
                    elif 'RMSE' in df.columns:
                        result_data['rmse'] = df['RMSE'].iloc[0] if len(df) > 0 else np.nan
                        
                    if 'r_square' in df.columns:
                        result_data['r2'] = df['r_square'].iloc[0] if len(df) > 0 else np.nan
                    elif 'R²' in df.columns:
                        result_data['r2'] = df['R²'].iloc[0] if len(df) > 0 else np.nan
                    elif 'r2' in df.columns:
                        result_data['r2'] = df['r2'].iloc[0] if len(df) > 0 else np.nan
                        
                    if 'smape' in df.columns:
                        result_data['mape'] = df['smape'].iloc[0] if len(df) > 0 else np.nan
                    elif 'MAPE' in df.columns:
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
    
    def run_gpu_only_experiments(self, project_ids: list = None):
        """运行GPU专用实验"""
        if project_ids is None:
            project_ids = self.checkpoint_manager.get_incomplete_projects()
        
        if not project_ids:
            logger.info("🎉 所有Project实验已完成!")
            return
        
        logger.info(f"🚀 开始GPU专用运行 {len(project_ids)} 个Project的实验")
        
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
        
        # 使用ThreadPoolExecutor进行GPU并行
        for i in range(0, len(all_experiments), self.batch_size):
            if not self.running:
                break
            
            batch = all_experiments[i:i + self.batch_size]
            logger.info(f"🚀 处理GPU批次 {i//self.batch_size + 1}: {len(batch)} 个实验")
            
            with ThreadPoolExecutor(max_workers=self.max_gpu_experiments) as executor:
                future_to_exp = {
                    executor.submit(self._run_single_experiment, exp): exp
                    for exp in batch
                }
                
                for future in as_completed(future_to_exp):
                    if not self.running:
                        break
                    
                    exp = future_to_exp[future]
                    try:
                        result = future.result()
                        
                        self.stats['completed_experiments'] += 1
                        if result['status'] == 'completed':
                            self.stats['gpu_experiments'] += 1
                        else:
                            self.stats['failed_experiments'] += 1
                        
                        # 结果已在_run_single_experiment中保存，无需重复保存
                        
                    except Exception as e:
                        logger.error(f"💥 GPU实验异常: {exp['project_id']} - {exp['config_name']}: {e}")
                        self.stats['failed_experiments'] += 1
            
            # 批次间同步
            self.drive_saver.sync_to_drive()
            
            logger.info(f"📊 GPU进度: {self.stats['gpu_experiments']}/{self.stats['total_experiments']}")
        
        # 最终统计
        self._print_final_statistics()
    
    def _print_final_statistics(self):
        """打印最终统计信息"""
        if self.stats['start_time']:
            total_duration = (datetime.now() - self.stats['start_time']).total_seconds()
            logger.info("\n" + "=" * 60)
            logger.info("🎉 GPU专用实验完成!")
            logger.info(f"📊 总实验数: {self.stats['total_experiments']}")
            logger.info(f"🚀 GPU实验: {self.stats['gpu_experiments']}")
            logger.info(f"✅ 成功完成: {self.stats['completed_experiments']}")
            logger.info(f"❌ 失败实验: {self.stats['failed_experiments']}")
            logger.info(f"⏱️ 总耗时: {total_duration/3600:.1f} 小时")
            logger.info(f"📈 成功率: {self.stats['completed_experiments']/self.stats['total_experiments']*100:.1f}%")
            logger.info(f"🚀 平均速度: {self.stats['completed_experiments']/total_duration*3600:.1f} 实验/小时")
            logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="GPU专用运行100个Project的消融实验")
    parser.add_argument("--drive-path", default="/content/drive/MyDrive/Solar PV electricity/ablation results",
                       help="Google Drive结果保存路径")
    parser.add_argument("--max-gpu-experiments", type=int, default=24,
                       help="GPU并行实验数")
    parser.add_argument("--batch-size", type=int, default=30,
                       help="批次大小")
    parser.add_argument("--project-ids", nargs="+",
                       help="指定要运行的Project ID列表")
    
    args = parser.parse_args()
    
    runner = GPUOnlyExperimentRunner(
        drive_path=args.drive_path,
        max_gpu_experiments=args.max_gpu_experiments,
        batch_size=args.batch_size
    )
    
    try:
        runner.run_gpu_only_experiments(args.project_ids)
    except KeyboardInterrupt:
        logger.info("🛑 用户中断，开始优雅关闭...")
        runner.running = False
    except Exception as e:
        logger.error(f"💥 运行异常: {e}")
        raise

if __name__ == "__main__":
    main()
