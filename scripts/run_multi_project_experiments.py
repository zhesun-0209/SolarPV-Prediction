#!/usr/bin/env python3
"""
运行100个Project的消融实验
支持断点续训、实时保存到Google Drive
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
import signal
import threading
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
        logging.FileHandler('multi_project_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiProjectExperimentRunner:
    """多Project实验运行器"""
    
    def __init__(self, 
                 drive_path: str = "/content/drive/MyDrive/Solar PV electricity/ablation results",
                 max_workers: int = 4,
                 batch_size: int = 10):
        self.drive_saver = DriveResultsSaver(drive_path)
        self.checkpoint_manager = CheckpointManager(drive_path)
        self.max_workers = max_workers
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
            'start_time': None
        }
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"多Project实验运行器初始化完成")
        logger.info(f"最大并发数: {max_workers}")
        logger.info(f"批次大小: {batch_size}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器，用于优雅关闭"""
        logger.info(f"收到信号 {signum}，开始优雅关闭...")
        self.running = False
    
    def run_single_experiment(self, project_id: str, config_info: Dict) -> Dict:
        """运行单个实验"""
        try:
            config_name = config_info['name']
            config = config_info['config']
            config_file = config_info['file_path']
            
            logger.info(f"🚀 开始实验: {project_id} - {config_name}")
            
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
                logger.info(f"✅ 实验完成: {project_id} - {config_name} (耗时: {duration:.1f}秒)")
                
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
                logger.error(f"❌ 实验失败: {project_id} - {config_name}")
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
            logger.error(f"⏰ 实验超时: {project_id} - {config_name}")
            
            result_data = {
                'config_name': config_name,
                'status': 'timeout',
                'duration': 3600,
                'error_message': 'Experiment timeout (1 hour)'
            }
            
            self.drive_saver.save_experiment_result(project_id, result_data)
            
            return {
                'project_id': project_id,
                'config_name': config_name,
                'status': 'timeout',
                'duration': 3600,
                'error': 'Timeout'
            }
        except Exception as e:
            logger.error(f"💥 实验异常: {project_id} - {config_name}: {e}")
            
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
                                  config: Dict, duration: float, 
                                  exp_results_dir: Path) -> Dict:
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
            # 检查是否有结果文件
            results_files = list(exp_results_dir.glob("*.csv"))
            if results_files:
                # 读取第一个CSV文件（通常是结果文件）
                df = pd.read_csv(results_files[0])
                
                # 尝试提取指标
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
    
    def run_project_experiments(self, project_id: str) -> Dict:
        """运行单个Project的所有实验"""
        logger.info(f"📁 开始处理Project: {project_id}")
        
        # 获取待执行的实验
        pending_configs = self.checkpoint_manager.get_pending_experiments(project_id)
        
        if not pending_configs:
            logger.info(f"✅ {project_id} 所有实验已完成")
            return {
                'project_id': project_id,
                'total_experiments': 0,
                'completed_experiments': 0,
                'failed_experiments': 0,
                'duration': 0
            }
        
        logger.info(f"📊 {project_id} 待执行实验: {len(pending_configs)}")
        
        # 运行实验
        start_time = time.time()
        completed_count = 0
        failed_count = 0
        
        for i, config_info in enumerate(pending_configs):
            if not self.running:
                logger.info(f"🛑 收到停止信号，中断 {project_id} 实验")
                break
            
            try:
                result = self.run_single_experiment(project_id, config_info)
                
                if result['status'] == 'completed':
                    completed_count += 1
                else:
                    failed_count += 1
                
                # 更新统计
                self.stats['completed_experiments'] += 1
                if result['status'] != 'completed':
                    self.stats['failed_experiments'] += 1
                
                # 定期同步到Drive
                if (i + 1) % 10 == 0:
                    self.drive_saver.sync_to_drive()
                    logger.info(f"📤 {project_id} 已同步 {i + 1}/{len(pending_configs)} 个实验")
                
            except Exception as e:
                logger.error(f"💥 {project_id} 实验异常: {e}")
                failed_count += 1
                self.stats['failed_experiments'] += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 最终同步
        self.drive_saver.sync_to_drive()
        
        logger.info(f"🎉 {project_id} 完成: {completed_count} 成功, {failed_count} 失败, 耗时: {duration:.1f}秒")
        
        return {
            'project_id': project_id,
            'total_experiments': len(pending_configs),
            'completed_experiments': completed_count,
            'failed_experiments': failed_count,
            'duration': duration
        }
    
    def run_all_projects(self, project_ids: List[str] = None):
        """运行所有Project的实验"""
        if project_ids is None:
            # 获取所有未完成的Project
            project_ids = self.checkpoint_manager.get_incomplete_projects()
        
        if not project_ids:
            logger.info("🎉 所有Project实验已完成!")
            return
        
        logger.info(f"🚀 开始运行 {len(project_ids)} 个Project的实验")
        
        self.stats['start_time'] = datetime.now()
        self.stats['total_experiments'] = sum(
            len(self.checkpoint_manager.get_pending_experiments(pid)) 
            for pid in project_ids
        )
        
        logger.info(f"📊 总实验数: {self.stats['total_experiments']}")
        
        # 按批次处理Project
        for i in range(0, len(project_ids), self.batch_size):
            if not self.running:
                logger.info("🛑 收到停止信号，中断所有实验")
                break
            
            batch = project_ids[i:i + self.batch_size]
            logger.info(f"📦 处理批次 {i//self.batch_size + 1}: {batch}")
            
            # 并行运行批次内的Project
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_project = {
                    executor.submit(self.run_project_experiments, project_id): project_id
                    for project_id in batch
                }
                
                for future in as_completed(future_to_project):
                    project_id = future_to_project[future]
                    try:
                        result = future.result()
                        logger.info(f"✅ {project_id} 批次完成: {result}")
                    except Exception as e:
                        logger.error(f"❌ {project_id} 批次失败: {e}")
            
            # 批次间同步
            self.drive_saver.sync_to_drive()
            
            # 生成进度报告
            self.checkpoint_manager.save_progress_report()
            
            logger.info(f"📊 当前进度: {self.stats['completed_experiments']}/{self.stats['total_experiments']}")
        
        # 最终统计
        self._print_final_statistics()
    
    def _print_final_statistics(self):
        """打印最终统计信息"""
        if self.stats['start_time']:
            total_duration = (datetime.now() - self.stats['start_time']).total_seconds()
            logger.info("\n" + "=" * 60)
            logger.info("🎉 所有实验完成!")
            logger.info(f"📊 总实验数: {self.stats['total_experiments']}")
            logger.info(f"✅ 成功完成: {self.stats['completed_experiments']}")
            logger.info(f"❌ 失败实验: {self.stats['failed_experiments']}")
            logger.info(f"⏱️ 总耗时: {total_duration/3600:.1f} 小时")
            logger.info(f"📈 成功率: {self.stats['completed_experiments']/self.stats['total_experiments']*100:.1f}%")
            logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="运行100个Project的消融实验")
    parser.add_argument("--drive-path", default="/content/drive/MyDrive/Solar PV electricity/ablation results",
                       help="Google Drive结果保存路径")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="最大并发数")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="批次大小")
    parser.add_argument("--project-ids", nargs="+",
                       help="指定要运行的Project ID列表")
    
    args = parser.parse_args()
    
    runner = MultiProjectExperimentRunner(
        drive_path=args.drive_path,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    try:
        runner.run_all_projects(args.project_ids)
    except KeyboardInterrupt:
        logger.info("🛑 用户中断，开始优雅关闭...")
        runner.running = False
    except Exception as e:
        logger.error(f"💥 运行异常: {e}")
        raise

if __name__ == "__main__":
    main()
