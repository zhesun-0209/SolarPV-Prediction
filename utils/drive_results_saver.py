#!/usr/bin/env python3
"""
Google Drive结果保存器
支持实时追加写入CSV结果，断点续训功能
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
try:
    from typing import Optional
except ImportError:
    Optional = None
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriveResultsSaver:
    """Google Drive结果保存器"""
    
    def __init__(self, drive_path: str = "/content/drive/MyDrive/Solar PV electricity/ablation results"):
        self.drive_path = Path(drive_path)
        self.drive_path.mkdir(parents=True, exist_ok=True)
        
        # 创建临时本地缓存目录
        self.local_cache_dir = Path("./temp_drive_cache")
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Drive结果保存器初始化完成")
        logger.info(f"Drive路径: {self.drive_path}")
        logger.info(f"本地缓存: {self.local_cache_dir}")
    
    def get_project_csv_path(self, project_id: str) -> Path:
        """获取Project的CSV结果文件路径"""
        return self.drive_path / f"{project_id}.csv"
    
    def get_local_cache_path(self, project_id: str) -> Path:
        """获取本地缓存文件路径"""
        return self.local_cache_dir / f"{project_id}.csv"
    
    def load_existing_results(self, project_id: str) -> pd.DataFrame:
        """加载已存在的结果"""
        csv_path = self.get_project_csv_path(project_id)
        local_cache_path = self.get_local_cache_path(project_id)
        
        # 优先从Drive加载
        if csv_path.exists():
            try:
                # 检查文件大小
                file_size = csv_path.stat().st_size
                if file_size == 0:
                    logger.info(f"Drive文件 {project_id} 为空，创建新文件")
                    return pd.DataFrame()
                
                df = pd.read_csv(csv_path)
                if df.empty:
                    logger.info(f"Drive文件 {project_id} 为空DataFrame")
                    return pd.DataFrame()
                
                # 同时保存到本地缓存
                df.to_csv(local_cache_path, index=False)
                logger.info(f"从Drive加载 {project_id} 结果: {len(df)} 条记录")
                return df
            except Exception as e:
                logger.warning(f"Drive文件加载失败: {e}")
                # 如果Drive文件损坏，尝试删除并重新开始
                try:
                    csv_path.unlink()
                    logger.info(f"已删除损坏的Drive文件: {csv_path}")
                except:
                    pass
        
        # 如果Drive没有，尝试从本地缓存加载
        if local_cache_path.exists():
            try:
                df = pd.read_csv(local_cache_path)
                logger.info(f"从本地缓存加载 {project_id} 结果: {len(df)} 条记录")
                return df
            except Exception as e:
                logger.warning(f"本地缓存加载失败: {e}")
        
        # 返回空的DataFrame
        logger.info(f"未找到 {project_id} 的已有结果，创建新文件")
        return pd.DataFrame()
    
    def get_completed_experiments(self, project_id: str) -> set:
        """获取已完成的实验配置名称"""
        df = self.load_existing_results(project_id)
        if df.empty:
            return set()
        
        if 'config_name' in df.columns:
            completed = set(df['config_name'].tolist())
            logger.info(f"{project_id} 已完成实验: {len(completed)} 个")
            return completed
        
        return set()
    
    def save_experiment_result(self, project_id: str, result_data: Dict[str, Any]) -> bool:
        """保存单个实验结果到CSV"""
        try:
            # 加载已有结果
            df = self.load_existing_results(project_id)
            
            # 准备新结果数据
            new_row = {
                'project_id': project_id,
                'config_name': result_data.get('config_name', ''),
                'status': result_data.get('status', 'completed'),
                'timestamp': datetime.now().isoformat(),
                'duration': result_data.get('duration', 0),
                
                # 性能指标
                'mae': result_data.get('mae', np.nan),
                'rmse': result_data.get('rmse', np.nan),
                'r2': result_data.get('r2', np.nan),
                'mape': result_data.get('mape', np.nan),
                
                # 训练信息
                'train_time_sec': result_data.get('train_time_sec', np.nan),
                'inference_time_sec': result_data.get('inference_time_sec', np.nan),
                'param_count': result_data.get('param_count', np.nan),
                'samples_count': result_data.get('samples_count', np.nan),
                
                # 配置信息
                'model': result_data.get('model', ''),
                'model_complexity': result_data.get('model_complexity', ''),
                'input_category': result_data.get('input_category', ''),
                'lookback_hours': result_data.get('lookback_hours', np.nan),
                'use_time_encoding': result_data.get('use_time_encoding', False),
                
                # 错误信息（如果有）
                'error_message': result_data.get('error_message', '')
            }
            
            # 追加新行
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            
            # 保存到本地缓存
            local_cache_path = self.get_local_cache_path(project_id)
            df.to_csv(local_cache_path, index=False)
            
            # 保存到Drive
            drive_path = self.get_project_csv_path(project_id)
            df.to_csv(drive_path, index=False)
            
            logger.info(f"✅ {project_id} 结果已保存: {result_data.get('config_name', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存 {project_id} 结果失败: {e}")
            return False
    
    def batch_save_results(self, project_id: str, results: List[Dict[str, Any]]) -> int:
        """批量保存多个实验结果"""
        saved_count = 0
        
        for result in results:
            if self.save_experiment_result(project_id, result):
                saved_count += 1
        
        logger.info(f"📊 {project_id} 批量保存完成: {saved_count}/{len(results)}")
        return saved_count
    
    def get_project_statistics(self, project_id: str) -> Dict[str, Any]:
        """获取Project的统计信息"""
        df = self.load_existing_results(project_id)
        
        if df.empty:
            return {
                'project_id': project_id,
                'total_experiments': 0,
                'completed_experiments': 0,
                'failed_experiments': 0,
                'completion_rate': 0.0,
                'best_mae': np.nan,
                'best_rmse': np.nan,
                'best_r2': np.nan
            }
        
        completed_df = df[df['status'] == 'completed']
        
        stats = {
            'project_id': project_id,
            'total_experiments': len(df),
            'completed_experiments': len(completed_df),
            'failed_experiments': len(df) - len(completed_df),
            'completion_rate': len(completed_df) / len(df) * 100 if len(df) > 0 else 0.0
        }
        
        if len(completed_df) > 0:
            stats.update({
                'best_mae': completed_df['mae'].min() if 'mae' in completed_df.columns else np.nan,
                'best_rmse': completed_df['rmse'].min() if 'rmse' in completed_df.columns else np.nan,
                'best_r2': completed_df['r2'].max() if 'r2' in completed_df.columns else np.nan,
                'avg_mae': completed_df['mae'].mean() if 'mae' in completed_df.columns else np.nan,
                'avg_rmse': completed_df['rmse'].mean() if 'rmse' in completed_df.columns else np.nan,
                'avg_r2': completed_df['r2'].mean() if 'r2' in completed_df.columns else np.nan
            })
        
        return stats
    
    def get_all_projects_statistics(self) -> pd.DataFrame:
        """获取所有Project的统计信息"""
        stats_list = []
        
        # 扫描所有CSV文件
        for csv_file in self.drive_path.glob("*.csv"):
            project_id = csv_file.stem
            stats = self.get_project_statistics(project_id)
            stats_list.append(stats)
        
        if stats_list:
            df = pd.DataFrame(stats_list)
            logger.info(f"📊 统计信息: {len(df)} 个Project")
            return df
        else:
            logger.info("📊 未找到任何Project结果")
            return pd.DataFrame()
    
    def cleanup_local_cache(self):
        """清理本地缓存"""
        try:
            import shutil
            shutil.rmtree(self.local_cache_dir)
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("🧹 本地缓存已清理")
        except Exception as e:
            logger.warning(f"清理本地缓存失败: {e}")
    
    def sync_to_drive(self):
        """同步所有本地缓存到Drive"""
        synced_count = 0
        
        for cache_file in self.local_cache_dir.glob("*.csv"):
            project_id = cache_file.stem
            drive_path = self.get_project_csv_path(project_id)
            
            try:
                # 复制到Drive
                import shutil
                shutil.copy2(cache_file, drive_path)
                synced_count += 1
                logger.info(f"📤 同步 {project_id} 到Drive")
            except Exception as e:
                logger.error(f"❌ 同步 {project_id} 失败: {e}")
        
        logger.info(f"📤 同步完成: {synced_count} 个文件")
        return synced_count

def test_drive_saver():
    """测试Drive保存器"""
    saver = DriveResultsSaver()
    
    # 测试保存结果
    test_result = {
        'config_name': 'Transformer_high_PV_plus_NWP_72h_TE',
        'status': 'completed',
        'duration': 120.5,
        'mae': 0.1234,
        'rmse': 0.1567,
        'r2': 0.8765,
        'mape': 5.4321,
        'train_time_sec': 95.2,
        'inference_time_sec': 0.8,
        'param_count': 1250000,
        'samples_count': 2400,
        'model': 'Transformer',
        'model_complexity': 'high',
        'input_category': 'PV_plus_NWP',
        'lookback_hours': 72,
        'use_time_encoding': True
    }
    
    # 保存测试结果
    success = saver.save_experiment_result('Project001', test_result)
    print(f"测试保存结果: {success}")
    
    # 获取统计信息
    stats = saver.get_project_statistics('Project001')
    print(f"Project001统计: {stats}")

if __name__ == "__main__":
    test_drive_saver()
