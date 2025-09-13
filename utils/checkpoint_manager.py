#!/usr/bin/env python3
"""
断点续训管理器
检查已完成的实验，支持从断点继续训练
"""

import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
try:
    from typing import Optional
except ImportError:
    Optional = None
import logging
from datetime import datetime

from utils.drive_results_saver import DriveResultsSaver

logger = logging.getLogger(__name__)

class CheckpointManager:
    """断点续训管理器"""
    
    def __init__(self, drive_path: str = "/content/drive/MyDrive/Solar PV electricity/ablation results"):
        self.drive_saver = DriveResultsSaver(drive_path)
        self.config_base_dir = Path("config/projects")
        
        logger.info("断点续训管理器初始化完成")
    
    def get_project_configs(self, project_id: str) -> List[Dict]:
        """获取Project的所有配置"""
        config_dir = self.config_base_dir / project_id
        index_file = config_dir / "config_index.yaml"
        
        if not index_file.exists():
            logger.error(f"配置索引文件不存在: {index_file}")
            return []
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = yaml.safe_load(f)
            
            configs = []
            for config_info in index_data.get('configs', []):
                config_file = config_dir / config_info['file']
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    configs.append({
                        'name': config_info['name'],
                        'config_id': config_info['config_id'],
                        'config': config,
                        'file_path': config_file
                    })
            
            logger.info(f"加载 {project_id} 配置: {len(configs)} 个")
            return configs
            
        except Exception as e:
            logger.error(f"加载 {project_id} 配置失败: {e}")
            return []
    
    def get_completed_experiments(self, project_id: str) -> Set[str]:
        """获取已完成的实验配置名称"""
        return self.drive_saver.get_completed_experiments(project_id)
    
    def get_pending_experiments(self, project_id: str) -> List[Dict]:
        """获取待执行的实验配置"""
        all_configs = self.get_project_configs(project_id)
        completed_configs = self.get_completed_experiments(project_id)
        
        pending_configs = []
        for config_info in all_configs:
            if config_info['name'] not in completed_configs:
                pending_configs.append(config_info)
        
        logger.info(f"{project_id} 待执行实验: {len(pending_configs)}/{len(all_configs)}")
        return pending_configs
    
    def check_missing_experiments(self, project_id: str) -> Dict:
        """检查缺失的实验组合"""
        # 每个项目应该有340个实验
        expected_experiments = 340
        
        # 获取当前已完成的实验数量
        completed_configs = self.get_completed_experiments(project_id)
        completed_count = len(completed_configs)
        
        # 计算缺失的实验数量
        missing_count = expected_experiments - completed_count
        
        result = {
            'project_id': project_id,
            'expected_experiments': expected_experiments,
            'completed_experiments': completed_count,
            'missing_experiments': missing_count,
            'completion_rate': completed_count / expected_experiments * 100 if expected_experiments > 0 else 0.0,
            'is_complete': missing_count == 0
        }
        
        logger.info(f"{project_id} 实验状态: {completed_count}/{expected_experiments} ({result['completion_rate']:.1f}%)")
        if missing_count > 0:
            logger.info(f"{project_id} 缺失实验: {missing_count} 个")
        
        return result
    
    def get_project_progress(self, project_id: str) -> Dict:
        """获取Project的进度信息"""
        all_configs = self.get_project_configs(project_id)
        completed_configs = self.get_completed_experiments(project_id)
        pending_configs = self.get_pending_experiments(project_id)
        
        progress = {
            'project_id': project_id,
            'total_experiments': len(all_configs),
            'completed_experiments': len(completed_configs),
            'pending_experiments': len(pending_configs),
            'completion_rate': len(completed_configs) / len(all_configs) * 100 if len(all_configs) > 0 else 0.0,
            'is_complete': len(pending_configs) == 0,
            'last_update': datetime.now().isoformat()
        }
        
        return progress
    
    def get_all_projects_progress(self) -> pd.DataFrame:
        """获取所有Project的进度信息"""
        progress_list = []
        
        # 扫描所有Project配置目录
        for project_dir in self.config_base_dir.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                project_id = project_dir.name
                progress = self.get_project_progress(project_id)
                progress_list.append(progress)
        
        if progress_list:
            df = pd.DataFrame(progress_list)
            df = df.sort_values('project_id')
            logger.info(f"📊 进度统计: {len(df)} 个Project")
            return df
        else:
            logger.info("📊 未找到任何Project配置")
            return pd.DataFrame()
    
    def get_next_experiment(self, project_id: str):
        """获取下一个待执行的实验"""
        pending_configs = self.get_pending_experiments(project_id)
        
        if pending_configs:
            # 按config_id排序，返回第一个
            pending_configs.sort(key=lambda x: x['config_id'])
            next_config = pending_configs[0]
            logger.info(f"{project_id} 下一个实验: {next_config['name']}")
            return next_config
        else:
            logger.info(f"{project_id} 所有实验已完成")
            return None
    
    def mark_experiment_completed(self, project_id: str, config_name: str, result_data: Dict):
        """标记实验为已完成"""
        success = self.drive_saver.save_experiment_result(project_id, result_data)
        if success:
            logger.info(f"✅ {project_id} 实验完成: {config_name}")
        else:
            logger.error(f"❌ {project_id} 实验结果保存失败: {config_name}")
        return success
    
    def get_incomplete_projects(self) -> List[str]:
        """获取未完成的Project列表"""
        progress_df = self.get_all_projects_progress()
        
        if progress_df.empty:
            return []
        
        incomplete_projects = progress_df[progress_df['is_complete'] == False]['project_id'].tolist()
        logger.info(f"未完成Project: {len(incomplete_projects)} 个")
        return incomplete_projects
    
    def get_all_missing_experiments(self) -> pd.DataFrame:
        """获取所有项目的缺失实验状态"""
        missing_list = []
        
        # 扫描所有Project配置目录
        for project_dir in self.config_base_dir.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                project_id = project_dir.name
                missing_info = self.check_missing_experiments(project_id)
                missing_list.append(missing_info)
        
        if missing_list:
            df = pd.DataFrame(missing_list)
            df = df.sort_values('project_id')
            logger.info(f"📊 缺失实验统计: {len(df)} 个Project")
            return df
        else:
            logger.info("📊 未找到任何Project配置")
            return pd.DataFrame()
    
    def get_projects_with_missing_experiments(self) -> List[str]:
        """获取有缺失实验的Project列表"""
        missing_df = self.get_all_missing_experiments()
        
        if missing_df.empty:
            return []
        
        projects_with_missing = missing_df[missing_df['missing_experiments'] > 0]['project_id'].tolist()
        logger.info(f"有缺失实验的Project: {len(projects_with_missing)} 个")
        return projects_with_missing
    
    def get_completed_projects(self) -> List[str]:
        """获取已完成的Project列表"""
        progress_df = self.get_all_projects_progress()
        
        if progress_df.empty:
            return []
        
        completed_projects = progress_df[progress_df['is_complete'] == True]['project_id'].tolist()
        logger.info(f"已完成Project: {len(completed_projects)} 个")
        return completed_projects
    
    def generate_progress_report(self) -> str:
        """生成进度报告"""
        progress_df = self.get_all_projects_progress()
        
        if progress_df.empty:
            return "未找到任何Project配置"
        
        # 总体统计
        total_projects = len(progress_df)
        completed_projects = len(progress_df[progress_df['is_complete'] == True])
        total_experiments = progress_df['total_experiments'].sum()
        completed_experiments = progress_df['completed_experiments'].sum()
        
        report = f"""
# Project1140 消融实验进度报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 总体进度

- **总Project数**: {total_projects}
- **已完成Project**: {completed_projects}
- **进行中Project**: {total_projects - completed_projects}
- **总实验数**: {total_experiments}
- **已完成实验**: {completed_experiments}
- **总体完成率**: {completed_experiments/total_experiments*100:.1f}%

## Project详细进度

| Project ID | 总实验数 | 已完成 | 待完成 | 完成率 |
|------------|----------|--------|--------|--------|
"""
        
        for _, row in progress_df.iterrows():
            report += f"| {row['project_id']} | {row['total_experiments']} | {row['completed_experiments']} | {row['pending_experiments']} | {row['completion_rate']:.1f}% |\n"
        
        # 添加最佳性能统计
        if completed_experiments > 0:
            report += "\n## 最佳性能统计\n\n"
            
            # 获取所有已完成实验的最佳结果
            best_results = []
            for project_id in progress_df['project_id']:
                stats = self.drive_saver.get_project_statistics(project_id)
                if stats['completed_experiments'] > 0:
                    best_results.append(stats)
            
            if best_results:
                best_results_df = pd.DataFrame(best_results)
                
                # 找出最佳MAE和RMSE
                best_mae_idx = best_results_df['best_mae'].idxmin()
                best_rmse_idx = best_results_df['best_rmse'].idxmin()
                best_r2_idx = best_results_df['best_r2'].idxmax()
                
                report += f"- **最佳MAE**: {best_results_df.loc[best_mae_idx, 'project_id']} ({best_results_df.loc[best_mae_idx, 'best_mae']:.4f})\n"
                report += f"- **最佳RMSE**: {best_results_df.loc[best_rmse_idx, 'project_id']} ({best_results_df.loc[best_rmse_idx, 'best_rmse']:.4f})\n"
                report += f"- **最佳R²**: {best_results_df.loc[best_r2_idx, 'project_id']} ({best_results_df.loc[best_r2_idx, 'best_r2']:.4f})\n"
        
        return report
    
    def save_progress_report(self, output_path: str = None):
        """保存进度报告"""
        if output_path is None:
            output_path = self.drive_saver.drive_path / "progress_report.md"
        
        report = self.generate_progress_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📊 进度报告已保存: {output_path}")
        return output_path

def test_checkpoint_manager():
    """测试断点续训管理器"""
    manager = CheckpointManager()
    
    # 测试获取Project进度
    progress_df = manager.get_all_projects_progress()
    print(f"Project进度统计:\n{progress_df}")
    
    # 测试获取待执行实验
    if not progress_df.empty:
        first_project = progress_df.iloc[0]['project_id']
        pending_configs = manager.get_pending_experiments(first_project)
        print(f"\n{first_project} 待执行实验: {len(pending_configs)}")
        
        if pending_configs:
            next_config = manager.get_next_experiment(first_project)
            print(f"下一个实验: {next_config['name']}")
    
    # 生成进度报告
    report = manager.generate_progress_report()
    print(f"\n进度报告:\n{report}")

if __name__ == "__main__":
    test_checkpoint_manager()
