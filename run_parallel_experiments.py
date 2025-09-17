#!/usr/bin/env python3
"""
并行实验脚本 - 使用GPU并行运行340个实验
"""

import os
import sys
import time
import yaml
from pathlib import Path
from parallel_experiment_system import ExperimentScheduler, GPUResourceManager

def get_project_configs(project_id):
    """获取项目的所有配置文件"""
    config_dir = Path(f"config/projects/{project_id}")
    if not config_dir.exists():
        return []
    
    config_files = []
    for config_file in sorted(config_dir.glob("*.yaml")):
        if config_file.name not in ['config_index.yaml']:
            config_files.append(str(config_file))
    
    return config_files

def main():
    """主函数"""
    print("🌟 SolarPV项目 - GPU并行实验系统")
    print("=" * 60)
    
    # 检查GPU
    gpu_manager = GPUResourceManager()
    if gpu_manager.gpu_count == 0:
        print("❌ 未检测到GPU，无法运行并行实验")
        return
    
    print(f"🎯 检测到 {gpu_manager.gpu_count} 个GPU")
    for gpu_id, memory_gb in gpu_manager.gpu_memory.items():
        print(f"   GPU {gpu_id}: {memory_gb:.1f}GB")
    
    # 设置并行数量（根据GPU数量和内存调整）
    max_parallel = min(4, gpu_manager.gpu_count * 2)
    print(f"📊 最大并行数: {max_parallel}")
    
    # 创建调度器
    scheduler = ExperimentScheduler(max_parallel=max_parallel, gpu_manager=gpu_manager)
    
    # 获取项目列表
    projects = [171, 172, 186]  # 可以根据需要调整
    data_dir = "data"
    
    # 添加实验到队列
    total_experiments = 0
    for project_id in projects:
        data_file = os.path.join(data_dir, f"Project{project_id}.csv")
        if not os.path.exists(data_file):
            print(f"⚠️ 数据文件不存在: {data_file}")
            continue
        
        config_files = get_project_configs(project_id)
        print(f"📁 项目 {project_id}: {len(config_files)} 个实验")
        
        for config_file in config_files:
            scheduler.add_experiment(config_file, data_file, project_id)
            total_experiments += 1
    
    print(f"📊 总实验数: {total_experiments}")
    
    # 启动调度器
    start_time = time.time()
    scheduler.run_scheduler()
    total_time = time.time() - start_time
    
    # 显示结果
    status = scheduler.get_status()
    print("\n" + "=" * 60)
    print("🎉 并行实验完成!")
    print(f"📊 总实验数: {total_experiments}")
    print(f"✅ 成功: {status['completed']}")
    print(f"❌ 失败: {status['failed']}")
    print(f"⏱️ 总时间: {total_time/3600:.2f}小时")
    print(f"🚀 平均速度: {total_experiments/(total_time/3600):.1f} 实验/小时")

if __name__ == "__main__":
    main()
