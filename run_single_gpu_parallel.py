#!/usr/bin/env python3
"""
单GPU并行实验脚本
在单块GPU上智能调度多个实验
"""

import os
import sys
import time
import yaml
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
import subprocess

class SingleGPUParallelScheduler:
    """单GPU并行调度器"""
    
    def __init__(self, max_parallel=2, gpu_memory_limit_mb=12000):
        self.max_parallel = max_parallel
        self.gpu_memory_limit_mb = gpu_memory_limit_mb
        self.experiment_queue = queue.Queue()
        self.running_experiments = {}
        self.completed_experiments = []
        self.failed_experiments = []
        self.lock = threading.Lock()
        
        # 模型内存需求估算（MB）
        self.model_memory_requirements = {
            'RF': 500,
            'XGB': 1000,
            'LGBM': 1000,
            'TCN_low': 1500,
            'LSTM_low': 2000,
            'GRU_low': 2000,
            'Transformer_low': 2500,
            'TCN_high': 3000,
            'LSTM_high': 4000,
            'GRU_high': 4000,
            'Transformer_high': 6000
        }
        
        # 实验优先级
        self.experiment_priorities = {
            'RF': 1,
            'XGB': 2,
            'LGBM': 3,
            'TCN_low': 4,
            'LSTM_low': 5,
            'GRU_low': 6,
            'Transformer_low': 7,
            'TCN_high': 8,
            'LSTM_high': 9,
            'GRU_high': 10,
            'Transformer_high': 11
        }
    
    def add_experiment(self, config_file, data_file, project_id):
        """添加实验到队列"""
        config_name = os.path.basename(config_file).replace('.yaml', '')
        model_name = config_name.split('_')[0]
        complexity = config_name.split('_')[1]
        
        priority_key = model_name if model_name in self.experiment_priorities else f"{model_name}_{complexity}"
        priority = self.experiment_priorities.get(priority_key, 999)
        memory_required = self.model_memory_requirements.get(f"{model_name}_{complexity}", 2000)
        
        experiment = {
            'config_file': config_file,
            'data_file': data_file,
            'project_id': project_id,
            'config_name': config_name,
            'model_name': model_name,
            'complexity': complexity,
            'priority': priority,
            'memory_required': memory_required,
            'status': 'queued',
            'start_time': None,
            'end_time': None
        }
        
        self.experiment_queue.put(experiment)
        return experiment
    
    def can_start_experiment(self, experiment):
        """检查是否可以启动实验"""
        if len(self.running_experiments) >= self.max_parallel:
            return False
        
        # 检查总内存使用量
        current_memory = sum(exp['memory_required'] for exp in self.running_experiments.values())
        if current_memory + experiment['memory_required'] > self.gpu_memory_limit_mb:
            return False
        
        return True
    
    def start_experiment(self, experiment):
        """启动实验"""
        with self.lock:
            experiment['status'] = 'running'
            experiment['start_time'] = time.time()
            self.running_experiments[experiment['config_name']] = experiment
        
        # 在新线程中运行实验
        thread = threading.Thread(
            target=self._run_experiment_thread,
            args=(experiment,),
            daemon=True
        )
        thread.start()
        
        return experiment
    
    def _run_experiment_thread(self, experiment):
        """在独立线程中运行实验"""
        try:
            # 运行实验
            success, stdout, stderr, duration = self._execute_experiment(experiment)
            
            # 更新实验状态
            with self.lock:
                experiment['status'] = 'completed' if success else 'failed'
                experiment['end_time'] = time.time()
                experiment['duration'] = duration
                experiment['stdout'] = stdout
                experiment['stderr'] = stderr
                
                # 从运行列表中移除
                if experiment['config_name'] in self.running_experiments:
                    del self.running_experiments[experiment['config_name']]
                
                # 添加到完成列表
                if success:
                    self.completed_experiments.append(experiment)
                    print(f"✅ 完成: {experiment['config_name']} ({duration:.1f}s)")
                else:
                    self.failed_experiments.append(experiment)
                    print(f"❌ 失败: {experiment['config_name']} - {stderr[:100]}")
                
        except Exception as e:
            with self.lock:
                experiment['status'] = 'error'
                experiment['error'] = str(e)
                if experiment['config_name'] in self.running_experiments:
                    del self.running_experiments[experiment['config_name']]
                self.failed_experiments.append(experiment)
            print(f"💥 错误: {experiment['config_name']} - {e}")
    
    def _execute_experiment(self, experiment):
        """执行单个实验"""
        try:
            # 加载配置文件
            with open(experiment['config_file'], 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 强制使用GPU，禁用CPU回退
            config['force_gpu'] = True
            
            # 修改配置文件中的data_path
            config['data_path'] = experiment['data_file']
            config['plant_id'] = experiment['project_id']
            
            # 创建临时配置文件
            temp_config = f"temp_config_{experiment['project_id']}_{experiment['config_name']}_{int(time.time())}.yaml"
            with open(temp_config, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 运行实验
            cmd = ['python', 'main.py', '--config', temp_config]
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            duration = time.time() - start_time
            
            # 清理临时文件
            if os.path.exists(temp_config):
                os.remove(temp_config)
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr, duration
            
        except Exception as e:
            return False, "", str(e), 0.0
    
    def run_scheduler(self):
        """运行调度器主循环"""
        print(f"🚀 启动单GPU并行调度器 (最大并行: {self.max_parallel})")
        
        while not self.experiment_queue.empty() or self.running_experiments:
            # 尝试启动新实验
            experiments_to_start = []
            temp_queue = queue.Queue()
            
            # 检查队列中的实验
            while not self.experiment_queue.empty():
                experiment = self.experiment_queue.get()
                if self.can_start_experiment(experiment):
                    experiments_to_start.append(experiment)
                    break
                else:
                    temp_queue.put(experiment)
            
            # 将未启动的实验放回队列
            while not temp_queue.empty():
                self.experiment_queue.put(temp_queue.get())
            
            # 启动新实验
            for experiment in experiments_to_start:
                self.start_experiment(experiment)
                current_memory = sum(exp['memory_required'] for exp in self.running_experiments.values())
                print(f"🔄 启动: {experiment['config_name']} (内存: {current_memory}/{self.gpu_memory_limit_mb}MB)")
            
            # 显示状态
            status = self.get_status()
            if status['running'] > 0 or status['queued'] > 0:
                print(f"📊 队列 {status['queued']} | 运行 {status['running']} | 完成 {status['completed']} | 失败 {status['failed']}")
            
            time.sleep(3)  # 等待3秒后再次检查
        
        print("✅ 所有实验完成!")
    
    def get_status(self):
        """获取调度器状态"""
        with self.lock:
            return {
                'queued': self.experiment_queue.qsize(),
                'running': len(self.running_experiments),
                'completed': len(self.completed_experiments),
                'failed': len(self.failed_experiments),
                'running_experiments': list(self.running_experiments.keys())
            }

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
    print("🌟 SolarPV项目 - 单GPU并行实验系统")
    print("=" * 60)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA GPU")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    print(f"🎯 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 根据GPU内存设置参数
    if gpu_memory >= 20:  # 20GB+
        max_parallel = 3
        memory_limit = int(gpu_memory * 1024 * 0.8)  # 80%内存使用率
    elif gpu_memory >= 12:  # 12GB+
        max_parallel = 2
        memory_limit = int(gpu_memory * 1024 * 0.8)
    else:  # 8GB
        max_parallel = 2
        memory_limit = int(gpu_memory * 1024 * 0.7)  # 70%内存使用率
    
    print(f"📊 最大并行数: {max_parallel}")
    print(f"📊 内存限制: {memory_limit}MB")
    
    # 创建调度器
    scheduler = SingleGPUParallelScheduler(max_parallel=max_parallel, gpu_memory_limit_mb=memory_limit)
    
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
    print("🎉 单GPU并行实验完成!")
    print(f"📊 总实验数: {total_experiments}")
    print(f"✅ 成功: {status['completed']}")
    print(f"❌ 失败: {status['failed']}")
    print(f"⏱️ 总时间: {total_time/3600:.2f}小时")
    print(f"🚀 平均速度: {total_experiments/(total_time/3600):.1f} 实验/小时")
    print(f"🎯 并行效率: {total_experiments/(total_time/3600)/max_parallel:.1f} 实验/小时/并行")

if __name__ == "__main__":
    main()
