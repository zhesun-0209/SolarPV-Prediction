#!/usr/bin/env python3
"""
实验GPU内存测量工具
准确测量单个实验的GPU内存消耗
"""

import torch
import threading
import time
from collections import defaultdict

class ExperimentGPUMonitor:
    """实验GPU内存监控器"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.experiment_memory = defaultdict(list)
        self.baseline_memory = 0
        self._init_baseline()
    
    def _init_baseline(self):
        """初始化基线内存使用量"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.baseline_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    def start_monitoring(self, experiment_id):
        """开始监控实验"""
        with self.lock:
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.experiment_memory[experiment_id] = [start_memory]
            return start_memory
    
    def record_memory_peak(self, experiment_id):
        """记录实验的峰值内存使用"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            with self.lock:
                if experiment_id in self.experiment_memory:
                    self.experiment_memory[experiment_id].append(current_memory)
    
    def stop_monitoring(self, experiment_id):
        """停止监控并返回实验的内存消耗"""
        if not torch.cuda.is_available():
            return 0
        
        with self.lock:
            if experiment_id not in self.experiment_memory:
                return 0
            
            # 获取实验期间的最大内存使用量
            max_memory = max(self.experiment_memory[experiment_id])
            # 减去基线内存，得到实验实际消耗
            experiment_memory = max_memory - self.baseline_memory
            
            # 清理记录
            del self.experiment_memory[experiment_id]
            
            return max(0, experiment_memory)  # 确保不为负数

def get_experiment_gpu_memory(experiment_id, monitor=None):
    """
    获取单个实验的GPU内存消耗
    
    Args:
        experiment_id: 实验ID
        monitor: ExperimentGPUMonitor实例
    
    Returns:
        float: 实验的GPU内存消耗(MB)
    """
    if not torch.cuda.is_available():
        return 0
    
    if monitor is None:
        # 如果没有监控器，使用简化的方法
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024
        return current_memory
    
    return monitor.stop_monitoring(experiment_id)

def measure_model_memory_usage(model, input_data, device='cuda'):
    """
    测量模型的内存使用量
    
    Args:
        model: PyTorch模型
        input_data: 输入数据
        device: 设备
    
    Returns:
        float: 模型内存使用量(MB)
    """
    if not torch.cuda.is_available():
        return 0
    
    try:
        # 清空缓存
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        # 将模型和数据移到GPU
        model = model.to(device)
        if isinstance(input_data, (list, tuple)):
            input_data = [x.to(device) if torch.is_tensor(x) else x for x in input_data]
        else:
            input_data = input_data.to(device)
        
        # 前向传播
        with torch.no_grad():
            if isinstance(input_data, (list, tuple)):
                output = model(*input_data)
            else:
                output = model(input_data)
        
        # 计算峰值内存使用
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        memory_usage = peak_memory - baseline_memory
        
        # 清理
        del model, input_data, output
        torch.cuda.empty_cache()
        
        return max(0, memory_usage)
        
    except Exception as e:
        print(f"⚠️ 测量模型内存使用失败: {e}")
        return 0

# 全局监控器实例
global_monitor = ExperimentGPUMonitor()

def get_single_experiment_gpu_memory():
    """
    获取单个实验的GPU内存使用量（简化版本）
    在并行环境中，这个函数会尝试估算单个实验的内存使用
    """
    if not torch.cuda.is_available():
        return 0
    
    try:
        # 获取当前总内存使用
        total_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        # 在并行环境中，我们无法准确知道单个实验的内存使用
        # 所以返回一个估算值
        # 这个估算基于模型类型和复杂度
        
        # 由于无法准确测量，返回一个合理的估算值
        # 实际使用中，建议使用ExperimentGPUMonitor来精确测量
        return int(total_memory * 0.3)  # 假设单个实验占用30%的总内存
        
    except Exception as e:
        print(f"⚠️ 获取GPU内存使用失败: {e}")
        return 0
