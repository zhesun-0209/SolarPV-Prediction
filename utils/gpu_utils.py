#!/usr/bin/env python3
"""
GPU工具函数
监控GPU内存使用情况
"""

import os
import subprocess
import psutil

def get_gpu_memory_used():
    """
    获取GPU内存使用量(MB)
    
    Returns:
        float: GPU内存使用量(MB)，如果无法获取则返回0
    """
    try:
        # 尝试使用nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            memory_used = float(result.stdout.strip().split('\n')[0])
            return memory_used
    except:
        pass
    
    try:
        # 尝试使用pynvml
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used = mem_info.used / 1024 / 1024  # 转换为MB
        return memory_used
    except:
        pass
    
    try:
        # 尝试使用torch
        import torch
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
            return memory_used
    except:
        pass
    
    # 如果都无法获取，返回0
    return 0.0

def get_system_memory_used():
    """
    获取系统内存使用量(MB)
    
    Returns:
        float: 系统内存使用量(MB)
    """
    try:
        memory = psutil.virtual_memory()
        return memory.used / 1024 / 1024  # 转换为MB
    except:
        return 0.0
