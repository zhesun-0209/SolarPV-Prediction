#!/usr/bin/env python3
"""
GPU优化器 - 针对A100等高端GPU的优化策略
动态调整并行数和内存使用
"""

import os
import sys
import time
import torch
import psutil
import subprocess
import logging
from pathlib import Path
import threading
import queue

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """GPU优化器"""
    
    def __init__(self):
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_memory_total = []
        self.gpu_memory_used = []
        
        if self.gpu_count > 0:
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3  # GB
                self.gpu_memory_total.append(total_memory)
                self.gpu_memory_used.append(0)
        
        logger.info(f"🎯 GPU优化器初始化: {self.gpu_count} 个GPU")
        for i, memory in enumerate(self.gpu_memory_total):
            logger.info(f"   GPU {i}: {memory:.1f}GB")
    
    def get_optimal_gpu_parallel_count(self, model_type='mixed'):
        """获取最优GPU并行数"""
        if self.gpu_count == 0:
            return 0
        
        # 获取当前GPU使用情况
        current_usage = self.get_gpu_memory_usage()
        
        # 根据GPU类型和内存使用情况计算最优并行数
        if self.gpu_memory_total[0] >= 80:  # A100 80GB
            base_parallel = 16
            memory_factor = 0.8
        elif self.gpu_memory_total[0] >= 40:  # A100 40GB 或其他高端GPU
            base_parallel = 12
            memory_factor = 0.7
        elif self.gpu_memory_total[0] >= 24:  # RTX 3090/4090等
            base_parallel = 8
            memory_factor = 0.6
        else:  # 其他GPU
            base_parallel = 4
            memory_factor = 0.5
        
        # 根据模型类型调整
        if model_type == 'transformer':
            model_factor = 0.8  # Transformer需要更多内存
        elif model_type == 'lstm':
            model_factor = 1.0  # LSTM内存需求中等
        elif model_type == 'tcn':
            model_factor = 1.2  # TCN内存需求较小
        else:  # mixed
            model_factor = 1.0
        
        # 根据当前内存使用情况调整
        if current_usage[0] > 0.8:  # 内存使用超过80%
            memory_factor *= 0.5
        elif current_usage[0] > 0.6:  # 内存使用超过60%
            memory_factor *= 0.7
        elif current_usage[0] > 0.4:  # 内存使用超过40%
            memory_factor *= 0.85
        
        optimal_parallel = max(1, int(base_parallel * memory_factor * model_factor))
        
        logger.info(f"🎯 最优GPU并行数: {optimal_parallel} (内存使用: {current_usage[0]:.1%})")
        return optimal_parallel
    
    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if self.gpu_count == 0:
            return [0]
        
        usage = []
        for i in range(self.gpu_count):
            try:
                # 使用nvidia-ml-py或nvidia-smi获取内存使用情况
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=memory.used,memory.total',
                    '--format=csv,noheader,nounits', f'--id={i}'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    used, total = map(int, result.stdout.strip().split(', '))
                    usage.append(used / total)
                else:
                    usage.append(0)
            except:
                usage.append(0)
        
        return usage
    
    def optimize_training_params(self, config, gpu_id=0):
        """优化训练参数"""
        if self.gpu_count == 0:
            return config
        
        # 根据GPU内存调整批处理大小
        gpu_memory = self.gpu_memory_total[gpu_id]
        current_batch_size = config.get('train_params', {}).get('batch_size', 32)
        
        if gpu_memory >= 80:  # A100 80GB
            optimal_batch_size = min(128, current_batch_size * 4)
        elif gpu_memory >= 40:  # A100 40GB
            optimal_batch_size = min(96, current_batch_size * 3)
        elif gpu_memory >= 24:  # RTX 3090/4090
            optimal_batch_size = min(64, current_batch_size * 2)
        else:
            optimal_batch_size = current_batch_size
        
        # 更新配置
        if 'train_params' not in config:
            config['train_params'] = {}
        
        config['train_params']['batch_size'] = optimal_batch_size
        
        # 启用混合精度训练
        config['train_params']['use_amp'] = True
        
        # 优化学习率
        base_lr = config['train_params'].get('learning_rate', 5e-4)
        if optimal_batch_size > current_batch_size:
            # 线性缩放学习率
            config['train_params']['learning_rate'] = base_lr * (optimal_batch_size / current_batch_size)
        
        # 设置GPU特定优化
        config['train_params']['gpu_optimizations'] = {
            'use_amp': True,
            'gradient_accumulation_steps': max(1, optimal_batch_size // 32),
            'dataloader_num_workers': min(8, os.cpu_count() // self.gpu_count),
            'pin_memory': True
        }
        
        logger.info(f"🎯 GPU {gpu_id} 优化: 批处理大小 {current_batch_size} -> {optimal_batch_size}")
        return config
    
    def monitor_gpu_usage(self, interval=30):
        """监控GPU使用情况"""
        def monitor_loop():
            while True:
                try:
                    usage = self.get_gpu_memory_usage()
                    for i, use in enumerate(usage):
                        if use > 0.9:  # 内存使用超过90%
                            logger.warning(f"⚠️ GPU {i} 内存使用过高: {use:.1%}")
                        elif use > 0.8:  # 内存使用超过80%
                            logger.info(f"📊 GPU {i} 内存使用: {use:.1%}")
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"GPU监控异常: {e}")
                    time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def get_system_recommendations(self):
        """获取系统优化建议"""
        recommendations = []
        
        # GPU建议
        if self.gpu_count > 0:
            if self.gpu_memory_total[0] >= 80:
                recommendations.append("🚀 A100 80GB检测到，建议GPU并行数: 16")
                recommendations.append("💡 可以启用大批次训练和混合精度")
            elif self.gpu_memory_total[0] >= 40:
                recommendations.append("🎯 A100 40GB检测到，建议GPU并行数: 12")
                recommendations.append("💡 可以启用中等批次训练")
            else:
                recommendations.append(f"💻 GPU检测到，建议GPU并行数: {min(8, self.gpu_count * 2)}")
        
        # CPU建议
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024**3
        
        if memory_gb >= 128:
            recommendations.append(f"💻 大内存系统，建议CPU并行数: {min(32, cpu_count * 2)}")
        elif memory_gb >= 64:
            recommendations.append(f"💻 中等内存系统，建议CPU并行数: {min(24, cpu_count)}")
        else:
            recommendations.append(f"💻 小内存系统，建议CPU并行数: {min(16, cpu_count // 2)}")
        
        # 环境变量建议
        recommendations.append("🔧 建议设置环境变量:")
        recommendations.append("   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
        recommendations.append("   export PYTORCH_CUDNN_V8_API_ENABLED=1")
        
        return recommendations

def test_gpu_optimizer():
    """测试GPU优化器"""
    optimizer = GPUOptimizer()
    
    print("🎯 GPU优化器测试")
    print("=" * 50)
    
    # 测试并行数计算
    optimal_parallel = optimizer.get_optimal_gpu_parallel_count()
    print(f"最优GPU并行数: {optimal_parallel}")
    
    # 测试内存使用情况
    usage = optimizer.get_gpu_memory_usage()
    print(f"GPU内存使用: {[f'{u:.1%}' for u in usage]}")
    
    # 测试训练参数优化
    test_config = {
        'train_params': {
            'batch_size': 32,
            'learning_rate': 5e-4
        }
    }
    
    optimized_config = optimizer.optimize_training_params(test_config)
    print(f"优化后批处理大小: {optimized_config['train_params']['batch_size']}")
    print(f"优化后学习率: {optimized_config['train_params']['learning_rate']}")
    
    # 获取系统建议
    recommendations = optimizer.get_system_recommendations()
    print("\n📋 系统优化建议:")
    for rec in recommendations:
        print(f"   {rec}")
    
    # 启动监控
    print("\n📊 启动GPU监控...")
    monitor_thread = optimizer.monitor_gpu_usage(interval=5)
    
    try:
        time.sleep(20)  # 监控20秒
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")

if __name__ == "__main__":
    test_gpu_optimizer()
