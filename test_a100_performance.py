#!/usr/bin/env python3
"""
A100性能测试脚本
快速测试系统性能并给出优化建议
"""

import torch
import time
import psutil
import subprocess
import sys
from pathlib import Path

def test_gpu_performance():
    """测试GPU性能"""
    print("🎯 GPU性能测试")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        if "A100" in props.name:
            print("   🚀 A100检测到！启用高性能模式")
            return True
    
    print("⚠️ 未检测到A100，将使用标准高性能模式")
    return True

def test_memory_performance():
    """测试内存性能"""
    print("\n💾 内存性能测试")
    print("-" * 40)
    
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1024**3
    
    print(f"总内存: {memory_gb:.1f}GB")
    print(f"可用内存: {memory.available / 1024**3:.1f}GB")
    print(f"内存使用率: {memory.percent:.1f}%")
    
    if memory_gb >= 128:
        print("✅ 大内存系统，支持高并行")
        return "high"
    elif memory_gb >= 64:
        print("✅ 中等内存系统，支持中等并行")
        return "medium"
    else:
        print("⚠️ 小内存系统，建议减少并行数")
        return "low"

def test_cpu_performance():
    """测试CPU性能"""
    print("\n💻 CPU性能测试")
    print("-" * 40)
    
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    print(f"CPU核心数: {cpu_count}")
    if cpu_freq:
        print(f"CPU频率: {cpu_freq.current:.0f}MHz")
    
    # 简单CPU性能测试
    start_time = time.time()
    for i in range(1000000):
        _ = i * i
    cpu_time = time.time() - start_time
    
    print(f"CPU计算性能: {cpu_time:.3f}秒 (100万次乘法)")
    
    if cpu_count >= 32:
        return "high"
    elif cpu_count >= 16:
        return "medium"
    else:
        return "low"

def test_disk_performance():
    """测试磁盘性能"""
    print("\n💿 磁盘性能测试")
    print("-" * 40)
    
    # 测试写入性能
    test_file = Path("test_disk_performance.tmp")
    
    start_time = time.time()
    with open(test_file, 'wb') as f:
        f.write(b'0' * 1024 * 1024)  # 1MB
    write_time = time.time() - start_time
    
    start_time = time.time()
    with open(test_file, 'rb') as f:
        f.read()
    read_time = time.time() - start_time
    
    test_file.unlink()  # 删除测试文件
    
    print(f"写入性能: {1/write_time:.1f}MB/s")
    print(f"读取性能: {1/read_time:.1f}MB/s")
    
    if write_time < 0.01:  # 100MB/s+
        return "high"
    elif write_time < 0.05:  # 20MB/s+
        return "medium"
    else:
        return "low"

def get_optimization_recommendations(gpu_available, memory_level, cpu_level, disk_level):
    """获取优化建议"""
    print("\n🎯 优化建议")
    print("-" * 40)
    
    recommendations = []
    
    # GPU建议
    if gpu_available:
        if memory_level == "high":
            recommendations.append("🚀 GPU并行数: 16")
            recommendations.append("💻 CPU并行数: 32")
            recommendations.append("📦 批处理大小: 128")
        elif memory_level == "medium":
            recommendations.append("🚀 GPU并行数: 12")
            recommendations.append("💻 CPU并行数: 24")
            recommendations.append("📦 批处理大小: 96")
        else:
            recommendations.append("🚀 GPU并行数: 8")
            recommendations.append("💻 CPU并行数: 16")
            recommendations.append("📦 批处理大小: 64")
        
        recommendations.append("⚡ 启用混合精度训练 (AMP)")
        recommendations.append("🔄 启用梯度累积")
    else:
        recommendations.append("💻 CPU并行数: 16")
        recommendations.append("📦 批处理大小: 32")
        recommendations.append("⚠️ 建议使用GPU加速")
    
    # 内存建议
    if memory_level == "low":
        recommendations.append("⚠️ 内存不足，建议减少并行数")
        recommendations.append("💾 启用梯度检查点")
    
    # 磁盘建议
    if disk_level == "low":
        recommendations.append("💿 磁盘性能较低，建议使用SSD")
        recommendations.append("📁 减少数据加载器线程数")
    
    # 环境变量建议
    recommendations.append("\n🔧 环境变量设置:")
    recommendations.append("export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
    recommendations.append("export PYTORCH_CUDNN_V8_API_ENABLED=1")
    recommendations.append("export OMP_NUM_THREADS=16")
    
    for rec in recommendations:
        print(f"   {rec}")

def estimate_completion_time(gpu_available, memory_level, cpu_level):
    """估算完成时间"""
    print("\n⏰ 完成时间预估")
    print("-" * 40)
    
    total_experiments = 36000  # 100个Project × 360个实验
    
    if gpu_available:
        if memory_level == "high":
            gpu_parallel = 16
            cpu_parallel = 32
            avg_time_per_experiment = 120  # 秒
        elif memory_level == "medium":
            gpu_parallel = 12
            cpu_parallel = 24
            avg_time_per_experiment = 150  # 秒
        else:
            gpu_parallel = 8
            cpu_parallel = 16
            avg_time_per_experiment = 200  # 秒
    else:
        cpu_parallel = 16
        avg_time_per_experiment = 300  # 秒
    
    # 计算总时间
    if gpu_available:
        # GPU和CPU并行运行
        gpu_experiments = 14400  # 40% 深度学习模型
        cpu_experiments = 21600  # 60% 传统ML模型
        
        gpu_time = (gpu_experiments * avg_time_per_experiment) / (gpu_parallel * 3600)  # 小时
        cpu_time = (cpu_experiments * avg_time_per_experiment) / (cpu_parallel * 3600)  # 小时
        
        total_hours = max(gpu_time, cpu_time)  # 取最大值，因为它们并行运行
    else:
        total_hours = (total_experiments * avg_time_per_experiment) / (cpu_parallel * 3600)
    
    total_days = total_hours / 24
    
    print(f"总实验数: {total_experiments:,}")
    if gpu_available:
        print(f"GPU并行数: {gpu_parallel}")
        print(f"CPU并行数: {cpu_parallel}")
    else:
        print(f"CPU并行数: {cpu_parallel}")
    
    print(f"预计完成时间: {total_hours:.0f} 小时 ({total_days:.1f} 天)")
    print(f"实验速度: {total_experiments / total_hours:.1f} 实验/小时")

def main():
    """主函数"""
    print("🚀 Project1140 A100性能测试")
    print("=" * 50)
    
    # 运行各项测试
    gpu_available = test_gpu_performance()
    memory_level = test_memory_performance()
    cpu_level = test_cpu_performance()
    disk_level = test_disk_performance()
    
    # 获取优化建议
    get_optimization_recommendations(gpu_available, memory_level, cpu_level, disk_level)
    
    # 估算完成时间
    estimate_completion_time(gpu_available, memory_level, cpu_level)
    
    print("\n" + "=" * 50)
    print("🎯 推荐启动命令:")
    
    if gpu_available:
        print("   ./run_a100_optimized.sh")
        print("   # 或")
        print("   python scripts/run_high_performance_experiments.py")
    else:
        print("   ./run_100_projects.sh")
        print("   # 或")
        print("   python scripts/run_multi_project_experiments.py")
    
    print("\n📊 性能监控:")
    print("   ./monitor_a100_experiment.sh")
    
    print("\n🔍 详细性能分析:")
    print("   python scripts/performance_estimator.py")

if __name__ == "__main__":
    main()
