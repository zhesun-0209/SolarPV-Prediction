#!/usr/bin/env python3
"""
性能预估器 - 估算不同配置下的实验完成时间
"""

import time
import torch
import psutil
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PerformanceEstimator:
    """性能预估器"""
    
    def __init__(self):
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / 1024**3
        
        # 实验配置统计
        self.total_experiments = 36000  # 100个Project × 360个实验
        self.gpu_experiments = 14400    # 40% 深度学习模型 (LSTM, GRU, TCN, Transformer)
        self.cpu_experiments = 21600    # 60% 传统ML模型 (LSR, RF, XGB, LGBM)
        
        # 基准性能数据（基于经验值）
        self.benchmark_times = {
            'gpu': {
                'LSTM': {'low': 180, 'high': 600},      # 秒
                'GRU': {'low': 150, 'high': 500},
                'TCN': {'low': 120, 'high': 400},
                'Transformer': {'low': 300, 'high': 900}
            },
            'cpu': {
                'LSR': {'low': 30, 'high': 30},         # 秒
                'RF': {'low': 60, 'high': 180},
                'XGB': {'low': 45, 'high': 120},
                'LGBM': {'low': 30, 'high': 90}
            }
        }
    
    def estimate_system_capability(self):
        """估算系统能力"""
        capability = {
            'gpu_parallel': 0,
            'cpu_parallel': 0,
            'memory_limitation': False
        }
        
        # GPU并行能力
        if self.gpu_count > 0:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory >= 80:  # A100 80GB
                capability['gpu_parallel'] = min(16, self.gpu_count * 4)
            elif gpu_memory >= 40:  # A100 40GB
                capability['gpu_parallel'] = min(12, self.gpu_count * 3)
            elif gpu_memory >= 24:  # RTX 3090/4090
                capability['gpu_parallel'] = min(8, self.gpu_count * 2)
            else:
                capability['gpu_parallel'] = min(4, self.gpu_count)
        
        # CPU并行能力
        if self.memory_gb >= 128:
            capability['cpu_parallel'] = min(32, self.cpu_count * 2)
        elif self.memory_gb >= 64:
            capability['cpu_parallel'] = min(24, self.cpu_count)
        else:
            capability['cpu_parallel'] = min(16, self.cpu_count // 2)
            capability['memory_limitation'] = True
        
        return capability
    
    def estimate_experiment_times(self, gpu_parallel, cpu_parallel):
        """估算实验时间"""
        times = {}
        
        # GPU实验时间估算
        gpu_models = ['LSTM', 'GRU', 'TCN', 'Transformer']
        gpu_experiments_per_model = self.gpu_experiments // len(gpu_models)
        
        total_gpu_time = 0
        for model in gpu_models:
            low_time = self.benchmark_times['gpu'][model]['low']
            high_time = self.benchmark_times['gpu'][model]['high']
            avg_time = (low_time + high_time) / 2
            
            # 考虑并行加速
            parallel_time = avg_time / gpu_parallel
            model_total_time = (gpu_experiments_per_model * parallel_time) / 3600  # 转换为小时
            
            times[f'gpu_{model}'] = model_total_time
            total_gpu_time += model_total_time
        
        # CPU实验时间估算
        cpu_models = ['LSR', 'RF', 'XGB', 'LGBM']
        cpu_experiments_per_model = self.cpu_experiments // len(cpu_models)
        
        total_cpu_time = 0
        for model in cpu_models:
            low_time = self.benchmark_times['cpu'][model]['low']
            high_time = self.benchmark_times['cpu'][model]['high']
            avg_time = (low_time + high_time) / 2
            
            # 考虑并行加速
            parallel_time = avg_time / cpu_parallel
            model_total_time = (cpu_experiments_per_model * parallel_time) / 3600  # 转换为小时
            
            times[f'cpu_{model}'] = model_total_time
            total_cpu_time += model_total_time
        
        # 总时间（取GPU和CPU的最大值，因为它们并行运行）
        total_time = max(total_gpu_time, total_cpu_time)
        
        times['total_gpu_time'] = total_gpu_time
        times['total_cpu_time'] = total_cpu_time
        times['total_time'] = total_time
        
        return times
    
    def estimate_completion_time(self, start_time=None):
        """估算完成时间"""
        if start_time is None:
            start_time = datetime.now()
        
        capability = self.estimate_system_capability()
        times = self.estimate_experiment_times(
            capability['gpu_parallel'], 
            capability['cpu_parallel']
        )
        
        completion_time = start_time + timedelta(hours=times['total_time'])
        
        return {
            'start_time': start_time,
            'completion_time': completion_time,
            'duration_hours': times['total_time'],
            'duration_days': times['total_time'] / 24,
            'capability': capability,
            'times': times
        }
    
    def compare_strategies(self):
        """比较不同策略的性能"""
        strategies = {
            'Standard': {'gpu_parallel': 4, 'cpu_parallel': 8},
            'High Performance': {'gpu_parallel': 8, 'cpu_parallel': 16},
            'A100 Optimized': {'gpu_parallel': 16, 'cpu_parallel': 32},
            'Maximum': {'gpu_parallel': 24, 'cpu_parallel': 48}
        }
        
        results = {}
        
        for strategy_name, params in strategies.items():
            times = self.estimate_experiment_times(
                params['gpu_parallel'], 
                params['cpu_parallel']
            )
            
            results[strategy_name] = {
                'gpu_parallel': params['gpu_parallel'],
                'cpu_parallel': params['cpu_parallel'],
                'total_hours': times['total_time'],
                'total_days': times['total_time'] / 24,
                'experiments_per_hour': self.total_experiments / times['total_time']
            }
        
        return results
    
    def generate_performance_report(self):
        """生成性能报告"""
        capability = self.estimate_system_capability()
        estimation = self.estimate_completion_time()
        strategies = self.compare_strategies()
        
        report = f"""
# Project1140 性能预估报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 系统配置

- **GPU数量**: {self.gpu_count}
- **CPU核心数**: {self.cpu_count}
- **内存**: {self.memory_gb:.1f}GB

## 系统能力评估

- **GPU并行数**: {capability['gpu_parallel']}
- **CPU并行数**: {capability['cpu_parallel']}
- **内存限制**: {'是' if capability['memory_limitation'] else '否'}

## 实验配置

- **总实验数**: {self.total_experiments:,}
- **GPU实验**: {self.gpu_experiments:,} (深度学习模型)
- **CPU实验**: {self.cpu_experiments:,} (传统ML模型)

## 性能预估

### 当前系统配置
- **预计完成时间**: {estimation['completion_time'].strftime('%Y-%m-%d %H:%M:%S')}
- **总耗时**: {estimation['duration_hours']:.1f} 小时 ({estimation['duration_days']:.1f} 天)
- **实验速度**: {self.total_experiments / estimation['duration_hours']:.1f} 实验/小时

### 详细时间分解

#### GPU实验
"""
        
        for model in ['LSTM', 'GRU', 'TCN', 'Transformer']:
            model_time = estimation['times'][f'gpu_{model}']
            report += f"- **{model}**: {model_time:.1f} 小时\n"
        
        report += f"""
#### CPU实验
"""
        
        for model in ['LSR', 'RF', 'XGB', 'LGBM']:
            model_time = estimation['times'][f'cpu_{model}']
            report += f"- **{model}**: {model_time:.1f} 小时\n"
        
        report += f"""
## 不同策略对比

| 策略 | GPU并行 | CPU并行 | 总耗时(小时) | 总耗时(天) | 实验/小时 |
|------|---------|---------|--------------|------------|-----------|
"""
        
        for strategy_name, result in strategies.items():
            report += f"| {strategy_name} | {result['gpu_parallel']} | {result['cpu_parallel']} | {result['total_hours']:.1f} | {result['total_days']:.1f} | {result['experiments_per_hour']:.1f} |\n"
        
        report += f"""
## 优化建议

### A100优化建议
- 使用混合精度训练 (AMP)
- 启用梯度累积
- 优化数据加载器
- 使用更大的批处理大小

### 内存优化建议
"""
        
        if capability['memory_limitation']:
            report += "- ⚠️ 内存可能成为瓶颈，建议减少并行数\n"
            report += "- 考虑使用梯度检查点\n"
            report += "- 启用数据并行而非模型并行\n"
        else:
            report += "- ✅ 内存充足，可以使用高并行配置\n"
        
        report += f"""
### 性能提升建议
- 使用SSD存储加速数据加载
- 启用CUDA优化
- 使用多GPU训练（如果可用）
- 优化网络配置

## 结论

基于当前系统配置，推荐使用 **A100 Optimized** 策略，预计完成时间为 **{strategies['A100 Optimized']['total_days']:.1f} 天**。

如果系统资源允许，可以考虑 **Maximum** 策略，进一步缩短到 **{strategies['Maximum']['total_days']:.1f} 天**。
"""
        
        return report

def main():
    """主函数"""
    estimator = PerformanceEstimator()
    
    print("🚀 Project1140 性能预估器")
    print("=" * 60)
    
    # 生成性能报告
    report = estimator.generate_performance_report()
    print(report)
    
    # 保存报告
    with open('performance_estimation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📊 性能报告已保存到: performance_estimation_report.md")

if __name__ == "__main__":
    main()
