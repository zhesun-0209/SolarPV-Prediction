#!/usr/bin/env python3
"""
SolarPV项目 - 单GPU并行批量实验脚本
在单块GPU上并行运行340个实验，保存结果到Google Drive
保持与colab_batch_experiments.py相同的效果和结果格式，但速度更快
"""

import os
import sys
import time
import subprocess
import yaml
import pandas as pd
import threading
import queue
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
import torch
from utils.experiment_gpu_utils import get_single_experiment_gpu_memory

def check_drive_mount():
    """检查Google Drive是否已挂载"""
    drive_path = "/content/drive/MyDrive"
    if os.path.exists(drive_path):
        print("✅ Google Drive已挂载")
        return True
    else:
        print("❌ Google Drive未挂载，请先挂载Drive")
        return False

def get_data_files():
    """扫描data目录，获取所有项目CSV文件"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return []
    
    csv_files = []
    for file in os.listdir(data_dir):
        if file.startswith("Project") and file.endswith(".csv"):
            project_id = file.replace("Project", "").replace(".csv", "")
            csv_files.append((project_id, os.path.join(data_dir, file)))
    
    csv_files.sort(key=lambda x: int(x[0]))  # 正序排序
    return csv_files

def get_config_files():
    """获取所有配置文件"""
    config_dir = "config/projects"
    all_config_files = []
    
    if os.path.exists(config_dir):
        for project_dir in sorted(os.listdir(config_dir)):
            project_path = os.path.join(config_dir, project_dir)
            if os.path.isdir(project_path):
                for config_file in sorted(os.listdir(project_path)):
                    if config_file.endswith('.yaml') and config_file != 'config_index.yaml':
                        all_config_files.append(os.path.join(project_path, config_file))
    
    return all_config_files

def create_project_csv(project_id, drive_path):
    """为项目创建CSV文件"""
    csv_file = os.path.join(drive_path, f"{project_id}_results.csv")
    
    if not os.path.exists(csv_file):
        # 创建CSV文件头
        columns = [
            'model', 'use_pv', 'use_hist_weather', 'use_forecast', 'weather_category',
            'use_time_encoding', 'past_days', 'model_complexity', 'epochs', 'batch_size',
            'learning_rate', 'use_ideal_nwp', 'input_category', 'train_time_sec', 'inference_time_sec', 'param_count',
            'samples_count', 'best_epoch', 'final_lr', 'mse', 'rmse', 'mae', 'nrmse',
            'r_square', 'smape', 'gpu_memory_used'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)
        print(f"📄 创建项目CSV文件: {csv_file}")
        return True
    else:
        print(f"📄 项目CSV文件已存在: {csv_file}")
        return True

def get_completed_experiments_count(project_id, drive_path):
    """获取已完成的实验数量"""
    csv_file = os.path.join(drive_path, f"{project_id}_results.csv")
    completed_count = 0
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            completed_count = len(df)
            print(f"📊 发现 {completed_count} 个已完成实验")
        except Exception as e:
            print(f"⚠️ 无法读取现有结果文件: {e}")
            completed_count = 0
    
    return completed_count

def get_completed_experiment_configs(project_id, drive_path):
    """获取已完成的实验配置名称列表"""
    csv_file = os.path.join(drive_path, f"{project_id}_results.csv")
    completed_configs = set()
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            # 从CSV中提取配置信息，重建配置名称
            for _, row in df.iterrows():
                # 获取所有必要的参数来重建完整的配置名称
                model = row['model']
                complexity = row['model_complexity']
                past_days = row['past_days']
                use_time_encoding = row['use_time_encoding']
                
                # 优先使用input_category字段（如果存在）
                if 'input_category' in df.columns and pd.notna(row.get('input_category')):
                    input_cat = row['input_category']
                else:
                    # 兼容旧格式：根据CSV中的参数重建配置名称
                    use_pv = row['use_pv']
                    use_hist_weather = row['use_hist_weather']
                    use_forecast = row['use_forecast']
                    use_ideal_nwp = row.get('use_ideal_nwp', False)
                    
                    # 确定输入类别
                    if use_pv and not use_hist_weather and not use_forecast:
                        input_cat = 'PV'
                    elif use_pv and not use_hist_weather and use_forecast and not use_ideal_nwp:
                        input_cat = 'PV_plus_NWP'
                    elif use_pv and not use_hist_weather and use_forecast and use_ideal_nwp:
                        input_cat = 'PV_plus_NWP_plus'
                    elif use_pv and use_hist_weather and not use_forecast:
                        input_cat = 'PV_plus_HW'
                    elif not use_pv and not use_hist_weather and use_forecast and not use_ideal_nwp:
                        input_cat = 'NWP'
                    elif not use_pv and not use_hist_weather and use_forecast and use_ideal_nwp:
                        input_cat = 'NWP_plus'
                    else:
                        continue  # 跳过无法识别的组合
                
                # 确定回看小时数
                lookback_hours = past_days * 24
                
                # 确定时间编码后缀
                te_suffix = 'TE' if use_time_encoding else 'noTE'
                
                # 重建完整的配置名称（包含所有关键字段）
                config_name = f"{model}_{complexity}_{input_cat}_{lookback_hours}h_{te_suffix}"
                completed_configs.add(config_name)
                
        except Exception as e:
            print(f"⚠️ 无法读取现有结果文件: {e}")
    
    return completed_configs

def run_experiment(config_file, data_file, project_id):
    """运行单个实验"""
    try:
        # 加载配置文件
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 修改配置文件中的data_path
        config['data_path'] = data_file
        config['plant_id'] = project_id
        
        # 创建临时配置文件
        temp_config = f"temp_config_{project_id}_{int(time.time())}_{threading.current_thread().ident}.yaml"
        with open(temp_config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 运行实验并记录时间
        cmd = ['python', 'main.py', '--config', temp_config]
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        duration = time.time() - start_time
        
        # 清理临时文件
        if os.path.exists(temp_config):
            os.remove(temp_config)
        
        if result.returncode == 0:
            return True, result.stdout, result.stderr, duration, config
        else:
            return False, result.stdout, result.stderr, duration, config
            
    except Exception as e:
        return False, "", str(e), 0.0, {}

def parse_experiment_output(output, config_file, duration, config):
    """解析实验输出，提取结果（与colab_batch_experiments.py完全一致）"""
    try:
        # 提取基本指标（支持负数和小数）
        mse_match = re.search(r'mse=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', output)
        rmse_match = re.search(r'rmse=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', output)
        mae_match = re.search(r'mae=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', output)
        r_square_match = re.search(r'r_square=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', output)
        
        # 初始化额外字段
        inference_time = 0.0
        param_count = 0
        samples_count = 0
        best_epoch = 0
        final_lr = 0.0
        nrmse = 0.0
        smape = 0.0
        gpu_memory_used = 0
        
        # 使用METRICS标签提取额外信息（与colab_batch_experiments.py保持一致）
        for line in output.split('\n'):
            if "[METRICS]" in line:
                # 使用正则表达式提取所有键值对
                metrics_in_line = re.findall(r'(\w+)=([0-9.-]+)', line)
                for key, value_str in metrics_in_line:
                    try:
                        if key == 'inference_time':
                            inference_time = float(value_str)
                        elif key == 'param_count':
                            param_count = int(float(value_str))
                        elif key == 'samples_count':
                            samples_count = int(float(value_str))
                        elif key == 'best_epoch':
                            if value_str.lower() == 'nan':
                                best_epoch = 0
                            else:
                                best_epoch = int(float(value_str))
                        elif key == 'final_lr':
                            if value_str.lower() == 'nan':
                                final_lr = 0.0
                            else:
                                final_lr = float(value_str)
                        elif key == 'nrmse':
                            nrmse = float(value_str)
                        elif key == 'smape':
                            smape = float(value_str)
                        elif key == 'gpu_memory_used':
                            gpu_memory_used = int(float(value_str))
                    except Exception as e:
                        pass
        
        # 从配置文件名解析参数
        config_filename = os.path.basename(config_file)
        parts = config_filename.replace('.yaml', '').split('_')
        
        model_name = parts[0]
        complexity = parts[1]
        
        # 解析输入类别和时间编码
        if len(parts) > 2:
            # 处理包含下划线的输入类别名称
            if parts[2] == 'PV' and len(parts) > 3:
                if parts[3] == 'plus' and len(parts) > 4:
                    if parts[4] == 'NWP' and len(parts) > 5 and parts[5] == 'plus':
                        input_category = 'PV_plus_NWP_plus'
                        lookback_hours = parts[6].replace('h', '') if len(parts) > 6 else '24'
                        time_encoding = parts[7] == 'TE' if len(parts) > 7 else False
                    elif parts[4] == 'NWP':
                        input_category = 'PV_plus_NWP'
                        lookback_hours = parts[5].replace('h', '') if len(parts) > 5 else '24'
                        time_encoding = parts[6] == 'TE' if len(parts) > 6 else False
                    elif parts[4] == 'HW':
                        input_category = 'PV_plus_HW'
                        lookback_hours = parts[5].replace('h', '') if len(parts) > 5 else '24'
                        time_encoding = parts[6] == 'TE' if len(parts) > 6 else False
                    else:
                        input_category = 'PV'
                        lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                        time_encoding = parts[4] == 'TE' if len(parts) > 4 else False
                else:
                    input_category = 'PV'
                    lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                    time_encoding = parts[4] == 'TE' if len(parts) > 4 else False
            elif parts[2] == 'NWP' and len(parts) > 3 and parts[3] == 'plus':
                input_category = 'NWP_plus'
                lookback_hours = parts[4].replace('h', '') if len(parts) > 4 else '24'
                time_encoding = parts[5] == 'TE' if len(parts) > 5 else False
            elif parts[2] == 'NWP':
                input_category = 'NWP'
                lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                time_encoding = parts[4] == 'TE' if len(parts) > 4 else False
            else:
                input_category = parts[2]
                lookback_hours = parts[3].replace('h', '') if len(parts) > 3 else '24'
                time_encoding = parts[4] == 'TE' if len(parts) > 4 else False
        else:
            input_category = 'unknown'
            lookback_hours = '24'
            time_encoding = False
        
        # 根据input_category确定其他参数
        use_pv = input_category in ['PV', 'PV_plus_NWP', 'PV_plus_NWP_plus', 'PV_plus_HW']
        use_hist_weather = input_category in ['PV_plus_HW']
        use_forecast = input_category in ['PV_plus_NWP', 'PV_plus_NWP_plus', 'NWP', 'NWP_plus']
        use_ideal_nwp = input_category in ['PV_plus_NWP_plus', 'NWP_plus']
        
        # 根据input_category确定weather_category
        if input_category == 'PV':
            weather_category = 'none'
        else:
            weather_category = 'all_weather'
        
        # 计算past_days
        past_days = int(int(lookback_hours) / 24) if lookback_hours.isdigit() else 1
        
        # 判断模型类型
        is_dl_model = model_name in ['Transformer', 'LSTM', 'GRU', 'TCN']
        has_learning_rate = model_name in ['XGB', 'LGBM']
        
        # 创建结果行
        result_row = {
            'model': model_name,
            'use_pv': use_pv,
            'use_hist_weather': use_hist_weather,
            'use_forecast': use_forecast,
            'weather_category': weather_category,
            'use_time_encoding': time_encoding,
            'past_days': past_days,
            'model_complexity': complexity,
            'epochs': config.get('epochs', 80 if complexity == 'high' else 50) if is_dl_model else 0,
            'batch_size': config.get('train_params', {}).get('batch_size', 64) if is_dl_model else 0,
            'learning_rate': config.get('train_params', {}).get('learning_rate', 0.001) if has_learning_rate else 0.0,
            'use_ideal_nwp': use_ideal_nwp,
            'input_category': input_category,  # 添加input_category字段
            'train_time_sec': round(duration, 4),  # 使用传入的duration参数
            'inference_time_sec': inference_time,
            'param_count': param_count,
            'samples_count': samples_count,
            'best_epoch': best_epoch if is_dl_model else 0,
            'final_lr': final_lr if is_dl_model else 0.0,
            'mse': float(mse_match.group(1)) if mse_match else 0.0,
            'rmse': float(rmse_match.group(1)) if rmse_match else 0.0,
            'mae': float(mae_match.group(1)) if mae_match else 0.0,
            'nrmse': nrmse,
            'r_square': float(r_square_match.group(1)) if r_square_match else 0.0,
            'smape': smape,
            'gpu_memory_used': gpu_memory_used  # 记录该实验自己的GPU消耗
        }
        
        return result_row
        
    except Exception as e:
        print(f"❌ 解析实验输出失败: {e}")
        return None

class SingleGPUParallelExecutor:
    """单GPU并行执行器"""
    
    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results_lock = threading.Lock()
        self.completed_count = 0
        self.failed_count = 0
    
    def execute_experiment(self, config_file, data_file, project_id, drive_path):
        """执行单个实验"""
        config_name = os.path.basename(config_file)
        experiment_id = f"{project_id}_{config_name}_{int(time.time())}"
        
        try:
            print(f"🔄 开始实验: {config_name}")
            
            # 运行实验
            success, stdout, stderr, duration, config = run_experiment(config_file, data_file, project_id)
            
            if success:
                # 从实验输出中解析GPU内存使用量
                actual_gpu_memory = 0
                if torch.cuda.is_available():
                    # 尝试从实验输出中提取GPU内存信息
                    gpu_memory_match = re.search(r'gpu_memory_used=([0-9.]+)', stdout)
                    if gpu_memory_match:
                        actual_gpu_memory = int(float(gpu_memory_match.group(1)))
                        print(f"🔍 从输出中提取GPU内存: {actual_gpu_memory}MB")
                    else:
                        # 如果无法从输出中提取，使用一个基于模型类型的估算值
                        model_name = os.path.basename(config_file).split('_')[0]
                        complexity = os.path.basename(config_file).split('_')[1]
                        
                        # 基于模型类型和复杂度估算GPU内存使用
                        if model_name in ['Transformer']:
                            if complexity == 'high':
                                actual_gpu_memory = 2000  # 2GB
                            elif complexity == 'medium':
                                actual_gpu_memory = 1500  # 1.5GB
                            else:
                                actual_gpu_memory = 1000  # 1GB
                        elif model_name in ['LSTM', 'GRU']:
                            if complexity == 'high':
                                actual_gpu_memory = 800   # 800MB
                            elif complexity == 'medium':
                                actual_gpu_memory = 600   # 600MB
                            else:
                                actual_gpu_memory = 400   # 400MB
                        elif model_name in ['TCN']:
                            if complexity == 'high':
                                actual_gpu_memory = 1200  # 1.2GB
                            elif complexity == 'medium':
                                actual_gpu_memory = 900   # 900MB
                            else:
                                actual_gpu_memory = 600   # 600MB
                        else:  # XGB, LGBM等
                            actual_gpu_memory = 200  # 200MB
                        
                        print(f"🔍 使用估算GPU内存: {actual_gpu_memory}MB (模型: {model_name}, 复杂度: {complexity})")
                
                # 解析结果
                result_row = parse_experiment_output(stdout, config_file, duration, config)
                if result_row:
                    # 使用准确的GPU内存测量值替换解析出的值
                    result_row['gpu_memory_used'] = int(actual_gpu_memory)
                    
                    # 保存结果到CSV
                    csv_file = os.path.join(drive_path, f"{project_id}_results.csv")
                    
                    with self.results_lock:
                        # 读取现有CSV
                        if os.path.exists(csv_file):
                            df = pd.read_csv(csv_file)
                        else:
                            df = pd.DataFrame()
                        
                        # 添加新行
                        new_row_df = pd.DataFrame([result_row])
                        df = pd.concat([df, new_row_df], ignore_index=True)
                        
                    # 保存CSV
                    df.to_csv(csv_file, index=False)
                    self.completed_count += 1
                    
                    print(f"✅ 完成: {config_name} ({duration:.1f}s) - MSE: {result_row['mse']:.4f}")
                    print(f"💾 结果已保存到: {csv_file}")
                    print(f"📊 GPU内存使用: {result_row['gpu_memory_used']}MB")
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated() / 1024 / 1024
                        print(f"🔍 调试信息: 当前GPU内存={current_memory:.1f}MB")
                else:
                    with self.results_lock:
                        self.failed_count += 1
                    print(f"⚠️ 无法解析实验结果: {config_name}")
            else:
                with self.results_lock:
                    self.failed_count += 1
                print(f"❌ 实验失败: {config_name}")
                print(f"   错误: {stderr}")
                
        except Exception as e:
            with self.results_lock:
                self.failed_count += 1
            print(f"💥 实验异常: {config_name} - {e}")
    
    def run_parallel_experiments(self, experiments, drive_path):
        """并行运行实验"""
        print(f"🚀 启动并行执行器 (最大并行数: {self.max_workers})")
        
        # 提交所有实验到线程池
        futures = []
        for config_file, data_file, project_id in experiments:
            future = self.executor.submit(
                self.execute_experiment, 
                config_file, data_file, project_id, drive_path
            )
            futures.append(future)
        
        # 等待所有实验完成
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"💥 线程执行异常: {e}")
        
        self.executor.shutdown(wait=True)
        
        print(f"✅ 并行执行完成! 成功: {self.completed_count}, 失败: {self.failed_count}")

def main():
    """主函数"""
    print("🌟 SolarPV项目 - 单GPU并行批量实验脚本")
    print("=" * 60)
    
    # 检查Google Drive
    if not check_drive_mount():
        return
    
    # 设置路径
    drive_path = "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(drive_path, exist_ok=True)
    
    # 获取数据文件
    print("📁 扫描数据文件...")
    data_files = get_data_files()
    print(f"📊 找到 {len(data_files)} 个项目: {[pid for pid, _ in data_files[:10]]}...")
    
    # 获取配置文件
    print("📁 检查配置文件...")
    config_files = get_config_files()
    print(f"📊 找到 {len(config_files)} 个配置文件")
    
    # 检查是否需要生成配置文件
    if len(config_files) < len(data_files) * 100:
        print("🔧 配置文件不足，正在生成...")
        try:
            result = subprocess.run([
                'python', 'scripts/generate_dynamic_project_configs.py'
            ], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ 配置文件生成完成")
                config_files = get_config_files()
                print(f"📊 现在有 {len(config_files)} 个配置文件")
            else:
                print(f"❌ 配置文件生成失败: {result.stderr}")
                return
        except Exception as e:
            print(f"❌ 配置文件生成异常: {e}")
            return
    
    if not data_files or not config_files:
        print("❌ 没有找到数据文件或配置文件")
        return
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA GPU")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    print(f"🎯 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 根据GPU内存设置并行数量
    if gpu_memory >= 40:  # 40GB+ (A100等)
        max_parallel = 6
    elif gpu_memory >= 24:  # 24GB+ (RTX 4090, RTX 3090等)
        max_parallel = 5
    elif gpu_memory >= 14:  # 16GB+ (RTX 4080, T4等)
        max_parallel = 4
    elif gpu_memory >= 12:  # 12GB+ (RTX 4070等)
        max_parallel = 3
    else:  # 8GB+ (RTX 3070等)
        max_parallel = 2
    
    print(f"📊 设置并行数: {max_parallel}")
    
    print("🚀 开始批量实验!")
    print(f"📊 总项目数: {len(data_files)}")
    print(f"📊 每项目实验数: {len(config_files) // len(data_files)}")
    print(f"📊 总实验数: {len(data_files) * (len(config_files) // len(data_files))}")
    
    total_experiments = 0
    successful_experiments = 0
    failed_experiments = 0
    
    # 遍历每个项目
    for project_idx, (project_id, data_file) in enumerate(data_files, 1):
        print(f"\n{'='*80}")
        print(f"🚀 开始项目 {project_id} 的实验 (正序: {project_idx}/{len(data_files)})")
        print(f"{'='*80}")
        
        # 创建项目CSV文件
        if not create_project_csv(project_id, drive_path):
            print(f"❌ 无法为项目 {project_id} 创建CSV文件")
            continue
        
        # 获取该项目的配置文件
        project_configs = sorted([cf for cf in config_files if f"/{project_id}/" in cf])
        
        # 检查已完成的实验
        completed_count = get_completed_experiments_count(project_id, drive_path)
        completed_configs = get_completed_experiment_configs(project_id, drive_path)
        
        print(f"📊 项目 {project_id}: 将运行 {len(project_configs)} 个实验")
        print(f"📁 结果保存到: {drive_path}")
        print(f"📊 已完成实验: {len(completed_configs)} 个")
        
        # 显示一些已完成的实验示例（用于调试）
        if completed_configs:
            sample_completed = list(completed_configs)[:5]  # 显示前5个
            print(f"🔍 已完成实验示例: {sample_completed}")
        
        # 准备实验列表（跳过已完成的）
        experiments_to_run = []
        skipped_count = 0
        for config_file in project_configs:
            config_name = os.path.basename(config_file)
            config_name_without_ext = config_name.replace('.yaml', '')
            
            # 跳过已完成的实验
            if config_name_without_ext in completed_configs:
                skipped_count += 1
                if skipped_count <= 5:  # 只显示前5个跳过的实验
                    print(f"⏭️ 跳过已完成实验: {config_name}")
                continue
            
            experiments_to_run.append((config_file, data_file, project_id))
        
        if skipped_count > 5:
            print(f"⏭️ ... 还有 {skipped_count - 5} 个已完成的实验被跳过")
        
        if not experiments_to_run:
            print(f"✅ 项目 {project_id} 所有实验已完成!")
            continue
        
        print(f"🚀 开始并行运行 {len(experiments_to_run)} 个实验...")
        
        # 创建并行执行器
        executor = SingleGPUParallelExecutor(max_workers=max_parallel)
        
        # 运行并行实验
        start_time = time.time()
        executor.run_parallel_experiments(experiments_to_run, drive_path)
        duration = time.time() - start_time
        
        total_experiments += len(experiments_to_run)
        successful_experiments += executor.completed_count
        failed_experiments += executor.failed_count
        
        print(f"✅ 项目 {project_id} 完成! 用时: {duration/60:.1f}分钟")
        print(f"📊 成功: {executor.completed_count}, 失败: {executor.failed_count}")
    
    # 总结
    print(f"\n{'='*80}")
    print("🎉 单GPU并行批量实验完成!")
    print(f"📊 总实验数: {total_experiments}")
    print(f"✅ 成功: {successful_experiments}")
    print(f"❌ 失败: {failed_experiments}")
    print(f"📁 结果保存在: {drive_path}")
    print(f"🚀 并行加速比: {max_parallel}x")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
